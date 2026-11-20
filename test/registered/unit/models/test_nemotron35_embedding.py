"""Native Nemotron-3.5 construction and checkpoint dispatch using tiny HF models."""

import unittest
from types import SimpleNamespace

import torch
from transformers import (
    Ministral3Config,
    Mistral3Config,
    Mistral3Model,
    PixtralVisionConfig,
)

from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.models.nemotron35_embedding import EmbeddingNemotron35Model
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _config(tied):
    return Mistral3Config(
        text_config=Ministral3Config(
            vocab_size=128,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            # The 1B checkpoint has a Q width different from its hidden size.
            num_attention_heads=3 if tied else 2,
            num_key_value_heads=1,
            head_dim=128,
            max_position_embeddings=512,
            original_max_position_embeddings=512,
            architectures=["Ministral3Model"],
            is_causal=False,
            pooling="avg",
            tie_word_embeddings=tied,
        ),
        vision_config=PixtralVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            head_dim=32,
            image_size=56,
            patch_size=14,
            hidden_act="silu",
        ),
        image_token_index=10,
        spatial_merge_size=2,
        vision_feature_layer=-1,
        multimodal_projector_bias=False,
        architectures=["Mistral3Model"],
        is_causal=False,
        pooling="avg",
        tie_word_embeddings=tied,
    )


def _native(config):
    pp = SimpleNamespace(
        rank_in_group=0, world_size=1, is_first_rank=True, is_last_rank=True
    )
    with parallel_scope(pp_group=pp), torch.device("cpu"):
        return EmbeddingNemotron35Model(config).eval()


class TestNemotron35Embedding(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy",
                device="cpu",
                tp_size=1,
                attention_backend="torch_native",
                mm_attention_backend="sdpa",
            ),
            role="test",
        )

    def test_both_checkpoint_layouts_load_every_parameter_without_a_separate_head(self):
        """Untied 8B configs must not allocate a random LM head; QKV shards must
        still use explicit head_dim for the unequal 1B hidden and Q widths.
        """
        for tied in (True, False):
            for outer_prefix in ("", "model."):
                with self.subTest(tied=tied, outer_prefix=outer_prefix):
                    config = _config(tied)
                    reference = Mistral3Model(config).eval()
                    model = _native(config)
                    model.load_weights(
                        (outer_prefix + name, value)
                        for name, value in reference.state_dict().items()
                    )
                    self.assertEqual(config.text_config.tie_word_embeddings, tied)
                    self.assertIs(
                        model.language_model.lm_head,
                        model.language_model.model.embed_tokens,
                    )
                    self.assertEqual(
                        sum(p.numel() for p in model.parameters()),
                        sum(p.numel() for p in reference.parameters()),
                    )
                    for layer in model.language_model.model.layers:
                        self.assertEqual(
                            layer.self_attn.attn.attn_type, AttentionType.ENCODER_ONLY
                        )
                    self._assert_checkpoint_values(model, reference)

    def _assert_checkpoint_values(self, model, reference):
        actual = dict(model.named_parameters())
        covered = set()
        for name, expected in reference.state_dict().items():
            target = name
            if name.startswith("language_model."):
                target = "language_model.model." + name[len("language_model.") :]
            tensor = None
            for source_suffix, fused_suffix, index, shards in (
                ("q_proj.weight", "qkv_proj.weight", 0, ("q_proj", "k_proj", "v_proj")),
                ("k_proj.weight", "qkv_proj.weight", 1, ("q_proj", "k_proj", "v_proj")),
                ("v_proj.weight", "qkv_proj.weight", 2, ("q_proj", "k_proj", "v_proj")),
                (
                    "gate_proj.weight",
                    "gate_up_proj.weight",
                    0,
                    ("gate_proj", "up_proj"),
                ),
                ("up_proj.weight", "gate_up_proj.weight", 1, ("gate_proj", "up_proj")),
            ):
                if name.endswith(source_suffix):
                    source_parent = name[: -len(source_suffix)]
                    sizes = [
                        reference.get_parameter(
                            source_parent + shard + ".weight"
                        ).shape[0]
                        for shard in shards
                    ]
                    target = target[: -len(source_suffix)] + fused_suffix
                    tensor = actual[target].split(sizes, dim=0)[index]
                    break
            if tensor is None:
                if name.startswith("vision_tower."):
                    target = target.replace(".attention.o_proj.", ".attention.proj.")
                tensor = actual[target]
            torch.testing.assert_close(tensor, expected, atol=0, rtol=0)
            covered.add(target)
        self.assertEqual(covered, actual.keys())

    def test_unknown_checkpoint_component_is_rejected(self):
        model = _native(_config(False))
        for name in ("lm_head.weight", "multi_modal_projector.wrong.weight"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    ValueError, "Unexpected Nemotron-3.5-Embed weight"
                ),
            ):
                model.load_weights([(name, torch.zeros(1))])

    def test_vision_projector_matches_hf_for_bundled_and_split_images(self):
        """Padding must not add image features or mix two images' attention."""
        config = _config(False)
        config._attn_implementation = "sdpa"
        reference = Mistral3Model(config).eval()
        model = _native(config)
        model.load_weights(reference.state_dict().items())
        pixels = torch.randn(2, 3, 56, 56)
        image_sizes = [(56, 28), (28, 56)]
        with torch.no_grad():
            expected = torch.cat(
                reference.get_image_features(
                    pixels, image_sizes, vision_feature_layer=-1
                ).pooler_output,
                dim=0,
            )
            for items in (
                [SimpleNamespace(feature=pixels.double(), image_sizes=image_sizes)],
                [
                    SimpleNamespace(
                        feature=pixels[i : i + 1].double(), image_sizes=[size]
                    )
                    for i, size in enumerate(image_sizes)
                ],
            ):
                actual = model.get_image_feature(items)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
