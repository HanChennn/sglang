"""Pinned Nemotron-3.5-Embed 1B/8B BF16 accuracy and HTTP input parity.

Run on one NVIDIA H100 (or a CUDA device supporting FA3). Reference weights
are released before launching SGLang. No SGLang preprocessing or pooling code
is used to construct the Hugging Face reference.
"""

import base64
import gc
import io
import json
import unittest

import requests
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=480, stage="extra-a", runner_config="1-gpu-large")


def _image(size, color):
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] // 2, size[1] // 2), fill="white")
    draw.line((0, size[1] - 1, size[0] - 1, 0), fill="black", width=3)
    return image


def _data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class _Nemotron35Accuracy:
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise unittest.SkipTest("requires a CUDA GPU with BF16 support")
        cls.process = None
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.snapshot = snapshot_download(
            cls.model,
            revision=cls.revision,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja"],
        )
        cls.processor = AutoProcessor.from_pretrained(cls.snapshot)
        cls.images = {
            "small": _image((56, 84), "red"),
            "wide": _image((280, 112), "blue"),
            # This exceeds processor longest_edge=1120 but not vision image_size=1540.
            # Using the latter to split multi-image rows corrupts feature placement.
            "tall": _image((560, 1400), "green"),
        }
        cls.urls = {name: _data_url(image) for name, image in cls.images.items()}
        # (task, text, image names). Shared prefixes exercise bidirectional caching;
        # the last long sample exposes packed BF16 mean-pooling cancellation.
        cls.samples = [
            ("query", "What does a GPU do?", []),
            ("document", "A GPU processes many operations in parallel.", []),
            ("document", "The animal in the picture is a red fox.", []),
            ("document", "The animal in the picture is a blue whale.", []),
            ("document", None, ["tall"]),
            ("document", "A blue rectangle with a white corner [detail].", ["wide"]),
            ("document", "Compare the two pictures.", ["small", "tall"]),
            (
                "document",
                "Parallel processors execute independent operations. " * 96,
                [],
            ),
        ]
        cls.prompts = [
            cls.processor.apply_chat_template(
                cls._messages(index), tokenize=False, add_generation_prompt=False
            )
            for index in range(len(cls.samples))
        ]
        cls.references, cls.token_counts = cls._reference()
        gc.collect()
        torch.cuda.empty_cache()
        # Auto-detection is part of the contract: do not pass --is-embedding.
        cls.process = popen_launch_server(
            cls.snapshot,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--served-model-name",
                cls.model,
                "--dtype",
                "bfloat16",
                "--tp",
                "1",
                "--context-length",
                "4096",
                "--mem-fraction-static",
                "0.7",
                "--attention-backend",
                "fa3",
                "--mm-attention-backend",
                "sdpa",
                "--enable-tokenizer-batch-encode",
            ],
        )

    @classmethod
    def _messages(cls, index):
        task, text, images = cls.samples[index]
        # Put text first deliberately: the checkpoint template canonicalizes
        # image-before-text for both the HF and /v2/embed paths.
        content = [] if text is None else [{"type": "text", "text": text}]
        content += [{"type": "image", "image": cls.images[name]} for name in images]
        return [
            {"role": "system", "content": "query:" if task == "query" else "passage:"},
            {"role": "user", "content": content},
        ]

    @classmethod
    def _reference(cls):
        model = AutoModel.from_pretrained(
            cls.snapshot,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        ).eval()
        references, token_counts = [], []
        try:
            with torch.inference_mode():
                for index in range(len(cls.samples)):
                    inputs = cls.processor.apply_chat_template(
                        cls._messages(index),
                        tokenize=True,
                        add_generation_prompt=False,
                        return_dict=True,
                        return_tensors="pt",
                        padding=True,
                    ).to("cuda:0")
                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
                    hidden = model(**inputs).last_hidden_state.float()
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1)
                    references.append(F.normalize(pooled, dim=-1)[0].cpu())
                    token_counts.append(int(mask.sum()))
        finally:
            del model
        return references, token_counts

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def _post(self, route, payload):
        response = requests.post(self.base_url + route, json=payload, timeout=180)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _native(self, order):
        results = self._post(
            "/encode",
            {
                "text": [self.prompts[index] for index in order],
                "image_data": [
                    [self.urls[name] for name in self.samples[index][2]] or None
                    for index in order
                ],
                "embedding_add_special_tokens": False,
            },
        )
        self.assertEqual(len(results), len(order))
        for result, index in zip(results, order):
            self.assertEqual(
                result["meta_info"]["prompt_tokens"], self.token_counts[index]
            )
        return [row["embedding"] for row in results]

    def _cohere(self, order):
        task = self.samples[order[0]][0]
        self.assertTrue(all(self.samples[index][0] == task for index in order))
        inputs = []
        for index in order:
            _, text, images = self.samples[index]
            content = [] if text is None else [{"type": "text", "text": text}]
            content += [
                {"type": "image_url", "image_url": {"url": self.urls[name]}}
                for name in images
            ]
            inputs.append({"content": content})
        result = self._post(
            "/v2/embed", {"inputs": inputs, "input_type": task, "truncate": "NONE"}
        )
        self.assertEqual(len(result["embeddings"]["float"]), len(order))
        return result["embeddings"]["float"]

    def _assert_reference(self, values, index, route):
        actual = torch.tensor(values, dtype=torch.float32)
        expected = self.references[index]
        self.assertEqual(tuple(actual.shape), (self.dimension,))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertAlmostEqual(actual.norm().item(), 1.0, delta=1e-5)
        delta = actual - expected
        metrics = {
            "model": self.model,
            "revision": self.revision,
            "sample": index,
            "route": route,
            "cosine": F.cosine_similarity(actual, expected, dim=0).item(),
            "max_abs": delta.abs().max().item(),
            "mean_abs": delta.abs().mean().item(),
            "relative_l2": (delta.norm() / expected.norm()).item(),
        }
        print(json.dumps(metrics), flush=True)
        self.assertGreaterEqual(metrics["cosine"], 0.999, metrics)
        self.assertLessEqual(metrics["max_abs"], 0.01, metrics)
        self.assertLessEqual(metrics["relative_l2"], 0.05, metrics)

    def test_reference_single_batch_and_http_parity(self):
        singles = []
        for index in range(len(self.samples)):
            with self.subTest(sample=index):
                native = self._native([index])[0]
                api = self._cohere([index])[0]
                self._assert_reference(native, index, "encode-single")
                self._assert_reference(api, index, "v2-single")
                torch.testing.assert_close(
                    torch.tensor(api), torch.tensor(native), atol=1e-5, rtol=1e-5
                )
                singles.append(native)

        # The query is kept out of the Cohere document batch, whose task is shared.
        # Long-first and short-first orderings exercise the packed mean reduction.
        for order in (
            list(range(1, len(self.samples))),
            list(range(len(self.samples) - 1, 0, -1)),
        ):
            for route, values in (
                ("encode-batch", self._native(order)),
                ("v2-batch", self._cohere(order)),
            ):
                for index, vector in zip(order, values):
                    with self.subTest(route=route, sample=index, order=order):
                        self._assert_reference(vector, index, route)
                        delta = torch.tensor(vector) - torch.tensor(singles[index])
                        self.assertLessEqual(delta.norm().item(), 0.05)

        # Exercise the dedicated texts/images adapters as well as inputs[].
        for index, field in ((0, "texts"), (1, "texts"), (4, "images")):
            task, text, images = self.samples[index]
            values = [text] if field == "texts" else [self.urls[images[0]]]
            result = self._post(
                "/v2/embed", {field: values, "input_type": task, "truncate": "NONE"}
            )
            self._assert_reference(result["embeddings"]["float"][0], index, field)


class TestNemotron35Embed1B(_Nemotron35Accuracy, CustomTestCase):
    model = "nvidia/Nemotron-3.5-Embed-1B-BF16"
    revision = "e24ca1e346bcc53572bd7bdc02b3e8ebf66c197c"
    dimension = 2048


class TestNemotron35Embed8B(_Nemotron35Accuracy, CustomTestCase):
    model = "nvidia/Nemotron-3.5-Embed-8B-BF16"
    revision = "781e779ee4ca9c2d2a4f38321993556baeab49a1"
    dimension = 4096


if __name__ == "__main__":
    unittest.main()
