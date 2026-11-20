"""Nemotron-3.5-Embed: Pixtral vision features and bidirectional text pooling."""

from array import array
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.nemotron3_embedding import EmbeddingNemotron3Model
from sglang.srt.models.pixtral import PixtralHFVisionModel
from sglang.srt.multimodal.modality import Modality


class EmbeddingNemotron35Model(nn.Module):
    """Shared native model for the 1B and 8B BF16 embedding checkpoints."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_tower = PixtralHFVisionModel(
            config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_tower", prefix),
        )
        self.multi_modal_projector = Mistral3MultiModalProjector(config)
        self.language_model = EmbeddingNemotron3Model(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

    def pad_input_ids(self, input_ids: array, mm_inputs: MultimodalInputs) -> array:
        # Pixtral already expands [IMG] into patch rows in the processor.
        # Replace only the image spans; retain [IMG_BREAK]/[IMG_END] embeddings.
        return self.vision_tower.pad_input_ids(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        features = []
        feature_layer = self.config.vision_feature_layer
        for item in items:
            pixel_values = torch.as_tensor(
                item.feature,
                device=self.vision_tower.device,
                dtype=self.vision_tower.dtype,
            )
            # The released checkpoints use the last layer; avoid retaining
            # every layer's activations for this common case.
            if feature_layer == -1:
                image_features = self.vision_tower(pixel_values, item.image_sizes)
            else:
                outputs = self.vision_tower(
                    pixel_values, item.image_sizes, output_hidden_states=True
                )
                if isinstance(feature_layer, int):
                    image_features = outputs.hidden_states[feature_layer]
                else:
                    image_features = torch.cat(
                        [outputs.hidden_states[index] for index in feature_layer],
                        dim=-1,
                    )
            features.append(
                self.multi_modal_projector(image_features.squeeze(0), item.image_sizes)
            )
        return torch.cat(features, dim=0)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = True,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={Modality.IMAGE: self.get_image_feature},
            positions=positions,
            get_embedding=True,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Mistral3Model stores bare submodules. Also accept the outer `model.`
        # prefix used when the same backbone is saved in a task wrapper.
        projector_params = dict(self.multi_modal_projector.named_parameters())
        for name, loaded_weight in weights:
            normalized_name = name.removeprefix("model.")
            part, _, rest = normalized_name.partition(".")
            if part in ("language_model", "vision_tower"):
                getattr(self, part).load_weights([(rest, loaded_weight)])
            elif part == "multi_modal_projector" and rest in projector_params:
                param = projector_params[rest]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                raise ValueError(f"Unexpected Nemotron-3.5-Embed weight: {name}")


EntryClass = [EmbeddingNemotron35Model]
