"""Pixtral preprocessing for the bidirectional Nemotron-3.5 embedder."""

from sglang.srt.models.nemotron35_embedding import EmbeddingNemotron35Model
from sglang.srt.multimodal.processors.pixtral import PixtralProcessor


class Nemotron35EmbeddingProcessor(PixtralProcessor):
    models = [EmbeddingNemotron35Model]
    # Pixtral expands each image into rows separated by break/end tokens.
    # Generic placeholder repetition cannot reconstruct that sequence.
    preserve_processor_input_ids = True

    def process_mm_data(self, input_text, **kwargs) -> dict:
        result = super().process_mm_data(input_text, **kwargs)
        # HF pads to a multiple of eight. SGLang packs valid tokens and pools
        # every position, so remove masked positions before computing offsets.
        keep = result["attention_mask"][0].bool()
        result["input_ids"] = result["input_ids"][:, keep]
        result["attention_mask"] = result["attention_mask"][:, keep]
        return result
