"""Regression tests for Pixtral multimodal item processing."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from PIL import Image

from sglang.srt.managers.mm_utils import get_new_expanded_mm_items
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.multimodal.modality import Modality, MultimodalInputFormat
from sglang.srt.multimodal.processors.nemotron35_embedding import (
    Nemotron35EmbeddingProcessor,
)
from sglang.srt.multimodal.processors.pixtral import PixtralProcessor
from sglang.srt.multimodal.transport.feature_transport import (
    MultimodalFeatureTransport,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPixtralProcessor(CustomTestCase):
    def _make_processor(self):
        processor = object.__new__(PixtralProcessor)
        processor._feature_transport = MultimodalFeatureTransport(
            "cuda_ipc", base_gpu_id=0, skip_mm_pool=True
        )
        processor.processor_fingerprint = "sha256:" + "12" * 32
        processor.image_size = 1024
        processor._effective_patch_size = 28
        processor._precompute_hashes_before_cpu_transfer = MagicMock()
        return processor

    def test_multi_image_features_are_split_before_transport(self):
        """CUDA IPC dispatch must receive per-image tensors, not a bundled proxy."""
        processor = self._make_processor()
        proxies = [object(), object()]
        processor._feature_transport._wrap_tensor_for_cuda_ipc = MagicMock(
            side_effect=proxies
        )

        feature = torch.arange(8).reshape(2, 4)
        image_sizes = torch.tensor([[56, 28], [84, 28]])
        bundled_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=feature,
            offsets=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
            format=MultimodalInputFormat.PROCESSOR_OUTPUT,
            model_specific_data={"image_sizes": image_sizes, "extra_key": "keep"},
        )
        images = [Image.new("RGB", (28, 56)), Image.new("RGB", (28, 84))]

        items = processor._finalize_mm_items(
            [bundled_item],
            images=images,
        )

        self.assertEqual(len(items), 2)
        self.assertEqual(
            [item.offsets for item in items],
            [[(0, 0), (1, 1)], [(2, 2), (3, 3), (4, 4)]],
        )
        self.assertTrue(
            all(item.format == MultimodalInputFormat.PROCESSOR_OUTPUT for item in items)
        )
        self.assertTrue(
            all(item.model_specific_data["extra_key"] == "keep" for item in items)
        )
        self.assertTrue(all(item.pad_value is not None for item in items))
        self.assertTrue(
            torch.equal(items[0].model_specific_data["image_sizes"], image_sizes[:1])
        )
        self.assertTrue(
            torch.equal(items[1].model_specific_data["image_sizes"], image_sizes[1:])
        )
        wrapped_features = [
            call.args[0]
            for call in processor._feature_transport._wrap_tensor_for_cuda_ipc.call_args_list
        ]
        self.assertTrue(torch.equal(wrapped_features[0], feature[:1]))
        self.assertTrue(torch.equal(wrapped_features[1], feature[1:]))
        self.assertEqual([item.feature for item in items], proxies)

    def test_multi_image_split_uses_processor_sizes_after_resize(self):
        """A 1120-pixel processor must not split rows at vision capacity 1540."""
        processor = self._make_processor()
        processor.image_size = 1540
        images = [Image.new("RGB", (1400, 1400)), Image.new("RGB", (2240, 1120))]
        offsets = [(index * 41, index * 41 + 39) for index in range(60)]
        feature = torch.zeros(2, 3, 1120, 1120)
        sizes = torch.tensor([[1120, 1120], [560, 1120]])
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=feature,
            offsets=offsets,
            model_specific_data={"image_sizes": sizes},
        )

        items = processor._postprocess_mm_items_before_transport([item], images=images)

        self.assertEqual([len(part.offsets) for part in items], [40, 20])
        self.assertEqual(items[0].offsets, offsets[:40])
        self.assertEqual(items[1].offsets, offsets[40:])
        self.assertTrue(torch.equal(items[0].image_sizes, sizes[:1]))
        self.assertTrue(torch.equal(items[1].image_sizes, sizes[1:]))

    def test_already_split_one_row_images_are_preserved(self):
        """Generic per-image splits must not be collapsed and re-sliced by Pixtral."""
        processor = self._make_processor()
        processor._feature_transport._wrap_tensor_for_cuda_ipc = MagicMock(
            side_effect=[object(), object()]
        )
        bundled_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.arange(8).reshape(2, 4),
            offsets=[(0, 0), (2, 2)],
        )
        items = get_new_expanded_mm_items([bundled_item])
        images = [Image.new("RGB", (28, 28)), Image.new("RGB", (56, 28))]

        items = processor._finalize_mm_items(items, images=images)

        self.assertEqual(len(items), 2)
        self.assertEqual([item.offsets for item in items], [[(0, 0)], [(2, 2)]])
        wrapped_features = [
            call.args[0]
            for call in processor._feature_transport._wrap_tensor_for_cuda_ipc.call_args_list
        ]
        self.assertTrue(torch.equal(wrapped_features[0], bundled_item.feature[:1]))
        self.assertTrue(torch.equal(wrapped_features[1], bundled_item.feature[1:]))

    def test_mismatched_patch_rows_fail_loudly(self):
        """Derived row counts cannot silently leave image placeholders unassigned."""
        processor = self._make_processor()
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.arange(8).reshape(2, 4),
            offsets=[(0, 0), (1, 1)],
        )
        images = [Image.new("RGB", (28, 56)), Image.new("RGB", (28, 84))]

        with self.assertRaisesRegex(ValueError, "patch rows"):
            processor._finalize_mm_items([item], images=images)


class TestNemotron35EmbeddingProcessor(CustomTestCase):
    def test_padding_mask_excludes_padding_but_preserves_valid_pad_token(self):
        """Only mask-zero padding may be removed from the packed mean pool."""
        processor = object.__new__(Nemotron35EmbeddingProcessor)
        processor._tokenizer = SimpleNamespace()
        processor._tokenizer_auto_adds_specials = False
        processor._feature_transport = MultimodalFeatureTransport(
            "cpu", base_gpu_id=0, skip_mm_pool=True
        )
        processor.FEATURE_NAMES = ("pixel_values",)
        processor.precompute_hash_before_cpu_transfer = False
        for input_ids, mask in (
            ([5, 10, 12, 10, 13, 11, 11, 11], [1, 1, 1, 1, 1, 1, 0, 0]),
            ([11, 11, 5, 10, 12, 10, 13, 11], [0, 0, 1, 1, 1, 1, 1, 1]),
        ):
            with self.subTest(mask=mask):
                pixels = torch.zeros(1, 3, 28, 28)
                # Mock the external HF call, retaining the real SGLang path.
                processor._processor = MagicMock(
                    return_value={
                        "input_ids": torch.tensor([input_ids]),
                        "attention_mask": torch.tensor([mask]),
                        "pixel_values": pixels,
                    }
                )
                result = processor.process_mm_data("passage: [IMG] text")
                self.assertEqual(
                    result["input_ids"].tolist(), [[5, 10, 12, 10, 13, 11]]
                )
                self.assertEqual(result["attention_mask"].tolist(), [[1] * 6])
                self.assertTrue(torch.equal(result["pixel_values"], pixels))


if __name__ == "__main__":
    unittest.main()
