# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the 2D detection transforms: every transform keeps the boxes on the pixels
they annotate, and the strong augmentation switch turns the strong ones off."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.detection2d.augmentation import (
    HSVAugmentation,
    MixUp,
    Mosaic,
    RandomAffine,
    RandomHorizontalFlip,
)
from autoware_ml.transforms.detection2d.formatting import FilterSmallBoxes2D, Letterbox
from autoware_ml.transforms.detection2d.loading import LoadImageAndResize
from autoware_ml.transforms.detection2d.multi_sample import strong_augmentation_switch
from autoware_ml.transforms.detection2d.sample_io import boxes_of, clip_boxes, image_of
from autoware_ml.transforms.detection2d.tests.marker_samples import (
    MARKER_COLORS,
    assert_boxes_cover_their_markers,
    file_sample,
    loaded_sample,
    marker_extent,
)

BOXES = np.array([[20, 30, 60, 50], [100, 20, 130, 80], [150, 90, 190, 110]], dtype=np.float32)
LABELS = np.array([0, 1, 0], dtype=np.int32)


class TestLoadImageAndResize(unittest.TestCase):
    """Unit tests for the loading transform."""

    def setUp(self) -> None:
        """Write a 200x120 marker image to a temporary file."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.sample = file_sample(Path(self.directory.name) / "image.png", 120, 200, BOXES, LABELS)

    def test_image_is_resized_to_fit_with_its_boxes(self) -> None:
        """Test that the image fits the target size with its aspect ratio and boxes kept.

        Check: a 200x120 image fits 100x100 as 100x60, the image is RGB in [0, 255], the boxes
        are halved and lie on their markers, and the augmentation matrix records the scale.
        """
        sample = LoadImageAndResize(image_size=[100, 100])(self.sample)

        image = image_of(sample)
        boxes, labels = boxes_of(sample)
        self.assertEqual(image.shape, (60, 100, 3))
        self.assertEqual(image.dtype, np.float32)
        self.assertGreater(image.max(), 200.0)
        np.testing.assert_allclose(boxes, BOXES / 2)
        self.assertEqual(labels.tolist(), LABELS.tolist())
        assert_boxes_cover_their_markers(sample, tolerance=1.0)
        matrix = sample.camera_image_data.image_augmentation_matrices[0]
        self.assertAlmostEqual(float(matrix[0, 0]), 0.5)
        self.assertAlmostEqual(float(matrix[1, 1]), 0.5)
        # The first marker is red: the channels are served as RGB.
        self.assertEqual(image[20, 20].tolist(), MARKER_COLORS[0].tolist())

    def test_image_keeps_its_size_without_a_target(self) -> None:
        """Test that no target size leaves the image and the boxes as they are."""
        sample = LoadImageAndResize(image_size=None)(self.sample)

        self.assertEqual(image_of(sample).shape, (120, 200, 3))
        np.testing.assert_allclose(boxes_of(sample)[0], BOXES)

    def test_reduced_decoding_serves_the_same_geometry(self) -> None:
        """Test that decoding at half size gives the image size and boxes of a full decoding."""
        full = LoadImageAndResize(image_size=[100, 100])(self.sample)
        reduced = LoadImageAndResize(image_size=[100, 100], decoding_reduction=2)(self.sample)

        self.assertEqual(image_of(reduced).shape, image_of(full).shape)
        np.testing.assert_allclose(boxes_of(reduced)[0], boxes_of(full)[0])
        assert_boxes_cover_their_markers(reduced, tolerance=1.0)
        with self.assertRaises(ValueError):
            LoadImageAndResize(image_size=[100, 100], decoding_reduction=3)

    def test_missing_file_is_reported(self) -> None:
        """Test that an unreadable image file raises instead of serving an empty image."""
        sample = file_sample(Path(self.directory.name) / "other.png", 120, 200, BOXES, LABELS)
        Path(sample.image_samples[0].image_path).unlink()

        with self.assertRaises(FileNotFoundError):
            LoadImageAndResize(image_size=[100, 100])(sample)


class TestGeometricTransforms(unittest.TestCase):
    """Unit tests for the transforms that move pixels: the boxes must move with them."""

    def setUp(self) -> None:
        """Seed the random source and build a 200x120 marker sample."""
        np.random.seed(0)
        self.sample = loaded_sample(120, 200, BOXES, LABELS)

    def test_flip_mirrors_image_and_boxes(self) -> None:
        """Test that a flipped sample has its boxes on the mirrored markers."""
        sample = RandomHorizontalFlip(probability=1.0)(self.sample)

        np.testing.assert_allclose(boxes_of(sample)[0][0], [140, 30, 180, 50])
        assert_boxes_cover_their_markers(sample, tolerance=0.0)
        self.assertEqual(float(sample.camera_image_data.image_augmentation_matrices[0, 0, 0]), -1.0)

    def test_flip_is_skipped_by_its_probability(self) -> None:
        """Test that a flip of probability 0 returns the sample unchanged."""
        sample = RandomHorizontalFlip(probability=0.0)(self.sample)

        np.testing.assert_allclose(boxes_of(sample)[0], BOXES)

    def test_letterbox_resizes_and_pads_bottom_right(self) -> None:
        """Test the letterbox: aspect ratio kept, top left aligned, padded with 114.

        Check: the 200x120 image becomes 100x60 inside a 100x100 image whose rows below it
        hold the pad value, and the boxes lie on their markers.
        """
        sample = Letterbox(image_size=[100, 100], pad_value=114.0)(self.sample)

        image = image_of(sample)
        self.assertEqual(image.shape, (100, 100, 3))
        self.assertTrue((image[60:] == 114.0).all())
        self.assertFalse((image[:60] == 114.0).all())
        np.testing.assert_allclose(boxes_of(sample)[0], BOXES / 2)
        assert_boxes_cover_their_markers(sample, tolerance=1.0)

    def test_letterbox_only_pads_an_image_that_fits(self) -> None:
        """Test that an image already fitting the target tightly is padded without resampling."""
        sample = Letterbox(image_size=[200, 200])(self.sample)

        image = image_of(sample)
        np.testing.assert_array_equal(image[:120], image_of(self.sample))
        np.testing.assert_allclose(boxes_of(sample)[0], BOXES)

    def test_random_affine_moves_the_boxes_with_the_pixels(self) -> None:
        """Test that the boxes follow a random rotation, scale, shear and shift.

        Setup: affines drawn with a scale near 1 so the markers stay in the output.
        Check: over several draws every kept box encloses its marker within two pixels, and
        the output has the configured size.
        """
        affine = RandomAffine(
            image_size=[160, 240],
            max_rotation_degrees=10.0,
            max_translation_ratio=0.1,
            scale_range=[0.8, 1.1],
            max_shear_degrees=2.0,
        )
        for _ in range(10):
            sample = affine(self.sample)
            image = image_of(sample)
            self.assertEqual(image.shape, (160, 240, 3))
            for box, label in zip(*boxes_of(sample), strict=True):
                # Boxes 0 and 2 share a label but not a color: find the marker by overlap.
                extents = [marker_extent(image, color) for color in MARKER_COLORS[:3]]
                distances = [
                    np.abs(box - extent).max() if extent is not None else np.inf
                    for extent in extents
                ]
                best = int(np.argmin(distances))
                self.assertLess(distances[best], 2.5)
                self.assertEqual(int(label), int(LABELS[best]))

    def test_mosaic_places_four_images_around_the_center(self) -> None:
        """Test that a mosaic holds the boxes of four images on their markers.

        Setup: the sample source hands out the same marker sample, and the center range is
        fixed to the middle of the canvas, so the four copies are placed whole.
        Check: the canvas is twice the input size and holds four copies of every box, each
        shifted by the offset of its image.
        """
        mosaic = Mosaic(image_size=[120, 200], center_ratio_range=[1.0, 1.0])
        mosaic.bind_sample_source(lambda: self.sample)

        sample = mosaic(self.sample)

        image = image_of(sample)
        boxes, labels = boxes_of(sample)
        self.assertEqual(image.shape, (240, 400, 3))
        self.assertEqual(len(boxes), 4 * len(BOXES))
        offsets = np.array([[0, 0], [200, 0], [0, 120], [200, 120]], dtype=np.float32)
        for index, offset in enumerate(offsets):
            placed = boxes[index * 3 : (index + 1) * 3]
            np.testing.assert_allclose(placed, BOXES + np.tile(offset, 2))
            for box, color in zip(placed.astype(int), MARKER_COLORS, strict=False):
                self.assertTrue((image[box[1] : box[3], box[0] : box[2]] == color).all())
        self.assertEqual(labels.tolist(), LABELS.tolist() * 4)

    def test_mosaic_clips_boxes_at_the_canvas_border(self) -> None:
        """Test that images cut by the canvas keep only the boxes still inside it."""
        mosaic = Mosaic(image_size=[120, 200], center_ratio_range=[0.5, 0.5])
        mosaic.bind_sample_source(lambda: self.sample)

        sample = mosaic(self.sample)

        boxes, _ = boxes_of(sample)
        image = image_of(sample)
        self.assertTrue((boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all())
        self.assertTrue((boxes >= 0).all())
        self.assertTrue((boxes[:, 2] <= 400).all() and (boxes[:, 3] <= 240).all())
        for box in boxes.astype(int):
            self.assertGreater(image[box[1] : box[3], box[0] : box[2]].sum(), 0.0)

    def test_mosaic_needs_a_sample_source(self) -> None:
        """Test that a mosaic no dataset bound a sample source to fails with a clear error."""
        with self.assertRaises(RuntimeError):
            Mosaic(image_size=[120, 200])(self.sample)

    def test_mixup_blends_the_partner_and_adds_its_boxes(self) -> None:
        """Test that mixup averages the two images and appends the boxes of the partner.

        Setup: the partner is a second marker sample with one box in the middle of its width,
        where a flip leaves it, mixed at a fixed scale of 1.
        Check: the sample keeps its boxes and gains the box of the partner on a pixel region
        that holds half the marker color, and its own markers are halved too.
        """
        partner = loaded_sample(120, 200, np.array([[90, 90, 110, 115]]), np.array([1]))
        mixup = MixUp(image_size=[120, 200], scale_range=[1.0, 1.0])
        mixup.bind_sample_source(lambda: partner)

        sample = mixup(self.sample)

        image = image_of(sample)
        boxes, labels = boxes_of(sample)
        self.assertEqual(image.shape, (120, 200, 3))
        self.assertEqual(len(boxes), 4)
        np.testing.assert_allclose(boxes[:3], BOXES)
        self.assertEqual(labels.tolist(), [0, 1, 0, 1])
        self.assertEqual(image[40, 40].tolist(), (MARKER_COLORS[0] / 2).tolist())
        x1, y1, x2, y2 = boxes[3].astype(int)
        self.assertEqual([x1, y1, x2, y2], [90, 90, 110, 115])
        self.assertTrue((image[y1:y2, x1:x2] == MARKER_COLORS[0] / 2).all())

    def test_mixup_skips_when_no_partner_has_boxes(self) -> None:
        """Test that mixup leaves the sample alone when the dataset serves no box."""
        empty = loaded_sample(120, 200, np.zeros((0, 4)), np.zeros(0))
        mixup = MixUp(image_size=[120, 200])
        mixup.bind_sample_source(lambda: empty)

        sample = mixup(self.sample)

        np.testing.assert_array_equal(image_of(sample), image_of(self.sample))
        self.assertEqual(len(boxes_of(sample)[0]), 3)

    def test_clip_boxes_drops_boxes_outside_the_image(self) -> None:
        """Test that boxes are clipped to the image and those left without an area dropped."""
        boxes = np.array(
            [[-10, -10, 20, 20], [90, 90, 120, 120], [150, 10, 180, 30]], dtype=np.float32
        )

        clipped, labels = clip_boxes(boxes, np.array([0, 1, 2]), width=100, height=100)

        np.testing.assert_allclose(clipped, [[0, 0, 20, 20], [90, 90, 100, 100]])
        self.assertEqual(labels.tolist(), [0, 1])


class TestPhotometricAndFilterTransforms(unittest.TestCase):
    """Unit tests for the transforms that leave the geometry alone."""

    def setUp(self) -> None:
        """Seed the random source and build a marker sample."""
        np.random.seed(0)
        self.sample = loaded_sample(120, 200, BOXES, LABELS)

    def test_hsv_changes_colors_only(self) -> None:
        """Test that the HSV shift keeps the boxes, the image size and the value range."""
        changed = False
        for _ in range(8):
            sample = HSVAugmentation()(self.sample)
            image = image_of(sample)
            np.testing.assert_allclose(boxes_of(sample)[0], BOXES)
            self.assertEqual(image.shape, (120, 200, 3))
            self.assertGreaterEqual(image.min(), 0.0)
            self.assertLessEqual(image.max(), 255.0)
            changed = changed or not np.array_equal(image, image_of(self.sample))
        self.assertTrue(changed)

    def test_small_boxes_are_filtered_by_their_shorter_side(self) -> None:
        """Test that a box is kept by its shorter side, not by its area.

        Setup: a 1.5 px wide sliver, a 2x2 px box and a regular box, with a threshold of 2 px.
        Check: the sliver is dropped although its area exceeds that of the 2x2 px box.
        """
        boxes = np.array([[10, 10, 11.5, 60], [30, 30, 32, 32], [50, 50, 90, 80]], dtype=np.float32)
        sample = loaded_sample(120, 200, boxes, np.array([0, 1, 0]))

        filtered = FilterSmallBoxes2D(min_side=2.0)(sample)

        kept, labels = boxes_of(filtered)
        np.testing.assert_allclose(kept, boxes[1:])
        self.assertEqual(labels.tolist(), [1, 0])


class TestStrongAugmentationSwitch(unittest.TestCase):
    """Unit tests for the switch that turns the strong augmentations off."""

    def setUp(self) -> None:
        """Seed the random source, build a marker sample and reset the test switch."""
        np.random.seed(0)
        self.sample = loaded_sample(120, 200, BOXES, LABELS)
        self.switch = strong_augmentation_switch("test_switch")
        self.switch.set(True)
        self.addCleanup(self.switch.set, True)

    def test_switch_is_shared_by_name(self) -> None:
        """Test that the same name yields the same switch and another name another one."""
        self.assertIs(strong_augmentation_switch("test_switch"), self.switch)
        self.assertIsNot(strong_augmentation_switch("other_test_switch"), self.switch)

    def test_switched_off_pipeline_only_letterboxes(self) -> None:
        """Test that the strong transforms run while the switch is on and are skipped once off.

        Setup: a pipeline of mosaic, random affine, mixup and letterbox on one switch.
        Check: with the switch on the sample holds the boxes of several images; with it off
        the sample is the letterboxed input, as an evaluation sample would be.
        """
        mosaic = Mosaic(image_size=[120, 200], switch_name="test_switch")
        mixup = MixUp(image_size=[120, 200], switch_name="test_switch")
        mosaic.bind_sample_source(lambda: self.sample)
        mixup.bind_sample_source(lambda: self.sample)
        pipeline = TransformsCompose(
            [
                mosaic,
                RandomAffine(
                    image_size=[120, 200], scale_range=[0.5, 0.5], switch_name="test_switch"
                ),
                mixup,
                Letterbox(image_size=[200, 200]),
            ]
        )

        strong = pipeline(self.sample)
        self.switch.set(False)
        plain = pipeline(self.sample)

        self.assertGreater(len(boxes_of(strong)[0]), len(BOXES))
        np.testing.assert_allclose(boxes_of(plain)[0], BOXES)
        np.testing.assert_array_equal(image_of(plain)[:120], image_of(self.sample))
        self.assertEqual(image_of(plain).shape, (200, 200, 3))

    def test_images_collate_after_the_pipeline(self) -> None:
        """Test that the image of a transformed sample is a float32 channel first tensor."""
        sample = Letterbox(image_size=[128, 224])(self.sample)

        images = sample.camera_image_data.images
        self.assertEqual(tuple(images.shape), (1, 3, 128, 224))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(tuple(torch.stack([images, images]).shape), (2, 1, 3, 128, 224))
