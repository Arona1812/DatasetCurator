"""Regressions for inner-photo refinement, without API requests."""
import unittest
import numpy as np
from PIL import Image
from photo_regions import refine_photo_region


class InnerPhotoRegions(unittest.TestCase):
    def setUp(self):
        pixels = np.full((700,500,3),240,dtype=np.uint8)
        pixels[180:520,120:420] = np.random.default_rng(7).integers(40,200,(340,300,3),dtype=np.uint8)
        self.image = Image.fromarray(pixels)
        self.photo = [120,180,420,520]

    def test_removes_canvas_left_and_right_inside_existing_horizontal_bounds(self):
        proposal = refine_photo_region(self.image,[0,180,500,520])
        self.assertIsNotNone(proposal)
        self.assertTrue(all(abs(a-b)<=2 for a,b in zip(proposal['bbox'],self.photo)),proposal)

    def test_removes_canvas_below_photo_inside_existing_vertical_bounds(self):
        proposal = refine_photo_region(self.image,[120,180,420,680])
        self.assertIsNotNone(proposal)
        self.assertTrue(all(abs(a-b)<=2 for a,b in zip(proposal['bbox'],self.photo)),proposal)

    def test_complete_photo_is_not_tightened_again(self):
        self.assertIsNone(refine_photo_region(self.image,self.photo))

    def test_no_existing_frame_does_not_create_an_object_crop(self):
        self.assertIsNone(refine_photo_region(self.image,None))


if __name__ == '__main__':
    unittest.main()
