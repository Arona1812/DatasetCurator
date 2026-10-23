"""Whole-frame and app-domain regressions without API calls or model weights."""
import unittest
import numpy as np
from PIL import Image
import frame_cleanup as frames
from photo_regions import complete_black_frame, app_photo_viewport, colour_inset_photo


class CompleteFrames(unittest.TestCase):
    def test_dark_content_seam_preserves_every_non_background_pixel(self):
        pixels=np.zeros((600,600,3),dtype=np.uint8)
        photo=np.random.default_rng(8).integers(14,25,(380,540,3),dtype=np.uint8)
        pixels[110:490,30:570]=photo
        box=complete_black_frame(Image.fromarray(pixels),10)
        self.assertIsNotNone(box)
        self.assertTrue(box[0]<=30 and box[1]<=110 and box[2]>=570 and box[3]>=490)
        self.assertLess(box[3],500)
        result=frames._analyse_preview(Image.fromarray(pixels),(600,600),frames.DetectorSettings(min_content_px=256))
        detected=result['candidate_bbox']
        self.assertTrue(detected[0]<=30 and detected[1]<=110 and detected[2]>=570 and detected[3]>=490)
        self.assertLess(result['candidate_bbox'][3],500)

    def test_single_dark_scene_edge_does_not_become_complete_frame(self):
        pixels=np.full((600,600,3),100,dtype=np.uint8)
        pixels[500:]=0
        self.assertIsNone(complete_black_frame(Image.fromarray(pixels),10))

    def test_small_full_photo_is_reviewed_instead_of_partial_auto_crop(self):
        pixels=np.zeros((640,640,3),dtype=np.uint8)
        pixels[130:490]=np.random.default_rng(8).integers(60,210,(360,640,3),dtype=np.uint8)
        result=frames._analyse_preview(Image.fromarray(pixels),(640,640),frames.DetectorSettings())
        self.assertLess(result['candidate_bbox'][3],500)
        self.assertIn('content_below_minimum',result['layout_flags'])
        self.assertIsNone(frames.automatic_crop_bbox(result,'auto_high_keep_medium',['uniform_canvas']))

    def test_app_viewport_keeps_the_entire_photo(self):
        pixels=np.full((1000,450,3),90,dtype=np.uint8)
        pixels[150:780]=np.random.default_rng(7).integers(30,210,(630,450,3),dtype=np.uint8)
        box=app_photo_viewport(Image.fromarray(pixels))
        self.assertIsNotNone(box)
        self.assertLessEqual(box[1],150)
        self.assertGreaterEqual(box[3],780)
        self.assertLessEqual(box[3],782)

    def test_colour_photo_inset_requires_four_supported_boundaries(self):
        pixels=np.full((1000,450,3),100,dtype=np.uint8)
        pixels[300:680,95:375]=np.random.default_rng(3).integers(30,240,(380,280,3),dtype=np.uint8)
        box=colour_inset_photo(Image.fromarray(pixels))
        self.assertIsNotNone(box)
        self.assertTrue(all(abs(a-b)<=6 for a,b in zip(box,[95,300,375,680])),box)
        self.assertTrue(box[0]<=95 and box[1]<=300 and box[2]>=375 and box[3]>=680,box)
        monochrome=np.full((1000,450,3),100,dtype=np.uint8)
        self.assertIsNone(colour_inset_photo(Image.fromarray(monochrome)))


if __name__=='__main__':
    unittest.main()
