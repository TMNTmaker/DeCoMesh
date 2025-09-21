import copy
import time
from yolox.utils import matrix_3diou,compute_map
import numpy as np

class SUNRGBeval_opt:
    """
    This class is a simplified version of the COCO evaluation API for SUNRGB dataset.
    It provides methods to evaluate detection results against ground truth annotations.
    """
    def __init__(self, sunrgbGt, sunrgbDt):
        self.sunrgbGt = sunrgbGt
        self.sunrgbDt = sunrgbDt
        self.evalImgs = []
        self.params = None

    def accumulate(self):
        """
        Run evaluation on the SUNRGB dataset.
        """
        
        # Here you would implement the logic to evaluate detections against ground truth
        # For simplicity, we will just simulate an evaluation result
        #self.evalImgs = [{"image_id": img_id, "score": 1} for img_id in list(self.sunrgbGt.keys())]
        scores = np.array([1.0] * len(self.sunrgbDt))
        
        ious= matrix_3diou(self.sunrgbGt, self.sunrgbDt)
        
        mAP50 = compute_map(iou_matrix, iou_thresholds=[0.5], pred_scores=scores)
        mAP = compute_map(iou_matrix, iou_thresholds=[0.25, 0.5, 0.75], pred_scores=scores)
        self.stats = [mAP50, mAP]