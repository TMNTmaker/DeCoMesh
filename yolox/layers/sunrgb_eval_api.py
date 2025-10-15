
from yolox.utils import evaluate_sunrgbd_3dIoU_mAP
from yolox.data import SUNRGBD_CLASSES_20,SUNRGBD_CLASSES_38
class SUNRGBDeval_opt:
    """
    This class is a simplified version of the COCO evaluation API for SUNRGB dataset.
    It provides methods to evaluate detection results against ground truth annotations.
    """
    def __init__(self, sunrgbdGt, sunrgbdDt,annType):
        self.sunrgbdGt = sunrgbdGt
        self.sunrgbdDt = sunrgbdDt
        self.evalImgs = []
        self.params = None
        self.annType = annType

    def accumulate(self):
        """
        Run evaluation on the SUNRGBD dataset.
        """
        
        # Here you would implement the logic to evaluate detections against ground truth
        # For simplicity, we will just simulate an evaluation result
        #self.evalImgs = [{"image_id": img_id, "score": 1} for img_id in list(self.sunrgbdGt.keys())]

        
        results = evaluate_sunrgbd_3dIoU_mAP(self.sunrgbdDt,self.sunrgbdGt,
                                             num_classes=len(SUNRGBD_CLASSES_20), 
                                             iou_thresh_list=[0.25,0.5])
        
        self.stats = [results[0.25]["mAP"], results[0.5]["mAP"]]