import os
import cv2
import numpy as np
import torch
import torchvision
from torch.nn import functional as F

_TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]
_TORCH11X = (_TORCH_VER >= [1, 10])

def meshgrid(*tensors):
    if _TORCH11X:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)

def extract_roi_otsu(img, gkernel=(5, 5)):
    """WARNING: this function modify input image inplace."""
    # Convert to grayscale if image has multiple channels
    if len(img.shape) > 2:
        if img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif img.shape[2] == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    ori_h, ori_w = img.shape[:2]
    
    # clip percentile: implant, white lines
    upper = np.percentile(img, 95)
    img[img > upper] = np.min(img)
    
    # Gaussian filtering to reduce noise (optional)
    if gkernel is not None:
        img = cv2.GaussianBlur(img, gkernel, 0)
    
    # Ensure image is uint8
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    
    _, img_bin = cv2.threshold(img, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # dilation to improve contours connectivity
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))
    img_bin = cv2.dilate(img_bin, element)
    
    cnts, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None, None, None
        
    areas = np.array([cv2.contourArea(cnt) for cnt in cnts])
    select_idx = np.argmax(areas)
    cnt = cnts[select_idx]
    area_pct = areas[select_idx] / (img.shape[0] * img.shape[1])
    x0, y0, w, h = cv2.boundingRect(cnt)
    
    # min-max for safety only
    # x0, y0, x1, y1
    x1 = min(max(int(x0 + w), 0), ori_w)
    y1 = min(max(int(y0 + h), 0), ori_h)
    x0 = min(max(int(x0), 0), ori_w)
    y0 = min(max(int(y0), 0), ori_h)
    
    return [x0, y0, x1, y1], area_pct, None

class RoiExtractor:
    def __init__(self,
                 engine_path=None,  # Not used in this version
                 input_size=[416, 416],
                 num_classes=1,
                 conf_thres=0.5,
                 nms_thres=0.9,
                 class_agnostic=False,
                 area_pct_thres=0.04,
                 hw=None,
                 strides=None,
                 exp=None):
        self.input_size = input_size
        self.input_h, self.input_w = input_size
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.class_agnostic = class_agnostic
        self.area_pct_thres = area_pct_thres
        
        # We'll use Otsu thresholding instead of the YOLOX model
        print("Using Otsu thresholding for ROI detection")

    def detect_single(self, img):
        """Detect ROI using Otsu thresholding"""
        # Convert tensor to numpy if needed
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Ensure image is in the right format
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
        # Convert to grayscale if needed
        if len(img.shape) > 2:
            if img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif img.shape[2] == 3:  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        ori_h, ori_w = img.shape[:2]
        
        # Use Otsu thresholding
        xyxy, area_pct, _ = extract_roi_otsu(img.copy())
        
        if xyxy is not None and area_pct >= self.area_pct_thres:
            print('ROI detection: using Otsu.')
            return xyxy, area_pct, None
        
        print('ROI detection: using full frame.')
        return [0, 0, ori_w, ori_h], 1.0, None