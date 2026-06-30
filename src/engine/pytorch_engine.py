import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .base import BaseSpatialEngine

DEPTH_RESOLUTION_PRESETS = {
    "FAST":     (518, 518),
    "BALANCED": (756, 756),
    "FULL":     None,
}

class PyTorchEngine(BaseSpatialEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        print("⚡ Initializing PyTorch VRAM Pipeline...")
        
        yolo_model = config['models']['yolo']['pytorch_weights']
        depth_model = config['models']['depth']['hf_model_id']
        warmup_runs = config['models']['depth']['warmup_runs']
        depth_res = config['models']['depth']['pytorch_resolution']
        
        self.depth_size = DEPTH_RESOLUTION_PRESETS.get(depth_res, None)
        
        if self.device != "cpu":
            cudnn.benchmark = True
            cudnn.deterministic = False

        print("-> Loading YOLO11n-Seg...")
        self.yolo = YOLO(yolo_model)
        self.yolo.to(self.device)
        self.yolo_names = self.yolo.names
        
        print("-> Loading Depth Anything V2 (FP16 fused)...")
        self.image_processor = AutoImageProcessor.from_pretrained(depth_model)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            depth_model, torch_dtype=torch.float16
        ).to(self.device)
        self.depth_model.eval()
        
        self._warmup(warmup_runs)
        print("✅ PyTorch Initialization Complete.\n")

    def _warmup(self, runs: int):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(runs):
            self._forward_depth(dummy)
            self._forward_yolo(dummy)
        if self.device != "cpu":
            torch.cuda.synchronize()

    def _forward_yolo(self, frame_bgr: np.ndarray):
        # Parse numeric device ID for ultralytics if CUDA is used
        dev_id = int(self.device.split(":")[1]) if "cuda" in self.device else "cpu"
        return self.yolo(frame_bgr, verbose=False, retina_masks=True, device=dev_id)[0]

    def _forward_depth(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = frame_rgb.shape[:2]

        if self.depth_size is not None:
            dh, dw = self.depth_size
            resized = cv2.resize(frame_rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame_rgb

        inputs = self.image_processor(images=resized, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device).half()

        outputs = self.depth_model(pixel_values)
        predicted_depth = outputs.predicted_depth

        # Upscale back to native frame resolution
        depth_tensor = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h_orig, w_orig),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Normalize [0, 1]
        d_min, d_max = depth_tensor.min(), depth_tensor.max()
        if d_max - d_min > 1e-5:
            depth_tensor = (depth_tensor - d_min) / (d_max - d_min)
        else:
            depth_tensor = torch.zeros_like(depth_tensor)

        return depth_tensor
