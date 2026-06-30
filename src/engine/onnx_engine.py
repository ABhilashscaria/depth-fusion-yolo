import cv2
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
from ultralytics import YOLO

from .base import BaseSpatialEngine

# ImageNet normalization constants
DEPTH_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEPTH_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class ONNXEngine(BaseSpatialEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        print("⚡ Initializing High-Performance ONNX Pipeline...")
        
        yolo_path = config['models']['yolo']['onnx_weights']
        depth_path = config['models']['depth']['onnx_weights']
        self.depth_input_size = config['models']['depth']['onnx_input_size']
        warmup_runs = config['models']['depth']['warmup_runs']

        # Arena strategy and 4GB memory limit to prevent OOM
        dev_id = int(self.device.split(":")[1]) if "cuda" in self.device else 0
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': dev_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            'CPUExecutionProvider',
        ]

        print("-> Loading YOLO11n-Seg ONNX Engine...")
        self.yolo = YOLO(yolo_path, task='segment')
        self.yolo_names = self.yolo.names
        
        print("-> Loading Depth Anything V2 ONNX Session...")
        self.depth_session = ort.InferenceSession(depth_path, providers=providers)
        self.depth_input_name = self.depth_session.get_inputs()[0].name
        
        active_providers = self.depth_session.get_providers()
        if active_providers[0] != 'CUDAExecutionProvider':
            print("⚠️ WARNING: CUDA Execution Provider not active. Running on CPU.")
            
        self._warmup(warmup_runs)
        print("✅ ONNX Pipeline Initialization Complete.\n")

    def _warmup(self, n: int):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(n):
            self._forward_yolo(dummy)
            self._forward_depth(dummy)
        if self.device != "cpu":
            torch.cuda.synchronize()

    def _forward_yolo(self, frame_bgr: np.ndarray):
        return self.yolo(frame_bgr, verbose=False, retina_masks=True)[0]

    def _preprocess_depth(self, frame_bgr: np.ndarray) -> np.ndarray:
        s = self.depth_input_size
        img = cv2.resize(frame_bgr, (s, s), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - DEPTH_MEAN) / DEPTH_STD
        return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    def _forward_depth(self, frame_bgr: np.ndarray) -> torch.Tensor:
        h, w = frame_bgr.shape[:2]
        img_input = self._preprocess_depth(frame_bgr)
        
        onnx_outputs = self.depth_session.run(None, {self.depth_input_name: img_input})
        raw_depth = onnx_outputs[0].squeeze()
        
        # Normalize to [0, 1]
        d_min, d_max = raw_depth.min(), raw_depth.max()
        depth_normalized = (
            (raw_depth - d_min) / (d_max - d_min)
            if d_max - d_min > 1e-5 else np.zeros_like(raw_depth)
        )
        
        # Upscale on GPU using PyTorch interpolation
        depth_tensor = torch.from_numpy(depth_normalized).to(self.device)
        depth_tensor = F.interpolate(
            depth_tensor.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        return depth_tensor
