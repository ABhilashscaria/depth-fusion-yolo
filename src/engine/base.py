import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseSpatialEngine(ABC):
    def __init__(self, config: dict):
        """
        Abstract base class for the Spatial Verification Engine.
        Handles the core mathematical fusion and physics gating natively on the GPU.
        """
        self.config = config
        self.device = config['system']['device']
        self.min_volume = config['spatial_physics']['min_volume']
        self.min_step_delta = config['spatial_physics']['min_step_delta']
        self.kernel_size = config['spatial_physics']['halo_kernel_size']
        self.iterations = config['spatial_physics']['halo_iterations']
        
        # Subclasses must populate this dict mapping class IDs to strings (e.g., {0: 'person'})
        self.yolo_names = {} 

    @abstractmethod
    def _forward_yolo(self, frame_bgr):
        """Must be implemented by subclass. Returns Ultralytics YOLO Results object."""
        pass

    @abstractmethod
    def _forward_depth(self, frame_bgr):
        """Must be implemented by subclass. Returns a [H, W] normalized torch.Tensor on self.device."""
        pass

    def _evaluate_spatial_physics(self, binary_mask: torch.Tensor, depth_tensor: torch.Tensor) -> dict:
        """
        Zero-Copy VRAM fusion mathematics. 
        Calculates depth variance (internal volume) and step-off (environmental context).
        """
        mask_float = binary_mask.float().unsqueeze(0).unsqueeze(0)
        
        # Mathematical Dilation to create the environmental Halo
        dilated_mask = mask_float
        for _ in range(self.iterations):
            dilated_mask = F.max_pool2d(
                dilated_mask, 
                kernel_size=self.kernel_size, 
                stride=1, 
                padding=self.kernel_size // 2
            )
        
        # XOR to get just the ring around the object
        halo_mask = (dilated_mask - mask_float).squeeze(0).squeeze(0).bool()
        
        # Native GPU Slicing
        object_depths = depth_tensor[binary_mask]
        halo_depths   = depth_tensor[halo_mask]
        
        if object_depths.numel() > 10 and halo_depths.numel() > 10:
            object_var    = float(torch.var(object_depths).item())
            object_median = float(torch.median(object_depths).item())
            halo_median   = float(torch.median(halo_depths).item())
            step_delta    = object_median - halo_median
        else:
            object_var = object_median = halo_median = step_delta = 0.0

        has_volume = object_var > self.min_volume
        pops_out   = step_delta > self.min_step_delta
        
        return {
            "object_variance": object_var,
            "step_delta": step_delta,
            "has_volume": has_volume,
            "pops_out": pops_out,
            "is_real_3d": has_volume and pops_out
        }

    @torch.inference_mode()
    def predict(self, frame_bgr) -> list:
        """
        The main inference pipeline. Standardized across all execution providers.
        """
        # 1. Forward Passes (Delegated to Subclass)
        yolo_results = self._forward_yolo(frame_bgr)
        depth_tensor = self._forward_depth(frame_bgr)
        
        fused_predictions = []
        
        # 2. VRAM Fusion & Payload Generation
        if yolo_results.boxes is not None and yolo_results.masks is not None:
            for i, box in enumerate(yolo_results.boxes):
                class_id = int(box.cls[0])
                class_name = self.yolo_names.get(class_id, "unknown")
                
                if class_name == 'person':
                    confidence = float(box.conf[0])
                    xyxy = box.xyxy[0].to(torch.int32)
                    x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
                    
                    binary_mask = yolo_results.masks.data[i].bool()
                    metrics = self._evaluate_spatial_physics(binary_mask, depth_tensor)
                    
                    fused_predictions.append({
                        "box": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class": class_name,
                        "depth_metrics": metrics,
                        "is_real_3d": metrics["is_real_3d"]
                    })
                    
        return fused_predictions
