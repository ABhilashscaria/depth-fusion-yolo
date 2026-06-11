import cv2
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
from ultralytics import YOLO
import time

class ONNXSpatialEngine:
    def __init__(self, yolo_onnx_path='yolo11n-seg.onnx', depth_onnx_path='depth_anything_v2.onnx'):
        """
        Initializes the Production ONNX Runtime Pipeline utilizing CUDA Execution Providers.
        """
        print("⚡ Initializing High-Performance ONNX Runtime Execution Context...")
        
        # Configure GPU execution settings for ONNX Runtime
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # 2GB Limit allocation
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            'CPUExecutionProvider'
        ]
        
        # 1. Load YOLO11-Seg ONNX Model
        print("-> Loading YOLO11n-Seg ONNX Engine...")
        self.yolo = YOLO(yolo_onnx_path, task='segment')
        
        # 2. Load Depth Anything V2 ONNX Session
        print("-> Loading Depth Anything V2 ONNX Session...")
        self.depth_session = ort.InferenceSession(depth_onnx_path, providers=providers)
        self.depth_input_name = self.depth_session.get_inputs()[0].name
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("✅ ONNX Pipeline Initialization Complete.")

    def predict(self, image_path_or_frame, min_volume=0.005, min_step_delta=0.05):
        """
        Executes inference using ONNX Runtime with zero-copy VRAM parsing for masking arrays.
        """
        if isinstance(image_path_or_frame, str):
            frame = cv2.imread(image_path_or_frame)
        else:
            frame = image_path_or_frame

        h, w, _ = frame.shape

        # --- 1. YOLO11-Seg ONNX Inference ---
        yolo_results = self.yolo(frame, verbose=False, retina_masks=True)[0]

        # --- 2. Depth Anything V2 ONNX Inference ---
        # Preprocessing matching the transformer specifications
        img_resized = cv2.resize(frame, (518, 518))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Change shape formatting from HWC to BCHW
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_input = np.expand_dims(img_transposed, axis=0)

        # Run ONNX Session
        onnx_outputs = self.depth_session.run(None, {self.depth_input_name: img_input})
        raw_depth = onnx_outputs[0].squeeze(0) # Shape: (518, 518)

        # Normalize depth map output down to [0, 1] range
        depth_min, depth_max = raw_depth.min(), raw_depth.max()
        if depth_max - depth_min > 1e-5:
            depth_normalized = (raw_depth - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(raw_depth)

        # Move array directly to VRAM to execute fast mask math
        depth_tensor = torch.from_numpy(depth_normalized).to(self.device)
        
        # Scale depth tensor back to native video frame coordinates natively on GPU
        depth_tensor = F.interpolate(
            depth_tensor.unsqueeze(0).unsqueeze(0), 
            size=(h, w), 
            mode="bilinear", 
            align_corners=False
        ).squeeze(0).squeeze(0)

        fused_predictions = []

        if yolo_results.boxes is not None and yolo_results.masks is not None:
            for i, box in enumerate(yolo_results.boxes):
                class_name = self.yolo.names[int(box.cls[0])]

                if class_name == 'person':
                    confidence = float(box.conf[0])
                    xyxy = box.xyxy[0].to(torch.int32)
                    x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
                    
                    # Core Mask Extraction
                    binary_mask = yolo_results.masks.data[i].bool()
                    mask_float = binary_mask.float().unsqueeze(0).unsqueeze(0)
                    
                    # Halo Mask Generation
                    kernel_size = 31 
                    dilated_mask = F.max_pool2d(
                        mask_float, 
                        kernel_size=kernel_size, 
                        stride=1, 
                        padding=kernel_size // 2
                    )
                    halo_mask = (dilated_mask - mask_float).squeeze(0).squeeze(0).bool()

                    # Tensor-level extraction
                    object_depths = depth_tensor[binary_mask]
                    halo_depths = depth_tensor[halo_mask]

                    if object_depths.numel() > 10 and halo_depths.numel() > 10:
                        object_var = float(torch.var(object_depths).item())
                        object_median = float(torch.median(object_depths).item())
                        halo_median = float(torch.median(halo_depths).item())
                        step_delta = object_median - halo_median
                    else:
                        object_var, object_median, halo_median, step_delta = 0.0, 0.0, 0.0, 0.0

                    has_volume = object_var > min_volume 
                    pops_out = step_delta > min_step_delta 
                    is_real_3d = has_volume and pops_out

                    fused_predictions.append({
                        "box": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class": class_name,
                        "depth_metrics": {
                            "object_variance": object_var,
                            "step_delta": step_delta,
                            "has_volume": has_volume,
                            "pops_out": pops_out
                        },
                        "is_real_3d": is_real_3d
                    })

        return fused_predictions

if __name__ == "__main__":
    import json
    
    # Instantiate the ONNX pipeline
    engine = ONNXSpatialEngine()
    
    # Warm up pass to remove initialization delay metrics
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _ = engine.predict(dummy_frame)
    
    # Benchmarking single-frame latencies
    print("\n🚀 Profiling ONNX Runtime Performance metrics...")
    start_time = time.perf_counter()
    
    results = engine.predict('rajatest.jpg')
    
    latency = (time.perf_counter() - start_time) * 1000
    print(f"⏱️ Single Frame Processing Latency: {latency:.2f} ms ({1000/latency:.1f} FPS equivalence)")
    
    print("\n--- Payload Verification ---")
    print(json.dumps(results, indent=4))
    
    # Check VRAM state
    allocated_vram = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"\nActive Helper VRAM Footprint: {allocated_vram:.2f} MB")
