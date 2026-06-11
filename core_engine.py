import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

class DepthConditionedYOLO:
    def __init__(self, yolo_model='yolo11n-seg.pt', depth_model='depth-anything/Depth-Anything-V2-Small-hf', device=0):
        """
        Initializes the Split-Stream Architecture:
        YOLO11-Seg for precise spatial tracking + Depth Anything V2 for 3D verification.
        """
        print("⚡ Initializing VRAM Concurrency & Dynamic Calibration...")
        self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        
        # 1. Load the Spatial Segmentation Brain
        print("-> Loading YOLO11n-Seg to VRAM...")
        self.yolo = YOLO(yolo_model)
        
        # 2. Load the 3D Brain
        print("-> Loading Depth Anything V2 to VRAM...")
        self.depth_estimator = pipeline(
            task="depth-estimation", 
            model=depth_model, 
            device=device
        )
        print("✅ Dual-Engine Initialization Complete.")

    @torch.inference_mode()
    def predict(self, image_path_or_frame, min_volume=0.005, min_step_delta=0.05):
        """
        Executes the edge pipeline using Zero-Copy Masking and Dual-Gate Physics.
        Self-calibrates to the environment by comparing the object to its immediate background.
        """
        # Handle OpenCV array or file path
        if isinstance(image_path_or_frame, str):
            frame = cv2.imread(image_path_or_frame)
        else:
            frame = image_path_or_frame

        # Format translations
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # --- 1. Concurrent Forward Pass ---
        # retina_masks=True forces pixel-perfect alignment with the high-res depth map
        yolo_results = self.yolo(frame, verbose=False, retina_masks=True)[0]
        depth_results = self.depth_estimator(pil_image)
        
        # Load normalized depth onto GPU
        depth_array = np.array(depth_results['depth'], dtype=np.float32) / 255.0
        depth_tensor = torch.from_numpy(depth_array).to(self.device)

        fused_predictions = []

        if yolo_results.boxes is not None and yolo_results.masks is not None:
            for i, box in enumerate(yolo_results.boxes):
                class_name = self.yolo.names[int(box.cls[0])]

                if class_name == 'person':
                    confidence = float(box.conf[0])
                    
                    # Store tracking coordinates
                    xyxy = box.xyxy[0].to(torch.int32)
                    x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
                    
                    # --- 2. Extract Core Mask & Generate Halo Ring ---
                    binary_mask = yolo_results.masks.data[i].bool()
                    
                    # Convert to [1, 1, H, W] for VRAM max pooling
                    mask_float = binary_mask.float().unsqueeze(0).unsqueeze(0)
                    
                    # Expand mask by ~15 pixels to capture the immediate background wall
                    kernel_size = 31 
                    dilated_mask = F.max_pool2d(
                        mask_float, 
                        kernel_size=kernel_size, 
                        stride=1, 
                        padding=kernel_size // 2
                    )
                    
                    # Subtract the core object to leave strictly the outer Halo
                    halo_mask_float = dilated_mask - mask_float
                    halo_mask = halo_mask_float.squeeze(0).squeeze(0).bool()

                    # --- 3. Zero-Copy Tensor Extraction ---
                    object_depths = depth_tensor[binary_mask]
                    halo_depths = depth_tensor[halo_mask]

                    # --- 4. Environmental Physics Math ---
                    if object_depths.numel() > 10 and halo_depths.numel() > 10:
                        object_var = float(torch.var(object_depths).item())
                        object_median = float(torch.median(object_depths).item())
                        halo_median = float(torch.median(halo_depths).item())
                        
                        # Calculate the physical step-off distance
                        step_delta = object_median - halo_median
                    else:
                        object_var, object_median, halo_median, step_delta = 0.0, 0.0, 0.0, 0.0

                    # --- 5. Dual-Logic Gating ---
                    # Gate 1: Object must have internal 3D curves (kills flat reflections)
                    has_volume = object_var > min_volume 
                    
                    # Gate 2: Object must stand in front of its background (kills flat posters)
                    pops_out = step_delta > min_step_delta 

                    is_real_3d = has_volume and pops_out

                    # --- 6. Compile Payload ---
                    prediction_payload = {
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
                    }
                    fused_predictions.append(prediction_payload)

        return fused_predictions


if __name__ == "__main__":
    import json
    
    # Initialize the engine
    engine = DepthConditionedYOLO()
    
    # Run the physical stress-test image
    print("\n🚀 Executing Dual-Gate Pipeline...")
    print("\n🚀 Profiling pt Runtime Performance metrics...")
    start_time = time.perf_counter()
    results = engine.predict('rajatest.jpg') # Swap this string to test your different edge-case images
    latency = (time.perf_counter() - start_time) * 1000
    print(f"⏱️ Single Frame Processing Latency: {latency:.2f} ms ({1000/latency:.1f} FPS equivalence)")
    print("\n--- Final Output Payload ---")
    print(json.dumps(results, indent=4))
    
    # Verify strict VRAM constraints
    allocated_vram = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"\nActive VRAM Footprint: {allocated_vram:.2f} MB")
