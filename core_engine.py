import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

class DepthConditionedYOLO:
    def __init__(self, yolo_model='yolo11n.pt', depth_model='depth-anything/Depth-Anything-V2-Small-hf', device=0):
        """
        Initializes both the 2D spatial detector and 3D depth estimator 
        concurrently into the designated GPU VRAM.
        """
        print("⚡ Initializing VRAM Concurrency...")
        self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        
        # 1. Load the 2D Brain (YOLO11)
        print("-> Loading YOLO11n to VRAM...")
        self.yolo = YOLO(yolo_model)
        
        # 2. Load the 3D Brain (Depth Anything V2)
        print("-> Loading Depth Anything V2 to VRAM...")
        self.depth_estimator = pipeline(
            task="depth-estimation", 
            model=depth_model, 
            device=device
        )
        print("✅ Dual-Engine Initialization Complete.")
    def _isolate_foreground_variance_gpu(self, depth_crop, tolerance=0.40):
        """
        Solves the 'Cardboard Cutout' background bleeding problem using pure VRAM tensor math.
        """
        # 1. Size check (if box is too small, skip to avoid tensor errors)
        if depth_crop.numel() < 10:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        # 2. Find the center of mass (Median Depth) natively on GPU
        median_depth = torch.median(depth_crop)

        # 3. Define the physical tolerance band
        lower_bound = median_depth - (median_depth * tolerance)
        upper_bound = median_depth + (median_depth * tolerance)

        # 4. Boolean Masking (Zero-Copy Isolation)
        foreground_mask = (depth_crop >= lower_bound) & (depth_crop <= upper_bound)
        foreground_pixels = depth_crop[foreground_mask]

        # 5. True Variance Calculation
        if foreground_pixels.numel() > 1: # Variance requires at least 2 pixels
            variance = torch.var(foreground_pixels)
        else:
            variance = torch.tensor(0.0, device=self.device)

        return variance, median_depth
    
    @torch.inference_mode()
    def predict(self, image_path_or_frame, variance_threshold=0.0005):
        """
        Executes the full edge pipeline: Ingestion -> Concurrency -> VRAM Fusion -> Logic Gating.
        Returns a structured list of verified detections.
        """
        from PIL import Image # Ensure this is imported at the top of your file
        
        if isinstance(image_path_or_frame, str):
            frame = cv2.imread(image_path_or_frame)
        else:
            frame = image_path_or_frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # --- 1. Concurrent Forward Pass ---
        yolo_results = self.yolo(frame, verbose=False)[0]
        depth_results = self.depth_estimator(pil_image)
        
        depth_array = np.array(depth_results['depth'], dtype=np.float32) / 255.0
        depth_tensor = torch.from_numpy(depth_array).to(self.device)

        fused_predictions = []

        # --- 2. Zero-Copy VRAM Fusion ---
        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.yolo.names[cls_id]

            if class_name == 'person':
                confidence = float(box.conf[0])
                
                # Keep coordinates on the GPU, cast to integers for tensor slicing
                xyxy = box.xyxy[0].to(torch.int32)
                x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

                # Slicing the depth tensor natively on CUDA
                depth_crop = depth_tensor[y1:y2, x1:x2]

                # Apply background isolation math
                variance, median_depth = self._isolate_foreground_variance_gpu(depth_crop)

                # --- 3. Logic Gating ---
                # Move scalar values back to CPU for the final Python dictionary payload
                variance_val = variance.item()
                is_real_3d = variance_val > variance_threshold

                prediction_payload = {
                    "box": [x1.item(), y1.item(), x2.item(), y2.item()],
                    "confidence": confidence,
                    "class": class_name,
                    "depth_metrics": {
                        "median": median_depth.item(),
                        "variance": variance_val
                    },
                    "is_real_3d": is_real_3d
                }
                fused_predictions.append(prediction_payload)

        return fused_predictions




if __name__ == "__main__":
    # Initialize the engine
    import json
    engine = DepthConditionedYOLO()
    
    print("\n🚀 Executing full VRAM fusion pipeline...")
    # Use the same image that gave you 2 detections earlier
    results = engine.predict('maxresdefault.jpg') 
    
    print("\n--- Final Output Payload ---")
    print(json.dumps(results, indent=4))
    
    # Check VRAM footprint
    allocated_vram = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"Active VRAM Footprint: {allocated_vram:.2f} MB")
