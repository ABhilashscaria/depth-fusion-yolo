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
    
    @torch.inference_mode()
    def predict_raw(self, image_path_or_frame):
        """
        Executes a concurrent forward pass through both models.
        Returns the raw 2D bounding boxes and the raw 3D depth tensor.
        """
        # Handle both file paths and raw OpenCV frames
        if isinstance(image_path_or_frame, str):
            frame = cv2.imread(image_path_or_frame)
        else:
            frame = image_path_or_frame

        # Hugging Face pipelines expect RGB images, while OpenCV loads in BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to a PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # 1. Execute 2D Spatial Pass
        # verbose=False stops YOLO from printing log spam every single frame
        yolo_results = self.yolo(frame, verbose=False)[0]

        # 2. Execute 3D Depth Pass
        depth_results = self.depth_estimator(pil_image)

        # 3. VRAM Tensor Extraction
        # We extract the depth map and immediately convert it to a normalized PyTorch tensor
        # sitting on the GPU, ready for Day 4's fusion math.
        depth_array = np.array(depth_results['depth'], dtype=np.float32) / 255.0
        depth_tensor = torch.from_numpy(depth_array).to(self.device)

        return yolo_results.boxes, depth_tensor




if __name__ == "__main__":
    # Initialize the engine
    engine = DepthConditionedYOLO()
    
    # Run a test frame
    print("\n🚀 Executing concurrent forward pass...")
    boxes, depth_tensor = engine.predict_raw('test.jpg')
    
    print("\n--- Pipeline Telemetry ---")
    print(f"Total Detections: {len(boxes)}")
    print(f"Depth Tensor Shape: {depth_tensor.shape}")
    print(f"Depth Tensor Device: {depth_tensor.device}")
    
    # Check VRAM footprint
    allocated_vram = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"Active VRAM Footprint: {allocated_vram:.2f} MB")