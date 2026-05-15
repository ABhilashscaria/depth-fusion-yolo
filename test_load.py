import torch
from ultralytics import YOLO
from transformers import pipeline

def test_vram_load():
    print("--- GPU Diagnostics ---")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("ERROR: CUDA not found. Exiting.")
        return
        
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

    try:
        print("--- Loading YOLO11n ---")
        # This will automatically download the yolo11n.pt weights if missing
        yolo = YOLO('yolo11n.pt')
        yolo.to('cuda') 
        print("✅ YOLO11n loaded successfully into VRAM.")

        print("\n--- Loading Depth Anything V2 (Small) ---")
        # This will download the HuggingFace weights if missing
        depth_estimator = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 # 0 refers to the first CUDA GPU
        )
        print("✅ Depth Anything V2 loaded successfully into VRAM.")
        
        print("\n🚀 SUCCESS: Both models are concurrently sitting in VRAM without OOM errors!")
        
    except RuntimeError as e:
        if "Out of memory" in str(e):
            print("\n❌ CRITICAL ERROR: VRAM Out of Memory (OOM). We need to optimize.")
        else:
            print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    test_vram_load()
