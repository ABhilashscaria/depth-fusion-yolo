import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

print("🚀 Initiating Phase 3: Hardware-Agnostic Compilation")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 1. YOLO11-Seg ONNX Export ---
print("\n[1/2] Exporting YOLO11-Seg (Spatial Engine)...")
yolo_model = YOLO("yolo11n-seg.pt")
# 'half=True' triggers FP16 compilation for 2x speed on RTX 4060
yolo_model.export(
    format="onnx", 
    half=True, 
    dynamic=True, # Allows variable image sizes
    opset=14
)
print("✅ YOLO ONNX Export Complete (Saved as yolo11n-seg.onnx)")


# --- 2. Depth Anything V2 ONNX Export ---
print("\n[2/2] Exporting Depth Anything V2 (Volumetric Engine)...")
model_id = "depth-anything/Depth-Anything-V2-Small-hf"

print("-> Loading Hugging Face transformer...")
depth_model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
depth_model.eval() # Lock dropout and batchnorm layers

# To trace a graph, PyTorch needs a "dummy" input to run through the layers.
# Depth Anything V2 natively expects a 518x518 tensor.
dummy_input = torch.randn(1, 3, 518, 518, device=device)

print("-> Tracing PyTorch graph and fusing layers...")
torch.onnx.export(
    depth_model,
    dummy_input,
    "depth_anything_v2.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True, # Merges static mathematical operations for speed
    input_names=['pixel_values'],
    output_names=['predicted_depth'],
    # Dynamic axes allow us to pass 1080p or 720p frames at runtime 
    # instead of being locked to the 518x518 dummy size
    dynamic_axes={
        'pixel_values': {0: 'batch_size', 2: 'height', 3: 'width'},
        'predicted_depth': {0: 'batch_size', 1: 'height', 2: 'width'}
    }
)
print("✅ Depth ONNX Export Complete (Saved as depth_anything_v2.onnx)")
print("\n🎉 Phase 3 Compilation Successful. You are ready to integrate ONNX Runtime.")
