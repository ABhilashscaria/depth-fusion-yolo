# 🛡️ Zero-Shot Spatial Verification Engine (Edge AI)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-GPU_Accelerated-005CED.svg)](https://onnxruntime.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

**A VRAM-optimized, split-stream Computer Vision architecture that extracts 3D spatial awareness from standard RGB feeds—eliminating 2D spatial spoofs, specular reflections, and hallucinations in industrial tracking pipelines without the need for additional hardware sensors.**

> **[🎥 Insert a GIF here showing a side-by-side Live Camera Test: A real human is bounded in Green, while a 2D poster/cutout of a human is bounded in Red]**

---

## 🧬 Project Origins & The Economic Premise

Originating from theoretical research conducted during my **MSc in Artificial Intelligence at the University of Southampton**, this architecture has been heavily refactored into a real-time, modular, production-grade edge system.

**The Economic Thesis:** Industrial autonomy (AGVs), perimeter security, and retail analytics currently rely heavily on standard 2D RGB camera infrastructure. Upgrading these systems to possess perfect 3D spatial awareness typically requires retrofitting environments with expensive LiDAR or Time-of-Flight (ToF) sensors. 

This project demonstrates a pure software solution: **extracting volumetric physics directly from existing 2D video streams**. By democratizing 3D spatial awareness, we achieve robust spoof rejection and depth verification on sub-$300 edge GPUs (e.g., RTX 4060, NVIDIA Jetson) without requiring a single hardware modification to the facility.

## ⚙️ The Engineering Problem

Modern visual pipelines rely on 2D object detection feeding into temporal trackers (e.g., DeepSORT, ByteTrack). However, these pipelines critically fail when encountering 2D environmental artifacts:
*   **The Ghost ID Crisis:** Specular reflections on glass or polished floors cause trackers to assign IDs to "ghost" entities, leading to track fragmentation and corrupted analytics.
*   **The False Stop Liability:** Autonomous Guided Vehicles (AGVs) trigger emergency brakes when 2D sensors detect human safety posters, wall murals, or reflections on shrink-wrapped pallets, causing massive operational downtime.

## 🚀 The Solution: Dual-Engine Spatial Architecture

This project introduces a **Spatial Verification Pipeline** that intercepts detections *before* they corrupt the tracking loop. By fusing YOLO11 Instance Segmentation with Depth Anything V2 natively in VRAM, the system calculates relative environmental physics without the PCIe bottleneck.

### Hardware Optimization & Zero-Copy Math
1.  **Split-Stream Processing:** Maintains standard YOLO bounding box coordinates `[x1, y1, x2, y2]` for legacy tracking loops, while routing high-resolution boolean segmentation masks to the 3D verification engine.
2.  **Zero-Copy VRAM Execution:** YOLO masks and Hugging Face depth tensors are cross-multiplied entirely on the GPU (`cuda:0`). By utilizing native boolean indexing (`depth_tensor[binary_mask]`), no data crosses to the CPU for NumPy processing until the final payload generation.
3.  **Dynamic Halo Calibration:** Eliminates brittle, hardcoded variance thresholds. The system uses PyTorch 2D Max Pooling to mathematically dilate the object's mask natively on the GPU, creating an environmental "Halo" ring. The system self-calibrates to any lighting or background texture by calculating the physical step-off between the object and this immediate background.

## 🚧 Overcoming Monocular Hallucination (Dual-Gate Physics)

During physical stress testing, a critical limitation of Vision Transformers was isolated: high-definition 2D cutouts with printed shadows trick monocular models into hallucinating 3D volume. To solve this in software, the architecture employs a **Dual-Gate Physics** approach:

*   **Gate 1 - Internal Volume (`has_volume`):** Measures the isolated depth variance of the silhouette. Evaluates if the object possesses internal 3D curves. *(Successfully filters out flat specular reflections and glass noise).*
*   **Gate 2 - Environmental Step-Off (`pops_out`):** Compares the object's median depth to the dilated Halo mask. *(Successfully filters out hallucinated depth printed flush against a wall, such as posters or murals).*

---

## 📊 System Telemetry & Benchmarks

*Tested on an NVIDIA RTX 4060 (8GB) utilizing CUDA-accelerated PyTorch / ONNX Runtime.*

| Metric | Performance | Business Impact |
| :--- | :--- | :--- |
| **Throughput** | `35.79 FPS` | Exceeds standard real-time 30 FPS camera hardware limits. |
| **Average Latency** | `27.94 ms` | Instantaneous spatial verification; prevents lag in control loops. |
| **P99 Latency (Tail)** | `30.86 ms` | Highly stable execution preventing garbage-collection spikes. |
| **Peak VRAM Active** | `310.42 MB` | Extremely lightweight; leaves 95% of GPU memory free for temporal tracking. |

---

## 🛠️ Quickstart

The repository uses a `config/default.yaml` to manage all execution parameters, allowing dynamic switching between PyTorch and ONNX backends without modifying source code.

### Option 1: Docker (Recommended)
Launch the pipeline instantly using the pre-configured CUDA runtime environment. This mounts your local webcam (`/dev/video0`) directly to the AI pipeline.

```bash
git clone https://github.com/yourusername/zero-shot-spatial-engine.git
cd zero-shot-spatial-engine
docker-compose up --build
```

### Option 2: Local Installation (Linux/Windows)
Ensure you have CUDA 12.1+ installed on your host system.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install PyTorch with specific CUDA bindings
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Run the stream manager
python -m src.pipeline.stream_manager
```

---

## 🏗️ Repository Architecture


```text
├── config/
│   └── default.yaml         # Centralized hyperparameters & thresholds
├── src/
│   ├── engine/
│   │   ├── base.py          # Abstract Base Class containing VRAM math
│   │   ├── pytorch_engine.py 
│   │   └── onnx_engine.py    
│   ├── pipeline/
│   │   └── stream_manager.py # Threaded ingestion with auto-reconnect
│   └── utils/
│       └── config_loader.py 
└── tests/                   # Pytest suite validating mathematical gating
```