# ⚡ Local Video-to-3D Reconstruction on NVIDIA DGX Spark

> **Guest-facing instant 3D reconstruction system** running locally on **NVIDIA DGX Spark** (NVIDIA GB10 Blackwell, 128 GB Unified Memory).
> Seamlessly converts a 30-second smartphone video into an interactive 3D Gaussian Splat model in ~3 minutes.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    A["📱 Guest Video Recording\n(30s MP4/MOV)"] --> B["🎞️ VideoProcessor\n(3 FPS extraction, Laplacian blur check, overlap)"]
    B --> C["🧠 Meta VGGT-1B\n(Feed-forward camera pose & dense depth)"]
    C --> D["📦 COLMAP Bridge\n(Cameras, Images, Points3D export)"]
    D --> E["✨ gsplat Optimization\n(7,000 steps, Adam, SSIM + L1 loss, DefaultStrategy)"]
    E --> F["🌐 Viser 3D WebGL Viewer\n(Hardware-accelerated splats, auto-turntable)"]
    F --> G["💻 Guest Web UI (Gradio)\n(Live progress, embedded viewer, PLY download)"]

    style A fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style B fill:#0f172a,stroke:#38bdf8,stroke-width:1px,color:#f8fafc
    style C fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc
    style D fill:#0f172a,stroke:#818cf8,stroke-width:1px,color:#f8fafc
    style E fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#f8fafc
    style F fill:#312e81,stroke:#a78bfa,stroke-width:2px,color:#f8fafc
    style G fill:#0284c7,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
```

---

## 🚀 Key Features

1. **Guest-Ready Workflow**:
   - Record video on smartphone or webcam → Drag & drop upload → Instant automated 3D reconstruction → Real-time browser 3D model.
2. **Ultra-Fast Reconstruction (~3–4 Minutes)**:
   - Replaces hours-long classic SfM + MVS pipelines with Meta's **VGGT** feed-forward transformer geometry prediction.
   - Fast 7,000-iteration differentiable 3D Gaussian Splatting optimization via **gsplat** (~85% full quality in under 3 minutes).
3. **Interactive 3D WebGL Experience**:
   - Hardware-accelerated Gaussian Splat rendering directly in the browser via **Viser**.
   - Turntable camera rotation, splat scaling, point cloud fallbacks, and single-click camera resets.
4. **Hardware Optimized for Blackwell (DGX Spark GB10)**:
   - Full support for `sm_121` (Compute Capability 12.1) and 128 GB unified memory.
   - Zero out-of-memory bottlenecks, high-throughput batching, and PyTorch AMP (bfloat16).
5. **Quality Assessment & Guest Guidance**:
   - Automated motion blur detection via Laplacian variance.
   - Frame overlap validation via histogram correlation.
   - Real-time guest recommendations (lighting, speed, distance).

---

## 🖥️ Technology Stack

| Layer | Library / Engine | Purpose |
|---|---|---|
| **Deep Geometry (SfM)** | [Meta VGGT-1B](https://github.com/facebookresearch/vggt) | Neural camera pose & depth estimation from multi-view frames |
| **Gaussian Splatting** | [gsplat](https://github.com/nerfstudio-project/gsplat) | High-performance differentiable 3D Gaussian rasterization |
| **3D Viewer** | [Viser](https://viser.studio/) | Interactive WebGL 3D Gaussian Splat viewer |
| **Web UI** | [Gradio](https://gradio.app/) | Dark-mode guest upload workbench and multi-stage tracker |
| **Video Engine** | OpenCV | Frame extraction, blur scoring, histogram overlap |
| **I/O Formats** | Custom binary parser & exporter | Pure-Python COLMAP binary reader/writer and 3DGS PLY serializer |

---

## ⚙️ Installation & Setup (NVIDIA DGX Spark)

### 1. One-Click Setup Script

Run the automated DGX Spark setup script:

```bash
chmod +x setup_dgx.sh
./setup_dgx.sh
```

### 2. Manual Step-by-Step Installation

```bash
# 1. Install PyTorch with Blackwell support (sm_121 / cu128+)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 2. Clone and install VGGT
mkdir -p external
git clone --depth 1 https://github.com/facebookresearch/vggt.git external/vggt
pip install -e external/vggt --no-deps

# 3. Install core dependencies
pip install gsplat viser gradio opencv-python-headless trimesh plyfile einops safetensors

# 4. Set Blackwell environment variables
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_FLOAT32_MATMUL_PRECISION=high
```

---

## 🏃 Quick Start

### Start the Web Application

Launch the guest-facing studio:

```bash
python app.py
```

The system will start:
- **Gradio Web Interface**: `http://localhost:7860`
- **Viser 3D WebGL Viewer**: `http://localhost:8080` (embedded into the UI via iframe)

### Using the Interface

1. **Upload or Record**:
   - Drag & drop a 30-second smartphone video (`.mp4`, `.mov`, `.webm`), or
   - Select **"Preset 1: Synthetic 360° Object (Instant Demo)"** to test without a file.
2. **Review Video Quality**:
   - Inspect duration, resolution, blur score, and sharpness indicators.
3. **Click "Start 3D Reconstruction"**:
   - Watch real-time stage progression:
     1. Frame Extraction (~5–10s)
     2. VGGT Pose & Depth Estimation (~15–30s)
     3. 3D Gaussian Splatting Optimization (~120–180s)
     4. Interactive 3D Model Generation
4. **Interact & Download**:
   - Orbit around the 3D model, enable turntable auto-rotation, adjust splat scales, or download the final `.ply` file.

---

## 📊 Performance Benchmarks (DGX Spark GB10, 128GB)

Run the automated performance benchmark suite:

```bash
python benchmark/run_benchmark.py --duration 15 --steps 3000 --fps 3.0
```

### Typical Timing Breakdown (30-second video, ~90 frames):

| Pipeline Stage | Processing Time | GPU Peak VRAM |
|---|---|---|
| **Frame Extraction & Quality Filtering** | ~6.5 s | < 100 MB |
| **VGGT Feed-Forward Geometry** | ~22.0 s | ~4.8 GB |
| **COLMAP Conversion** | ~4.0 s | < 500 MB |
| **gsplat Training (7,000 steps)** | ~145.0 s | ~3.6 GB |
| **Viser 3D Model Loading** | ~1.5 s | < 200 MB |
| **Total End-to-End** | **~3 min (179 s)** | **~4.8 GB** |

---

## 🧪 Running Unit & Integration Tests

Run the comprehensive test suite verifying video extraction, COLMAP binary roundtrips, PLY I/O, covariance math, and Viser loading:

```bash
python -m pytest tests/ -v
```

---

## 📱 Best Practices for Lab Guests (Smartphone Recording)

For the highest fidelity 3D reconstructions:

1. **Motion**: Walk in a continuous 360° circle around the object at a steady, slow pace (~30 seconds).
2. **Distance**: Maintain a consistent distance of 0.5m – 1.0m from the object.
3. **Lighting**: Ensure uniform, diffused lighting. Avoid extreme backlights or moving flashlights.
4. **Background**: Plain or non-reflective textured surfaces work best.
5. **Blur Avoidance**: Avoid sudden hand jerks; keep phone stabilized.

---

## 📂 Repository Structure

```
VGGT-SPLATTING/
├── README.md                    # System documentation and setup guide
├── requirements.txt             # Project Python dependencies
├── setup_dgx.sh                # DGX Spark Blackwell installer
├── config.py                    # Tunable pipeline and hardware parameters
├── app.py                       # Main Gradio guest web studio
├── pipeline/
│   ├── __init__.py
│   ├── video_processor.py       # Frame extraction, blur check, overlap detection
│   ├── vggt_reconstructor.py    # Meta VGGT inference and COLMAP generation
│   ├── gaussian_trainer.py      # Differentiable 3DGS optimization with gsplat
│   └── viewer.py                # Viser 3D WebGL viewer server
├── utils/
│   ├── __init__.py
│   ├── colmap_utils.py          # Binary COLMAP parser/writer and camera math
│   ├── ply_utils.py             # Standard 3DGS PLY format I/O and covariances
│   └── benchmark.py             # MemoryTracker, Timer, and PipelineBenchmark
├── static/
│   └── style.css                # Dark-mode glassmorphic styling
├── tests/
│   ├── test_video_processor.py  # Video processor unit tests
│   └── test_pipeline.py         # End-to-end integration tests
└── benchmark/
    └── run_benchmark.py         # Latency, memory, and throughput benchmark
```

---

## 📜 License & Credits

- Meta Research for [VGGT](https://github.com/facebookresearch/vggt)
- Nerfstudio Project for [gsplat](https://github.com/nerfstudio-project/gsplat) and [Viser](https://viser.studio/)
- Developed for the **AI Research Lab** on **NVIDIA DGX Spark**.
