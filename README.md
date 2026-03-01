# 🎬 AI Subtitle Remover (CRAFT + LaMa Cleaner)

A high-performance, AI-driven video tool that automatically detects subtitles and removes them using state-of-the-art inpainting. Unlike simple blurring, this tool "reconstructs" the background behind the text for a seamless, professional look.

---

## ✨ Key Features

* **🔍 Precision Detection**: Uses the **CRAFT** (Character Region Awareness for Text Detection) model to find text with pixel-perfect accuracy.
* **🖼️ Professional Inpainting**: Powered by **LaMa** (Resolution-robust Large Mask Inpainting) to fill in the gaps without ghosting or artifacts.
* **🚀 GPU Optimized**: Automatically detects and leverages **CUDA (NVIDIA GPUs)** for 10-20x faster processing.
* **⚡ Pipelined Workflow**: Uses a **Producer-Consumer** architecture. While the GPU is "inpainting" frame 1, the CPU is already "detecting" text for frame 2, ensuring zero idle time.
* **🧹 Auto-Cleaning**: All temporary frames (thousands of images) are handled in system temp folders and automatically wiped upon completion.
* **📊 Real-time Progress**: Integrated `tqdm` progress bars with accurate ETA tracking.

---

## 🛠️ Installation

This project uses [uv](https://github.com/astral-sh/uv) for ultra-fast, reliable dependency management.

1. **Clone the Repository**

2. **Setup Environment**:

   ```bash
   pip install uv  # If you don't have it
   uv sync         # Automatically creates .venv and installs everything
   ```

---

## 🚀 Usage

### Local Run

1. Place your video in the `videos/` directory.
2. Update the `VIDEO_PATH` in `src/subtitle_remover.py` if needed.
3. Run the script:

   ```bash
   chmod +x run.sh
   ./run.sh
   ```

### Google Colab (Recommended for Free T4 GPU)

1. Open a new Colab notebook with a **T4 GPU Runtime**.
2. Run the following commands:

   ```bash
   !git clone <your-repo-url>
   %cd <your-repo-name>
   !pip install uv
   !uv sync
   !./run.sh
   ```

---

## 🧠 How It Works

1. **Extraction**: The original audio is ripped, and the video is split into a stream of high-quality `.png` frames.
2. **Detection (Producer)**: A background thread runs the CRAFT model to identify text bounding boxes and pushes them to a processing queue.
3. **Removal (Consumer)**: The main GPU thread pulls frames from the queue, creates a binary mask, and runs the LaMa inpainting model to reconstruct the background.
4. **Re-assembly**: FFmpeg stitches the "cleaned" frames back together and re-attaches the original audio with zero quality loss.

---

## 📝 License

This project is open-source and available under the MIT License.
