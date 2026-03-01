import cv2
import os
import tempfile
import subprocess
import shutil
import torch
import numpy as np
import time
import sys

# --- FIX: craft-text-detector compatibility with newer torchvision ---
# craft-text-detector tries to import 'model_urls' from 'torchvision.models.vgg',
# which was removed in recent versions. We monkey patch it here before importing Craft.
try:
    import torchvision.models.vgg as vgg
    if not hasattr(vgg, 'model_urls'):
        vgg.model_urls = {
            'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        }
    elif 'vgg16_bn' not in vgg.model_urls:
         vgg.model_urls['vgg16_bn'] = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
except ImportError:
    pass
# ---------------------------------------------------------------------

from pathlib import Path
from PIL import Image
from craft_text_detector import Craft
from simple_lama_inpainting import SimpleLama
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread


# --- CONFIGURATION ---
VIDEO_PATH = "videos/2.mp4"
OUTPUT_VIDEO = "output.mp4"
# ---------------------


def extract_frames(video_path, frames_dir):
  os.makedirs(frames_dir, exist_ok=True)
  vidcap = cv2.VideoCapture(video_path)
  
  fps = vidcap.get(cv2.CAP_PROP_FPS)
  total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
  
  count = 0
  with tqdm(total=total_frames, desc="[1] Extracting Frames", leave=False) as pbar:
    while True:
      success, frame = vidcap.read()
      if not success:
        break
      cv2.imwrite(os.path.join(frames_dir, f"{count:06d}.png"), frame)
      count += 1
      pbar.update(1)
      
  vidcap.release()
  return count, fps


def save_image(image, path):
  image.save(path)


def detector_worker(frame_paths, craft_model, queue):
  """Producer: Detects text and pushes to queue."""
  for frame_path in frame_paths:
    prediction = craft_model.detect_text(str(frame_path))
    # Simplify prediction for the consumer
    queue.put({
      "frame_path": frame_path,
      "boxes": prediction["boxes"],
      "has_text": len([p for p in prediction["polys"] if p is not None and len(p) > 0]) > 0
    })
  queue.put(None)  # Sentinel to stop


def detect_and_remove_subtitles(frames_dir, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  
  # 1. Smart Device Detection
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device == "cuda":
    # Log GPU info for the user in Colab
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[*] Found GPU: {gpu_name}. Performance will be tuned for CUDA.")
  else:
    print("[!] No GPU found. Falling back to CPU (Processing will be SLOW).")
  
  # 2. Initialize Models
  craft = Craft(output_dir=None, crop_type="poly", cuda=(device == "cuda"))
  lama = SimpleLama(device=device)

  frame_paths = sorted(Path(frames_dir).iterdir())
  total_frames = len(frame_paths)
  
  # 3. Setup Pipeline Queue
  # We limit queue size to prevent RAM overflow in Colab
  task_queue = Queue(maxsize=10) 
  
  # Start Detector Thread (Producer)
  detector_thread = Thread(target=detector_worker, args=(frame_paths, craft, task_queue))
  detector_thread.start()

  # 4. Processing Loop (Consumer)
  # Using ThreadPool to background the disk I/O while GPU is busy
  with ThreadPoolExecutor(max_workers=4) as io_executor:
    with tqdm(total=total_frames, desc="[3] Removing Subtitles", unit="frame") as pbar:
      while True:
        task = task_queue.get()
        if task is None: # Finished
          break
        
        frame_path = task["frame_path"]
        save_path = os.path.join(output_dir, frame_path.name)

        if not task["has_text"]:
          # Copy skip if no text
          shutil.copy(frame_path, save_path)
          pbar.update(1)
          continue

        # Prepare Mask & Inpaint
        image = Image.open(frame_path).convert("RGB")
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        
        for box in task["boxes"]:
          points = np.array(box, dtype=np.int32)
          cv2.fillPoly(mask, [points], 255)

        mask_pil = Image.fromarray(mask)
        inpainted = lama(image, mask_pil)
        
        # Dispatch save to I/O thread
        io_executor.submit(save_image, inpainted, save_path)
        pbar.update(1)

  # Cleanup models from VRAM
  craft.unload_craftnet_model()
  craft.unload_refinenet_model()
  del lama
  if device == "cuda":
    torch.cuda.empty_cache()


def frames_to_video(frames_dir, audio_path, output_path, fps):
  cmd = [
    "ffmpeg", "-y",
    "-framerate", str(fps),
    "-i", os.path.join(frames_dir, "%06d.png"),
    "-i", audio_path,
    "-c:v", "libx264", 
    "-preset", "faster",
    "-crf", "18",
    "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "192k",
    output_path
  ]
  # Suppress FFmpeg output but keep progress tracking
  subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_audio(video_path, audio_path):
  cmd = [
    "ffmpeg", "-y",
    "-i", video_path,
    "-q:a", "0", "-map", "a",
    audio_path
  ]
  subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
  if not os.path.exists(VIDEO_PATH):
    print(f"[!] Error: Input video {VIDEO_PATH} not found.")
    return

  start_time = time.time()

  with tempfile.TemporaryDirectory() as tmpdir:
    frames_dir = os.path.join(tmpdir, "frames")
    clean_frames_dir = os.path.join(tmpdir, "clean_frames")
    audio_path = os.path.join(tmpdir, "audio.aac")

    # Step 1: Prep
    total_frames, fps = extract_frames(VIDEO_PATH, frames_dir)

    # Step 2: Audio
    print("[2] Extracting original audio...")
    extract_audio(VIDEO_PATH, audio_path)

    # Step 3: Pipelined Processing
    detect_and_remove_subtitles(frames_dir, clean_frames_dir)

    # Step 4: Final Encode
    print("[4] High-quality re-encoding...")
    frames_to_video(clean_frames_dir, audio_path, OUTPUT_VIDEO, fps)

  total_time = time.time() - start_time
  print(f"\n[Done] Finished in {total_time:.1f}s.")
  print(f"[*] Average speed: {total_frames/total_time:.2f} frames per second.")
  print(f"[*] Processed file saved as: {OUTPUT_VIDEO}")


if __name__ == "__main__":
  main()
