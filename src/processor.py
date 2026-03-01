import os
import shutil
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from craft_text_detector import Craft
from simple_lama_inpainting import SimpleLama
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
import cv2

from utils import save_image

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

def detect_and_remove_subtitles(frames_dir, output_dir, mask_expansion=5):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Smart Device Detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
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
    task_queue = Queue(maxsize=10) 
    
    # Start Detector Thread (Producer)
    detector_thread = Thread(target=detector_worker, args=(frame_paths, craft, task_queue))
    detector_thread.start()

    # 4. Processing Loop (Consumer)
    with ThreadPoolExecutor(max_workers=4) as io_executor:
        with tqdm(total=total_frames, desc="[3] Removing Subtitles", unit="frame") as pbar:
            while True:
                task = task_queue.get()
                if task is None: # Finished
                    break
                
                frame_path = task["frame_path"]
                save_path = os.path.join(output_dir, frame_path.name)

                if not task["has_text"]:
                    shutil.copy(frame_path, save_path)
                    pbar.update(1)
                    continue

                # Prepare Mask
                image = Image.open(frame_path).convert("RGB")
                mask = np.zeros((image.height, image.width), dtype=np.uint8)
                
                for box in task["boxes"]:
                    points = np.array(box, dtype=np.int32)
                    cv2.fillPoly(mask, [points], 255)

                # DILATION: Expand the mask to cover shadows/glow
                if mask_expansion > 0:
                    kernel = np.ones((mask_expansion, mask_expansion), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)

                # 5. Removal (LaMa)
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
