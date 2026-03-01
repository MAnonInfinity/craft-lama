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
                image_np = np.array(Image.open(frame_path).convert("RGB"))
                h, w = image_np.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                
                for box in task["boxes"]:
                    points = np.array(box, dtype=np.int32)
                    cv2.fillPoly(mask, [points], 255)

                # --- HYBRID REFINEMENT ---
                # 1. Color Thresholding: Catch bright white pixels CRAFT missed
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                
                # Intersect: Only keep white pixels that are near the CRAFT boxes
                # (We dilate the CRAFT boxes a bit first to create a 'search zone')
                kernel_zone = np.ones((10, 10), np.uint8)
                search_zone = cv2.dilate(mask, kernel_zone, iterations=1)
                refined_white = cv2.bitwise_and(white_mask, search_zone)
                
                # Combine original CRAFT mask with the refined white pixels
                mask = cv2.bitwise_or(mask, refined_white)

                # 2. Dilation: Expand to cover the glow/shadow
                if mask_expansion > 0:
                    kernel = np.ones((mask_expansion, mask_expansion), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)
                
                # 3. Feathering: Blur the mask edges for seamless blending
                mask = cv2.GaussianBlur(mask, (5, 5), 0)

                # 5. Removal (LaMa)
                image_pil = Image.fromarray(image_np)
                mask_pil = Image.fromarray(mask)
                inpainted = lama(image_pil, mask_pil)
                
                # Dispatch save to I/O thread
                io_executor.submit(save_image, inpainted, save_path)
                pbar.update(1)

    # Cleanup models from VRAM
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    del lama
    if device == "cuda":
        torch.cuda.empty_cache()
