import os
import sys
import time
import tempfile
import cv2

# Ensure the script can import local modules when running from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from patches import apply_patches
from utils import extract_frames, extract_audio, frames_to_video
from processor import detect_and_remove_subtitles

# --- CONFIGURATION ---
VIDEO_PATH = "videos/2.mp4"
OUTPUT_VIDEO = "output.mp4"
# ---------------------

def main():
    # 1. Apply compatibility patches before doing anything else
    apply_patches()

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
