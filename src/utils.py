import cv2
import os
import subprocess
from tqdm import tqdm

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

def extract_audio(video_path, audio_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-q:a", "0", "-map", "a",
        audio_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
