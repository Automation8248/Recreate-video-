import os
import csv
import time
import random
import shutil
import warnings
import subprocess
import requests
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# PyTorch/EasyOCR CPU warnings suppress karne ke liye
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_DIR = Path("videos")
PROCESSED_DIR = Path("processed_videos")
DONE_DIR = Path("done_originals")
LINKS_CSV = Path("output_links.csv")

MAX_DURATION_SEC = 125          # 2 minute max (+5s buffer)
REMOVE_BURNED_CAPTIONS = True   # Auto-captions hatane ke liye
AUDIO_MODE = "no_vocals"        # 'no_vocals' (voiceover hataye), 'deepfilter' (bg noise hataye), 'keep_vocals' (music hataye)
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

for d in (INPUT_DIR, PROCESSED_DIR, DONE_DIR):
    d.mkdir(exist_ok=True)

OCR_READER = None
if REMOVE_BURNED_CAPTIONS:
    import easyocr
    OCR_READER = easyocr.Reader(["en"], gpu=False, quantize=False)


# ==========================================
# HELPER: VIDEO DURATION
# ==========================================
def get_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    return float(subprocess.check_output(cmd).decode().strip())


# ==========================================
# STEP 1: ZERO-LEAK AUTO-CAPTION REMOVAL
# ==========================================
def remove_shorts_captions(input_video: str, output_video: str):
    if not REMOVE_BURNED_CAPTIONS:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_video,
            "-map", "0:v", "-map", "0:a?", "-c", "copy", "-sn", output_video
        ], check=True)
        return

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Top se bottom tak lagbhag poora screen scan (8% to 96%)
    y_start, y_end = int(h * 0.08), int(h * 0.96)

    # PASS 1: Har 2nd frame par OCR chala kar saare text boxes collect karein
    print(f"    -> Pass 1/2: Scanning {total_frames} frames for captions...", flush=True)
    detected_boxes = {}  # frame_idx -> list of polygon points
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 2 == 0:
            roi = frame[y_start:y_end, 0:w]
            # Confidence threshold 0.15 rakha hai taaki stylish/faint captions bhi catch hon
            results = OCR_READER.readtext(roi)
            boxes = []
            for (bbox, text, conf) in results:
                if conf > 0.15:
                    pts = np.array(bbox, dtype=np.int32)
                    pts[:, 1] += y_start
                    boxes.append(pts)
            if boxes:
                detected_boxes[frame_idx] = boxes

        frame_idx += 1

    cap.release()

    # PASS 2: Temporal Buffer (±3 frames) + Heavy Dilation ke saath inpaint karein
    print("    -> Pass 2/2: Removing captions with temporal buffer...", flush=True)
    cap = cv2.VideoCapture(input_video)
    temp_silent = f"temp_silent_{int(time.time())}.mp4"
    out = cv2.VideoWriter(temp_silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    kernel = np.ones((21, 21), np.uint8)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Current frame ke aage-peeche 3 frames ke saare text boxes merge karein
        mask = np.zeros((h, w), dtype=np.uint8)
        has_text = False
        for offset in range(-3, 4):
            boxes = detected_boxes.get(frame_idx + offset)
            if boxes:
                has_text = True
                for pts in boxes:
                    cv2.fillPoly(mask, [pts], 255)

        if has_text:
            # 2 iterations dilation taaki text ka shadow/outline bilkul na bache
            mask = cv2.dilate(mask, kernel, iterations=2)
            frame = cv2.inpaint(frame, mask, 7, cv2.INPAINT_TELEA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    subprocess.run([
        "ffmpeg", "-y", "-i", temp_silent, "-i", input_video,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?", "-sn", output_video
    ], check=True)
    os.remove(temp_silent)


# ==========================================
# STEP 2: REMOVE BACKGROUND VOICEOVER / NOISE
# ==========================================
def clean_audio(input_video: str, output_video: str, mode: str = AUDIO_MODE):
    raw_wav = f"temp_audio_{int(time.time())}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", input_video,
        "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2", raw_wav
    ], check=True)

    if mode == "deepfilter":
        os.makedirs("df_out", exist_ok=True)
        subprocess.run(["deepFilter", raw_wav, "-o", "df_out"], check=True)
        clean_wav = str(Path("df_out") / Path(raw_wav).name)
    elif mode in ("keep_vocals", "no_vocals"):
        subprocess.run(["demucs", "--two-stems=vocals", "-d", "cpu", raw_wav, "-o", "demucs_out"], check=True)
        stem = "vocals.wav" if mode == "keep_vocals" else "no_vocals.wav"
        clean_wav = f"demucs_out/htdemucs/{Path(raw_wav).stem}/{stem}"
    else:
        raise ValueError("Invalid AUDIO_MODE")

    subprocess.run([
        "ffmpeg", "-y", "-i", input_video, "-i", clean_wav,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0", "-sn", output_video
    ], check=True)

    if os.path.exists(raw_wav):
        os.remove(raw_wav)


# ==========================================
# STEP 3: CHANGE VIDEO FINGERPRINT & PIXELS
# ==========================================
def alter_fingerprint_and_pixels(input_video: str, output_video: str):
    zoom = round(random.uniform(1.015, 1.028), 3)
    brightness = round(random.uniform(-0.02, 0.02), 3)
    contrast = round(random.uniform(0.98, 1.03), 3)
    saturation = round(random.uniform(0.98, 1.04), 3)
    speed = round(random.uniform(1.005, 1.02), 3)
    pts_factor = round(1.0 / speed, 4)

    vf_chain = (
        f"scale=iw*{zoom}:ih*{zoom},"
        f"crop=iw/{zoom}:ih/{zoom},"
        f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation},"
        f"noise=alls=2:allf=t+u,"
        f"setpts={pts_factor}*PTS"
    )
    af_chain = f"atempo={speed}"

    subprocess.run([
        "ffmpeg", "-y", "-i", input_video,
        "-vf", vf_chain,
        "-af", af_chain,
        "-map_metadata", "-1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_video
    ], check=True)


# ==========================================
# STEP 4: 12 SERVERS FALLBACK UPLOADER
# (Ek fail hoga tabhi dusra try karega)
# ==========================================
HEADERS = {"User-Agent": "Mozilla/5.0 (ShortsAutoBot/1.0)"}

def upload_catbox(p):
    with open(p, "rb") as f:
        r = requests.post("https://catbox.moe/user/api.php", data={"reqtype": "fileupload"}, files={"fileToUpload": f}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.text.strip()

def upload_litterbox(p):
    with open(p, "rb") as f:
        r = requests.post("https://litterbox.catbox.moe/resources/internals/api.php", data={"reqtype": "fileupload", "time": "72h"}, files={"fileToUpload": f}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.text.strip()

def upload_0x0(p):
    with open(p, "rb") as f:
        r = requests.post("https://0x0.st", files={"file": f}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.text.strip()

def upload_uguu(p):
    with open(p, "rb") as f:
        r = requests.post("https://uguu.se/upload", files={"files[]": f}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.json()["files"][0]["url"]

def upload_quax(p):
    with open(p, "rb") as f:
        r = requests.post("https://qu.ax/upload.php", files={"files[]": f}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.json()["files"][0]["url"]

def upload_tmpfiles(p):
    with open(p, "rb") as f:
        r = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": f}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.json()["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")

def upload_tempsh(p):
    with open(p, "rb") as f:
        r = requests.post("https://temp.sh/upload", files={"file": f}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.text.strip()

def upload_x0at(p):
    with open(p, "rb") as f:
        r = requests.post("https://x0.at/", files={"file": f}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.text.strip()

def upload_bashupload(p):
    with open(p, "rb") as f:
        r = requests.put(f"https://bashupload.com/{os.path.basename(p)}", data=f, headers=HEADERS, timeout=120)
    r.raise_for_status()
    for line in r.text.splitlines():
        if "https://bashupload.com/" in line:
            return line.split()[-1].strip()
    raise RuntimeError("No URL returned")

def upload_fileio(p):
    with open(p, "rb") as f:
        r = requests.post("https://file.io", files={"file": f}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.json()["link"]

def upload_pomf(p):
    with open(p, "rb") as f:
        r = requests.post("https://pomf.lain.la/upload.php", files={"files[]": f}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.json()["files"][0]["url"]

def upload_gofile(p):
    with open(p, "rb") as f:
        r = requests.post("https://upload.gofile.io/uploadfile", files={"file": f}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.json()["data"]["downloadPage"]

SERVERS = [
    ("Catbox.moe", upload_catbox),
    ("Litterbox", upload_litterbox),
    ("0x0.st", upload_0x0),
    ("Uguu.se", upload_uguu),
    ("Qu.ax", upload_quax),
    ("Tmpfiles.org", upload_tmpfiles),
    ("Temp.sh", upload_tempsh),
    ("x0.at", upload_x0at),
    ("Bashupload", upload_bashupload),
    ("File.io", upload_fileio),
    ("Pomf.lain.la", upload_pomf),
    ("GoFile.io", upload_gofile),
]


def upload_with_fallback(video_path: str):
    """
    Servers ko ek-ek karke try karega.
    Jaise hi koi 1 server valid link dega, wahin stop ho jayega.
    """
    for name, func in SERVERS:
        try:
            print(f"    -> Trying {name}...", flush=True)
            url = func(video_path)
            if url and url.startswith("http"):
                print(f"    [SUCCESS] {name}: {url}", flush=True)
                return name, url
            else:
                print(f"    [FAIL] {name}: Invalid response ({url})", flush=True)
        except Exception as e:
            print(f"    [FAIL] {name}: {e}", flush=True)
    return "ALL_FAILED", "FAILED"


def save_link_to_csv(filename: str, server_name: str, direct_link: str):
    file_exists = LINKS_CSV.exists()
    headers = ["Timestamp", "Video_File", "Server_Used", "Direct_Link"]
    with open(LINKS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            server_name,
            direct_link
        ])


# ==========================================
# MAIN BATCH RUNNER
# ==========================================
def process_all_videos():
    videos = [
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    if not videos:
        print("No valid video files found in videos/. Exiting.", flush=True)
        return

    for video_file in videos:
        print(f"\n>>> Processing: {video_file.name}", flush=True)
        try:
            duration = get_duration(str(video_file))
            if duration > MAX_DURATION_SEC:
                print(f"  [SKIP] {video_file.name} is {duration:.1f}s (2 min limit se zyada).", flush=True)
                continue

            step1 = str(PROCESSED_DIR / f"s1_{video_file.stem}.mp4")
            step2 = str(PROCESSED_DIR / f"s2_{video_file.stem}.mp4")
            final_out = str(PROCESSED_DIR / f"final_{video_file.stem}.mp4")

            print("  [1/4] Removing auto-captions (Zero-Leak 2-Pass)...", flush=True)
            remove_shorts_captions(str(video_file), step1)

            print(f"  [2/4] Removing background voiceover/noise ({AUDIO_MODE})...", flush=True)
            clean_audio(step1, step2)

            print("  [3/4] Changing video fingerprint & pixels...", flush=True)
            alter_fingerprint_and_pixels(step2, final_out)

            print("  [4/4] Uploading with 12-server fallback...", flush=True)
            server_name, direct_link = upload_with_fallback(final_out)
            save_link_to_csv(video_file.name, server_name, direct_link)

            for tmp in (step1, step2):
                if os.path.exists(tmp):
                    os.remove(tmp)

            shutil.move(str(video_file), str(DONE_DIR / video_file.name))
            print(f"<<< DONE: {video_file.name} | {server_name} -> {direct_link}", flush=True)

        except Exception as e:
            print(f"[ERROR] Failed on {video_file.name}: {e}", flush=True)


if __name__ == "__main__":
    process_all_videos()
