import os
import re
import csv
import time
import uuid
import random
import shutil
import tempfile
import warnings
import subprocess
import requests
import gdown
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# PyTorch/EasyOCR CPU warnings suppress karne ke liye
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION (ALL OPTIONS KEPT INTACT)
# ==========================================
INPUT_DIR = Path("videos")
DONE_DIR = Path("done_originals")
MUSIC_FILE = Path("music.txt")
LINKS_CSV = Path("output_links.csv")

# Target Repo Settings (Final video file yahan save hogi)
TARGET_REPO = "Automation8248/Postza"
TARGET_BRANCH = "main"
TARGET_SUBFOLDER = "Facebook Intresting Fact/Videos"
GH_TOKEN = os.environ.get("GH_TOKEN", "").strip()

MAX_DURATION_SEC = 125          # 2 minute max (+5s buffer)
AUTO_DELETE_DAYS = 30           # 30 days baad processed videos auto-delete
REMOVE_BURNED_CAPTIONS = True   # Zero-leak 2-pass auto-caption removal
AUDIO_MODE = "no_vocals"        # 'no_vocals' (voiceover hataye), 'deepfilter' (bg noise hataye), 'keep_vocals' (music hataye)
ADD_BG_MUSIC = True             # music.txt se random Drive music add karega
BG_MUSIC_VOLUME = 0.45          # Clear audible background music volume (0.1 to 1.0)
ENABLE_SERVER_UPLOAD = True     # 12-server fallback uploader option
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

for d in (INPUT_DIR, DONE_DIR):
    d.mkdir(exist_ok=True)

OCR_READER = None
if REMOVE_BURNED_CAPTIONS:
    import easyocr
    OCR_READER = easyocr.Reader(["en"], gpu=False, quantize=False)


# ==========================================
# HELPER: VIDEO DURATION & AUDIO CHECK
# ==========================================
def get_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    return float(subprocess.check_output(cmd).decode().strip())


def has_audio_stream(media_path: str) -> bool:
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", media_path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return len(out) > 0
    except Exception:
        return False


# ==========================================
# STEP 1: ZERO-LEAK AUTO-CAPTION REMOVAL
# ==========================================
def remove_shorts_captions(input_video: str, output_video: str, work_dir: Path):
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
    y_start, y_end = int(h * 0.08), int(h * 0.96)

    print(f"    -> Pass 1/2: Scanning {total_frames} frames for captions...", flush=True)
    detected_boxes = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 2 == 0:
            roi = frame[y_start:y_end, 0:w]
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

    print("    -> Pass 2/2: Removing captions with temporal buffer...", flush=True)
    cap = cv2.VideoCapture(input_video)
    temp_silent = str(work_dir / f"temp_silent_{int(time.time())}.mp4")
    out = cv2.VideoWriter(temp_silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    kernel = np.ones((21, 21), np.uint8)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = np.zeros((h, w), dtype=np.uint8)
        has_text = False
        for offset in range(-3, 4):
            boxes = detected_boxes.get(frame_idx + offset)
            if boxes:
                has_text = True
                for pts in boxes:
                    cv2.fillPoly(mask, [pts], 255)

        if has_text:
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


# ==========================================
# STEP 2: REMOVE BACKGROUND VOICEOVER / NOISE
# ==========================================
def clean_audio(input_video: str, output_video: str, work_dir: Path, mode: str = AUDIO_MODE):
    if not has_audio_stream(input_video):
        shutil.copy(input_video, output_video)
        return

    raw_wav = str(work_dir / f"temp_audio_{int(time.time())}.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_video,
        "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2", raw_wav
    ], check=True)

    if mode == "deepfilter":
        df_dir = work_dir / "df_out"
        df_dir.mkdir(exist_ok=True)
        subprocess.run(["deepFilter", raw_wav, "-o", str(df_dir)], check=True)
        clean_wav = str(df_dir / Path(raw_wav).name)
    elif mode in ("keep_vocals", "no_vocals"):
        demucs_dir = work_dir / "demucs_out"
        subprocess.run(["demucs", "--two-stems=vocals", "-d", "cpu", raw_wav, "-o", str(demucs_dir)], check=True)
        stem = "vocals.wav" if mode == "keep_vocals" else "no_vocals.wav"
        clean_wav = str(demucs_dir / "htdemucs" / Path(raw_wav).stem / stem)
    else:
        raise ValueError("Invalid AUDIO_MODE")

    subprocess.run([
        "ffmpeg", "-y", "-i", input_video, "-i", clean_wav,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0", "-sn", output_video
    ], check=True)


# ==========================================
# STEP 3: DEEP UNIQUENESS (METADATA, HASH, FINGERPRINT, PIXELS, AUDIO)
# ==========================================
def make_video_fully_unique(input_video: str, output_video: str):
    zoom = round(random.uniform(1.015, 1.030), 4)
    angle_rad = round(random.uniform(-0.005, 0.005), 4)
    brightness = round(random.uniform(-0.025, 0.025), 3)
    contrast = round(random.uniform(0.97, 1.04), 3)
    saturation = round(random.uniform(0.97, 1.05), 3)
    gamma = round(random.uniform(0.98, 1.03), 3)
    hue_shift = round(random.uniform(-2.0, 2.0), 2)
    noise_str = random.choice([2, 3])
    speed = round(random.uniform(1.005, 1.022), 4)
    pts_factor = round(1.0 / speed, 5)
    crf_val = str(random.choice([20, 21, 22]))

    vf_chain = (
        f"scale=iw*{zoom}:ih*{zoom},"
        f"rotate={angle_rad}:fillcolor=black@0,"
        f"crop=iw/{zoom}:ih/{zoom},"
        f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}:gamma={gamma},"
        f"hue=h={hue_shift},"
        f"noise=alls={noise_str}:allf=t+u,"
        f"setpts={pts_factor}*PTS"
    )

    cmd = ["ffmpeg", "-y", "-i", input_video, "-vf", vf_chain]

    if has_audio_stream(input_video):
        pitch = round(random.uniform(0.985, 1.015), 4)
        tempo_adj = round(speed / pitch, 4)
        bass_gain = round(random.uniform(-1.2, 1.5), 2)
        af_chain = (
            f"asetrate=48000*{pitch},aresample=48000,"
            f"atempo={tempo_adj},"
            f"bass=g={bass_gain}"
        )
        cmd += ["-af", af_chain, "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]

    unique_id = uuid.uuid4().hex[:12]
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000Z")
    chosen_encoder = random.choice(["Apple iPhone 15 Pro", "Samsung Galaxy S24", "Google Pixel 9", "Sony Alpha ZV-E10"])

    cmd += [
        "-map_metadata", "-1",
        "-metadata", f"title=Fact_{unique_id}",
        "-metadata", f"comment=UID_{unique_id}",
        "-metadata", f"encoder={chosen_encoder}",
        "-metadata", f"creation_time={now_iso}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", crf_val,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_video
    ]
    subprocess.run(cmd, check=True)


# ==========================================
# STEP 4: FINAL RANDOM BACKGROUND MUSIC (music.txt)
# (Sabhi work ke baad final video mein add hoga)
# ==========================================
def extract_drive_id(url: str):
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"/uc\?.*id=([a-zA-Z0-9_-]+)"
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def download_valid_music(work_dir: Path):
    if not MUSIC_FILE.exists():
        print("    [WARN] music.txt file nahi mili!", flush=True)
        return None

    links = [
        line.strip() for line in MUSIC_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not links:
        print("    [WARN] music.txt khali hai!", flush=True)
        return None

    random.shuffle(links)

    for idx, url in enumerate(links):
        music_path = str(work_dir / f"bg_music_{idx}_{int(time.time())}.audio")
        print(f"    -> Trying music link ({idx + 1}/{len(links)}): {url}", flush=True)

        # Method 1: gdown (best for Google Drive links)
        try:
            file_id = extract_drive_id(url)
            if file_id:
                gdown.download(id=file_id, output=music_path, quiet=True)
            else:
                gdown.download(url=url, output=music_path, quiet=True, fuzzy=True)
        except Exception as e:
            print(f"    -> gdown failed: {e}", flush=True)

        # Verify ki valid audio file hi download hui hai (HTML page nahi)
        if os.path.exists(music_path) and has_audio_stream(music_path):
            print("    [OK] Valid music file downloaded via gdown.", flush=True)
            return music_path

        # Method 2: Direct requests fallback
        try:
            direct_url = url
            file_id = extract_drive_id(url)
            if file_id:
                direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            r = requests.get(direct_url, timeout=60, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 5000:
                with open(music_path, "wb") as f:
                    f.write(r.content)
                if has_audio_stream(music_path):
                    print("    [OK] Valid music file downloaded via direct request.", flush=True)
                    return music_path
        except Exception as e:
            print(f"    -> Direct download failed: {e}", flush=True)

        print("    [FAIL] Invalid audio or private Drive link, next link try kar rahe hain...", flush=True)

    return None


def add_final_bg_music(input_video: str, output_video: str, work_dir: Path):
    if not ADD_BG_MUSIC:
        shutil.copy(input_video, output_video)
        return

    music_path = download_valid_music(work_dir)
    if not music_path:
        print("    [WARN] Koi bhi music download nahi ho paya, bina music ke aage badh rahe hain.", flush=True)
        shutil.copy(input_video, output_video)
        return

    unique_id = uuid.uuid4().hex[:12]
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000Z")
    chosen_encoder = random.choice(["Apple iPhone 15 Pro", "Samsung Galaxy S24", "Google Pixel 9", "Sony Alpha ZV-E10"])

    # Auto-repeat (-stream_loop -1) agar music chhota hai, Auto-trim (-shortest) agar music bada hai
    if has_audio_stream(input_video):
        filter_complex = (
            f"[0:a]volume=0.75[a0];"
            f"[1:a]volume={BG_MUSIC_VOLUME}[a1];"
            f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-stream_loop", "-1", "-i", music_path,
            "-filter_complex", filter_complex,
            "-map", "0:v:0", "-map", "[aout]",
            "-map_metadata", "-1",
            "-metadata", f"title=Fact_{unique_id}",
            "-metadata", f"comment=UID_{unique_id}",
            "-metadata", f"encoder={chosen_encoder}",
            "-metadata", f"creation_time={now_iso}",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-shortest", "-movflags", "+faststart",
            output_video
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-stream_loop", "-1", "-i", music_path,
            "-filter_complex", f"[1:a]volume={BG_MUSIC_VOLUME}[aout]",
            "-map", "0:v:0", "-map", "[aout]",
            "-map_metadata", "-1",
            "-metadata", f"title=Fact_{unique_id}",
            "-metadata", f"comment=UID_{unique_id}",
            "-metadata", f"encoder={chosen_encoder}",
            "-metadata", f"creation_time={now_iso}",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-shortest", "-movflags", "+faststart",
            output_video
        ]

    subprocess.run(cmd, check=True)
    print("    [SUCCESS] Background music merged into final video!", flush=True)


# ==========================================
# STEP 5: 12 SERVERS FALLBACK UPLOADER
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
    if not ENABLE_SERVER_UPLOAD:
        return "DISABLED", "DISABLED"
    for name, func in SERVERS:
        try:
            print(f"    -> Trying {name}...", flush=True)
            url = func(video_path)
            if url and url.startswith("http"):
                print(f"    [SUCCESS] {name}: {url}", flush=True)
                return name, url
        except Exception as e:
            print(f"    [FAIL] {name}: {e}", flush=True)
    return "ALL_FAILED", "FAILED"


def save_link_to_csv(filename: str, server_name: str, direct_link: str, target_file: str):
    file_exists = LINKS_CSV.exists()
    headers = ["Timestamp", "Video_File", "Server_Used", "Direct_Link", "Target_Repo_Path"]
    with open(LINKS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            server_name,
            direct_link,
            target_file
        ])


# ==========================================
# STEP 6: PUSH VIDEO FILE TO TARGET GITHUB REPO
# ==========================================
def push_video_to_target_repo(video_path: str, output_filename: str) -> str:
    if not GH_TOKEN:
        raise RuntimeError("GH_TOKEN secret missing! GitHub Secrets mein GH_TOKEN add karein.")

    remote_url = f"https://x-access-token:{GH_TOKEN}@github.com/{TARGET_REPO}.git"

    with tempfile.TemporaryDirectory() as clone_dir:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", TARGET_BRANCH, remote_url, clone_dir],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        dest_folder = Path(clone_dir) / TARGET_SUBFOLDER
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest_file = dest_folder / output_filename
        shutil.copy2(video_path, dest_file)

        subprocess.run(["git", "-C", clone_dir, "config", "user.email", "action@github.com"], check=True)
        subprocess.run(["git", "-C", clone_dir, "config", "user.name", "GitHub Action"], check=True)
        subprocess.run(["git", "-C", clone_dir, "add", str(dest_file)], check=True)
        subprocess.run(["git", "-C", clone_dir, "commit", "-m", f"Auto: Add processed video {output_filename}"], check=True)
        subprocess.run(["git", "-C", clone_dir, "push", "origin", TARGET_BRANCH], check=True)

        print(f"    [PUSHED] Saved video to {TARGET_REPO}/{TARGET_SUBFOLDER}/{output_filename}", flush=True)
        return f"{TARGET_SUBFOLDER}/{output_filename}"


# ==========================================
# STEP 7: 30-DAY AUTO-DELETE (SOURCE DONE_ORIGINALS)
# ==========================================
def cleanup_old_videos(folder: Path, max_days: int = AUTO_DELETE_DAYS):
    if not folder.exists():
        return

    now_ts = time.time()
    cutoff_sec = max_days * 86400

    for f in folder.iterdir():
        if not f.is_file() or f.suffix.lower() not in VIDEO_EXTS:
            continue
        try:
            out = subprocess.check_output(
                ["git", "log", "-1", "--format=%ct", "--", str(f)]
            ).decode().strip()
            file_ts = int(out) if out else int(f.stat().st_mtime)
        except Exception:
            file_ts = int(f.stat().st_mtime)

        age_days = (now_ts - file_ts) / 86400
        if (now_ts - file_ts) >= cutoff_sec:
            print(f"[AUTO-DELETE] {f.name} ({age_days:.1f} days old) deleted.", flush=True)
            f.unlink()


# ==========================================
# MAIN BATCH RUNNER
# ==========================================
def process_all_videos():
    cleanup_old_videos(DONE_DIR, AUTO_DELETE_DAYS)

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

            with tempfile.TemporaryDirectory() as tmp:
                work_dir = Path(tmp)
                step1 = str(work_dir / "s1_nocap.mp4")
                step2 = str(work_dir / "s2_noaudio.mp4")
                step3 = str(work_dir / "s3_unique.mp4")
                unique_tag = uuid.uuid4().hex[:6]
                final_name = f"{video_file.stem}_{unique_tag}.mp4"
                final_out = str(work_dir / final_name)

                print("  [1/6] Removing auto-captions (Zero-Leak 2-Pass)...", flush=True)
                remove_shorts_captions(str(video_file), step1, work_dir)

                print(f"  [2/6] Removing background voiceover/noise ({AUDIO_MODE})...", flush=True)
                clean_audio(step1, step2, work_dir)

                print("  [3/6] Applying unique metadata, hash, fingerprint, pixel & audio tweaks...", flush=True)
                make_video_fully_unique(step2, step3)

                print("  [4/6] Adding final random background music from music.txt...", flush=True)
                add_final_bg_music(step3, final_out, work_dir)

                print(f"  [5/6] Pushing final video file directly to {TARGET_REPO}...", flush=True)
                target_path = push_video_to_target_repo(final_out, final_name)

                print("  [6/6] Uploading with 12-server fallback...", flush=True)
                server_name, direct_link = upload_with_fallback(final_out)
                save_link_to_csv(video_file.name, server_name, direct_link, target_path)

            shutil.move(str(video_file), str(DONE_DIR / video_file.name))
            print(f"<<< DONE: {video_file.name} -> {target_path}", flush=True)

        except Exception as e:
            print(f"[ERROR] Failed on {video_file.name}: {e}", flush=True)


if __name__ == "__main__":
    process_all_videos()
