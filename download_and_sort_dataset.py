import os
import zipfile
import requests
from tqdm import tqdm
import shutil

# -----------------------------
# 1. DOWNLOAD RAVDESS ZIP FILE
# -----------------------------

url = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
zip_path = "ravdess.zip"

print("Downloading RAVDESS dataset...")
response = requests.get(url, stream=True)

total_size = int(response.headers.get('content-length', 0))
block_size = 1024

with open(zip_path, "wb") as file:
    for data in tqdm(response.iter_content(block_size), total=total_size // block_size, unit='KB'):
        file.write(data)

print("Download complete!")

# -----------------------------
# 2. EXTRACT ZIP FILE
# -----------------------------

extract_folder = "ravdess_raw"

print("Extracting ZIP file...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

print("Extraction complete!")

# -----------------------------
# 3. CREATE TARGET DATASET FOLDERS
# -----------------------------

target_root = "dataset"

emotions_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
}

for emotion in emotions_map.values():
    os.makedirs(os.path.join(target_root, emotion), exist_ok=True)

# -----------------------------
# 4. SORT AUDIO FILES INTO FOLDERS
# -----------------------------

print("Sorting audio files...")

for root, dirs, files in os.walk(extract_folder):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split("-")

            if len(parts) < 3:
                continue

            emotion_code = parts[2]

            if emotion_code in emotions_map:
                emotion = emotions_map[emotion_code]

                src = os.path.join(root, file)
                dest = os.path.join(target_root, emotion, file)

                shutil.copy(src, dest)
                print(f"Copied {file} → {emotion}")

print("\n✔ DONE! All files sorted into:")
print("dataset/angry")
print("dataset/happy")
print("dataset/sad")
print("dataset/neutral")
