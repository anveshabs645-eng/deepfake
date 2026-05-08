import os
import csv
import librosa
import numpy as np
from video_model import get_video_score
from audio_features import (extract_audio, extract_features,
                             train_audio_classifier, load_audio_classifier,
                             get_audio_score)

dataset_path = "dataset"
output_csv = "scores.csv"
failed_log = "failed_files.txt"


def log_failed(video_path, reason):
    with open(failed_log, "a") as f:
        f.write(f"{video_path} | Reason: {reason}\n")


def train_audio():
    features_list = []
    labels_list = []

    if os.path.exists(failed_log):
        os.remove(failed_log)

    for label_name, label_value in [("real", 0), ("fake", 1)]:
        folder = os.path.join(dataset_path, label_name)
        if not os.path.exists(folder):
            print(f"Warning: folder '{folder}' not found, skipping.")
            continue

        videos = [v for v in os.listdir(folder) if v.lower().endswith((".mp4", ".avi", ".mov"))]
        total = len(videos)
        print(f"Found {total} {label_name} videos for audio training.")

        for i, video in enumerate(videos, 1):
            video_path = os.path.join(folder, video)
            print(f"[{i}/{total}] Extracting audio features: {video}...")

            try:
                wav = extract_audio(video_path)
            except Exception as e:
                print(f"  [SKIP] extract_audio raised an exception: {e}")
                log_failed(video_path, f"extract_audio exception: {e}")
                continue

            if wav is None:
                print(f"  [SKIP] extract_audio returned None for: {video}")
                log_failed(video_path, "extract_audio returned None")
                continue

            if not os.path.exists(wav):
                print(f"  [SKIP] WAV file not found on disk: {wav}")
                log_failed(video_path, f"WAV not found at path: {wav}")
                continue

            try:
                y, sr = librosa.load(wav, duration=10)
                features = extract_features(y, sr)
                features_list.append(features)
                labels_list.append(label_value)
            except Exception as e:
                print(f"  [SKIP] Feature extraction error: {e}")
                log_failed(video_path, f"librosa/feature error: {e}")
            finally:
                if wav and os.path.exists(wav):
                    os.remove(wav)

    if len(features_list) > 0:
        train_audio_classifier(features_list, labels_list)
        print(f"Audio classifier trained on {len(features_list)} samples → audio_model.pkl")
    else:
        print("No audio features extracted. Check your dataset and failed_files.txt.")


def generate_csv():
    load_audio_classifier()

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "V_score", "A_score", "label"])

        for label_name, label_value in [("real", 0), ("fake", 1)]:
            folder = os.path.join(dataset_path, label_name)
            if not os.path.exists(folder):
                print(f"Warning: folder '{folder}' not found, skipping.")
                continue

            videos = [v for v in os.listdir(folder) if v.lower().endswith((".mp4", ".avi", ".mov"))]
            total = len(videos)
            print(f"Found {total} {label_name} videos for scoring.")

            for i, video in enumerate(videos, 1):
                video_path = os.path.join(folder, video)
                print(f"[{i}/{total}] Processing: {video}...")

                try:
                    v_score = get_video_score(video_path)
                except Exception as e:
                    print(f"  [SKIP] Video score failed: {e}")
                    log_failed(video_path, f"get_video_score error: {e}")
                    continue

                try:
                    a_score, _ = get_audio_score(video_path)
                except Exception as e:
                    print(f"  [SKIP] Audio score failed: {e}")
                    log_failed(video_path, f"get_audio_score error: {e}")
                    continue

                if v_score is None or a_score is None:
                    print(f"  [SKIP] Got None score — V={v_score}, A={a_score}")
                    log_failed(video_path, f"None score: V={v_score}, A={a_score}")
                    continue

                writer.writerow([video, v_score, a_score, label_value])
                print(f"  ✓ V={v_score:.3f}  A={a_score:.3f}  label={label_value}")

    print(f"\nCSV saved to '{output_csv}'")
    if os.path.exists(failed_log):
        print(f"Some files were skipped — see '{failed_log}' for details.")


if __name__ == "__main__":
    print("=== STEP 1: Training audio classifier ===")
    train_audio()

    print("\n=== STEP 2: Generating scores.csv ===")
    generate_csv()

    print("\nDone. Run train_mlp.py next.")