import os
import librosa
import math
import json
import numpy as np

DATASET_PATH = "data/genres_original"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            semantic_label = os.path.basename(dirpath)
            data["mapping"].append(semantic_label)
            print(f"\nProcessing: {semantic_label}")

            for f in filenames:
                # The 'jazz.00054.wav' file in GTZAN is notoriously corrupted
                if f == "jazz.00054.wav":
                    continue
                
                file_path = os.path.join(dirpath, f)
                
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                except Exception as e:
                    print(f"Skipping {f} due to load error.")
                    continue

                for s in range(num_segments):
                    start = samples_per_segment * s
                    finish = start + samples_per_segment

                    # Extract MFCC
                    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
                    mfcc = mfcc.T

                    # Standardize/Normalize MFCCs (Great for Accuracy Boost)
                    # mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                
                print(f"Processed: {f}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    print("\nPre-processing complete. data.json saved.")

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)