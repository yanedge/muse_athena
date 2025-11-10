import pandas as pd
import OpenMuse 
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from scipy.signal import find_peaks, resample_poly
import json

files = ["data3.txt", "data4.txt", "data5.txt", "data6.txt"]
label_files = ["data3.label.json", "data4.label.json", "data5.label.json", "data6.label.json"]
# files = ["yan1.txt"]


def prepare():
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            messages = f.readlines()
        data = OpenMuse.decode_rawdata(messages)

        eeg_cols = ["EEG_TP9", "EEG_TP10"] # ignore AF7 and AF8, which don't contain alpha waves
        eeg_raw = data["EEG"][eeg_cols].values
        eeg_raw = eeg_raw - np.mean(eeg_raw, axis=0)
        eeg_resampled = resample_poly(eeg_raw, up=128, down=256, axis=0)

        label_file = file.replace(".txt", ".label.json")
        with open(label_file, "r", encoding="utf-8") as lf:
            labels = json.load(lf)
            closed_segments = labels.get("close", [])

            eeg_df = pd.DataFrame(eeg_resampled, columns=eeg_cols)
            eeg_df["label"] = 0  # Default to open eyes
            for segment in closed_segments:
                start, end = segment
                eeg_df.loc[start:end, "label"] = 1  # Mark closed eyes

            file_name = file.split("/")[-1]
            out_path = "output/" + file_name.replace(".txt", "_eeg_128_labeled.csv")
            eeg_df.to_csv(out_path, index=False)


def analyze():
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            messages = f.readlines()
        data = OpenMuse.decode_rawdata(messages)
        file_name = "output/" + file.split("/")[-1]

        eeg_cols = ["EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10"]
        eeg_raw = data["EEG"][eeg_cols].values
        eeg_raw = pd.DataFrame(eeg_raw, columns=eeg_cols)
        eeg_raw_resampled = pd.DataFrame(
            signal.resample(eeg_raw.values, int(len(eeg_raw) * 128 / 256)),
            columns=eeg_cols
        )
        eeg_raw = eeg_raw_resampled
        eeg_raw = eeg_raw - np.mean(eeg_raw, axis=0)
        eeg_raw.to_csv(file_name.replace(".txt", "_eeg_128.csv"), index=False)

        channel0 = eeg_raw[eeg_cols[0]]
        peaks, _ = find_peaks(np.abs(channel0), height=100, distance=50)
        peak_values = channel0.iloc[peaks]

        # Add peak markers to the raw plot
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(eeg_raw[eeg_cols[0]])
        axs[0].set_title(f'EEG Raw - {eeg_cols[0]}')
        axs[0].set_xlabel('Sample')
        axs[0].set_xlim(0, len(eeg_raw))
        axs[0].set_ylabel('Amplitude')
        axs[0].plot(peaks, peak_values, "rx", label="Peaks")
        for peak_idx, peak in zip(peaks, peak_values):
            axs[0].annotate(
                f"{peak_idx}\n{peak:.0f}",
                xy=(peak_idx, peak),
                xytext=(0, 10 if peak > 0 else -20),
                textcoords="offset points",
                ha="center",
                color="red",
                fontsize=8,
            )
        axs[0].legend()
        axs[1].specgram(eeg_raw[eeg_cols[0]], NFFT=256, Fs=128, noverlap=128, cmap='viridis')
        axs[1].set_title(f'Spectrogram - {eeg_cols[0]}')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Frequency (Hz)')
        plt.savefig(file_name.replace(".txt", "_eeg_raw_subplots_peaks.png"))
        plt.close()

        continue

        fig, axs = plt.subplots(len(eeg_cols), 1, figsize=(10, 8))
        for i, col in enumerate(eeg_cols):
            axs[i].plot(eeg_raw[col])
            axs[i].set_title(f'EEG Raw - {col}')
            axs[i].set_xlabel('Sample')
            axs[i].set_ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig(file_name.replace(".txt", "_eeg_raw_subplots.png"))
        plt.close()

        fig, axs = plt.subplots(len(eeg_cols), 1, figsize=(10, 8))
        for i, col in enumerate(eeg_cols):
            axs[i].specgram(eeg_raw[col], NFFT=256, Fs=128, noverlap=128, cmap='viridis')
            axs[i].set_title(f'Spectrogram - {col}')
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.savefig(file_name.replace(".txt", "_eeg_specgram.png"))
        plt.close()


if __name__ == "__main__":
    prepare()
    #analyze()
