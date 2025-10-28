import pandas as pd
import OpenMuse 
import matplotlib.pyplot as plt
import scipy.signal as signal

files = ["data3.txt", "data4.txt", "data5.txt", "data6.txt"]

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        messages = f.readlines()
    data = OpenMuse.decode_rawdata(messages)

    eeg_cols = ["EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10"]
    eeg_raw = data["EEG"][eeg_cols].values
    eeg_raw = pd.DataFrame(eeg_raw, columns=eeg_cols)
    eeg_raw_resampled = pd.DataFrame(
        signal.resample(eeg_raw.values, int(len(eeg_raw) * 128 / 256)),
        columns=eeg_cols
    )
    eeg_raw = eeg_raw_resampled
    eeg_raw.to_csv(file.replace(".txt", "_eeg_128.csv"), index=False)

    data["EEG"].plot(
        x="time",  # Assuming 'time' is the x-axis column
        y=data["EEG"].columns.drop("time"),  # Plot all columns except 'time'
        subplots=True,
    )

    plt.savefig(file.replace(".txt", "_eeg_plot.png"))
    plt.close()
