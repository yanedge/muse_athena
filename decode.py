import pandas as pd
import OpenMuse 
import matplotlib.pyplot as plt

with open("data2.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()
data = OpenMuse.decode_rawdata(messages)

# Inspect the structure of `data`
print("Type of data:", type(data))
print("Keys in data:", data.keys())

# Display all decoded column names for the "ACCGYRO" DataFrame
if "ACCGYRO" in data:
    print("Decoded column names:", data["ACCGYRO"].columns.tolist())
else:
    print("Key 'ACCGYRO' not found in data.")

# Display all decoded column names
# print("Decoded column names:", data.columns.tolist())

# Plot Movement Data
data["ACCGYRO"].plot(
    x="time",
    y=["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"],
    subplots=True
)
plt.show()

# Plot EEG Data
if "EEG" in data:
    print("Decoded column names for EEG:", data["EEG"].columns.tolist())
    # Plot EEG data
    data["EEG"].plot(
        x="time",  # Assuming 'time' is the x-axis column
        y=data["EEG"].columns.drop("time"),  # Plot all columns except 'time'
        subplots=True
    )
    plt.show()
else:
    print("Key 'EEG' not found in data.")