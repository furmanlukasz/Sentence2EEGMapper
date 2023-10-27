import numpy as np
from pylsl import StreamInfo, StreamOutlet
import time

# Define the stream parameters
n_channels = 32  # Let's assume 32 EEG channels for this example
sampling_rate = 512  # in Hz
channel_format = 'float32'
stream_name = 'TestEEG'
stream_type = 'EEG'

# Create a StreamInfo object
info = StreamInfo(stream_name, stream_type, n_channels, sampling_rate, channel_format)

# Create a StreamOutlet
outlet = StreamOutlet(info)

print("Now sending synthetic EEG data... Press Ctrl+C to stop.")

try:
    while True:
        # Generate random EEG data for the sake of this example
        eeg_data = np.random.rand(n_channels) * 100  # Random values between 0 and 100
        eeg_data *= 1e-6 # Scale down the data to microvolts
        # Send the data
        outlet.push_sample(eeg_data)
        # Sleep for the duration of 1 sample
        time.sleep(1.0 / sampling_rate)
except KeyboardInterrupt:
    print("Stopping stream...")

