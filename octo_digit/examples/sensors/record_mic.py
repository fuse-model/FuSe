# import sounddevice as sd
# import numpy as np
# import matplotlib.pyplot as plt

# # Settings
# duration = 3  # Duration of the recording in seconds
# fs = 44100  # Sampling frequency

# # Record audio
# print("Recording...")
# audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
# sd.wait()  # Wait for recording to finish

# # Plot the waveform of the recorded audio
# plt.figure(figsize=(10, 6))
# time_axis = np.linspace(0, duration, len(audio_data))
# plt.plot(time_axis, audio_data)
# plt.xlabel("Time (seconds)")
# plt.ylabel("Amplitude")
# plt.title("Recorded Audio Waveform")
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

# Settings
duration = 3  # Duration of the recording in seconds
fs = 44100  # Sampling frequency

# Record audio
print("Recording...")
audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
sd.wait()  # Wait for recording to finish

# Perform FFT (Fast Fourier Transform) to analyze frequency content
n = len(audio_data)
frequencies = np.fft.rfftfreq(n, d=1 / fs)  # Real FFT frequencies
magnitude_spectrum = np.abs(np.fft.rfft(audio_data, axis=0))

# Plot the waveform and frequency spectrum of the recorded audio
plt.figure(figsize=(12, 6))

# Plot waveform
plt.subplot(2, 1, 1)
time_axis = np.linspace(0, duration, n)
plt.plot(time_axis, audio_data)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Recorded Audio Waveform")

# Plot frequency spectrum
plt.subplot(2, 1, 2)
plt.plot(frequencies, magnitude_spectrum)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum")
plt.tight_layout()
plt.show()
