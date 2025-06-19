import sounddevice as sd
import librosa
import numpy as np
import scipy.io.wavfile as wavfile
import os
# Parameters
DURATION = 3  # seconds
SAMPLERATE = 22050  # Hz
FILENAME = "speech_recording.wav"

# Step 1: Record audio from mic
print(f"Recording for {DURATION} seconds...")
audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1)
sd.wait()
print("Recording complete.")

# Convert to 1D and save
audio = audio.flatten()
wavfile.write(FILENAME, SAMPLERATE, (audio * 32767).astype(np.int16))

# Step 2: Load audio with librosa
y, sr = librosa.load(FILENAME, sr=SAMPLERATE)

# Step 3: Extract features
features = {}

# Pitch (estimated using librosa.yin)
f0 = librosa.yin(y, fmin=50, fmax=300)
features['pitch_mean'] = np.mean(f0)
features['pitch_std'] = np.std(f0)

# Energy (RMS)
rms = librosa.feature.rms(y=y)[0]
features['rms_mean'] = np.mean(rms)

# Tempo (BPM)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
features['tempo'] = tempo

# Zero-Crossing Rate
zcr = librosa.feature.zero_crossing_rate(y)[0]
features['zcr_mean'] = np.mean(zcr)

# Spectral features
features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

# MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
for i in range(mfccs.shape[0]):
    features[f'mfcc_{i+1}'] = np.mean(mfccs[i])

# Chroma
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
features['chroma_mean'] = np.mean(chroma)

# Output features
print("\n🔍 Extracted Speech Features:")
for k, v in features.items():

    print(f"{k}: {float(v):.3f}" if np.isscalar(v) else f"{k}: {v}")

# Optional: delete file
os.remove(FILENAME)



import sounddevice as sd
import numpy as np
import librosa
import time

sr = 22050  # Sampling rate

def extract_features(audio, sr):
    features = {}
    # Ensure mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Extract features
    pitch, _ = librosa.piptrack(y=audio, sr=sr)
    pitches = pitch[pitch > 0]
    features["pitch_mean"] = np.mean(pitches) if len(pitches) > 0 else 0
    features["pitch_std"] = np.std(pitches) if len(pitches) > 0 else 0
    
    features["rms_mean"] = np.mean(librosa.feature.rms(y=audio))
    features["zcr_mean"] = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    features["tempo"] = librosa.beat.tempo(y=audio, sr=sr)[0]
    features["mfccs"] = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)

    return features

duration = 3  # seconds per chunk
print("🎤 Listening... (Ctrl+C to stop)")

try:
    while True:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        chunk = audio.flatten()

        features = extract_features(chunk, sr)
        
        # Display (replace with streamlit/web socket later)
        print(f"🎧 PITCH: {features['pitch_mean']:.1f} Hz | ENERGY: {features['rms_mean']:.3f} | TEMPO: {features['tempo']:.1f} BPM")
        print(f"ZCR: {features['zcr_mean']:.3f}")
        print("MFCCs:", ", ".join([f"{v:.2f}" for v in features["mfccs"][:5]]), "...")
        print("------")
        
        time.sleep(0.2)  # Optional: slight buffer

except KeyboardInterrupt:
    print("🛑 Stopped.")
