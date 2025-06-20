# # import sounddevice as sd
# # import librosa
# # import numpy as np
# # import scipy.io.wavfile as wavfile
# # import os
# # # Parameters
# # DURATION = 3  # seconds
# # SAMPLERATE = 22050  # Hz
# # FILENAME = "speech_recording.wav"

# # # Step 1: Record audio from mic
# # print(f"Recording for {DURATION} seconds...")
# # audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1)
# # sd.wait()
# # print("Recording complete.")

# # # Convert to 1D and save
# # audio = audio.flatten()
# # wavfile.write(FILENAME, SAMPLERATE, (audio * 32767).astype(np.int16))

# # # Step 2: Load audio with librosa
# # y, sr = librosa.load(FILENAME, sr=SAMPLERATE)

# # # Step 3: Extract features
# # features = {}

# # # Pitch (estimated using librosa.yin)
# # f0 = librosa.yin(y, fmin=50, fmax=300)
# # features['pitch_mean'] = np.mean(f0)
# # features['pitch_std'] = np.std(f0)

# # # Energy (RMS)
# # rms = librosa.feature.rms(y=y)[0]
# # features['rms_mean'] = np.mean(rms)

# # # Tempo (BPM)
# # tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
# # features['tempo'] = tempo

# # # Zero-Crossing Rate
# # zcr = librosa.feature.zero_crossing_rate(y)[0]
# # features['zcr_mean'] = np.mean(zcr)

# # # Spectral features
# # features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
# # features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
# # features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

# # # MFCCs
# # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
# # for i in range(mfccs.shape[0]):
# #     features[f'mfcc_{i+1}'] = np.mean(mfccs[i])

# # # Chroma
# # chroma = librosa.feature.chroma_stft(y=y, sr=sr)
# # features['chroma_mean'] = np.mean(chroma)

# # # Output features
# # print("\n🔍 Extracted Speech Features:")
# # for k, v in features.items():

# #     print(f"{k}: {float(v):.3f}" if np.isscalar(v) else f"{k}: {v}")

# # # Optional: delete file
# # os.remove(FILENAME)



# import sounddevice as sd
# import numpy as np
# import librosa
# import time

# sr = 22050  # Sampling rate

# def extract_features(audio, sr):
#     features = {}
#     # Ensure mono
#     if len(audio.shape) > 1:
#         audio = np.mean(audio, axis=1)
    
#     # Extract features
#     pitch, _ = librosa.piptrack(y=audio, sr=sr)
#     pitches = pitch[pitch > 0]
#     features["pitch_mean"] = np.mean(pitches) if len(pitches) > 0 else 0
#     features["pitch_std"] = np.std(pitches) if len(pitches) > 0 else 0
    
#     features["rms_mean"] = np.mean(librosa.feature.rms(y=audio))
#     features["zcr_mean"] = np.mean(librosa.feature.zero_crossing_rate(y=audio))
#     features["tempo"] = librosa.beat.tempo(y=audio, sr=sr)[0]
#     features["mfccs"] = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)

#         # Additional features for Clarity
#     features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
#     features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
#     features["chroma_mean"] = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr))

#     return features
# def analyze_speech(features):
#     analysis = []

#     # Pitch
#     if features["pitch_mean"] < 100:
#         analysis.append("⚠️ Voice may sound dull or low-pitched.")
#     elif features["pitch_mean"] > 250:
#         analysis.append("⚠️ Voice might be too high-pitched.")
#     else:
#         analysis.append("✅ Pitch is within a natural speaking range.")

#     if features["pitch_std"] < 10:
#         analysis.append("⚠️ Try adding more pitch variation for expressiveness.")
#     else:
#         analysis.append("✅ Good pitch variation detected.")

#     # Loudness / Energy
#     if features["rms_mean"] < 0.02:
#         analysis.append("⚠️ Speaking too softly. Increase volume.")
#     elif features["rms_mean"] > 0.1:
#         analysis.append("⚠️ Voice might be too loud or harsh.")
#     else:
#         analysis.append("✅ Volume level seems appropriate.")

#     # Tempo (words per minute approximation)
#     if features["tempo"] < 90:
#         analysis.append("⚠️ You may be speaking too slowly.")
#     elif features["tempo"] > 160:
#         analysis.append("⚠️ You may be speaking too fast.")
#     else:
#         analysis.append("✅ Speaking pace looks natural.")

#     # Optional: ZCR can indicate articulation or sharpness
#     if features["zcr_mean"] > 0.1:
#         analysis.append("⚠️ Speech may be sharp or hissy (check articulation).")
#     else:
#         analysis.append("✅ Speech articulation is within a normal range.")
    
#     if features["spectral_centroid"] < 1500:
#         print("🔈 Try to speak more clearly or with more energy.")

#     if features["spectral_bandwidth"] < 1800:
#         print("📉 Your voice may sound muffled or dull — increase enunciation.")

#     if features["chroma_mean"] < 0.3:
#         print("🎵 Add more variation to your pitch for expressive delivery.")


#     return analysis

# duration = 3  # seconds per chunk
# print("🎤 Listening... (Ctrl+C to stop)")

# try:
#     while True:
#         audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
#         sd.wait()
#         chunk = audio.flatten()

#         features = extract_features(chunk, sr)
        
#         analysis= analyze_speech(features)
#         # Display (replace with streamlit/web socket later)
#         print(f"🎧 PITCH: {features['pitch_mean']:.1f} Hz | ENERGY: {features['rms_mean']:.3f} | TEMPO: {features['tempo']:.1f} BPM")
#         print(f"ZCR: {features['zcr_mean']:.3f}")
#         print("MFCCs:", ", ".join([f"{v:.2f}" for v in features["mfccs"][:5]]), "...")
#         print("Analysis:")
#         for a in analysis:
#             print("•", a)
#         print("------")
        
#         time.sleep(0.2)  # Optional: slight buffer

# except KeyboardInterrupt:
#     print("🛑 Stopped.")

# import librosa.display
# import matplotlib.pyplot as plt

# mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mfcc, x_axis='time')
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# plt.show()
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

        # Additional features for Clarity
    features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    features["chroma_mean"] = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr))

    return features
def analyze_speech(features):
    alerts = []

    # 🎵 Pitch
    if features["pitch_mean"] < 100:
        alerts.append("📢 Your pitch is quite low. Try speaking with more energy.")
    elif features["pitch_mean"] > 250:
        alerts.append("📢 Your pitch is unusually high. Consider toning it down.")

    if features["pitch_std"] < 10:
        alerts.append("🎙️ Your voice lacks variation. Try adding some pitch dynamics.")

    # 🔊 Loudness / Energy
    if features["rms_mean"] < 0.02:
        alerts.append("🔈 You're speaking too softly. Increase your volume.")
    elif features["rms_mean"] > 0.1:
        alerts.append("🔊 Your volume is too high. Lower it slightly for comfort.")

    # 🕒 Tempo
    if features["tempo"] < 90:
        alerts.append("🐢 You may be speaking too slowly. Increase your pace.")
    elif features["tempo"] > 160:
        alerts.append("⚡ You're speaking too fast. Try slowing down.")

    # 🫧 Articulation via ZCR
    if features["zcr_mean"] > 0.1:
        alerts.append("💨 Your speech has high sibilance or sharpness. Speak more clearly.")
 
    if features["spectral_centroid"] < 1500:
        print("🔈 Try to speak more clearly or with more energy.")

    if features["spectral_bandwidth"] < 1800:
        print("📉 Your voice may sound muffled or dull — increase enunciation.")

    if features["chroma_mean"] < 0.3:
        print("🎵 Add more variation to your pitch🎵 Let your voice rise and fall a bit more — it'll keep your listeners hooked.")

    return alerts

def send_to_gemini(all_results):
    """Send the accumulated results to Gemini for post-analysis.
    Replace this method with actual Gemini API call."""
    prompt = "Here are voice metrics every 3 seconds:\n\n"
    for i, result in enumerate(all_results, 1):
        prompt += f"Chunk {i}: pitch_mean={result['pitch_mean']:.2f}, rms_mean={result['rms_mean']:.3f}, zcr_mean={result['zcr_mean']:.3f}, " \
                  f"tempo={result['tempo']:.1f}, spectral_centroid={result['spectral_centroid']:.1f}, spectral_bandwidth={result['spectral_bandwidth']:.1f}, chroma_mean={result['chroma_mean']:.3f}\n"

    prompt += (
        "\nPlease analyze this data as a public speaking coach and give overall feedback. "
        "Mention areas of improvement for pitch, volume, tempo, and clarity."
    )
    print("\n=== Sending the following prompt to Gemini ===\n", prompt)

    # Here you can implement Gemini API call
    # For now, we'll just save it to a file
    #response = model.generate_content(prompt)
    #print("\n=== Gemini's Analysis ===\n", response.text)
    with open("gemini_prompt.txt", "w") as f:
        f.write(prompt)

duration = 3  # seconds per chunk
all_results=[]
print("🎤 Listening... (Ctrl+C to stop)")

try:
    while True:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        chunk = audio.flatten()

        features = extract_features(chunk, sr)
        #collecting the results for post analysis
        all_results.append(features)

        
        # Display (replace with streamlit/web socket later)
        alerts = analyze_speech(features)

        print(f"🎧 PITCH: {features['pitch_mean']:.1f} Hz | ENERGY: {features['rms_mean']:.3f} | TEMPO: {features['tempo']:.1f} BPM")
        print(f"ZCR: {features['zcr_mean']:.3f}")
        print("MFCCs:", ", ".join([f"{v:.2f}" for v in features["mfccs"][:5]]), "...")
        print("Analysis:")
        for a in alerts:
            print("•", a)
        print("------")
        
        time.sleep(0.2)  # Optional: slight buffer

except KeyboardInterrupt:
    print("🛑 Stopped.")

send_to_gemini(all_results)