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
# Or more directly:
import google.genai as genai

client = genai.Client(api_key="GEMINI_API_KEY")


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
        "\nPlease review this data as a public speaking coach and give overall feedback. "
        "Make it very simple and human-readable — no technical jargon or metric names. "
        "Just give clear, actionable advice about how the person can speak more confidently, clearly, and consistently."
    )

    
    print("\n=== Sending the following prompt to Gemini ===\n", prompt)

    # Here you can implement Gemini API call

    response = client.models.generate_content(
        model="gemini-1.5-flash", 
        contents=prompt
    )
    print("\n=== Gemini's Analysis ===\n", response.text)
    return response.text


    # response = client.models.generate_content(
    #     model="gemini-1.5-flash",
    #     contents="Hello, world!"
    
    # with open("gemini_prompt.txt", "w") as f:
    #     f.write(prompt)

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