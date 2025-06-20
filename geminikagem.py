import asyncio
import numpy as np
import librosa
import parselmouth
import sounddevice as sd
import websockets
import json
import time
import os
import google.genai as genai
import soniox as stt
# Assuming Soniox library is installed
#import soniox.speech_to_text as stt
#fom soniox.speech_to_text import AudioChunk, AudioChunkMark, Segment, SpeechContext

#from soniox import AudioChunk, AudioChunkMark, Segment, SpeechContext

# --- Configuration ---
# Replace with your actual Soniox API Key
SONIOX_API_KEY = os.environ.get("SONIOX_API_KEY", "0c8a505b0b7b3850353998ada7b3c95e78ba39bd4beed9152bb14b41e0740fa1") # Recommended: Use environment variable
if SONIOX_API_KEY == "YOUR_SONIOX_API_KEY":
    print("WARNING: Soniox API Key is not set. Please set the SONIOX_API_KEY environment variable or replace 'YOUR_SONIOX_API_KEY'.")

# Audio settings
SAMPLING_RATE = 16000  # Soniox recommends 16kHz for best results, adjust if your source is different
CHUNK_SIZE_SECONDS = 3 # Analyze chunks every 3 seconds
BUFFER_SIZE_FRAMES = int(SAMPLING_RATE * CHUNK_SIZE_SECONDS)

# Queues for inter-task communication
audio_queue = asyncio.Queue() # Raw audio chunks from recorder
soniox_queue = asyncio.Queue() # Results from Soniox
analysis_queue = asyncio.Queue() # Results from librosa analysis

# --- Your existing librosa/parselmouth functions (slightly adapted for async) ---

def extract_formants(audio, sr, time_step=0.01, max_formant=5500, n_formants=3):
    """Extract average F1, F2, and F3 from an audio chunk."""
    # Ensure audio is float64 for parselmouth
    audio_float64 = audio.astype(np.float64)
    snd = parselmouth.Sound(audio_float64, sr)
    formant = snd.to_formant_burg(time_step=time_step,
                                   max_number_of_formants=n_formants,
                                   maximum_formant=max_formant)

    times = np.arange(0, snd.get_total_duration(), time_step)
    formant_values = []
    for t in times:
        vals = []
        for i in range(1, n_formants + 1):
            val = formant.get_value_at_time(i, t)
            if not np.isnan(val): # Filter out NaN values from parselmouth
                vals.append(val)
        if vals:
            formant_values.append(vals)

    if not formant_values:
        return {"F1_mean": 0.0, "F2_mean": 0.0, "F3_mean": 0.0}

    formant_values = np.array(formant_values, dtype=np.float32)
    mean_values = np.nanmean(formant_values, axis=0) # Use nanmean

    return {
        "F1_mean": float(mean_values[0]) if len(mean_values) > 0 and not np.isnan(mean_values[0]) else 0.0,
        "F2_mean": float(mean_values[1]) if len(mean_values) > 1 and not np.isnan(mean_values[1]) else 0.0,
        "F3_mean": float(mean_values[2]) if len(mean_values) > 2 and not np.isnan(mean_values[2]) else 0.0,
    }


def extract_features(audio, sr):
    """Extract relevant audio features, including formants."""
    features = {}

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Ensure audio is float32 for librosa
    audio = audio.astype(np.float32)

    # Pitch tracking - handle potential no-pitch
    # f0, _, = librosa.pyin(y=audio, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C5'))
    f0, voiced_flag, voiced_probabilities = librosa.pyin(
        y=audio, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C5')
    )
    pitches = f0[~np.isnan(f0)] # Filter out NaN values
    
    features["pitch_mean"] = float(np.mean(pitches)) if len(pitches) > 0 else 0.0
    features["pitch_std"] = float(np.std(pitches)) if len(pitches) > 0 else 0.0

    features["rms_mean"] = float(np.mean(librosa.feature.rms(y=audio)))
    features["zcr_mean"] = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    
    # Tempo estimation can be less reliable on short chunks, but we'll include it
    try:
        features["tempo"] = float(librosa.beat.tempo(y=audio, sr=sr)[0])
    except Exception:
        features["tempo"] = 0.0 # Handle case where tempo cannot be estimated

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features["mfccs"] = [float(val) for val in np.mean(mfccs, axis=1)]

    features["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    features["spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    features["chroma_mean"] = float(np.mean(librosa.feature.chroma_stft(y=audio, sr=sr)))

    # Formant Features
    formants = extract_formants(audio, sr)
    features.update(formants)

    return features

def analyze_speech_local_alerts(features):
    """Check various vocal characteristics and alert the user accordingly."""
    alerts = []
    # Make sure to handle potential None/NaN values from feature extraction gracefully
    pitch_mean = features.get("pitch_mean", 0)
    pitch_std = features.get("pitch_std", 0)
    rms_mean = features.get("rms_mean", 0)
    tempo = features.get("tempo", 0)
    zcr_mean = features.get("zcr_mean", 0)
    spectral_centroid = features.get("spectral_centroid", 0)
    spectral_bandwidth = features.get("spectral_bandwidth", 0)
    chroma_mean = features.get("chroma_mean", 0)


    if pitch_mean > 0: # Only check if pitch was detected
        if pitch_mean < 100: # General low male voice, adjust as per target audience
            alerts.append("📢 Your pitch is quite low. Try speaking with more energy.")
        elif pitch_mean > 250: # General high female voice, adjust as per target audience
            alerts.append("📢 Your pitch is unusually high. Consider toning it down.")

    if pitch_std < 10 and pitch_mean > 0: # Only if pitch was detected and is too flat
        alerts.append("🎙️ Your voice lacks variation. Try adding some pitch dynamics.")

    if rms_mean < 0.02: # Adjust threshold as needed
        alerts.append("🔈 You're speaking too softly. Increase your volume.")
    elif rms_mean > 0.15: # Adjust threshold as needed
        alerts.append("🔊 You're too loud. Lower your volume slightly for comfort.")

    if tempo > 0: # Only if tempo was estimated
        if tempo < 90:
            alerts.append("🐢 You're speaking too slowly. Try increasing your pace.")
        elif tempo > 160:
            alerts.append("⚡ You're speaking too quickly. Try slowing down.")

    # ZCR can indicate speech clarity/noisiness
    if zcr_mean > 0.15: # High ZCR might indicate fricatives or noise
        alerts.append("💨 High sibilance or sharpness detected. Speak more clearly.")

    # Spectral centroid and bandwidth for voice quality/brightness
    if spectral_centroid > 0 and spectral_centroid < 1500:
        alerts.append("🔈 Try to speak more clearly or with more energy. Your voice might sound a bit muffled.")

    if spectral_bandwidth > 0 and spectral_bandwidth < 1800:
        alerts.append("📉 Your voice may sound dull or lack fullness — try increasing enunciation.")

    if chroma_mean > 0 and chroma_mean < 0.3:
        alerts.append("🎵 Add more pitch variation for a dynamic voice.")

    # Formant alerts (these are more advanced and might require careful tuning)
    if features.get("F1_mean", 0) > 0 and features.get("F1_mean", 0) < 300:
        alerts.append("👄 Your F1 is low, which might affect open vowel pronunciation.")
    if features.get("F2_mean", 0) > 0 and features.get("F2_mean", 0) < 1000:
        alerts.append("👅 Your F2 is low, which might affect front vowel articulation.")
    if features.get("F3_mean", 0) > 0 and features.get("F3_mean", 0) < 2500:
        alerts.append("🗣️ Your F3 is low, which might affect overall resonance clarity.")

    return alerts

# --- Async Tasks ---

async def soniox_transcription_task():
    """Handles real-time transcription using Soniox WebSocket API."""
    print("Soniox Transcription Task: Starting...")
    try:
        async with stt.SpeechToTextClient(SONIOX_API_KEY) as client:
            request = stt.TranscribeStreamRequest(
                enable_partials=True,
                include_word_info=True,
                audio_config=stt.AudioConfig(
                    encoding=stt.AudioEncoding.LINEAR16, # Raw PCM
                    sample_rate_hertz=SAMPLING_RATE,
                    num_channels=1
                )
            )
            async with client.transcribe_stream(request) as stream:
                print("Soniox Transcription Task: Connected to Soniox API.")
                while True:
                    # Get audio from the queue
                    audio_chunk = await audio_queue.get()
                    if audio_chunk is None: # Sentinel for end of stream
                        print("Soniox Transcription Task: Received stop signal. Closing stream.")
                        await stream.close()
                        break

                    # Send audio to Soniox
# DIRECTLY send the raw PCM data
                    await stream.send(audio_chunk.tobytes())

                    # Process responses from Soniox
                    response = await stream.recv()
                    if response.is_partial:
                        # print(f"Soniox Partial: {response.transcript}")
                        await soniox_queue.put({"type": "partial", "transcript": response.transcript, "words": []})
                    else:
                        full_transcript = response.transcript
                        words_info = [{"word": w.word, "start": w.start_time_seconds, "end": w.end_time_seconds} for w in response.words]
                        print(f"Soniox Final: {full_transcript}")
                        await soniox_queue.put({"type": "final", "transcript": full_transcript, "words": words_info})

    except Exception as e:
        print(f"Soniox Transcription Task Error: {e}")
    finally:
        print("Soniox Transcription Task: Exiting.")

async def librosa_analysis_task():
    """Handles real-time librosa/parselmouth audio analysis."""
    print("Librosa Analysis Task: Starting...")
    chunk_buffer = np.array([], dtype=np.float32)
    chunk_counter = 0
    try:
        while True:
            # Get audio from the queue (raw audio bytes)
            audio_data = await audio_queue.get()
            if audio_data is None: # Sentinel for end of stream
                print("Librosa Analysis Task: Received stop signal. Processing remaining buffer.")
                if len(chunk_buffer) > 0:
                    features = extract_features(chunk_buffer, SAMPLING_RATE)
                    alerts = analyze_speech_local_alerts(features)
                    await analysis_queue.put({"chunk_id": chunk_counter, "features": features, "alerts": alerts})
                    print(f"Librosa Analysis Task: Processed final buffered chunk {chunk_counter}")
                break

            # Add to buffer
            chunk_buffer = np.concatenate((chunk_buffer, audio_data))

            # Process chunks of specific duration
            while len(chunk_buffer) >= BUFFER_SIZE_FRAMES:
                current_chunk = chunk_buffer[:BUFFER_SIZE_FRAMES]
                chunk_buffer = chunk_buffer[BUFFER_SIZE_FRAMES:]

                chunk_counter += 1
                
                # Perform analysis
                features = extract_features(current_chunk, SAMPLING_RATE)
                alerts = analyze_speech_local_alerts(features)
                
                # Put results into analysis queue
                await analysis_queue.put({"chunk_id": chunk_counter, "features": features, "alerts": alerts})
                
                # Print real-time alerts
                print(f"\n--- Analysis Chunk {chunk_counter} ---")
                print(f"🎧 PITCH: {features['pitch_mean']:.1f} Hz | ENERGY: {features['rms_mean']:.3f} | TEMPO: {features['tempo']:.1f} BPM")
                print(f"ZCR: {features['zcr_mean']:.3f} | F1: {features.get('F1_mean', 0):.1f} Hz | F2: {features.get('F2_mean', 0):.1f} Hz | F3: {features.get('F3_mean', 0):.1f} Hz ")
                # print("MFCCs:", ", ".join([f"{v:.2f}" for v in features["mfccs"][:5]]), "...") # MFCCs are complex for real-time display
                print("Analysis Alerts:")
                if alerts:
                    for a in alerts:
                        print("•", a)
                else:
                    print("• No immediate alerts for this chunk.")
                print("---------------------")

    except Exception as e:
        print(f"Librosa Analysis Task Error: {e}")
    finally:
        print("Librosa Analysis Task: Exiting.")

async def audio_recorder_task():
    """Records audio from microphone and puts chunks into a queue."""
    print("Audio Recorder Task: Starting...")
    try:
        with sd.InputStream(samplerate=SAMPLING_RATE, channels=1, dtype='float32', blocksize=BUFFER_SIZE_FRAMES) as stream:
            print(f"🎤 Recording audio at {SAMPLING_RATE} Hz in chunks of {CHUNK_SIZE_SECONDS} seconds. Press Ctrl+C to stop.")
            while stream.active:
                try:
                    data, overflowed = stream.read(BUFFER_SIZE_FRAMES) # Read exactly BUFFER_SIZE_FRAMES
                    if overflowed:
                        print("Audio Recorder Task: Input stream overflowed!")
                    
                    # Put audio chunk into the queue for both tasks to consume
                    # Use audio_queue.put_nowait if you want to drop frames if consumers are slow
                    await audio_queue.put(data.flatten())
                except sd.PortAudioError as e:
                    print(f"Audio Recorder Task: PortAudioError: {e}")
                    break # Exit on audio error
                except Exception as e:
                    print(f"Audio Recorder Task: Unexpected error during recording: {e}")
                    break
                
                # Small sleep to allow other tasks to run, though not strictly needed with async queues
                await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\nAudio Recorder Task: KeyboardInterrupt detected.")
    except Exception as e:
        print(f"Audio Recorder Task Error: {e}")
    finally:
        print("Audio Recorder Task: Exiting.")
        # Signal consumers to stop by putting None
        await audio_queue.put(None) # Signal for Soniox task
        await audio_queue.put(None) # Signal for Librosa task (since both consume from the same queue)
        print("Audio Recorder Task: Sent stop signals to consumers.")


async def results_aggregator_task():
    """Aggregates results from Soniox and Librosa and potentially sends to Gemini for final analysis."""
    print("Results Aggregator Task: Starting...")
    all_librosa_results = []
    transcripts = [] # To store full transcripts
    
    # Store pending tasks for graceful shutdown
    pending_tasks = {
        asyncio.create_task(soniox_transcription_task()),
        asyncio.create_task(librosa_analysis_task()),
        asyncio.create_task(audio_recorder_task())
    }

    try:
        while True:
            # Wait for any result from either queue
            done, pending_tasks = await asyncio.wait(
                [
                    asyncio.create_task(soniox_queue.get()),
                    asyncio.create_task(analysis_queue.get()),
                    *pending_tasks # Include other long-running tasks to monitor their state
                ],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=1.0 # Timeout to check if main tasks are still running
            )

            # Process completed tasks
            for task in done:
                if task in pending_tasks: # If it's one of the long-running tasks, it finished
                    pending_tasks.remove(task)
                    continue # Already processed its completion state
                
                try:
                    result = task.result() # Get result from the completed queue .get() task
                    if result is None: # A queue returned None, might indicate a producer finished
                        # Check if all producers are done, if so, exit aggregator
                        if audio_queue.empty() and soniox_queue.empty() and analysis_queue.empty() and not any(t.done() for t in pending_tasks):
                            print("Results Aggregator Task: All queues empty and producers seem done. Exiting.")
                            return
                        continue

                    if "type" in result and result["type"] in ["partial", "final"]:
                        # print(f"Received Soniox Result: {result['transcript']}")
                        if result["type"] == "final":
                            transcripts.append(result["transcript"])
                    elif "chunk_id" in result:
                        # print(f"Received Librosa Analysis for Chunk {result['chunk_id']}")
                        all_librosa_results.append(result["features"])

                except asyncio.CancelledError:
                    print("Results Aggregator Task: Task was cancelled.")
                    return # Exit gracefully
                except Exception as e:
                    print(f"Results Aggregator Task: Error processing result: {e}")
            
            # If all producer tasks are done and queues are empty, we can exit
            if all(t.done() for t in pending_tasks) and audio_queue.empty() and soniox_queue.empty() and analysis_queue.empty():
                print("Results Aggregator Task: All producer tasks completed and queues are empty. Finalizing.")
                break

    except asyncio.CancelledError:
        print("Results Aggregator Task: Cancelled.")
    except Exception as e:
        print(f"Results Aggregator Task Error: {e}")
    finally:
        print("Results Aggregator Task: Exiting.")
        # Ensure all pending tasks are cancelled on exit
        for task in pending_tasks:
            task.cancel()
        await asyncio.gather(*pending_tasks, return_exceptions=True) # Wait for them to truly finish cancelling

        # Perform final Gemini analysis if data was collected
        if all_librosa_results:
            print("\n=== All recording and analysis finished. Sending final data to Gemini ===\n")
            await send_to_gemini_async(all_librosa_results) # Call async version

async def send_to_gemini_async(all_results):
    """Asynchronously sends the accumulated results to Gemini for post-analysis."""
    prompt = "Here are voice metrics every 3 seconds:\n\n"
    for i, result in enumerate(all_results, 1):
        # Format the features for Gemini, ensuring float conversion and handling missing keys
        formatted_features = {k: f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v) for k, v in result.items()}
        
        # Ensure MFCCs are properly formatted if they exist
        mfccs_str = ", ".join([f"{val:.2f}" for val in formatted_features.get('mfccs', [])[:5]])
        
        prompt += (
            f"Chunk {i}: Vocal Pitch Avg={formatted_features.get('pitch_mean', 'N/A')}, Pitch Var={formatted_features.get('pitch_std', 'N/A')}, "
            f"Loudness Avg={formatted_features.get('rms_mean', 'N/A')}, Clarity (ZCR Avg)={formatted_features.get('zcr_mean', 'N/A')}, "
            f"Pace (BPM)={formatted_features.get('tempo', 'N/A')}, "
            f"F1 Avg={formatted_features.get('F1_mean', 'N/A')}, F2 Avg={formatted_features.get('F2_mean', 'N/A')}, F3 Avg={formatted_features.get('F3_mean', 'N/A')}, "
            f"Vocal Brightness (Centroid)={formatted_features.get('spectral_centroid', 'N/A')}, "
            f"Vocal Fullness (Bandwidth)={formatted_features.get('spectral_bandwidth', 'N/A')}, "
            f"Vocal Expressiveness (Chroma)={formatted_features.get('chroma_mean', 'N/A')}\n"
        )

    # The coach prompt is exactly as requested in the previous turn
    prompt += (
        "You are a highly experienced public speaking coach. I need your expert analysis of a speech, based on the vocal characteristics I provide. "
        "Please avoid technical jargon and focus on actionable advice.\n\n"
        "For each speech, I will give you values for:\n"
        "* **Vocal Pitch (Average and Variation):** How high or low the speaker's voice generally is, and how much it changes.\n"
        "* **Vocal Loudness (Overall):** The general volume and power of the speaker's voice.\n"
        "* **Speech Clarity (Pronunciation and Crispness):** How clear and distinct the speaker's words are.\n"
        "* **Speaking Pace (Speed):** How fast or slow the speaker talks.\n"
        "* **Vocal Resonance (Warmth and Fullness):** The richness and depth of the speaker's voice.\n\n"
        "Based on these, please provide:\n\n"
        "1. **Strengths:** What did the speaker do well vocally?\n"
        "2. **Areas for Improvement:** Where could their voice be more effective?\n"
        "3. **Tips for Improvement:** Specific, easy-to-understand actions the speaker can take to enhance their vocal delivery.\n\n"
        "Your feedback should be concise, direct, and practical."
    )
    
    print("\n=== Sending the following prompt to Gemini ===\n")
    try:
        # Use a non-blocking API call for Gemini
        response = await asyncio.to_thread(lambda: genai.GenerativeModel("gemini-2.0-flash").generate_content(contents=prompt))
        print("\n=== Gemini's Analysis ===\n", response.text)
        return response.text
    except Exception as e:
        print(f"Error sending to Gemini: {e}")
        return f"Error: Could not get analysis from Gemini: {e}"


# --- Main Execution ---
async def main():
    print("Starting Public Speaking Trainer Backend...")
    # Start all tasks concurrently
    aggregator_task = asyncio.create_task(results_aggregator_task())

    try:
        # Wait for the aggregator task to complete, which signals overall completion
        await aggregator_task
    except KeyboardInterrupt:
        print("\nMain: KeyboardInterrupt detected. Shutting down.")
    except Exception as e:
        print(f"Main: An error occurred: {e}")
    finally:
        print("Main: All tasks are shutting down.")
        # Ensure all tasks are cancelled and cleaned up
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        print("Main: All tasks stopped.")

if __name__ == "__main__":
    # Set your Soniox API Key (recommended via environment variable)
    os.environ["SONIOX_API_KEY"] = "0c8a505b0b7b3850353998ada7b3c95e78ba39bd4beed9152bb14b41e0740fa1" # Uncomment and set if not using system env
    
    # Initialize Google Gemini client
    # Assuming client is set up globally or passed around if needed for other Gemini calls
    # For this specific example, the `send_to_gemini_async` function creates its own model instance.
    client = genai.Client(api_key="AIzaSyCqq5SRE6mU9Fh8jpduWGXvdDKgzl0KZIY")

    asyncio.run(main())