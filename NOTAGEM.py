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

# Correct Soniox imports
import soniox as stt
#from soniox import AudioChunk, AudioChunkMark, Segment, SpeechContext

# --- Configuration ---
SONIOX_API_KEY = os.environ.get("SONIOX_API_KEY", "sonikey")
if SONIOX_API_KEY == "YOUR_SONIOX_API_KEY":
    print("WARNING: Soniox API Key is not set. Please set the SONIOX_API_KEY environment variable or replace 'YOUR_SONIOX_API_KEY' with your actual key.")

# Audio settings
SAMPLING_RATE = 16000  # Soniox recommends 16kHz for best results
CHUNK_SIZE_SECONDS = 0.5 # Reduced chunk size for potentially lower latency and better responsiveness
BUFFER_SIZE_FRAMES = int(SAMPLING_RATE * CHUNK_SIZE_SECONDS) # Block size for sounddevice

# Queues for inter-task communication
# Separate queues for each consumer of audio data
soniox_audio_queue = asyncio.Queue()
librosa_audio_queue = asyncio.Queue()

# Queues for results from processing tasks
soniox_result_queue = asyncio.Queue()
analysis_result_queue = asyncio.Queue()

# --- Your existing librosa/parselmouth functions (with robustness improvements) ---

# import numpy as np
# import parselmouth

def extract_formants(audio, sr, time_step=0.01, max_formant=5500, n_formants=3):
    """Extract average F1, F2, and F3 from an audio chunk."""
    if len(audio) == 0:
        return {"F1_mean": 0.0, "F2_mean": 0.0, "F3_mean": 0.0}

    audio_float64 = audio.astype(np.float64)
    snd = parselmouth.Sound(audio_float64, sr)
    
    if snd.get_total_duration() == 0:
        return {"F1_mean": 0.0, "F2_mean": 0.0, "F3_mean": 0.0}

    formant = snd.to_formant_burg(time_step=time_step,
                                   max_number_of_formants=n_formants,
                                   maximum_formant=max_formant)

    times = np.arange(0, snd.get_total_duration(), time_step)
    if len(times) == 0:
        return {"F1_mean": 0.0, "F2_mean": 0.0, "F3_mean": 0.0}
        
    formant_values = []
    for t in times:
        current_time_formants = []
        for i in range(1, n_formants + 1):
            val = formant.get_value_at_time(i, t)
            current_time_formants.append(val if not np.isnan(val) else np.nan) # Append NaN if value is NaN
        
        # Only append if at least one formant was valid for this time slice
        # If all were NaN, don't include this time slice in the average calculation
        if any(not np.isnan(f) for f in current_time_formants):
            formant_values.append(current_time_formants)
        
    if not formant_values:
        return {"F1_mean": 0.0, "F2_mean": 0.0, "F3_mean": 0.0}

    formant_values_np = np.array(formant_values, dtype=np.float32)
    
    # Ensure mean_values has at least n_formants elements, padding with 0.0 if not
    mean_values = np.nanmean(formant_values_np, axis=0)
    
    # Pad mean_values with 0.0 if fewer than n_formants were calculated
    f_means = [float(mean_values[i]) if i < len(mean_values) and not np.isnan(mean_values[i]) else 0.0 for i in range(n_formants)]

    return {
        "F1_mean": f_means[0],
        "F2_mean": f_means[1],
        "F3_mean": f_means[2],
    }


def extract_features(audio, sr):
    """Extract relevant audio features, including formants."""
    features = {}

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # Handle very short or silent audio chunks at the beginning
    if len(audio) < sr * 0.1 or np.all(np.abs(audio) < 1e-6): # If very short or near silent
        return {
            "pitch_mean": 0.0, "pitch_std": 0.0, "rms_mean": 0.0,
            "zcr_mean": 0.0, "tempo": 0.0, "mfccs": [0.0]*13,
            "spectral_centroid": 0.0, "spectral_bandwidth": 0.0, "chroma_mean": 0.0,
            "F1_mean": 0.0, "F2_mean": 0.0, "F3_mean": 0.0
        }

    # Pitch tracking - Now correctly unpacking all 3 return values
    f0, voiced_flag, voiced_probabilities = librosa.pyin(
        y=audio, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C5'),
        frame_length=2048, # Increase frame_length for more stable pitch tracking on longer chunks
        hop_length=512 # Smaller hop_length for more detail
    )
    pitches = f0[~np.isnan(f0)]
    
    features["pitch_mean"] = float(np.mean(pitches)) if len(pitches) > 0 else 0.0
    features["pitch_std"] = float(np.std(pitches)) if len(pitches) > 0 else 0.0

    features["rms_mean"] = float(np.mean(librosa.feature.rms(y=audio)))
    features["zcr_mean"] = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    
    try:
        # tempo estimation requires a minimum audio length
        if len(audio) >= sr * 2: # At least 2 seconds for tempo estimation
            features["tempo"] = float(librosa.beat.tempo(y=audio, sr=sr)[0])
        else:
            features["tempo"] = 0.0
    except Exception:
        features["tempo"] = 0.0

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features["mfccs"] = [float(val) for val in np.mean(mfccs, axis=1)]

    features["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    features["spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    features["chroma_mean"] = float(np.mean(librosa.feature.chroma_stft(y=audio, sr=sr)))

    formants = extract_formants(audio, sr)
    features.update(formants)

    return features

def analyze_speech_local_alerts(features):
    """Check various vocal characteristics and alert the user accordingly."""
    alerts = []
    
    # Use .get with default 0.0 for all feature accesses to prevent KeyError if a feature is missing
    pitch_mean = features.get("pitch_mean", 0.0)
    pitch_std = features.get("pitch_std", 0.0)
    rms_mean = features.get("rms_mean", 0.0)
    tempo = features.get("tempo", 0.0)
    zcr_mean = features.get("zcr_mean", 0.0)
    spectral_centroid = features.get("spectral_centroid", 0.0)
    spectral_bandwidth = features.get("spectral_bandwidth", 0.0)
    chroma_mean = features.get("chroma_mean", 0.0)

    # Alerts logic remains similar, but ensure thresholds are appropriate for 16kHz audio
    if pitch_mean > 0:
        if pitch_mean < 80: # Lowered for more robust detection of low pitch
            alerts.append("📢 Your pitch is quite low. Try speaking with more energy.")
        elif pitch_mean > 280: # Adjusted for potential higher pitch
            alerts.append("📢 Your pitch is unusually high. Consider toning it down.")

    if pitch_std < 15 and pitch_mean > 0: # Increased threshold for variation
        alerts.append("🎙️ Your voice lacks variation. Try adding some pitch dynamics.")

    if rms_mean < 0.015: # Slightly lower threshold for loudness
        alerts.append("🔈 You're speaking too softly. Increase your volume.")
    elif rms_mean > 0.18: # Slightly higher threshold for loudness
        alerts.append("🔊 You're too loud. Lower your volume slightly for comfort.")

    if tempo > 0:
        if tempo < 90:
            alerts.append("🐢 You're speaking too slowly. Try increasing your pace.")
        elif tempo > 160:
            alerts.append("⚡ You're speaking too quickly. Try slowing down.")

    if zcr_mean > 0.15:
        alerts.append("💨 High sibilance or sharpness detected. Speak more clearly.")

    if spectral_centroid > 0 and spectral_centroid < 1500:
        alerts.append("🔈 Your voice might sound a bit muffled. Try to speak more clearly or with more energy.")

    if spectral_bandwidth > 0 and spectral_bandwidth < 1800:
        alerts.append("📉 Your voice may sound dull or lack fullness — try increasing enunciation.")

    if chroma_mean > 0 and chroma_mean < 0.3:
        alerts.append("🎵 Add more pitch variation for a dynamic voice.")

    if features.get("F1_mean", 0.0) > 0 and features.get("F1_mean", 0.0) < 300:
        alerts.append("👄 Your F1 is low, which might affect open vowel pronunciation.")
    if features.get("F2_mean", 0.0) > 0 and features.get("F2_mean", 0.0) < 1000:
        alerts.append("👅 Your F2 is low, which might affect front vowel articulation.")
    if features.get("F3_mean", 0.0) > 0 and features.get("F3_mean", 0.0) < 2500:
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
                    encoding=stt.AudioEncoding.LINEAR16,
                    sample_rate_hertz=SAMPLING_RATE,
                    num_channels=1
                )
            )
            async with client.transcribe_stream(request) as stream:
                print("Soniox Transcription Task: Connected to Soniox API.")
                while True:
                    audio_chunk_np = await soniox_audio_queue.get() # Get from specific Soniox queue
                    if audio_chunk_np is None:
                        print("Soniox Transcription Task: Received stop signal. Closing stream.")
                        await stream.close()
                        break

                    # Convert float32 numpy array to 16-bit signed integer bytes for LINEAR16
                    # Scale to range -32768 to 32767 and convert to int16
                    audio_bytes = (audio_chunk_np * 32767).astype(np.int16).tobytes()
                    
# DIRECTLY send the raw PCM data
                    await stream.send(audio_chunk_np.tobytes())

                    response = await stream.recv()
                    if response.is_partial:
                        await soniox_result_queue.put({"type": "partial", "transcript": response.transcript, "words": []})
                    else:
                        full_transcript = response.transcript
                        words_info = [{"word": w.word, "start": w.start_time_seconds, "end": w.end_time_seconds} for w in response.words]
                        print(f"Soniox Final: {full_transcript}")
                        await soniox_result_queue.put({"type": "final", "transcript": full_transcript, "words": words_info})

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
            audio_data = await librosa_audio_queue.get() # Get from specific Librosa queue
            if audio_data is None:
                print("Librosa Analysis Task: Received stop signal. Processing remaining buffer.")
                if len(chunk_buffer) > 0:
                    features = extract_features(chunk_buffer, SAMPLING_RATE)
                    alerts = analyze_speech_local_alerts(features)
                    await analysis_result_queue.put({"chunk_id": chunk_counter, "features": features, "alerts": alerts})
                    print(f"Librosa Analysis Task: Processed final buffered chunk {chunk_counter}")
                break

            chunk_buffer = np.concatenate((chunk_buffer, audio_data))

            while len(chunk_buffer) >= BUFFER_SIZE_FRAMES:
                current_chunk = chunk_buffer[:BUFFER_SIZE_FRAMES]
                chunk_buffer = chunk_buffer[BUFFER_SIZE_FRAMES:]

                chunk_counter += 1
                
                features = extract_features(current_chunk, SAMPLING_RATE)
                alerts = analyze_speech_local_alerts(features)
                
                await analysis_result_queue.put({"chunk_id": chunk_counter, "features": features, "alerts": alerts})
                
                print(f"\n--- Analysis Chunk {chunk_counter} ---")
                print(f"🎧 PITCH: {features['pitch_mean']:.1f} Hz | ENERGY: {features['rms_mean']:.3f} | TEMPO: {features['tempo']:.1f} BPM")
                print(f"ZCR: {features['zcr_mean']:.3f} | F1: {features.get('F1_mean', 0.0):.1f} Hz | F2: {features.get('F2_mean', 0.0):.1f} Hz | F3: {features.get('F3_mean', 0.0):.1f} Hz ")
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
    """Records audio from microphone and puts chunks into separate queues."""
    print("Audio Recorder Task: Starting...")
    try:
        # Use a larger blocksize for sounddevice if overflows persist, e.g., BUFFER_SIZE_FRAMES * 2
        # Or, manage internal stream buffer if blocksize is set to 0.
        with sd.InputStream(samplerate=SAMPLING_RATE, channels=1, dtype='float32', blocksize=BUFFER_SIZE_FRAMES) as stream:
            print(f"🎤 Recording audio at {SAMPLING_RATE} Hz in chunks of {CHUNK_SIZE_SECONDS} seconds. Press Ctrl+C to stop.")
            while stream.active:
                try:
                    data, overflowed = stream.read(BUFFER_SIZE_FRAMES)
                    if overflowed:
                        print("Audio Recorder Task: Input stream overflowed!")
                    
                    # Put the same audio chunk into both specific queues
                    audio_flat = data.flatten()
                    await soniox_audio_queue.put(audio_flat)
                    await librosa_audio_queue.put(audio_flat)
                except sd.PortAudioError as e:
                    print(f"Audio Recorder Task: PortAudioError: {e}")
                    break
                except Exception as e:
                    print(f"Audio Recorder Task: Unexpected error during recording: {e}")
                    break
                
                await asyncio.sleep(0.01) # Yield control

    except KeyboardInterrupt:
        print("\nAudio Recorder Task: KeyboardInterrupt detected.")
    except Exception as e:
        print(f"Audio Recorder Task Error: {e}")
    finally:
        print("Audio Recorder Task: Exiting.")
        # Signal consumers to stop by putting None into their respective queues
        await soniox_audio_queue.put(None)
        await librosa_audio_queue.put(None)
        print("Audio Recorder Task: Sent stop signals to consumers.")


async def results_aggregator_task():
    """Aggregates results from Soniox and Librosa and potentially sends to Gemini for final analysis."""
    print("Results Aggregator Task: Starting...")
    all_librosa_results = []
    transcripts = [] # To store full transcripts
    
    producer_tasks = [
        asyncio.create_task(soniox_transcription_task()),
        asyncio.create_task(librosa_analysis_task()),
        asyncio.create_task(audio_recorder_task())
    ]

    try:
        while True:
            # Prepare tasks to wait for results from queues or for producers to finish
            wait_tasks = [
                asyncio.create_task(soniox_result_queue.get()),
                asyncio.create_task(analysis_result_queue.get()),
            ] + producer_tasks

            done, _pending = await asyncio.wait(
                wait_tasks,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=1.0 # Timeout to prevent indefinite blocking and check producer status
            )

            # Process completed tasks
            for task in done:
                try:
                    if task in producer_tasks:
                        # A producer task finished. Remove it from the active list.
                        producer_tasks.remove(task)
                        # print(f"Producer task {task.get_name()} finished.") # Debugging
                        continue

                    result = task.result()
                    if result is None:
                        continue # A queue might have returned None if its producer finished

                    if "type" in result and result["type"] in ["partial", "final"]:
                        # print(f"Received Soniox Result: {result['transcript']}")
                        if result["type"] == "final":
                            transcripts.append(result["transcript"])
                    elif "chunk_id" in result:
                        # print(f"Received Librosa Analysis for Chunk {result['chunk_id']}")
                        all_librosa_results.append(result["features"])

                except asyncio.CancelledError:
                    print("Results Aggregator Task: Sub-task was cancelled.")
                except Exception as e:
                    print(f"Results Aggregator Task: Error processing result from sub-task: {e}")
            
            # Global completion check: all producers are done AND all queues are empty.
            if all(t.done() for t in producer_tasks) and \
               soniox_audio_queue.empty() and librosa_audio_queue.empty() and \
               soniox_result_queue.empty() and analysis_result_queue.empty():
                print("Results Aggregator Task: All producer tasks completed and all queues are empty. Finalizing.")
                break # Exit the while True loop

            # If no tasks completed in the timeout, and producers are still running, continue.
            # If all producers are done, but queues aren't empty, keep processing queues.
            # If all producers are done AND queues are empty, the `break` above handles it.

    except asyncio.CancelledError:
        print("Results Aggregator Task: Cancelled.")
    except Exception as e:
        print(f"Results Aggregator Task Error: {e}")
    finally:
        print("Results Aggregator Task: Exiting.")
        # Ensure remaining producer tasks are cancelled (if any are still running)
        for task in producer_tasks:
            task.cancel()
        await asyncio.gather(*producer_tasks, return_exceptions=True)

        print(f"DEBUG: soniox_audio_queue size: {soniox_audio_queue.qsize()}")
        print(f"DEBUG: librosa_audio_queue size: {librosa_audio_queue.qsize()}")
        print(f"DEBUG: soniox_result_queue size: {soniox_result_queue.qsize()}")
        print(f"DEBUG: analysis_result_queue size: {analysis_result_queue.qsize()}")
        print(f"DEBUG: Remaining producer tasks: {[t.get_name() for t in producer_tasks if not t.done()]}")

        # Perform final Gemini analysis if data was collected
        if all_librosa_results:
            print("\n=== All recording and analysis finished. Sending final data to Gemini ===\n")
            await send_to_gemini_async(all_librosa_results)

async def send_to_gemini_async(all_results):
    """Asynchronously sends the accumulated results to Gemini for post-analysis."""
    prompt = "Here are voice metrics every 3 seconds:\n\n"
    for i, result in enumerate(all_results, 1):
        formatted_features = {}
        for k, v in result.items():
            if isinstance(v, (float, np.floating)):
                formatted_features[k] = f"{v:.2f}"
            elif isinstance(v, list): # For MFCCs which are a list of floats
                formatted_features[k] = [f"{val:.2f}" for val in v]
            else:
                formatted_features[k] = str(v)
        
        mfccs_str = ", ".join(formatted_features.get('mfccs', [])[:5])
        
        prompt += (
            f"Chunk {i}: Vocal Pitch Avg={formatted_features.get('pitch_mean', 'N/A')}, Pitch Var={formatted_features.get('pitch_std', 'N/A')}, "
            f"Loudness Avg={formatted_features.get('rms_mean', 'N/A')}, Clarity (ZCR Avg)={formatted_features.get('zcr_mean', 'N/A')}, "
            f"Pace (BPM)={formatted_features.get('tempo', 'N/A')}, "
            f"F1 Avg={formatted_features.get('F1_mean', 'N/A')}, F2 Avg={formatted_features.get('F2_mean', 'N/A')}, F3 Avg={formatted_features.get('F3_mean', 'N/A')}, "
            f"Vocal Brightness (Centroid)={formatted_features.get('spectral_centroid', 'N/A')}, "
            f"Vocal Fullness (Bandwidth)={formatted_features.get('spectral_bandwidth', 'N/A')}, "
            f"Vocal Expressiveness (Chroma)={formatted_features.get('chroma_mean', 'N/A')}\n"
        )

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
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = await asyncio.to_thread(lambda: model.generate_content(contents=prompt))
        print("\n=== Gemini's Analysis ===\n", response.text)
        return response.text
    except Exception as e:
        print(f"Error sending to Gemini: {e}")
        return f"Error: Could not get analysis from Gemini: {e}"


async def main():
    print("Starting Public Speaking Trainer Backend...")
    aggregator_task = asyncio.create_task(results_aggregator_task())

    try:
        await aggregator_task
    except KeyboardInterrupt:
        print("\nMain: KeyboardInterrupt detected. Shutting down.")
    except Exception as e:
        print(f"Main: An error occurred: {e}")
    finally:
        print("Main: All tasks are shutting down.")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        print("Main: All tasks stopped.")

if __name__ == "__main__":
    os.environ["SONIOX_API_KEY"] = "sonikey"
    client = genai.Client(api_key="gemini_api_key")    
    asyncio.run(main())