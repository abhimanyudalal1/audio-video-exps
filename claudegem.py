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
SONIOX_API_KEY = os.environ.get("SONIOX_API_KEY", "0c8a505b0b7b3850353998ada7b3c95e78ba39bd4beed9152bb14b41e0740fa1")
if SONIOX_API_KEY == "YOUR_SONIOX_API_KEY":
    print("WARNING: Soniox API Key is not set. Please set the SONIOX_API_KEY environment variable or replace 'YOUR_SONIOX_API_KEY' with your actual key.")

# Audio settings
SAMPLING_RATE = 16000  # Soniox recommends 16kHz for best results
CHUNK_SIZE_SECONDS = 2.5 # Reduced chunk size for potentially lower latency and better responsiveness
BUFFER_SIZE_FRAMES = int(SAMPLING_RATE * CHUNK_SIZE_SECONDS) # Block size for sounddevice

# Queues for inter-task communication
# Separate queues for each consumer of audio data
soniox_audio_queue = asyncio.Queue(maxsize=150)  # Add maxsize to prevent memory issues
librosa_audio_queue = asyncio.Queue(maxsize=150)

# Queues for results from processing tasks
soniox_result_queue = asyncio.Queue(maxsize=150)
analysis_result_queue = asyncio.Queue(maxsize=150)

# Global shutdown flag
shutdown_flag = asyncio.Event()

# --- Your existing librosa/parselmouth functions (with robustness improvements) ---

def extract_formants(audio, sr, time_step=0.01, max_formant=5500, n_formants=3):
    """Extract average F1, F2, and F3 from an audio chunk."""
    try:
        if len(audio) == 0:
            return {"F1_mean": 0.0, "F2_mean": 0.0, "F3_mean": 0.0}

        audio_float64 = audio.astype(np.float64)
        snd = parselmouth.Sound(audio_float64, sr)
        
        if snd.get_total_duration() == 0:
            return {"F1_mean": 0.0, "F2_mean": 0.0, "F3_mean": 0.0}

        # Add timeout protection for formant analysis
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
                current_time_formants.append(val if not np.isnan(val) else np.nan)
            
            if any(not np.isnan(f) for f in current_time_formants):
                formant_values.append(current_time_formants)
            
        if not formant_values:
            return {"F1_mean": 0.0, "F2_mean": 0.0, "F3_mean": 0.0}

        formant_values_np = np.array(formant_values, dtype=np.float32)
        mean_values = np.nanmean(formant_values_np, axis=0)
        
        f_means = [float(mean_values[i]) if i < len(mean_values) and not np.isnan(mean_values[i]) else 0.0 for i in range(n_formants)]

        return {
            "F1_mean": f_means[0],
            "F2_mean": f_means[1],
            "F3_mean": f_means[2],
        }
    except Exception as e:
        print(f"Error in extract_formants: {e}")
        return {"F1_mean": 0.0, "F2_mean": 0.0, "F3_mean": 0.0}


def extract_features(audio, sr):
    """Extract relevant audio features, including formants."""
    try:
        features = {}

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32)

        # Handle very short or silent audio chunks at the beginning
        if len(audio) < sr * 0.1 or np.all(np.abs(audio) < 1e-6):
            return {
                "pitch_mean": 0.0, "pitch_std": 0.0, "rms_mean": 0.0,
                "zcr_mean": 0.0, "tempo": 0.0, "mfccs": [0.0]*13,
                "spectral_centroid": 0.0, "spectral_bandwidth": 0.0, "chroma_mean": 0.0,
                "F1_mean": 0.0, "F2_mean": 0.0, "F3_mean": 0.0
            }

        # Pitch tracking with error handling
        try:
            f0, voiced_flag, voiced_probabilities = librosa.pyin(
                y=audio, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C5'),
                frame_length=2048,
                hop_length=512
            )
            pitches = f0[~np.isnan(f0)]
            
            features["pitch_mean"] = float(np.mean(pitches)) if len(pitches) > 0 else 0.0
            features["pitch_std"] = float(np.std(pitches)) if len(pitches) > 0 else 0.0
        except Exception as e:
            print(f"Error in pitch tracking: {e}")
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0

        # Other features with error handling
        try:
            features["rms_mean"] = float(np.mean(librosa.feature.rms(y=audio)))
        except Exception:
            features["rms_mean"] = 0.0

        try:
            features["zcr_mean"] = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
        except Exception:
            features["zcr_mean"] = 0.0
        
        try:
            if len(audio) >= sr * 2:
                tempo_result = librosa.beat.tempo(y=audio, sr=sr)
                features["tempo"] = float(tempo_result[0]) if len(tempo_result) > 0 else 0.0
            else:
                features["tempo"] = 0.0
        except Exception as e:
            print(f"Error in tempo estimation: {e}")
            features["tempo"] = 0.0

        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features["mfccs"] = [float(val) for val in np.mean(mfccs, axis=1)]
        except Exception:
            features["mfccs"] = [0.0] * 13

        try:
            features["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        except Exception:
            features["spectral_centroid"] = 0.0

        try:
            features["spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
        except Exception:
            features["spectral_bandwidth"] = 0.0

        try:
            features["chroma_mean"] = float(np.mean(librosa.feature.chroma_stft(y=audio, sr=sr)))
        except Exception:
            features["chroma_mean"] = 0.0

        # Formant extraction with timeout protection
        try:
            formants = extract_formants(audio, sr)
            features.update(formants)
        except Exception as e:
            print(f"Error in formant extraction: {e}")
            features.update({"F1_mean": 0.0, "F2_mean": 0.0, "F3_mean": 0.0})

        return features
    
    except Exception as e:
        print(f"Critical error in extract_features: {e}")
        return {
            "pitch_mean": 0.0, "pitch_std": 0.0, "rms_mean": 0.0,
            "zcr_mean": 0.0, "tempo": 0.0, "mfccs": [0.0]*13,
            "spectral_centroid": 0.0, "spectral_bandwidth": 0.0, "chroma_mean": 0.0,
            "F1_mean": 0.0, "F2_mean": 0.0, "F3_mean": 0.0
        }

def analyze_speech_local_alerts(features):
    """Check various vocal characteristics and alert the user accordingly."""
    alerts = []
    
    pitch_mean = features.get("pitch_mean", 0.0)
    pitch_std = features.get("pitch_std", 0.0)
    rms_mean = features.get("rms_mean", 0.0)
    tempo = features.get("tempo", 0.0)
    zcr_mean = features.get("zcr_mean", 0.0)
    spectral_centroid = features.get("spectral_centroid", 0.0)
    spectral_bandwidth = features.get("spectral_bandwidth", 0.0)
    chroma_mean = features.get("chroma_mean", 0.0)

    if pitch_mean > 0:
        if pitch_mean < 80:
            alerts.append("📢 Your pitch is quite low. Try speaking with more energy.")
        elif pitch_mean > 280:
            alerts.append("📢 Your pitch is unusually high. Consider toning it down.")

    if pitch_std < 15 and pitch_mean > 0:
        alerts.append("🎙️ Your voice lacks variation. Try adding some pitch dynamics.")

    if rms_mean < 0.015:
        alerts.append("🔈 You're speaking too softly. Increase your volume.")
    elif rms_mean > 0.18:
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
                while not shutdown_flag.is_set():
                    try:
                        audio_chunk_np = await asyncio.wait_for(
                            soniox_audio_queue.get(), timeout=1.0
                        )
                        if audio_chunk_np is None:
                            print("Soniox Transcription Task: Received stop signal. Closing stream.")
                            await stream.close()
                            break

                        # Convert float32 numpy array to 16-bit signed integer bytes for LINEAR16
                        # Convert float32 numpy array to 16-bit signed integer bytes for LINEAR16
                        audio_int16 = (audio_chunk_np * 32767).astype(np.int16)
                        await stream.send(stt.AudioChunk(audio=audio_int16.tobytes()))                        
                        response = await asyncio.wait_for(stream.recv(), timeout=0.5)
                        if response.is_partial:
                            await soniox_result_queue.put({"type": "partial", "transcript": response.transcript, "words": []})
                        else:
                            full_transcript = response.transcript
                            words_info = [{"word": w.word, "start": w.start_time_seconds, "end": w.end_time_seconds} for w in response.words]
                            print(f"Soniox Final: {full_transcript}")
                            print(f"DEBUG: Added transcript to list: {full_transcript}")
                            await soniox_result_queue.put({"type": "final", "transcript": full_transcript, "words": words_info})

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"Soniox Transcription Task Inner Error: {e}")
                        break

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
        while not shutdown_flag.is_set():
            try:
                # Add timeout to prevent hanging
                audio_data = await asyncio.wait_for(
                    librosa_audio_queue.get(), timeout=1.0
                )
                
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
                    
                    print(f"Processing chunk {chunk_counter}...")
                    
                    # Add timeout protection for feature extraction
                    try:
                        features = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, extract_features, current_chunk, SAMPLING_RATE
                            ), timeout=5.0
                        )
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
                        
                    except asyncio.TimeoutError:
                        print(f"Librosa Analysis Task: Timeout processing chunk {chunk_counter}, skipping...")
                        continue
                    except Exception as e:
                        print(f"Librosa Analysis Task: Error processing chunk {chunk_counter}: {e}")
                        continue

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Librosa Analysis Task: Error getting audio data: {e}")
                continue

    except Exception as e:
        print(f"Librosa Analysis Task Error: {e}")
    finally:
        print("Librosa Analysis Task: Exiting.")

async def audio_recorder_task():
    """Records audio from microphone and puts chunks into separate queues."""
    print("Audio Recorder Task: Starting...")
    try:
        with sd.InputStream(samplerate=SAMPLING_RATE, channels=1, dtype='float32', blocksize=BUFFER_SIZE_FRAMES) as stream:
            print(f"🎤 Recording audio at {SAMPLING_RATE} Hz in chunks of {CHUNK_SIZE_SECONDS} seconds. Press Ctrl+C to stop.")
            while not shutdown_flag.is_set() and stream.active:
                try:
                    data, overflowed = stream.read(BUFFER_SIZE_FRAMES)
                    if overflowed:
                        print("Audio Recorder Task: Input stream overflowed!")
                    
                    audio_flat = data.flatten()
                    
                    # Use try/except for queue puts to handle full queues
                    try:
                        soniox_audio_queue.put_nowait(audio_flat)
                    except asyncio.QueueFull:
                        print("Soniox audio queue full, skipping chunk")
                    
                    try:
                        librosa_audio_queue.put_nowait(audio_flat)
                    except asyncio.QueueFull:
                        print("Librosa audio queue full, skipping chunk")
                        
                except sd.PortAudioError as e:
                    print(f"Audio Recorder Task: PortAudioError: {e}")
                    break
                except Exception as e:
                    print(f"Audio Recorder Task: Unexpected error during recording: {e}")
                    break
                
                await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\nAudio Recorder Task: KeyboardInterrupt detected.")
    except Exception as e:
        print(f"Audio Recorder Task Error: {e}")
    finally:
        print("Audio Recorder Task: Exiting.")
        shutdown_flag.set()
        # Signal consumers to stop
        try:
            await soniox_audio_queue.put(None)
            await librosa_audio_queue.put(None)
        except:
            pass
        print("Audio Recorder Task: Sent stop signals to consumers.")


async def results_aggregator_task():
    """Aggregates results from Soniox and Librosa and potentially sends to Gemini for final analysis."""
    print("Results Aggregator Task: Starting...")
    all_librosa_results = []
    transcripts = []
    
    producer_tasks = [
        asyncio.create_task(soniox_transcription_task()),
        asyncio.create_task(librosa_analysis_task()),
        asyncio.create_task(audio_recorder_task())
    ]

    try:
        while not shutdown_flag.is_set():
            # Check if all producer tasks are done
            if all(t.done() for t in producer_tasks):
                print("Results Aggregator Task: All producer tasks completed.")
                break
                
            # Process results with timeout
            try:
                # Check for Soniox results
                try:
                    soniox_result = soniox_result_queue.get_nowait()
                    if "type" in soniox_result and soniox_result["type"] == "final":
                        transcripts.append(soniox_result["transcript"])
                except asyncio.QueueEmpty:
                    pass
                
                # Check for analysis results
                try:
                    analysis_result = analysis_result_queue.get_nowait()
                    if "chunk_id" in analysis_result:
                        all_librosa_results.append(analysis_result["features"])
                except asyncio.QueueEmpty:
                    pass
                    
            except Exception as e:
                print(f"Results Aggregator Task: Error processing results: {e}")
            
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

    except asyncio.CancelledError:
        print("Results Aggregator Task: Cancelled.")
    except Exception as e:
        print(f"Results Aggregator Task Error: {e}")
    finally:
        print("Results Aggregator Task: Exiting.")
        shutdown_flag.set()
        
        # Cancel remaining producer tasks
        for task in producer_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*producer_tasks, return_exceptions=True)

        if transcripts:
            print("\n=== All Transcriptions ===")
        for i, transcript in enumerate(transcripts, 1):
            print(f"{i}. {transcript}")
            print("=" * 50)

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
            elif isinstance(v, list):
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
        response = await asyncio.to_thread(
        lambda: client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    )
    
        print("\n=== Gemini's Analysis ===\n", response.text)
        return response.text
    except Exception as e:
        print(f"Error sending to Gemini: {e}")
        return f"Error: Could not get analysis from Gemini: {e}"


async def main():
    print("Starting Public Speaking Trainer Backend...")
    
    try:
        aggregator_task = asyncio.create_task(results_aggregator_task())
        await aggregator_task
    except KeyboardInterrupt:
        print("\nMain: KeyboardInterrupt detected. Shutting down.")
        shutdown_flag.set()
    except Exception as e:
        print(f"Main: An error occurred: {e}")
        shutdown_flag.set()
    finally:
        print("Main: All tasks are shutting down.")
        shutdown_flag.set()
        
        # Cancel all remaining tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        print("Main: All tasks stopped.")

if __name__ == "__main__":
    #os.environ["SONIOX_API_KEY"] = "0c8a505b0b7b3850353998ada7b3c95e78ba39bd4beed9152bb14b41e0740fa1"
    client = genai.Client(api_key="AIzaSyCqq5SRE6mU9Fh8jpduWGXvdDKgzl0KZIY")    
    asyncio.run(main())