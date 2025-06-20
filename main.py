from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import aiofiles
import uuid
import requests
import json
import time
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import wave
import struct

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def get_index():
    return FileResponse("index.html")

ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")

base_url = "https://api.assemblyai.com"
headers = {"authorization": ASSEMBLY_API_KEY}

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"WebSocket connection accepted")
    
    audio_filename = f"recorded_audio_{uuid.uuid4().hex}.mp3"

    audio_chunks = []
    
    try:
        print("Waiting for audio data...")
        
        # Receive audio data until we get the END signal
        async def receive_audio_data():
            nonlocal audio_chunks
            try:
                while True:
                    # Try to receive both binary and text data
                    try:
                        # First try to receive text (for END signal)
                        data = await websocket.receive_text()
                        if data == "END_OF_AUDIO":
                            print("Received END_OF_AUDIO signal")
                            break
                        else:
                            print(f"Received unexpected text: {data}")
                    except:
                        # If it's not text, try to receive bytes
                        try:
                            data = await websocket.receive_bytes()
                            if data and len(data) > 0:
                                audio_chunks.append(data)
                                print(f"Received audio chunk: {len(data)} bytes (total chunks: {len(audio_chunks)})")
                            else:
                                print("Received empty data chunk")
                        except Exception as e:
                            print(f"Error receiving data: {e}")
                            break
                            
            except WebSocketDisconnect as e:
                print(f"WebSocket disconnected: {e.code}")
            except Exception as e:
                print(f"Error in receive_audio_data: {e}")
        
        # Wait for audio data with timeout
        try:
            await asyncio.wait_for(receive_audio_data(), timeout=150)  # 2.5 minutes timeout
        except asyncio.TimeoutError:
            print("Audio reception timed out")
            await websocket.send_text("Recording session timed out. Please try again.")
            return
        
        print(f"Audio reception finished. Total chunks received: {len(audio_chunks)}")
        
        if not audio_chunks:
            print("No audio data received")
            await websocket.send_text("No audio data received. Please check your microphone and try again.")
            return
        
        # Combine all audio chunks
        print("Combining audio chunks...")
        audio_data = b''.join(audio_chunks)
        print(f"Total audio data size: {len(audio_data)} bytes")
        
        if len(audio_data) == 0:
            print("Combined audio data is empty")
            await websocket.send_text("No audio data captured. Please ensure your microphone is working.")
            return
        
        # Create a proper WAV file from the raw audio data
        audio_path = await create_mp3_file(audio_data, audio_filename)
        if not audio_path:
            await websocket.send_text("Failed to save audio file.")
            return
        
        print(f"WAV file created: {audio_filename}")
        
        # Process the audio
        await process_audio(websocket, audio_filename)
        
    except Exception as e:
        print(f"Unexpected error in websocket_endpoint: {e}")
        import traceback
        traceback.print_exc()
        try:
            if websocket.client_state.name != 'DISCONNECTED':
                await websocket.send_text(f"Server error: {str(e)}")
        except:
            pass
    finally:
        # Clean up
        if os.path.exists(audio_filename):
            try:
                os.remove(audio_filename)
                print(f"Cleaned up file: {audio_filename}")
            except Exception as e:
                print(f"Error cleaning up file: {e}")

import os
from pydub import AudioSegment
import io

async def create_mp3_file(raw_audio_data: bytes, filename: str) -> str:
    try:
        # Create directory if it doesn't exist
        save_dir = "recordings"
        os.makedirs(save_dir, exist_ok=True)

        # Temporary WebM file to decode
        temp_webm_path = os.path.join(save_dir, "temp.webm")
        with open(temp_webm_path, "wb") as f:
            f.write(raw_audio_data)

        # Decode from WebM to AudioSegment
        audio_segment = AudioSegment.from_file(temp_webm_path, format="webm")

        # Save as MP3
        full_mp3_path = os.path.join(save_dir, filename)
        audio_segment.export(full_mp3_path, format="mp3")
        print(f"MP3 file created successfully: {full_mp3_path}")

        # Clean up temp file
        os.remove(temp_webm_path)

        return full_mp3_path

    except Exception as e:
        print(f"Error creating MP3 file: {e}")
        return None

async def process_audio(websocket: WebSocket, audio_filename: str):
    try:
        print("Starting transcription process...")
        
        # Check if API key is available
        if not ASSEMBLY_API_KEY or ASSEMBLY_API_KEY == "your_api_key_here":
            await websocket.send_text("AssemblyAI API key not configured. Please set ASSEMBLY_API_KEY environment variable.")
            return
        
        # Upload the audio to AssemblyAI
        print("Uploading audio to AssemblyAI...")
        with open(audio_filename, 'rb') as f:
            response = requests.post(base_url + "/v2/upload", headers=headers, data=f, timeout=60)
            
        if response.status_code != 200:
            error_msg = f"Upload failed: {response.status_code} - {response.text}"
            print(error_msg)
            await websocket.send_text(error_msg)
            return
            
        upload_response = response.json()
        upload_url = upload_response["upload_url"]
        print(f"Audio uploaded successfully: {upload_url}")

        # Request transcription
        print("Requesting transcription...")
        data = {
            "audio_url": upload_url,
            "speech_model": "best"
        }
        response = requests.post(base_url + "/v2/transcript", json=data, headers=headers, timeout=30)
        
        if response.status_code != 200:
            error_msg = f"Transcription request failed: {response.status_code} - {response.text}"
            print(error_msg)
            await websocket.send_text(error_msg)
            return
            
        transcript_response = response.json()
        transcript_id = transcript_response['id']
        polling_endpoint = base_url + "/v2/transcript/" + transcript_id
        print(f"Transcription requested. ID: {transcript_id}")

        # Poll until complete
        print("Polling for transcription completion...")
        max_polls = 60
        poll_count = 0
        
        while poll_count < max_polls:
            try:
                result = requests.get(polling_endpoint, headers=headers, timeout=30).json()
                print(f"Transcription status: {result['status']} (poll {poll_count + 1})")
                
                if result['status'] == 'completed':
                    transcript = result.get('text', '')
                    print(f"Transcription completed: {transcript[:100]}...")
                    break
                elif result['status'] == 'error':
                    error_msg = "Transcription failed: " + result.get('error', 'Unknown error')
                    print(error_msg)
                    await websocket.send_text(error_msg)
                    return
                else:
                    await asyncio.sleep(3)
                    poll_count += 1
                    
            except requests.RequestException as e:
                print(f"Polling request failed: {e}")
                await asyncio.sleep(5)
                poll_count += 1
        
        if poll_count >= max_polls:
            await websocket.send_text("Transcription timed out. Please try again.")
            return
        
        if transcript:
            print("-------------------------------------------------------------")
            print("THE TRANSCRIPT IS:")
            print(transcript)
            print("-------------------------------------------------------------")
            # Generate feedback
            print("Generating feedback...")
            feedback = generate_feedback(transcript)
            
            # Check if websocket is still connected before sending
            if websocket.client_state.name != 'DISCONNECTED':
                await websocket.send_text(feedback)
                print("Feedback sent successfully")
            else:
                print("WebSocket disconnected, cannot send feedback")
        else:
            if websocket.client_state.name != 'DISCONNECTED':
                await websocket.send_text("Transcription completed but no text was detected. Please speak more clearly and try again.")
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error: {str(e)}"
        print(error_msg)
        if websocket.client_state.name != 'DISCONNECTED':
            await websocket.send_text(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid response from API: {str(e)}"
        print(error_msg)
        if websocket.client_state.name != 'DISCONNECTED':
            await websocket.send_text(error_msg)
    except Exception as e:
        error_msg = f"Error processing audio: {str(e)}"
        print(error_msg)
        if websocket.client_state.name != 'DISCONNECTED':
            await websocket.send_text(error_msg)

def generate_feedback(transcript: str) -> str:
    """Generate public speaking feedback based on the transcript"""
    
    # Basic analysis
    words = transcript.split()
    word_count = len(words)
    sentence_count = len([s for s in transcript.split('.') if s.strip()])
    
    # Filler words analysis
    filler_words = ["um", "uh", "like", "you know", "so", "actually", "basically", "literally"]
    filler_count = 0
    filler_details = {}
    
    transcript_lower = transcript.lower()
    for filler in filler_words:
        count = transcript_lower.count(filler)
        if count > 0:
            filler_count += count
            filler_details[filler] = count
    
    # Calculate speaking rate (words per minute, assuming 2 minutes)
    speaking_rate = word_count / 2
    
    # Generate feedback
    feedback = f"SPEECH ANALYSIS RESULTS\n"
    feedback += f"=" * 50 + "\n\n"
    feedback += f"📝 TRANSCRIPTION:\n\"{transcript}\"\n\n"
    feedback += f"📊 STATISTICS:\n"
    feedback += f"• Word count: {word_count}\n"
    feedback += f"• Estimated sentences: {sentence_count}\n"
    feedback += f"• Speaking rate: ~{speaking_rate:.1f} words per minute\n"
    feedback += f"• Filler words used: {filler_count}\n"
    
    if filler_details:
        feedback += f"• Filler word breakdown: {', '.join([f'{word}({count})' for word, count in filler_details.items()])}\n"
    
    feedback += f"\n💡 FEEDBACK:\n"
    
    # Speaking rate feedback
    if speaking_rate < 120:
        feedback += "• Consider speaking a bit faster to maintain audience engagement.\n"
    elif speaking_rate > 180:
        feedback += "• Try slowing down slightly to ensure clarity and comprehension.\n"
    else:
        feedback += "• Good speaking pace! You're maintaining an appropriate speed.\n"
    
    # Filler words feedback
    filler_percentage = (filler_count / word_count * 100) if word_count > 0 else 0
    if filler_percentage > 5:
        feedback += f"• Try to reduce filler words ({filler_percentage:.1f}% of your speech). Practice pausing instead of using fillers.\n"
    elif filler_percentage < 2:
        feedback += "• Excellent! You kept filler words to a minimum.\n"
    else:
        feedback += "• Good control of filler words. Keep practicing to minimize them further.\n"
    
    # Word count feedback
    if word_count < 200:
        feedback += "• Consider expanding your content to fill the time more effectively.\n"
    elif word_count > 400:
        feedback += "• Great content volume! Make sure you're allowing time for emphasis and pauses.\n"
    else:
        feedback += "• Good content length for a 2-minute speech.\n"
    
    return feedback

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")