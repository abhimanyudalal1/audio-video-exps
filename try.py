import asyncio
from dotenv import load_dotenv
import os
import traceback

load_dotenv()
API_KEY = os.getenv("DEEPGRAM_API_KEY")

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

async def main():
    try:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(API_KEY, config)
        dg_connection = deepgram.listen.asyncwebsocket.v("1")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) > 0:
                print(sentence)

        async def on_utterance_end(self, utterance_end, **kwargs):
            print("utterance detected")

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)

        options: LiveOptions = LiveOptions(
            model="nova-3",
            language="en-US",
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            interim_results=True,
            filler_words=True,
            utterance_end_ms="5000",  # 5 second gap triggers UtteranceEnd
        )

        if await dg_connection.start(options) is False:
            print("Failed to connect to Deepgram")
            return

        microphone = Microphone(dg_connection.send)
        microphone.start()

        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            microphone.finish()
            await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        traceback.print_exc()

asyncio.run(main())