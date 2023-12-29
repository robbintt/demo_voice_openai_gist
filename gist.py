import openai
import elevenlabs
import os

# set from env
openai.api_key = os.environ['OPENAI_API_KEY']
elevenlabs.set_api_key(os.environ['ELEVEN_LABS_API_KEY'])

def write(prompt: str):
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        # Extract the content from the chunk if available
        if (text_chunk := chunk["choices"][0]["delta"].get("content")) is not None:
            yield text_chunk

# Generate a text stream
text = "A five sentence story about The Fonz and his cool adventures."
text_stream = write(text)

# Convert the text stream to an audio stream
#    stream=True,
#    latency=4,
#    text=text_stream,
audio_stream = elevenlabs.generate(
    text = text,
    voice="Sarah",
    model="eleven_multilingual_v2",
)

# Stream the audio
# requires mpv
#output = elevenlabs.stream(audio_stream)

elevenlabs.save(audio_stream, "speech.mp3")
