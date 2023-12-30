from openai import OpenAI
import elevenlabs
import os

# set from env
elevenlabs.set_api_key(os.environ['ELEVEN_LABS_API_KEY'])
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

'''
def write(prompt: str):
    for chunk in client.chat.completions.create(model="gpt-3.5-turbo-0301",
    messages=[{"role": "user", "content": prompt}],
    stream=True):
        # Extract the content from the chunk if available
        if (text_chunk := chunk["choices"][0]["delta"].get("content")) is not None:
            yield text_chunk
'''

'''
# gpt4 first try
def write(prompt: str):
    for chunk in client.chat.completions.create(model="gpt-3.5-turbo-0301",
    messages=[{"role": "user", "content": prompt}],
    stream=True):
        # Check if 'delta' exists and contains 'content'
        print(chunk)
        if hasattr(chunk.choices[0], 'delta') and 'content' in chunk.choices[0].delta:
            text_chunk = chunk.choices[0].delta['content']
            print(text_chunk)
            yield text_chunk
'''

def write(prompt: str):
    for chunk in client.chat.completions.create(model="gpt-3.5-turbo-0301",
    messages=[{"role": "user", "content": prompt}],
    stream=True):
        # Check if 'delta' exists and contains 'content'
        if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content is not None:
            text_chunk = chunk.choices[0].delta.content
            yield text_chunk
        # Stop the generator if the finish reason is 'stop'
        if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason == 'stop':
            break


# Generate a text stream
text = "A five sentence story about The Fonz and his cool adventures."


#text_stream = write(text)
text_stream = ("").join(write(text))
print(str(text_stream))

# Convert the text stream to an audio stream
#    stream=True,
#    latency=4,
#    text=text_stream,
audio_stream = elevenlabs.generate(
    text = text_stream,
    voice="Sarah",
    model="eleven_multilingual_v2",
)

# Stream the audio
# requires mpv
#output = elevenlabs.stream(audio_stream)

elevenlabs.save(audio_stream, "speech.mp3")
