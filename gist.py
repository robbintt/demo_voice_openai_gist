''' GPT4 + Eleven Labs Modified Gist

Original Gist: https://gist.github.com/NN1985/a0712821269259061177c6abb08e8e0a
GPT-4 repaired 2023/12/29
'''
from openai import OpenAI
import elevenlabs
import os

# set from env
elevenlabs.set_api_key(os.environ['ELEVEN_LABS_API_KEY'])
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# gpt4 generated and debugged its own code here
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
text = "A three sentence story about space travel maintenance on a generation ship."


#text_stream = write(text)
#text_stream = ("").join(write(text))
#print(str(text_stream))

# Convert the text stream to an audio stream
# eleven_multilingual_v1 best first (female): Matilda, Glinda (fast), Dorothy, Emily (slow), Gigi (fast), Grace
# eleven_multilingual_v1 best first (male): Jessie (fast)
# eleven_multilingual_v2 best first (female): Sarah

# generate the stream, this doesn't actually work according to the elevenlabs API, we can only send a string
text_stream = write(text)

# lets generate audio as it comes in?

#
buffer = []
fragment = []
fulltext = []
for text in write(text):
    buffer.append(text)
    if len(buffer) < 5:
        continue
    else:
        fragment.extend(buffer)
        fulltext.extend(buffer)
        buffer.clear()
        audio_stream = elevenlabs.generate(
            text = "".join(fragment),
            voice="Sarah",
            model="eleven_multilingual_v2",
            stream=True,
            latency=4,
        )
        # Stream the audio
        # requires mpv for `stream`
        output = elevenlabs.stream(audio_stream)
        fragment.clear()

print(fulltext)


'''
# save the output for testing or later use
audio_file = elevenlabs.generate(
    text = text_stream,
    voice="Sarah",
    model="eleven_multilingual_v2",
)
elevenlabs.save(audio_file, "speech.mp3")
'''

