""" Use GPT-4 with Tortoise TTS.

TODO:
* This script needs run from Tortoise project's root directory because tortoise tts python module isn't installed yet.

Conda Environment:
    This is generated from tortoise-tts setup and then `pip install openai`
    conda activate tortoise
"""
import os
import re
import time
from datetime import datetime

from openai import OpenAI

import torch
import torchaudio

from api import TextToSpeech, MODELS_DIR
from utils.audio import load_voices

# carve off args
cvvp_amount = .0 # .0 is disabled, this is currently broken: https://github.com/neonbjb/tortoise-tts/issues/692
seed = None
model_dir = MODELS_DIR
output_path = "results/"
kv_cache = True
half = True # float 16 / half precision (True for half / faster)
produce_debug_state = True
preset = "ultra_fast" # high_quality, standard, fast, ultra_fast
voice = ["emma"]
use_deepspeed = True
# check use_deepspeed
if torch.backends.mps.is_available():
    print("Warning: Deepspeed not available, resetting to deepspeed=False.")
    use_deepspeed = False

def generate_filename(filename_base="audio"):
    ''' Generate a base filename with timestamp
    '''
    extension = "wav" # might support multiple formats, if so move to args
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename_base:
        pattern = r'^[a-zA-Z0-9_.-]+$'
        if not re.match(pattern, filename_base):
            filename_base = "audio"
    filename = f"{filename_base}_{timestamp}.{extension}"
    return filename

def save_audio(gen, dbg_state):
    '''
    '''
    print("Saving audio...")
    if isinstance(gen, list):
        for j, g in enumerate(gen):
            print("Debug: Using generator path.")
            torchaudio.save(
                os.path.join(output_path, generate_filename()),
                g.squeeze(0).cpu(),
                24000,
            )
    else:
        print("Debug: Using non-generator path")
        torchaudio.save(
            os.path.join(output_path, generate_filename()),
            gen.squeeze(0).cpu(),
            24000,
        )

    if produce_debug_state:
        print("Warning: Saving debug states.")
        os.makedirs("debug_states", exist_ok=True)
        torch.save(dbg_state, "debug_states/do_tts_debug_{}.pth".format(datetime.now().strftime("%Y%m%d_%H%M%S")))

    return

def generate_audio(content):
    ''' Migrate the original cli args into module layout

    Before upgrading this:
        - check if tortoise-fast is worth it, and has better tooling
        - see if it is mappable to "tortoise fast"
    '''
    if not content:
        print("No content provided.")
        return
    if not isinstance(content, str):
        print("Content is not a string")
        return

    print("Loading voices")
    voice_samples, conditioning_latents = load_voices(voice)

    # tts, rather than tts_with_preset, has way more parameters here:
    # tts_with_preset wraps tts
    # token count allowed is actually too high at 400, and I would like a fast tokenizer available to check this: https://github.com/neonbjb/tortoise-tts/blob/1e061bc6752f05bccb59748c8bd7c7fc85d54988/tortoise/api.py#L381C16-L381C20
    # https://github.com/neonbjb/tortoise-tts/blob/1e061bc6752f05bccb59748c8bd7c7fc85d54988/tortoise/api.py#L333
    print("tts_with_preset, includes generating audio apparently...")
    gen, dbg_state = tts.tts_with_preset(
        content,
        k=1,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        preset=preset,
        use_deterministic_seed=seed,
        return_deterministic_state=True,
        cvvp_amount=cvvp_amount,
    )

    save_audio(gen, dbg_state)

    return

def manual_input_loop():
    while True:
        try:
            user_input = input("Enter text to tts (Ctrl+C to quit):")
            #print(f"You entered: {user_input}")
            # needs split to appropriate segments based on token count and sentence breaks, for sufficient context for generation.
            start_time = time.time()
            generate_audio(user_input)
            end_time = time.time()
            print("Time elapsed: {}".format(end_time - start_time))
        except KeyboardInterrupt:
            print("\nExiting gracefully")
            break
    return

def collect_audio_segment(gpt4_generator):
    ''' Collect enough audio from the buffer for context but not too many tokens for the model.

    We are not concerned about any other maxima, but we want as much as possible.
    Another constraint would be the wait time per segment,
    but it is a tradeoff with context/quality so discarding for now.
    '''


# gpt4 generated and debugged its own code here
def gpt4_generate(prompt: str):
    ''' yield each piece of text from GPT-4
    '''
    for chunk in openai_client.chat.completions.create(model="gpt-3.5-turbo-0301",
    messages=[{"role": "user", "content": prompt}],
    stream=True):
        # Check if 'delta' exists and contains 'content'
        if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content is not None:
            text_chunk = chunk.choices[0].delta.content
            yield text_chunk
        # Stop the generator if the finish reason is 'stop'
        if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason == 'stop':
            break

if __name__ == "__main__":
    # need to check the length, even split on sentences and then by length at some level of the stack.
    #content = "She initiates a desperate override sequence, sweat beading on her brow, as the ship's hull groans under the strain of an unexpected course correction towards a black hole."
    #content = "As the ship hurtled towards the rogue planet, Alvarez initiated a risky gambit, overriding the AI with a forgotten command code, gambling the fate of thousands on a hunch about the ship's ancient, hidden protocols."
    #generate_audio(content)

    # set from direnv/.envrc
    openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    os.makedirs(output_path, exist_ok=True)

    # Generate a text stream
    prompt = "Write a 4 sentence scifi story segment from the climax of a generation ship scifi thriller."

    text_generator = gpt4_generate(prompt)

    text_ready = False
    while not text_ready:
        collect_audio_segment(text_generator)
        print(text_generator)
        time.sleep(0.1)

    '''
    tts = TextToSpeech(
        models_dir=model_dir, use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=half
    )
    print("Finished setting up tts object.")
    '''

    # this is for directly generating text, next we want to generate a separate prompt input loop.
    #manual_input_loop()
