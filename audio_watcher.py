''' Watch for each new wav file to play, play them in timestamp sequence (alphanumeric sort)
'''
import os
import subprocess
import time

def play_audio(file_path, blocking=True):
    try:
        # Determine the player based on file extension
        _, ext = os.path.splitext(file_path)
        player = 'aplay' if ext == '.wav' else 'afplay'  # Default to 'afplay' for non-wav files

        proc = subprocess.Popen([player, file_path])
        if blocking:
            proc.wait()  # Wait for the audio player process to finish if blocking
    except Exception as e:
        print(f"Error playing file: {e}")

def get_audio_files(directory, extension):
    """List all audio files with the given extension in the directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]

def main():
    directory_to_watch = "/home/robbintt/clones/tortoise-tts/results/"
    file_extension = ".wav"  # Change to '.mp3' or other audio formats as needed

    if not os.path.exists(directory_to_watch):
        os.makedirs(directory_to_watch)

    # Preload the processed_files set with existing files
    processed_files = set(get_audio_files(directory_to_watch, file_extension))

    try:
        while True:
            audio_files = get_audio_files(directory_to_watch, file_extension)
            audio_files.sort()  # Sort files based on the embedded timestamp in the filename

            for full_path in audio_files:
                if full_path not in processed_files:
                    play_audio(full_path, blocking=True)
                    processed_files.add(full_path)

            time.sleep(1)  # Polling interval
    except KeyboardInterrupt:
        print("Program interrupted and stopped.")

if __name__ == "__main__":
    main()
