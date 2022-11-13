from pydub import AudioSegment

import os
import subprocess

iteration = 1

for file_name in os.listdir('./midis'):
    print(f"Iteration: {iteration}")

    # song = AudioSegment.from_wav('./wavs/' + wav_file)
    # song.export('./mp3s/' + wav_file[:-3] + 'mp3', format="mp3")

    subprocess.run(["timidity",  "./midis/" + file_name, "-Ow", "-o", "./wavs/" + file_name[:-3] + "wav"])
    iteration += 1