# regressor

Create the following directories:
```
models/
midis/
wavs/
```

Download the following models from [here](https://essentia.upf.edu/models/) and install them inside `models/`.

1. [deam + musicnn](https://essentia.upf.edu/models/classification-heads/deam/deam-musicnn-msd-1.pb)
2. [deam + vggish](https://essentia.upf.edu/models/classification-heads/deam/deam-vggish-audioset-1.pb)
3. [emomusic + musicnn](https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-musicnn-msd-1.pb)
4. [emomusic + vggish](https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-vggish-audioset-1.pb)
5. [muse + musicnn](https://essentia.upf.edu/models/classification-heads/muse/muse-musicnn-msd-1.pb)
6. [muse + vggish](https://essentia.upf.edu/models/classification-heads/muse/muse-vggish-audioset-1.pb)

NOTE: These have to be renamed - in each filename, replace the sequence `musicnn-msd` with `msd-musicnn` and the sequence `vggish-audioset` with `audioset-vggish`
(Alternatively I could just rework the code but meh)

The raw models themselves:

1. [vggish](https://essentia.upf.edu/models/feature-extractors/vggish/audioset-vggish-3.pb)
2. [musicnn](https://essentia.upf.edu/models/autotagging/msd/msd-musicnn-1.pb)
 

Download the midis directory from [here](https://drive.google.com/drive/folders/1-sEBh0iLo0wNAu1BRgupRan7BW_cM614?usp=sharing) and store them in your `midis/` directory.
 
Make sure [Timidity++](https://manpages.ubuntu.com/manpages/bionic/man1/timidity.1.html) is installed on your system (the next script runs it as a subprocess).

Next, run:
```
python mp3.py
```

This will convert all the MIDI files to WAV fiiles (it's misnamed because it was originally converting to MP3, but this format works too). 

Finally, install the essentia dependencies using:
```
pip install essentia
pip install essentia-tensorflow
```

and run

```
python predict.py
```
