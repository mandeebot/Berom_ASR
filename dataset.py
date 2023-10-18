import pandas as pd
import numpy as np

import os, json
#from datasets import load_dataset, load_metric, DatasetDict, Audio

#def _load_audio(filename):
#    """Loads an audio file into a `datasets.Audio` object."""
#    audio = Audio(
#        data=tf.io.read_file(filename),
##   return audio

# Clone the GitHub repo to our local machine
os.system("git clone https://github.com/mandeebot/Berom_Speech_Dataset.git")

# Find the audio files in the repo
dataset_dir = os.path.join("Berom_Speech_Dataset", "dataset")