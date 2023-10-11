import os, json
from datasets import load_dataset, load_metric, DatasetDict, Audio

def _load_audio(filename):
    """Loads an audio file into a `datasets.Audio` object."""
    audio = Audio(
        data=tf.io.read_file(filename),
        sampling_rate=16000,
    )
    return audio

# Clone the GitHub repo to our local machine
os.system("git clone https://github.com/mandeebot/Berom_Speech_Dataset.git")

# Find the audio files in the repo
dataset_dir = os.path.join("Berom_Speech_Dataset", "dataset")

# Load the audio files into a datasets.Dataset object
voice_train = load_dataset(
    ["csv","wav"],
    data_files={"train": os.path.join(dataset_dir, "train.csv")},
    delimiter="\t",
    features={
        "audio": lambda x: _load_audio(x["audio"]),
        "text": lambda x: x["text"],
    },
)

voice_test = load_dataset(
    "csv",
    data_files={"test": os.path.join(dataset_dir, "test.csv")},
    delimiter="\t",
    features={
        "audio": lambda x: _load_audio(x["audio"]),
        "text": lambda x: x["text"],
    },
)