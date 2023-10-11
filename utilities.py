import os, pandas as pd,  json
import tensorflow as tf

import datasets
from datasets import load_dataset, DatasetDict


class BeromSpeechDataset(tf.keras.utils.DatasetV2):
    """beromSpeech dataset"""

    def __init__(self, data_dir, device):
        """
        Args:
            data_dir: The directory where the dataset is stored.
            device: The device to use for training.
        """
        super().__init__()
        self.data_dir = data_dir
        self.data = None
        self.device = device
        self.train_data = None
        self.eval_data = None

    def data_processer(self):
        """
        Converts the dataset files to a Pandas DataFrame.

        Returns:
            A Pandas DataFrame containing the dataset files.
        """
        wav_data = os.path.join(self.data_dir, 'wav')
        txt_data = os.path.join(self.data_dir, 'trans')
        data_train, data_test = [], []
        wav_set = os.listdir(wav_data)
        train_size = int(len(wav_set)*0.8)
        for i, filename in enumerate(wav_set):
            txt = os.path.join(txt_data, f"{filename.rstrip('.wav')}.txt")
            with open(txt) as f:
                text = f.read()
            waveform, sr = tf.audio.decode_wav(tf.io.read_file(os.path.join(wav_data, filename)))
            if i < train_size:
                data_train.append((os.path.join(wav_data, filename), text))
            else:
                data_test.append((os.path.join(wav_data, filename), text))
        df_train = pd.DataFrame(data_train, columns=['audio', 'text'])
        df_test = pd.DataFrame(data_test, columns=['audio', 'text'])
        df_train.to_csv("train.csv", sep="\t", encoding="utf-8", index=False)
        df_test.to_csv("test.csv", sep="\t", encoding="utf-8", index=False)

        return df_train, df_test

    def get_dataset(self):
        """
        Loads the dataset from the Pandas DataFrame.

        Returns:
            A DatasetDict containing the train and eval datasets.
        """
        self.files2df()
        berom_train = load_dataset("csv", data_files={"train": "train.csv"}, delimiter="\t")["train"]
        berom_test = load_dataset("csv", data_files={"test": "test.csv"}, delimiter="\t")["test"]
        self.data = DatasetDict({k: dt for k, dt in {'train': berom_train, 'test': berom_test}.items()})
        self.train_data = self.data['train']
        self.eval_data = self.data['test']
        return self

    def extract_all_chars(self, batch):
        """
        Extracts all the characters in the batch.

        Args:
            batch: A batch of data.

        Returns:
            A dictionary containing the vocabulary and all the text in the batch.
        """
        all_text = " ".join(batch["text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    def speech_file_to_array_fn(self, batch):
        """
        Converts the speech files in the batch to arrays.

        Args:
            batch: A batch of data.

        Returns:
            A batch of data with the speech files converted to arrays.
        """
        resampler = tf.audio.resample(tf.io.decode_wav(tf.io.read_file(batch["audio"])), 16000,
                                     batch["audio"])
        batch["speech"] = resampler.squeeze().numpy()
        batch["sampling_rate"] = 16000
        batch["target_text"] = batch["text"]
        return batch

    def split_train_test(self):
        """Splits the dataset into train and eval sets."""
        self.train_data = self.train_data.map(
            self.speech_file_to_array_fn,
            remove_columns=self.train_data.column_names,
            num_proc=4,
        )
        self.eval_data = self.eval_data.map(
            self.speech_file_to_array_fn,
            remove_columns=self.eval_data.column_names,
            num_proc=4,
        )

    def prepare_dataset(self, batch):
        """Prepares the dataset for training."""

        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {self.processor.feature_extractor.sampling_rate}."

        batch["input_values"] = self.processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["target_text"]).input_ids

        return batch

    def convert_to_ids(self, processor):
        """
        Converts the dataset to IDs using the given processor.

        Args:
            processor: The processor to use to convert the dataset to IDs.
        """
        self.processor = processor

        self.train_data = self.train_data.map(self.prepare_dataset,
                                            remove_columns=self.train_data.column_names,
                                            batch_size=32,
                                            batched=True,
                                            num_proc=4)
        self.eval_data = self.eval_data.map(self.prepare_dataset,
                                            remove_columns=self.eval_data.column_names,
                                            batch_size=32,
                                            batched=True,
                                            num_proc=4)

    def get_vocab(self):
        """
        Gets the vocabulary of the dataset.

        Returns:
            A dictionary containing the vocabulary.
        """

        vocabs = self.data.map(self.extract_all_chars,
                            batched=True,
                            batch_size=-1,
                            keep_in_memory=True,
                            remove_columns=self.data.column_names["train"])
        vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)

        with open('vocab.json', 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)

        return vocab_dict