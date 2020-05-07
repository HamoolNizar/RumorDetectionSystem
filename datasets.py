import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 2412798343


class DataLoader():

    @staticmethod
    def _veracity(path, topic=None):
        def clean_ascii(text):
            # function to remove non-ASCII chars from data
            return ''.join(i for i in text if ord(i) < 128)

        orig = pd.read_csv(path, delimiter='\t', header=0, encoding="latin-1")
        orig['Tweet'] = orig['Tweet'].apply(clean_ascii)
        df = orig
        # Get only those tweets that pertain to a single topic in the training data
        if topic is not None:
            df = df.loc[df['Target'] == topic]
        X = df.Tweet.values
        veracity_states = ["RUMOR", "NON-RUMOR", "UNKNOWN"]
        class_nums = {s: i for i, s in enumerate(veracity_states)}
        Y = np.array([class_nums[s] for s in df.Veracity])
        return X, Y

    def veracity(self, data_dir, topic=None):
        path = Path(data_dir)
        training_data_file = 'RumorsAndNonRumors-TrainingData.txt'
        testing_data_file = 'RumorsAndNonRumors-TestingData.txt'

        # X = Tweet Text , Y = Tweet Veracity State
        X, Y = self._veracity(path / training_data_file, topic=topic)
        test_text, _ = self._veracity(path / testing_data_file, topic=topic)
        tr_text, va_text, tr_state, va_state = train_test_split(X, Y, test_size=0.2, random_state=seed)
        train_text = []
        train_state = []
        for text, state in zip(tr_text, tr_state):
            train_text.append(text)
            train_state.append(state)

        validate_text = []
        validate_state = []
        for text, state in zip(va_text, va_state):
            validate_text.append(text)
            validate_state.append(state)
        train_state = np.asarray(train_state, dtype=np.int32)
        validate_state = np.asarray(validate_state, dtype=np.int32)
        return (train_text, train_state), (validate_text, validate_state), (test_text,)


# Testing Purpose
if __name__ == "__main__":
    ## Test
    data_dir = "./data"

    dataLoader = DataLoader()
    (trX, trY), (vaX, vaY), teX = dataLoader.veracity(data_dir)

    print(trX[:5], trY[:5])
    print(len(trX))
    print(len(teX))
