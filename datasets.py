import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445


def _rocstories(path):
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f)
        st = []
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = ' '.join(line[1:5])
                c1 = line[5]
                c2 = line[6]
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1]) - 1)
        return st, ct1, ct2, y


def rocstories(data_dir, n_train=1497, n_valid=374):
    storys, comps1, comps2, ys = _rocstories(
        os.path.join(data_dir, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'))
    teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(storys, comps1,
                                                                                                      comps2, ys,
                                                                                                      test_size=n_valid,
                                                                                                      random_state=seed)
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)


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

    # (trX, trY), (vaX, vaY), teX = veracity(data_dir)

    dataLoader = DataLoader()
    (trX, trY), (vaX, vaY), teX = dataLoader.veracity(data_dir)

    print(trX[:5], trY[:5])
    print(len(trX))
    print(len(teX))
