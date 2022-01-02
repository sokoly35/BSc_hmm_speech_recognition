import os
import numpy as np
import pandas as pd
import librosa
from sklearn.utils import shuffle
from typing import Optional, Tuple, Dict
import colorednoise as cn


def create_dataframe(dir_name: Optional[str] = 'data',
                     how_many: Optional[int] = None,
                     test_size: Optional[float] = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Creating train and test datasets with references to observations. Furthemore we create
    label encoder, mapper between word classes and id.

    Parameters
    ----------
    how_many (int): Number of samples which you want to use during training.
        For example you have 1000 observations, but for learning you want to use only 100.

    Returns
    -------
        train_df (pd.DataFrame): train set with labels and paths to files
        test_df (pd.DataFrame): test set with labels and paths to files
        mapper (dict): mapps word class --> id label
    """
    # Full dataframe holder
    df = pd.DataFrame({'path': [],
                       'word': [],
                       'label': [],
                       'set': []})

    # List of all classes it depends from directory structure
    classes = os.listdir(os.path.join(dir_name, 'train'))
    # We create mapping from word calsses and id
    mapper = {j: i for i, j in enumerate(sorted(classes))}

    for subdir, dirs, files in os.walk(dir_name):
        if os.path.join(dir_name, 'train') in subdir or os.path.join(dir_name, 'test') in subdir:
            if files:
                for filename in files:
                    if filename.endswith('.wav'):
                        path = os.path.join(subdir, filename)
                        # Files are in form <class_name>_id.format so we want only <class_name>
                        word = filename.split('_')[0]
                        label = mapper[word]
                        # path is in form like data\\<train or test>\\<class>\\...
                        # We want to extract information if it is train or test
                        set_ = 'train' if 'train' in path else 'test'
                        # Temporary row holder
                        temp = pd.DataFrame({'path': [path],
                                             'word': [word],
                                             'label': [label],
                                             'set': [set_]})
                        df = df.append(temp)
    # We divide set with respect of train and test set
    train_df = df.loc[df['set'] == 'train', :'label']
    test_df = df.loc[df['set'] == 'test', :'label']
    # If we specified how_many then we sample only fraction of sets
    if how_many:
        train_df = train_df.groupby('word').sample(how_many)
        test_df = test_df.groupby('word').sample(int(np.ceil(test_size * how_many)))

    # Shuffle and reseting index
    train_df = shuffle(train_df).reset_index(drop=True)
    test_df = shuffle(test_df).reset_index(drop=True)

    return train_df, test_df, mapper


class DataPreprocessor:
    """
    A class used to preprocess dataframe

    Attributes
    ----------
    df : pd.DataFrame
        list of word classes, each class represents word spoken in single audio file
    max_len : int
        len of the longest file, we assume that each file has the same length
    n : str
        number of observations

    Methods
    -------
    mfcc
        calculate mfcc coefficients
    mfcc_noised
        calculate mfcc coefficients but for noised signals
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Parameters
        ----------
        df (pd.DataFrame): dataframe with path to files and labels of class
        """
        self.df = df

        # We assume that each file have the same len, so we take the first and
        # write down the max length
        y, sr = librosa.load(df.loc[0, 'path'])
        self.max_len = len(y)
        self.n = len(df)

    def mfcc(self, M: int, S: Optional[float] = 0.025,
             R: Optional[float] = 0.01) -> Tuple[np.array, np.array]:
        """
        Calculate the mfcc coefficients

        Parameters
        ----------
        M (int): number of mfcc coefficients
        S (float): window length in ms
        R (float): size of window's hop in ms

        Returns
        -------
            X (np.array): array of shape (n_observations, M, n_windows)
            y (np.array): word class labels
        """
        # Empty matrix for signals
        signals = np.zeros((self.n, self.max_len))

        # Loading every signal
        for i, file in enumerate(self.df.path):
            y, _ = librosa.load(file)
            signals[i, :] = y
        # defining parameters
        sr = 22050
        hop_length = int(np.floor(sr * R))
        win_length = int(np.floor(sr * S))

        # For each signal and signal's window we calculate M mfcc coefficients
        X = [librosa.feature.mfcc(signals[i, :], sr=sr, n_mfcc=M,
                                  hop_length=hop_length, win_length=win_length)
             for i in range(self.n)]
        # converting X to 3D array
        X = np.atleast_3d(X)

        # labels
        y = self.df.label.to_numpy()

        # Normalizing each window coefficients
        for n in range(len(X)):
            X[n] /= X[n].sum(axis=0)

        return X, y

    def mfcc_noised(self, M: int, beta: int, SNR: float, S: Optional[float] = 0.025,
                    R: Optional[float] = 0.01) -> Tuple[np.array, np.array]:
        """
        Calculate the mfcc coefficients

        Parameters
        ----------
        M (int): number of mfcc coefficients
        beta (int): rank of noise, read more in noising function
        SNR (float): signal noise ration, the ratio of power between signal and noise
        S (float): window length in ms
        R (float): size of window's hop in ms

        Returns
        -------
            X (np.array): array of shape (n_observations, M, n_windows)
            y (np.array): word class labels
        """
        # Empty matrix for signals
        signals = np.zeros((self.n, self.max_len))

        # Loading every signal
        for i, file in enumerate(self.df.path):
            y, _ = librosa.load(file)
            y = noising(y, beta, SNR)
            signals[i, :] = y
        # defining parameters
        sr = 22050
        hop_length = int(np.floor(sr * R))
        win_length = int(np.floor(sr * S))

        # For each noised signal and signal's window we calculate M mfcc coefficients
        X = [librosa.feature.mfcc(signals[i, :], sr=sr, n_mfcc=M,
                                  hop_length=hop_length, win_length=win_length)
             for i in range(self.n)]
        # converting X to 3D array
        X = np.atleast_3d(X)

        # labels
        y = self.df.label.to_numpy()

        # Normalizing each window coefficients
        for n in range(len(X)):
            X[n] /= X[n].sum(axis=0)

        return X, y


def noising(y: np.array, beta: int, SNR: float) -> np.array:
    """Noising signal with respect to SNR ratio

    Args:
    ----------------
        y (np.array): signal
        beta (int): coefficient, defines which noise is currently used
                -2 - violet noise
                -1 - blue noise
                0 - white noise
                1 - pink noise
                2 - Brownian noise
        SNR (int): signal noise ration, the lower the value the louder noise
    Returns
    -------
        y + noise (np.array): noised signal"""
    n = len(y)

    # average power of discrete signal
    signal_avg_power = np.mean(y ** 2)
    # converting to dB
    signal_avg_power_db = 10 * np.log10(signal_avg_power)
    # SNR = Ps dB - Pn db (power of signal and power of noise)
    noise_power_db = signal_avg_power_db - SNR
    # dB --> Hz
    noise_power = 10 ** (noise_power_db / 10)
    # generating noise sample
    noise = cn.powerlaw_psd_gaussian(beta, n)
    # specifing noise power with respect to SNR
    noise *= noise_power / np.std(noise)
    return y + noise
