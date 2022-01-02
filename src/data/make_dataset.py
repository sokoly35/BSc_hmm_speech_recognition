from gtts import gTTS
import os
import time
import numpy as np
import librosa
from pydub import AudioSegment
from scipy.ndimage.interpolation import shift
import soundfile as sf
from typing import Optional, Tuple


class DataCreator:
    """
    A class used to create Isolated Word Recognition dataset with Google Text to Speech tool.

    Attributes
    ----------
    dir_name : str
        name of main directory, default 'data'
    classes : list
        list of word classes, each class represents word spoken in single audio file
    train_path : str
        reference to train directory
    test_path : str
        reference to test directory
    max_len : int
        len of the longest .wav file with sampling rate 22050 Hz

    Methods
    -------
    generate_observations
        creating audio files for each class with fixed length and randomization
    _convert_mp3_to_wav
        converts mp3 files to wav format then removes mp3 version
    _randomize_observation
        after creating audio file we can add some random shifts or rescaling signal
    _normalize_signal_len
        fixing length of audio files with respect to the longest
    """

    def __init__(self, dir_name: Optional[str] = 'data', classes: Optional[list] = None):
        """
        Parameters
        ----------
        classes : optional, list
            list of word classes
        """

        # If classes weren't provided then create default list
        if not classes:
            self.classes = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
        else:
            self.classes = classes

        self.dir_name = dir_name

        # Creating directory template if it doesn't exist
        self.train_path = os.path.join(self.dir_name, 'train')
        self.test_path = os.path.join(self.dir_name, 'test')
        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
            os.mkdir(self.train_path)
            os.mkdir(self.test_path)
            for class_ in self.classes:
                os.mkdir(os.path.join(self.train_path, class_))
                os.mkdir(os.path.join(self.test_path, class_))

        # The length of the longest file
        # We set it to 0 and it will increase with further files
        self.max_len = 0

    def generate_observations(self, n: int,
                              sleep : Optional[int]=0,
                              test_size: Optional[float] = 0.2,
                              language: Optional[str] = 'en',
                              tld: Optional[str] = 'com',
                              silent_ratio: Optional[float] = 0.2,
                              horizontal_shift_range: Optional[Tuple[float, float]] = (-0.1, 0.1),
                              vertical_expand_range: Optional[Tuple] = (1, 1)) -> None:
        """
        creating audio files for each class with fixed length and randomization

        Parameters
        ----------
        n (int): number of observation in each class
        test_size (float): ratio of test size and train size
        language (str): language of gTTS generator
        tld (str): accent of lector in gTTS generator
        silent_ratio (float): this ratio tells the fraction of silence which will be added to signal.
            for example if signal has 10 observation, and silen_ratio=0.2, then one zero will be added
            at the begining and at the end
        horizontal_shift_range (tuple): range of percent values from which we choose the value which shift the
            signal to the left or right
        vertical_expand_range (tuple): range of values from which we should scale the signal

        Returns
        -------
            None
        """

        # We generate n signals from each word class in train set
        # test_size * n in test set
        for class_ in self.classes:
            time.sleep(sleep)
            # We want to extract max id of single class file
            # We need only one class because number of observation in each class should be equal
            # format of filenames should look like <class_name>_id.wav
            files = os.listdir(os.path.join(self.train_path, class_))
            files_test = os.listdir(os.path.join(self.test_path, class_))
            ids_test = [int(filename.split('.')[0].split('_')[-1]) for filename in files_test]
            ids = [int(filename.split('.')[0].split('_')[-1]) for filename in files]
            if ids:
                max_id = max(ids)
            else:
                max_id = 0

            if ids_test:
                max_id_test = max(ids_test)
            else:
                max_id_test = 0

            # Creating speech object generator
            text_obj = gTTS(text=class_, tld=tld, lang=language, slow=False)

            if max_id < n:
                # With respect to max id files which exists we create n mp3 files
                for i in range(max_id + 1, max_id + n + 1):
                    filename = f'{class_}_{i}.mp3'
                    text_obj.save(os.path.join(self.train_path, class_, filename))

            if max_id_test < int(np.ceil(test_size * n)):
                # Then test_size * n mp3 files to test set
                for i in range(max_id_test + 1, int(np.ceil(test_size * n)) + 1):
                    filename = f'{class_}_{i}.mp3'
                    text_obj.save(os.path.join(self.test_path, class_, filename))

        # We convert .mp3 --> .wav
        for subdir, dirs, files in os.walk(self.dir_name):
            # If in directory are files
            if files:
                for filename in files:
                    if filename.endswith('.mp3'):
                        self._convert_mp3_to_wav(filename, subdir)

        # We randomize our observations to make them more diverse
        for subdir, dirs, files in os.walk(self.dir_name):
            if files:
                for filename in files:
                    if filename.endswith('.wav'):
                        path = os.path.join(subdir, filename)
                        # Randomizing observations
                        self._randomize_observation(path,
                                                    silent_ratio=silent_ratio,
                                                    horizontal_shift_range=horizontal_shift_range,
                                                    vertical_expand_range=vertical_expand_range)

        for subdir, dirs, files in os.walk(self.dir_name):
            if files:
                for filename in files:
                    if filename.endswith('.wav'):
                        path = os.path.join(subdir, filename)
                        # Fixing len of each file
                        self._normalize_signal_len(path)

    def _convert_mp3_to_wav(self, filename: str, subdir: str) -> None:
        """
        converts mp3 files to wav format then removes mp3 version

        Parameters
        ----------
        filename (str): name of file
        subdir (str): name of path to the file

        Returns
        -------
            None
        """
        # Full path
        path = os.path.join(subdir, filename)
        # Name of file before extension
        base = filename.split('.')[0]
        # Converting mp3 to wav
        sound = AudioSegment.from_mp3(path)
        sound.export(os.path.join(subdir, f'{base}.wav'), format='wav')
        # Removing mp3 file
        os.remove(path)

    def _randomize_observation(self, path: str,
                               silent_ratio: Optional[float] = 0.2,
                               horizontal_shift_range: Optional[Tuple[float, float]] = (-0.1, 0.1),
                               vertical_expand_range: Optional[Tuple[float, float]] = (1, 1)) -> None:
        """
        after creating audio file we can add some random shifts or rescaling signal

        Parameters
        ----------
        path (str): path to file
        silent_ratio (float): this ratio tells the fraction of silence which will be added to signal.
            for example if signal has 10 observation, and silen_ratio=0.2, then one zero will be added
            at the begining and at the end
        horizontal_shift_range (tuple): range of percent values from which we choose the value which shift the
            signal to the left or right
        vertical_expand_range (tuple): range of values from which we should scale the signal

        Returns
        -------
            None
        """
        # Loading audio file
        y, sr = librosa.load(path, sr=22050)
        # Length of single zeros vector
        silence = int(np.ceil(silent_ratio / 2 * len(y)))
        # Silence vector of zeros
        zeros = np.zeros(silence)
        # Stacking silence at beginning and ending of signal
        y = np.hstack([zeros, y, zeros])
        # New len of signal
        n = len(y)
        # Random change parameters
        random_hshift = np.random.uniform(*horizontal_shift_range)
        random_vexpand = np.random.uniform(*vertical_expand_range)
        # Rescaling signal
        y *= random_vexpand
        # Random shifting
        y = shift(y, int(random_hshift * n))
        # Rewriting file
        sf.write(path, y, sr)

        # If file is longer than max_len we fix max_len
        if n > self.max_len:
            self.max_len = n

    def _normalize_signal_len(self, path: str) -> None:
        """
        fixing length of audio files with respect to the longest

        Parameters
        ----------
        path (str): path to audio file

        Returns
        -------
            None
        """
        # Loading audio file
        y, sr = librosa.load(path, sr=22050)
        # Number of samples which remain to the longes
        remain = self.max_len - len(y)
        # Creating zero vector of remain length
        zeros = np.zeros(remain)
        # Fixing signal length
        y = np.hstack([y, zeros])
        # Rewriting file
        sf.write(path, y, sr)
