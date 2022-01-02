import numpy as np
import scipy.stats as st
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from typing import Optional, List, Dict


class gmmhmm:
    # This class converted with modifications from https://code.google.com/p/hmm-speech-recognition/source/browse/Word.m
    def __init__(self, n_states, n_iter=15):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = np.random.RandomState(0)

        # Normalize random initial state
        self.prior = self._normalize(self.random_state.rand(self.n_states, 1))
        self.A = self._stochasticize(self.random_state.rand(self.n_states, self.n_states))

        self.mu = None
        self.covs = None
        self.n_dims = None

    def _forward(self, B):
        log_likelihood = 0.
        T = B.shape[1]
        alpha = np.zeros(B.shape)
        for t in range(T):
            if t == 0:
                alpha[:, t] = B[:, t] * self.prior.ravel()
            else:
                alpha[:, t] = B[:, t] * np.dot(self.A.T, alpha[:, t - 1])
            alpha[:, t] = alpha[:, t] + (alpha[:, t] == 0)
            alpha_sum = np.sum(alpha[:, t])
            alpha_sum = np.where(alpha_sum, alpha_sum, t)
            alpha[:, t] /= alpha_sum
            log_likelihood = log_likelihood + np.log(alpha_sum)
        return log_likelihood, alpha

    def _backward(self, B):
        T = B.shape[1]
        beta = np.zeros(B.shape)

        beta[:, -1] = np.ones(B.shape[0])

        for t in range(T - 1)[::-1]:
            beta[:, t] = np.dot(self.A, (B[:, t + 1] * beta[:, t + 1]))
            beta[:, t] = beta[:, t] + (beta[:, t] == 0)
            beta_sum = np.sum(beta[:, t])
            beta[:, t] /= np.where(beta_sum, beta_sum, t)
        return beta

    def _state_likelihood(self, obs):
        obs = np.atleast_2d(obs)
        B = np.zeros((self.n_states, obs.shape[1]))
        for s in range(self.n_states):
            # Needs scipy 0.14
            np.random.seed(self.random_state.randint(1))
            B[s, :] = st.multivariate_normal.pdf(
                obs.T, mean=self.mu[:, s].T, cov=self.covs[:, :, s].T)
            # This function can (and will!) return values >> 1
            # See the discussion here for the equivalent matlab function
            # https://groups.google.com/forum/#!topic/comp.soft-sys.matlab/YksWK0T74Ak
            # Key line: "Probabilities have to be less than 1,
            # Densities can be anything, even infinite (at individual points)."
            # This is evaluating the density at individual points...
        return B

    def _normalize(self, x):
        return (x + (x == 0)) / np.where(np.sum(x), np.sum(x), len(x))

    def _stochasticize(self, x):
        # robi to samo co _normalize tylko dla macierzy przejścia
        sum_of_rows = x.sum(axis=1)
        sum_of_rows = np.where(sum_of_rows, sum_of_rows, len(x))
        return (x + (x == 0)) / sum_of_rows[:, np.newaxis]

    def _em_init(self, obs):
        # Using this _em_init function allows for less required constructor args
        if self.n_dims is None:
            self.n_dims = obs.shape[0]
        if self.mu is None:
            subset = self.random_state.choice(np.arange(self.n_dims), size=self.n_states, replace=False)
            self.mu = obs[:, subset]
        if self.covs is None:
            self.covs = np.zeros((self.n_dims, self.n_dims, self.n_states))
            self.covs += np.diag(np.diag(np.cov(obs)))[:, :, None]
        return self

    def _em_step(self, obs):
        obs = np.atleast_2d(obs)
        B = self._state_likelihood(obs)
        T = obs.shape[1]

        log_likelihood, alpha = self._forward(B)
        beta = self._backward(B)

        xi_sum = np.zeros((self.n_states, self.n_states))
        gamma = np.zeros((self.n_states, T))

        for t in range(T - 1):
            partial_sum = self.A * np.dot(alpha[:, t], (beta[:, t] * B[:, t + 1]).T)
            xi_sum += self._normalize(partial_sum)
            partial_g = alpha[:, t] * beta[:, t]
            gamma[:, t] = self._normalize(partial_g)

        partial_g = alpha[:, -1] * beta[:, -1]
        gamma[:, -1] = self._normalize(partial_g)

        expected_prior = gamma[:, 0]
        expected_A = self._stochasticize(xi_sum)

        expected_mu = np.zeros((self.n_dims, self.n_states))
        expected_covs = np.zeros((self.n_dims, self.n_dims, self.n_states))

        gamma_state_sum = np.sum(gamma, axis=1)
        # Set zeros to 1 before dividing
        gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)

        for s in range(self.n_states):
            gamma_obs = obs * gamma[s, :]
            expected_mu[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]
            partial_covs = np.dot(gamma_obs, obs.T) / gamma_state_sum[s] - np.dot(expected_mu[:, s],
                                                                                  expected_mu[:, s].T)
            # Symmetrize
            partial_covs = np.triu(partial_covs) + np.triu(partial_covs).T - np.diag(partial_covs)

        # Ensure positive semidefinite by adding diagonal loading
        expected_covs += .01 * np.eye(self.n_dims)[:, :, None]

        self.prior = expected_prior
        self.mu = expected_mu
        self.covs = expected_covs
        self.A = expected_A
        return log_likelihood

    def fit(self, obs):
        # Support for 2D and 3D arrays
        # 2D should be n_features, n_dims
        # 3D should be n_examples, n_features, n_dims
        # For example, with 6 features per speech segment, 105 different words
        # this array should be size
        # (105, 6, X) where X is the number of frames with features extracted
        # For a single example file, the array should be size (6, X)
        if len(obs.shape) == 2:
            for i in range(self.n_iter):
                self._em_init(obs)
                log_likelihood = self._em_step(obs)
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            for n in range(count):
                for i in range(self.n_iter):
                    self._em_init(obs[n, :, :])
                    log_likelihood = self._em_step(obs[n, :, :])
        return self

    def transform(self, obs):
        # Support for 2D and 3D arrays
        # 2D should be n_features, n_dims
        # 3D should be n_examples, n_features, n_dims
        # For example, with 6 features per speech segment, 105 different words
        # this array should be size
        # (105, 6, X) where X is the number of frames with features extracted
        # For a single example file, the array should be size (6, X)
        if len(obs.shape) == 2:
            B = self._state_likelihood(obs)
            log_likelihood, _ = self._forward(B)
            return log_likelihood
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            out = np.zeros((count,))
            for n in range(count):
                B = self._state_likelihood(obs[n, :, :])
                log_likelihood, _ = self._forward(B)
                out[n] = log_likelihood
            return out


class IwrGaussianHMMModel:
    """
        A class used to create Isolated Word Recognition dataset with Google Text to Speech tool

        ...

        Attributes
        ----------
        n_states : int
            number of hidden states in hmm model
        n_iter : int
            number of iterations in Baum-Welch algorithm
        model : List[gmmhmm, ...]
            list of hmm models, each model represents chain of single word class

        Methods
        -------
        fit
            fitting list of models
        _divide_dataset_by_class
            creating list of datasets for each word class
        predict
            predicting word class label
        score
            calculating solution score
        """

    def __init__(self, n_states: int, n_iter: Optional[int] = 10) -> None:
        """

        Parameters
        ----------
        n_states (int): number of hidden states in hmm
        n_iter (int): number of iterations in Baum-Welch algorithm
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.model = None

    def fit(self, X: np.array, y: np.array) -> None:
        """
        fitting list of models

        Parameters
        ----------
        X (np.array): array of shape (n_observations, n_mfcc_coef, n_windows)
        y (np.array): array of labels for each word class

        Returns
        -------
            None
        """
        # Unique labels
        labels = list(set(y))
        # list of datasets for each unique class
        datasets = self._divide_dataset_by_class(X, y, labels)

        # list of hmm models, each model represents one word class
        self.model = [gmmhmm(n_states=self.n_states,
                             n_iter=self.n_iter)
                      for _ in range(len(labels))]
        # fitting each model
        self.model = [m.fit(x) for m, x in zip(self.model, datasets)]

    def _divide_dataset_by_class(self, X: np.array, y: np.array, labels: List[float]) -> List[np.array]:
        """
        creating list of datasets for each word class

        Parameters
        ----------
        X (np.array): array of shape (n_observations, n_mfcc_coef, n_windows)
        y (np.array): array of labels for each word class
        labels (list): list of unique labels

        Returns
        -------
            datasets (list): list of array, each array contain single word class signal
        """
        datasets = [X[y == label, :, :] for label in labels]
        return datasets

    def predict(self, X: np.array) -> np.array:
        """
        predicting word class label

        Parameters
        ----------
        X (np.array): array of shape (n_observations, n_mfcc_coef, n_windows)

        Returns
        -------
        preds (np.array): array with prediction label
        """
        n, *_ = X.shape
        # if model exists
        if self.model:
            # each log likelihood values from models
            ps = [m.transform(X) for m in self.model]
            # stacking observations
            res = np.vstack(ps)
            # taking the max index
            preds = np.argmax(res, axis=0)
            return preds
        else:
            print('First of all you must fit a model')

    def score(self, X: np.array, y: np.array) -> Dict[str, float]:
        """
        calculating solution score

        Parameters
        ----------
        X (np.array): array of shape (n_observations, n_mfcc_coef, n_windows)
        y (np.array): array of labels for each word class

        Returns
        -------
            acc (float): accuracy of solutions
        """
        results = {}
        preds = self.predict(X)
        accuracy = accuracy_score(y, preds)
        recall = recall_score(y, preds, average='weighted', zero_division=1)
        precision = precision_score(y, preds, average='weighted', zero_division=1)
        f1 = f1_score(y, preds, average='weighted', zero_division=1)

        results['accuracy'] = accuracy
        results['recall'] = recall
        results['precision'] = precision
        results['f1'] = f1
        return results

