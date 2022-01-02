from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from collections import defaultdict
import time
import os
from ..data.make_dataset import DataCreator
from ..features.preprocess import create_dataframe, DataPreprocessor
from ..models.model_setup import IwrGaussianHMMModel
from typing import Optional, Tuple, Dict, List, DefaultDict, Union


class GridSearch:
    """
    Class to conducting experiments via grid search method

     Attributes
    ----------
    random_state : int
        seed which specifies random seed

    Methods
    -------
    create_random_datasets
        create isolated word recognition datasets with specified random parameters
    grid_search_randomness
        function performs grid search with multiple datasets to find
        various results from each combination of parameters
    grid_search_model
        function performs grid search with multiple models to find
        various results from each combination of parameters
    _process_cv_results
        function updates solutions from cross validation to default dicts
    _process_params
        function structures results in dataframes
    _find_rhs_rvs
        function find rhs and rvs parameters from specified path
    """
    def __init__(self, random_state: Optional[int] = None) -> None:
        if not random_state:
            self.random_state = random_state
        else:
            self.random_state = None
        # specyfing random seed
        if not self.random_state:
            np.random.seed(self.random_state)

    def create_random_datasets(self, params: Dict[str, List],
                               sleep: Optional[int] = 0) -> Union[List[str], bool]:
        """
        create isolated word recognition datasets with specified random parameters

        Parameters
        ----------
        params : dict
            maps name_of_parameter -> possible  values
        sleep : int, optional
            during creating datasets there may occur too many request error. YOu can set
            sleep parameter to constrain density of requests

        Returns
        -------
            paths : list
                paths to datasets directories. If there don't exist data directory
                error will occur
        """
        # Every dataset is saved to data\random directory
        path_random = os.path.join('data', 'random')

        # If there isn't data directory
        if not os.path.exists('data'):
            print('There is no data directory')
            return True
        # If there wasn't created any random dataset before
        # then we will create random directory under data
        elif not os.path.exists(path_random):
            os.mkdir(path_random)

        # Unmpacking params
        random_horizontal_shifts = params['random_horizontal_shift']
        random_vertical_scalings = params['random_vertical_scaling']
        paths = []

        for rhs in random_horizontal_shifts:
            for rvs in random_vertical_scalings:
                # We will create paths in pattern like data\random\data_rhs_<number>_<number>_rvs_<number>_<number>
                rhs_str = str(rhs).replace('.', '_')
                rvs_str = str(rvs).replace('.', '_')
                # These are the default parameters from our main dataset, we don't need another
                if rhs == 0.1 and rvs == 1:
                    paths.append('data')
                else:
                    # Creating directory name corresponding to rhs and rvs
                    dir_name = f"data_rhs_{rhs_str}_rvs_{rvs_str}"
                    path = os.path.join(path_random, dir_name)
                    paths.append(path)
                    # If there currently exists needed directory we don't need to create it
                    # The fact that directory is correctly created is checked by calculating
                    # number of .wav files. If there are 1200 files then it is correct
                    file_count = sum([len([f for f in fs if f.lower().endswith('.wav')]) for _, _, fs in os.walk(path)])
                    if file_count != 1200:
                        dc = DataCreator(dir_name=path)
                        # we need sorted rvs values, because we will generate scaling factor
                        # from range (min(rvs, 1), max(rvs, 1))
                        sorted_rvs = tuple(sorted([rvs, 1.0]))
                        dc.generate_observations(100,
                                                 sleep=sleep,
                                                 horizontal_shift_range=(-rhs, rhs),
                                                 vertical_expand_range=sorted_rvs)
        return paths

    def grid_search_randomness(self, params: Dict[str, List], sleep: Optional[int] = 0) -> pd.DataFrame:
        """
        unction performs grid search with multiple datasets to find
        various results from each combination of parameters

        Parameters
        ----------
        params : dict
            maps name_of_parameter -> possible  values
        sleep : int, optional
            during creating datasets there may occur too many request error. YOu can set
            sleep parameter to constrain density of requests

        Returns
        -------
            results : pd.DataFrame
                each row is combination of params and scores from evaluation
        """
        # Creating datasets and paths to them
        paths = self.create_random_datasets(params, sleep=sleep)
        # Unpacking params
        n_iters = params['n_iter']
        n_hiddens = params['n_hiddens']

        results = pd.DataFrame({'random_horizontal_shift': [],
                                'random_vertical_scaling': [],
                                'n_iter': [],
                                'n_hiddens': [],
                                'accuracy': [],
                                'recall': [],
                                'precision': [],
                                'F1_score': []})

        # For each dataset
        for path in paths:
            # creating dataframes
            train_df, test_df, mapper = create_dataframe(dir_name=path, how_many=100)
            train_dp = DataPreprocessor(train_df)
            test_dp = DataPreprocessor(test_df)
            # Processing data with respect to best found params
            X_train, y_train = train_dp.mfcc(13, S=0.025, R=0.01)
            X_test, y_test = test_dp.mfcc(13, S=0.025, R=0.01)
            # We choosing best model params
            for n_iter in n_iters:
                for n_hidden in n_hiddens:
                    model = IwrGaussianHMMModel(n_states=n_hidden, n_iter=n_iter)
                    model.fit(X_train, y_train)
                    scores = model.score(X_test, y_test)
                    # Finding rhs and rvs used to create dataset
                    rhs, rvs = self._find_rhs_rvs(path)
                    temp = self._process_params(rhs, rvs, n_iter, n_hidden, scores, which_grid='randomness')
                    results = results.append(temp)

        results.reset_index(drop=True, inplace=True)
        return results

    def grid_search_model(self, params: Dict[str, List], k: Optional[int] = 5) -> pd.DataFrame:
        """
        function performs grid search with multiple models to find
        various results from each combination of parameters

        Parameters
        ----------
        params : dict
            maps name_of_parameter -> possible  values
        k : int, optional
            number of folds in stratified cross validation

        Returns
        -------
            results : pd.DataFrame
                each row is combination of params and scores from evaluation
        """
        # Number of iterations
        max_iterations = np.prod([len(i) for i in params.values()])
        i = 1
        # Unpacking params
        n_samples = params['n_samples']
        n_mfcc = params['n_mfcc']
        hop_wins = params['hop_wins']
        len_wins = params['len_wins']
        n_iters = params['n_iters']
        n_hiddens = params['n_hiddens']

        results = pd.DataFrame({'n_samples': [],
                                'n_mfcc': [],
                                'hop_win': [],
                                'len_win': [],
                                'n_iter': [],
                                'n_hidden': [],
                                'accuracy': [],
                                'recall': [],
                                'precision': [],
                                'F1_score': []})
        # Deifining cross validation instance
        skf = StratifiedKFold(n_splits=k)

        print(f"Iteration {i}/{max_iterations}.")
        for n in n_samples:
            # Creating dataframes
            train_df, test_df, mapper = create_dataframe(how_many=n)
            train_dp = DataPreprocessor(train_df)
            for hop_win in hop_wins:
                for len_win in len_wins:
                    for M in n_mfcc:
                        # Processing data
                        X, y = train_dp.mfcc(M, S=len_win, R=hop_win)
                        for n_iter in n_iters:
                            for n_hidden in n_hiddens:
                                # Modelling
                                model = IwrGaussianHMMModel(n_states=n_hidden, n_iter=n_iter)
                                # Defining result dicts from cross validation
                                cv_scores = defaultdict(list)
                                for train_index, test_index in skf.split(X, y):
                                    X_train, X_test = X[train_index], X[test_index]
                                    y_train, y_test = y[train_index], y[test_index]
                                    model.fit(X_train, y_train)
                                    self._process_cv_results(cv_scores, model.score(X_test, y_test))
                                # Raporting progress with each 5 iterations
                                if i % 5 == 0:
                                    print(f"Iteration {i}/{max_iterations}.")
                                i += 1
                                temp = self._process_params(n, M, hop_win, len_win, n_iter, n_hidden, cv_scores)
                                results = results.append(temp)

        results.reset_index(drop=True, inplace=True)
        return results

    def _process_cv_results(self, cv_scores: DefaultDict[str, List], scores: Dict[str, float]) -> None:
        """
        function updates solutions from cross validation to default dicts

        Parameters
        ----------
        cv_scores : defaultdict
            maps score --> list of results
        scores : dict
            maps score --> result

        Returns
        -------
            None
        """
        scoring = ['accuracy', 'recall', 'precision', 'f1']
        for name in scoring:
            cv_scores[name].append(scores[name])

    def _process_params(self, *args, which_grid: Optional[str] = 'model') -> pd.DataFrame:
        """
        function structures results in dataframes

        Parameters
        ----------
        args : tuple
            arguments from simulation. Length of args depends from which_grid param
        which_grid : str
            which type of grid is currently used. Possible options ["model", "randomness"]

        Returns
        -------
            pd.DataFrame
                each row is model param combination and scores
        """
        scoring = ['accuracy', 'recall', 'precision', 'f1']
        if which_grid == 'model':
            scores_ = [args[6][name] for name in scoring]
            return pd.DataFrame({'n_samples': [args[0]],
                                 'n_mfcc': [args[1]],
                                 'hop_win': [args[2]],
                                 'len_win': [args[3]],
                                 'n_iter': [args[4]],
                                 'n_hidden': [args[5]],
                                 'accuracy': [scores_[0]],
                                 'recall': [scores_[1]],
                                 'precision': [scores_[2]],
                                 'F1_score': [scores_[3]]})
        elif which_grid == 'randomness':
            scores_ = [args[4][name] for name in scoring]
            return pd.DataFrame({'random_horizontal_shift': [args[0]],
                                 'random_vertical_scaling': [args[1]],
                                 'n_iter': [args[2]],
                                 'n_hiddens': [args[3]],
                                 'accuracy': [scores_[0]],
                                 'recall': [scores_[1]],
                                 'precision': [scores_[2]],
                                 'F1_score': [scores_[3]]})

    def _find_rhs_rvs(self, path: str) -> Tuple[float, float]:
        """
        function find rhs and rvs parameters from specified path

        Parameters
        ----------
        path : str
            path to specific dataset, it should contain rhs and rvs numbers

        Returns
        -------
            rvs : float
                random vertical scaling
            rhs : float
                random horisontal shift
        """
        # If it is default dataset, then the rhs=0.1 and rvs=1
        if path == 'data':
            return 0.1, 1.0
        # path is in patern like data\random\data_rhs_<number>_<number>_rvs_<number>_<number>
        # we want to extract <number>s, remove underscores and convert them to floats
        rhs_end_index = path.find('rhs') + 4
        rvs_start_index = path.find('rvs')
        # in case to convert to float we need to replace _ with .
        rhs = float(path[rhs_end_index: rvs_start_index - 1].replace('_', '.'))
        rvs = float(path[rvs_start_index + 4:].replace('_', '.'))
        return rhs, rvs


class Noiser(GridSearch):
    """
    Noiser class, children of GridSearch
    Used for experiments combined with noised signals

     Attributes
    ----------
    random_state : int
        seed which specifies random seed

    Methods
    -------
    noise_hmm_results
        function calculates results of model with specific parameters
        for test sets with different noises
    _process_params
        function structures results in dataframes
    best_noise_on_noise_grid_search
        experiment depends on model trained on concrete noise get better results than base model.
        The aim is to find optimal noise params to train model on.
    """
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(random_state)
        if not random_state:
            self.random_state = random_state
        else:
            self.random_state = None

        if not self.random_state:
            np.random.seed(self.random_state)

    def noise_hmm_results(self, params: Dict[str, List],
                          path: Optional[str] = 'data',
                          train_on_noise: Optional[bool] = False,
                          beta_noise: Optional[int] = 0,
                          snr_noise: Optional[int] = 8,
                          verbose: Optional[int] = 1) -> pd.DataFrame:
        """
        function calculates results of model with specific parameters
        for test sets with different noises

        Parameters
        ----------
        params : dict
            maps name_of_parameter -> possible  values
        path : str
            path to dataset on which model should be trained
        train_on_noise : bool
            if model should be trained on noise observation or not
        beta_noise : int
            order of noise on which model should be trained. Used only if train_on_noise=True.
        snr_noise : int
            signal noise ration on which model should be trained. Used only if train_on_noise=True.
        verbose : int
            controls if raport should be printed or not

        Returns
        -------
            results : pd.DataFrame
                each row is model param combination and scores
        """
        # Setting random state
        if not self.random_state:
            np.random.seed(self.random_state)

        # iteration of model
        max_iterations = np.prod([len(i) for i in params.values()])
        i = 1
        # unpacking params
        SNR = params['SNR']
        beta = params['beta']
        n_iters = params['n_iters']
        n_hiddens = params['n_hiddens']

        results = pd.DataFrame({'beta': [],
                                'SNR': [],
                                'n_iter': [],
                                'n_hidden': [],
                                'accuracy': [],
                                'recall': [],
                                'precision': [],
                                'F1_score': []})

        # If raport should be printed
        if verbose:
            print(f"Iteration {i}/{max_iterations}.")
        # creating dataframes
        train_df, test_df, mapper = create_dataframe(dir_name=path, how_many=100)
        train_dp = DataPreprocessor(train_df)
        test_dp = DataPreprocessor(test_df)
        # If we want to train model on noised data
        if train_on_noise:
            X_train, y_train = train_dp.mfcc_noised(13, beta_noise, snr_noise, S=0.025, R=0.01)
        # Training on clean observations
        else:
            X_train, y_train = train_dp.mfcc(13, S=0.025, R=0.01)
        for n_hidden in n_hiddens:
            for n_iter in n_iters:
                # Modelling
                model = IwrGaussianHMMModel(n_states=n_hidden, n_iter=n_iter)
                model.fit(X_train, y_train)
                for b in beta:
                    for snr in SNR:
                        # Printing progress each 5 iterations
                        if i % 5 == 0 and verbose:
                            print(f"Iteration {i}/{max_iterations}.")
                        i += 1
                        # Testing model on noised observations
                        X_test, y_test = test_dp.mfcc_noised(13, b, snr, S=0.025, R=0.01)
                        scores = model.score(X_test, y_test)
                        temp = self._process_params(b, snr, n_iter, n_hidden, scores, which_grid='noise')
                        results = results.append(temp)

        results.reset_index(drop=True, inplace=True)
        return results

    def _process_params(self, *args, which_grid: Optional[str] = 'model') -> pd.DataFrame:
        """
                function structures results in dataframes

                Parameters
                ----------
                args : tuple
                    arguments from simulation. Length of args depends from which_grid param
                which_grid : str
                    which type of grid is currently used. Possible options ["model", "randomness", "noise"]

                Returns
                -------
                    pd.DataFrame
                        each row is model param combination and scores
                """
        if not self.random_state:
            np.random.seed(self.random_state)
        scoring = ['accuracy', 'recall', 'precision', 'f1']
        if which_grid == 'model':
            scores_ = [args[6][name] for name in scoring]
            return pd.DataFrame({'n_samples': [args[0]],
                                 'n_mfcc': [args[1]],
                                 'hop_win': [args[2]],
                                 'len_win': [args[3]],
                                 'n_iter': [args[4]],
                                 'n_hidden': [args[5]],
                                 'accuracy': [scores_[0]],
                                 'recall': [scores_[1]],
                                 'precision': [scores_[2]],
                                 'F1_score': [scores_[3]]})
        elif which_grid == 'randomness':
            scores_ = [args[4][name] for name in scoring]
            return pd.DataFrame({'random_horizontal_shift': [args[0]],
                                 'random_vertical_scaling': [args[1]],
                                 'n_iter': [args[2]],
                                 'n_hiddens': [args[3]],
                                 'accuracy': [scores_[0]],
                                 'recall': [scores_[1]],
                                 'precision': [scores_[2]],
                                 'F1_score': [scores_[3]]})
        elif which_grid == 'noise':
            scores_ = [args[4][name] for name in scoring]
            return pd.DataFrame({'beta': [args[0]],
                                 'SNR': [args[1]],
                                 'n_iter': [args[2]],
                                 'n_hidden': [args[3]],
                                 'accuracy': [scores_[0]],
                                 'recall': [scores_[1]],
                                 'precision': [scores_[2]],
                                 'F1_score': [scores_[3]]})

    def best_noise_on_noise_grid_search(self, params: Dict[str, List]) -> Tuple[pd.Series, pd.Series]:
        """
        experiment depends on model trained on concrete noise get better results than base model.
        The aim is to find optimal noise params to train model on.

        Parameters
        ----------
        params : dict
            maps name_of_parameter -> possible  values

        Returns
        -------
        best : pd.Series
            model trained on best combination (beta, snr) noised data which was better than based model
        cost : pd.Series
            comparison of models based on noised data with the base model
        """
        print('Training base result...')
        # results for base model
        base_results = self.noise_hmm_results(params, verbose=0)
        # unpacking params
        chosen_snr = params['chosen_snr']
        beta = params['beta']
        comparison = base_results[['accuracy']].copy()

        # calculating number of iterations
        max_iterations = np.prod([len(i) for i in params.values()]) * len(beta)
        iteration_hop = max_iterations / len(beta) / len(chosen_snr)
        i = 0
        print(f"Iteration {i}/{max_iterations}.")
        for b in beta:
            for snr in chosen_snr:
                # measuring times
                t1 = time.time()
                # result for trained on noise model
                result = self.noise_hmm_results(params,
                                                path='data',
                                                train_on_noise=True,
                                                beta_noise=b,
                                                snr_noise=snr,
                                                verbose=0)
                t2 = time.time()
                model = f"beta_{b}_snr_{snr}"
                # Saving results
                path = os.path.join('results', f'result_noise_{model}.csv')
                result.to_csv(path)
                # If model is better than base depends on the difference in accuracy solutions
                comparison[model] = result['accuracy'] - base_results['accuracy']
                i += iteration_hop
                t = t2 - t1
                # printing raport
                print(f"Iteration {int(i)}/{max_iterations}. Evaluation time: {t // 60:.0f} min {t % 60:.2f} s")
        # We create comparison dataframe based on accuracy column, we drop it
        comparison = comparison.drop(['accuracy'], axis=1)
        # Calculating how many times models were better than base
        cost = (comparison > 0).sum()
        # selecting best model
        best = cost.nlargest(1)
        return best, cost
