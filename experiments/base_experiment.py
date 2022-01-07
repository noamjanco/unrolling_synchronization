import pandas as pd
import os.path
from pathlib import Path


BASE_PATH = 'C:/Users/Noam/Desktop/University/Research/MRA/unrolling_synchronization/results/'


class Experiment:
    def __init__(self, params: dict):
        self.params = params
        self.results = pd.DataFrame({})
        self.results_path = ''
        self.set_results_path()

        rerun_experiment = self.params['rerun_experiment'] if 'rerun_experiment' in self.params.keys() else False

        if not self.results_exist() or rerun_experiment:
            self.run_experiment()
            self.save_results()
        else:
            self.load_results()
        self.plot_results()

    def set_results_path(self):
        experiment_path =  BASE_PATH + '/' + self.__class__.__name__ + '/'
        Path(experiment_path).mkdir(parents=True, exist_ok=True)
        param_str = ''
        for key, val in zip(self.params.keys(), self.params.values()):
            param_str += str(key) + '_' + str(val) + '_'
        param_str = param_str.replace('[','')
        param_str = param_str.replace(']','')
        param_str = param_str.replace(' ','_')

        self.results_path = experiment_path + param_str

    def get_results_path(self) -> str:
        return self.results_path

    def results_exist(self):
        return os.path.isfile(self.get_results_path()+'.pickle')

    def run_experiment(self):
        raise NotImplementedError

    def save_results(self):
        self.results.to_pickle(self.get_results_path()+'.pickle')

    def load_results(self):
        self.results = pd.read_pickle(self.get_results_path()+'.pickle')

    def plot_results(self):
        raise NotImplementedError
