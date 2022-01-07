from experiments.compare_gaussian_u_1 import CompareU1Experiment
from experiments.compare_z_over_2 import CompareZOver2Experiment
import numpy as np


if __name__ == '__main__':
    CompareZOver2Experiment(params={'N': 2000, 'R': 10, 'Lambda_range': np.arange(1.2, 3.2, 0.2), 'seed': 2})
    CompareU1Experiment(params={'N': 20, 'R': 100, 'Lambda_range': np.arange(.4,3.,0.2), 'seed': 1, 'L': 10, 'tol': 1e-3})

