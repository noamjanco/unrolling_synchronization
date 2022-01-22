from experiments.compare_antipodal_unrolling import CompareAntipodalUnrollingExperiment
from experiments.compare_gaussian_u_1 import CompareU1Experiment
from experiments.compare_gaussian_unrolling import CompareGaussianUnrollingExperiment
from experiments.compare_z_over_2 import CompareZOver2Experiment
from experiments.compare_z_over_2_unrolling import CompareZOver2UnrollingExperiment
import numpy as np


if __name__ == '__main__':
    CompareZOver2Experiment(params={'N': 2000, 'R': 10, 'Lambda_range': np.arange(1.2, 3.2, 0.2), 'seed': 2})
    CompareU1Experiment(params={'N': 20, 'R': 10, 'Lambda_range': np.arange(.4,3.,0.2), 'seed': 1, 'L': 10, 'tol': 1e-3})
    CompareZOver2UnrollingExperiment(params={'N': 20, 'R': 20000, 'num_trials': 1, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 300, 'Lambda': 1.25})
    CompareZOver2UnrollingExperiment(params={'N': 20, 'R': 20000, 'num_trials': 1, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 300, 'Lambda': 1.5})
    CompareZOver2UnrollingExperiment(params={'N': 20, 'R': 20000, 'num_trials': 1, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 300, 'Lambda': 2})
    CompareAntipodalUnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 5, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 300, 'Lambda': 0.3, 'L': 21})
    CompareGaussianUnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 300, 'Lambda': 1.2, 'L': 10})
    CompareGaussianUnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 300, 'Lambda': 1.5, 'L': 10})


