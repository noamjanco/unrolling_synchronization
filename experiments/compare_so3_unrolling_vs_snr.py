import numpy as np
import pandas as pd

from common.math_utils import initialize_matrix
from data_generation.generate_data_so3 import generate_training_data_so3
from experiments.base_experiment import Experiment
from models.unrolling_synchronization_so3 import BuildModel, TrainModel, EvaluateModel, loss_so3
from synchronization.ppm import ppm_so3
from synchronization.pim import pim_so3
import matplotlib.pyplot as plt

def initialize_batch_matrix(R: int, N: int) -> np.ndarray:
    return np.asarray([initialize_matrix(N) for _ in range(R)])

def task(N, R, Lambda, DEPTH, seed, epochs, num_iterations):
    # Initialize random seed
    np.random.seed(seed)
    s, Y = generate_training_data_so3(N,Lambda,R)
    s_val, Y_val = generate_training_data_so3(N,Lambda,R)

    training_loss = []
    models = []
    for t in range(3):
        model = BuildModel(N, Lambda, DEPTH)
        s_init = initialize_batch_matrix(R,N)
        s_init2 = initialize_batch_matrix(R,N)
        s_val_init = initialize_batch_matrix(R,N)
        s_val_init2 = initialize_batch_matrix(R,N)
        TrainModel(model, Y, s, s_init, s_init2, Y_val,s_val,s_val_init,s_val_init2,epochs)
        models.append(model)
        x_est, loss_nn = EvaluateModel(model, Y_val, s_val, s_val_init, s_val_init2)
        training_loss.append(loss_nn)

    model = models[np.argmin(training_loss)]

    # Generate test data
    s, Y = generate_training_data_so3(N,Lambda,R)
    s_init = initialize_batch_matrix(R,N)
    s_init2 = initialize_batch_matrix(R,N)
    x_est, loss_nn = EvaluateModel(model, Y, s, s_init, s_init2)

    z_total = []
    for r in range(R):
        z,num_iter = ppm_so3(Y[r], z_init=s_init[r,:], num_iterations=num_iterations)
        z_total.append(z)
    z1 = np.asarray(z_total)
    loss_ppm = loss_so3(s.astype(np.float32), z1.astype(np.float32))
    print('[PPM] loss = %f' % loss_ppm)

    z_total = []
    for r in range(R):
        z = pim_so3(Y[r])
        z_total.append(z)
    z1 = np.asarray(z_total)
    loss_pim = loss_so3(s.astype(np.float32), z1.astype(np.float32))
    print('[PIM] loss = %f' % loss_pim)



    return loss_ppm, loss_pim, None, loss_nn

class CompareSO3UnrollingVsSNRExperiment(Experiment):
    def __init__(self, params: dict):
        super().__init__(params=params)

    def run_experiment(self):
        N = self.params['N']
        R = self.params['R'] # Number of training batches
        Lambda_range = self.params['Lambda_range']
        num_trials = self.params['num_trials']
        depth = self.params['depth']
        epochs = self.params['epochs']
        num_iterations = self.params['num_iterations']
        df = pd.DataFrame()
        for Lambda in Lambda_range:
            for t in range(num_trials):
                try:
                    print('running N=',N,'R=',R,'Lambda=',Lambda,'Depth= ',depth)
                    loss_ppm, loss_pim, loss_amp, loss_nn = task(N=N, R=R, Lambda=Lambda, DEPTH=depth, seed=t, epochs=epochs, num_iterations=num_iterations)
                    df = df.append({'loss_ppm': loss_ppm,
                                    'loss_pim': loss_pim,
                                    'loss_amp': loss_amp,
                                    'loss_nn': loss_nn,
                                    'DEPTH': depth,
                                    'trial': t,
                                    'Lambda': Lambda,
                                    'R': R,
                                    'N': N,
                                    'epochs': epochs}, ignore_index=True)
                except Exception as e:
                    print(e)
        self.results = df

    def plot_results(self):
        df = self.results
        Lambda_range = self.params['Lambda_range']
        avg_loss_ppm_vs_d = [np.mean(df[df.Lambda == x].loss_ppm) for x in Lambda_range]
        avg_loss_pim_vs_d = [np.mean(df[df.Lambda == x].loss_pim) for x in Lambda_range]
        avg_loss_amp_vs_d = [np.mean(df[df.Lambda == x].loss_amp) for x in Lambda_range]
        avg_loss_nn_vs_d = [np.mean(df[df.Lambda == x].loss_nn) for x in Lambda_range]
        plt.plot(Lambda_range, avg_loss_ppm_vs_d)
        plt.plot(Lambda_range, avg_loss_pim_vs_d)
        # plt.plot(depth_range, avg_loss_amp_vs_d)
        plt.plot(Lambda_range, avg_loss_nn_vs_d)
        plt.legend(['PPM', 'PIM', 'NN'])
        plt.xlabel('SNR')
        plt.ylabel('Mean Error')
        plt.savefig(self.get_results_path()+'_vs_snr.eps')
        plt.savefig(self.get_results_path()+'_vs_snr.png')
        plt.clf()


if __name__ == '__main__':
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 1000, 'num_trials': 1, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 200, 'Lambda': 1.2})
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1,3,5,9,15,20,50], 'epochs': 500, 'Lambda': 1.5})
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1,3,5,9,15,20], 'epochs': 500, 'Lambda': 1.2})
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1,3,5,9,15,20], 'epochs': 100, 'Lambda': 1.2})
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1,3,5,9,15,20], 'epochs': 100, 'Lambda': 1.5})
    # CompareSO3UnrollingVsSNRExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'Lambda_range': np.arange(0.25,3.5,0.25), 'depth': 9, 'epochs': 20, 'num_iterations': 100})
    # CompareSO3UnrollingVsSNRExperiment(params={'N': 20, 'R': 1000, 'num_trials': 1, 'Lambda_range': np.arange(0.25,3.5,0.25), 'depth': 9, 'epochs': 20, 'num_iterations': 100})
    CompareSO3UnrollingVsSNRExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'Lambda_range': np.arange(0.25,3.5,0.25), 'depth': 9, 'epochs': 100, 'num_iterations': 100})
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [9], 'epochs': 10, 'Lambda': 1.2})
    # CompareGaussianUnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 300, 'Lambda': 1.5, 'L': 10})
