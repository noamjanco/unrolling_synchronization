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

def task(N, R, Lambda, DEPTH, seed, epochs):
    # Initialize random seed
    np.random.seed(seed)
    s, Y = generate_training_data_so3(N,Lambda,R)
    s_val, Y_val = generate_training_data_so3(N,Lambda,R)

    model = BuildModel(N, Lambda, DEPTH)
    s_init = initialize_batch_matrix(R,N)
    s_init2 = initialize_batch_matrix(R,N)
    s_val_init = initialize_batch_matrix(R,N)
    s_val_init2 = initialize_batch_matrix(R,N)
    TrainModel(model, Y, s, s_init, s_init2, Y_val,s_val,s_val_init,s_val_init2,epochs)


    # Generate test data
    s, Y = generate_training_data_so3(N,Lambda,R)
    s_init = initialize_batch_matrix(R,N)
    s_init2 = initialize_batch_matrix(R,N)
    x_est, loss_nn = EvaluateModel(model, Y, s, s_init, s_init2)

    z_total = []
    for r in range(R):
        z,num_iter = ppm_so3(Y[r], z_init=s_init[r,:], num_iterations=DEPTH)
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

class CompareSO3UnrollingExperiment(Experiment):
    def __init__(self, params: dict):
        super().__init__(params=params)

    def run_experiment(self):
        N = self.params['N']
        R = self.params['R'] # Number of training batches
        depth_range = self.params['depth_range']
        Lambda = self.params['Lambda']
        num_trials = self.params['num_trials']
        epochs = self.params['epochs']
        df = pd.DataFrame()
        for d in depth_range:
            for t in range(num_trials):
                try:
                    print('running N=',N,'R=',R,'Lambda=',Lambda,'Depth= ',d)
                    loss_ppm, loss_pim, loss_amp, loss_nn = task(N=N, R=R, Lambda=Lambda, DEPTH=d, seed=t, epochs=epochs)
                    df = df.append({'loss_ppm': loss_ppm,
                                    'loss_pim': loss_pim,
                                    'loss_amp': loss_amp,
                                    'loss_nn': loss_nn,
                                    'DEPTH': d,
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
        depth_range = self.params['depth_range']
        avg_loss_ppm_vs_d = [np.mean(df[df.DEPTH == x].loss_ppm) for x in depth_range]
        avg_loss_pim_vs_d = [np.mean(df[df.DEPTH == x].loss_pim) for x in depth_range]
        avg_loss_amp_vs_d = [np.mean(df[df.DEPTH == x].loss_amp) for x in depth_range]
        avg_loss_nn_vs_d = [np.mean(df[df.DEPTH == x].loss_nn) for x in depth_range]
        plt.plot(depth_range, avg_loss_ppm_vs_d)
        plt.plot(depth_range, avg_loss_pim_vs_d)
        # plt.plot(depth_range, avg_loss_amp_vs_d)
        plt.plot(depth_range, avg_loss_nn_vs_d)
        plt.legend(['PPM', 'PIM', 'NN'])
        plt.xlabel('Depth')
        plt.ylabel('Mean Error')
        plt.savefig(self.get_results_path()+'.eps')
        plt.savefig(self.get_results_path()+'.png')
        plt.clf()


if __name__ == '__main__':
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 1000, 'num_trials': 1, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 200, 'Lambda': 1.2})
    CompareSO3UnrollingExperiment(params={'N': 20, 'R': 1000, 'num_trials': 1, 'depth_range': [1,3,5,9,15,20,50], 'epochs': 200, 'Lambda': 1.2})
    # CompareGaussianUnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 300, 'Lambda': 1.5, 'L': 10})
