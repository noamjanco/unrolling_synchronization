import numpy as np
import pandas as pd
import time

from common.math_utils import initialize_matrix
from data_generation.generate_data_so3 import generate_training_data_so3
from experiments.base_experiment import Experiment
from models.unrolling_synchronization_so3 import BuildModel, TrainModel, EvaluateModel, loss_so3
from synchronization.ppm import ppm_so3
from synchronization.pim import pim_so3
import matplotlib.pyplot as plt

def initialize_batch_matrix(R: int, N: int) -> np.ndarray:
    return np.asarray([initialize_matrix(N) for _ in range(R)])

def task(N, N_train, R, Lambda, DEPTH, seed, epochs):
    # Initialize random seed
    np.random.seed(seed)
    s, Y = generate_training_data_so3(N_train,Lambda,R)
    s_val, Y_val = generate_training_data_so3(N_train,Lambda,R)

    model = BuildModel(N_train, Lambda, DEPTH)
    s_init = initialize_batch_matrix(R,N_train)
    s_init2 = initialize_batch_matrix(R,N_train)
    s_val_init = initialize_batch_matrix(R,N_train)
    s_val_init2 = initialize_batch_matrix(R,N_train)
    TrainModel(model, Y, s, s_init, s_init2, Y_val,s_val,s_val_init,s_val_init2,epochs)


    # Generate test data
    model = BuildModel(N, Lambda, DEPTH)
    model.load_weights('tmp/checkpoint')
    s, Y = generate_training_data_so3(N,Lambda,R)
    s_init = initialize_batch_matrix(R,N)
    s_init2 = initialize_batch_matrix(R,N)
    x_est, loss_nn = EvaluateModel(model, Y, s, s_init, s_init2)

    z_total = []
    t0 = time.time()
    for r in range(R):
        z,num_iter = ppm_so3(Y[r], z_init=s_init[r,:], num_iterations=DEPTH)
        z_total.append(z)
    prediction_time = time.time() - t0
    print('PPM took ', prediction_time, ' seconds for batch size: ', len(Y))
    z1 = np.asarray(z_total)
    loss_ppm = loss_so3(s.astype(np.float32), z1.astype(np.float32))
    print('[PPM] loss = %f' % loss_ppm)

    z_total = []
    t0 = time.time()
    for r in range(R):
        z = pim_so3(Y[r])
        z_total.append(z)
    prediction_time = time.time() - t0
    print('PIM took ', prediction_time, ' seconds for batch size: ', len(Y))
    z1 = np.asarray(z_total)
    loss_pim = loss_so3(s.astype(np.float32), z1.astype(np.float32))
    print('[PIM] loss = %f' % loss_pim)



    return loss_ppm, loss_pim, None, loss_nn

class CompareSO3UnrollingDifferentNExperiment(Experiment):
    def __init__(self, params: dict):
        super().__init__(params=params)

    def run_experiment(self):
        N = self.params['N']
        N_train = self.params['N_train']
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
                    loss_ppm, loss_pim, loss_amp, loss_nn = task(N=N,N_train=N_train, R=R, Lambda=Lambda, DEPTH=d, seed=t, epochs=epochs)
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
        plt.plot(depth_range, avg_loss_ppm_vs_d ,'-o')
        plt.plot(depth_range, avg_loss_pim_vs_d, '-^')
        # plt.plot(depth_range, avg_loss_amp_vs_d)
        plt.plot(depth_range, avg_loss_nn_vs_d, '-*',color=u'#d62728')
        plt.legend(['PPM', 'spectral method', 'unrolled algorithm'],fontsize=16,loc='upper right')
        plt.xlabel('Depth / Iterations',fontsize=16)
        plt.ylabel('Mean Error',fontsize=16)
        plt.savefig(self.get_results_path()+'.eps')
        plt.savefig(self.get_results_path()+'.png')
        plt.clf()


if __name__ == '__main__':
    # CompareSO3UnrollingDifferentNExperiment(params={'N_train': 20,'N':40, 'R': 10000, 'num_trials': 1, 'depth_range': [9], 'epochs': 100, 'Lambda': 1.5})
    CompareSO3UnrollingDifferentNExperiment(params={'N_train': 40,'N':40, 'R': 10000, 'num_trials': 1, 'depth_range': [9], 'epochs': 100, 'Lambda': 1.5})

