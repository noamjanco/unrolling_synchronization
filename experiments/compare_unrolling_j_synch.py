import numpy as np
import pandas as pd

from common.math_utils import initialize_matrix
from data_generation.generate_data_so3 import generate_training_data_so3, generate_training_data_so3_with_j_ambiguity
from experiments.base_experiment import Experiment
from experiments.j_synchronization_forward import j_synch_forward
from models.unrolling_j_synchronization import BuildModel, TrainModel, EvaluateModel, loss_z_over_2
from synchronization.ppm import ppm_so3
from synchronization.pim import pim_so3
import matplotlib.pyplot as plt

def initialize_batch_matrix(R: int, N: int) -> np.ndarray:
    return np.asarray([initialize_matrix(N) for _ in range(R)])

def initialize_j_est(R: int, N: int) -> np.ndarray:
    return 1e-3*np.ones((R, int(N*(N-1)/2), 1))

def task(N, R, Lambda, DEPTH, seed, epochs):
    # Initialize random seed
    np.random.seed(seed)
    s, Y, j_gt = generate_training_data_so3_with_j_ambiguity(N,Lambda,R)
    s_val, Y_val, j_gt_val = generate_training_data_so3_with_j_ambiguity(N,Lambda,R)

    batchsize = 10
    model = BuildModel(N, DEPTH, batchsize)
    # s_init = initialize_batch_matrix(R,N)
    # s_init2 = initialize_batch_matrix(R,N)
    # s_val_init = initialize_batch_matrix(R,N)
    # s_val_init2 = initialize_batch_matrix(R,N)
    # TrainModel(model, Y, s, s_init, s_init2, Y_val,s_val,s_val_init,s_val_init2,epochs, batchsize)

    # ----
    # Generate test data
    s, Y, j_gt = generate_training_data_so3_with_j_ambiguity(N,Lambda,batchsize)
    j_init = initialize_j_est(batchsize,N)
    x_est, loss_nn = EvaluateModel(model, Y, j_gt, j_init)
    print('[Learend J-Synch] loss = %f' % loss_nn)

    _, u_s = j_synch_forward(Y)
    loss_j_synch = loss_z_over_2(j_gt, u_s)
    print('[J-Synch] loss = %f' % loss_j_synch)


    return loss_nn, loss_j_synch

class CompareSO3UnrollingJSynchExperiment(Experiment):
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
                    loss_nn, loss_j_synch = task(N=N, R=R, Lambda=Lambda, DEPTH=d, seed=t, epochs=epochs)
                    df = df.append({'loss_j_synch': loss_j_synch,
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
        avg_loss_loss_j_synch_vs_d = [np.mean(df[df.DEPTH == x].loss_j_synch) for x in depth_range]
        avg_loss_nn_vs_d = [np.mean(df[df.DEPTH == x].loss_nn) for x in depth_range]
        plt.plot(depth_range, avg_loss_loss_j_synch_vs_d, color=u'#d62728')
        plt.plot(depth_range, avg_loss_nn_vs_d, color=u'#d62728')
        plt.legend(['J-Synch', 'Learend J-Synch'],fontsize=16,loc='upper right')
        plt.xlabel('Depth / Iterations',fontsize=16)
        plt.ylabel('Mean Error',fontsize=16)
        plt.savefig(self.get_results_path()+'.eps')
        plt.savefig(self.get_results_path()+'.png')
        plt.clf()


if __name__ == '__main__':
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 1000, 'num_trials': 1, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 200, 'Lambda': 1.2})
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1,3,5,9,15,20,50], 'epochs': 500, 'Lambda': 1.5})
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1,3,5,9,15,20], 'epochs': 500, 'Lambda': 1.2})
    CompareSO3UnrollingJSynchExperiment(params={'N': 20, 'R': 100, 'num_trials': 1, 'depth_range': [9], 'epochs': 100, 'Lambda': 9.})
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1,3,5,9,15,20], 'epochs': 100, 'Lambda': 1.5})
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [9], 'epochs': 20, 'Lambda': 1.5})
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [9], 'epochs': 10, 'Lambda': 1.2})
    # CompareGaussianUnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 300, 'Lambda': 1.5, 'L': 10})
