import numpy as np
import pandas as pd

from common.math_utils import initialize_matrix
from data_generation.generate_data_so3 import generate_training_data_so3, generate_training_data_so3_with_j_ambiguity
from experiments.base_experiment import Experiment
from models.unrolling_j_synchronization import BuildModel, TrainModel, EvaluateModel, loss_z_over_2
from synchronization.ppm import ppm_so3
from synchronization.pim import pim_so3
import matplotlib.pyplot as plt

def initialize_batch_matrix(R: int, N: int) -> np.ndarray:
    return np.asarray([initialize_matrix(N) for _ in range(R)])

def task(N, R, Lambda, DEPTH, seed, epochs):
    # Initialize random seed
    np.random.seed(seed)
    s, Y, j_gt = generate_training_data_so3_with_j_ambiguity(N,Lambda,R)
    s_val, Y_val, j_gt_val = generate_training_data_so3_with_j_ambiguity(N,Lambda,R)

    batchsize = 10
    model = BuildModel(N, DEPTH, batchsize)
    s_init = initialize_batch_matrix(R,N)
    s_init2 = initialize_batch_matrix(R,N)
    s_val_init = initialize_batch_matrix(R,N)
    s_val_init2 = initialize_batch_matrix(R,N)
    TrainModel(model, Y, s, s_init, s_init2, Y_val,s_val,s_val_init,s_val_init2,epochs, batchsize)



    # --
    # train without j ambiguity
    s2, Y2 = generate_training_data_so3(N,Lambda,R)
    s_val2, Y_val2 = generate_training_data_so3(N,Lambda,R)

    model2 = BuildModel(N, Lambda, DEPTH)
    s_init2 = initialize_batch_matrix(R,N)
    s_init22 = initialize_batch_matrix(R,N)
    s_val_init_ = initialize_batch_matrix(R,N)
    s_val_init2_ = initialize_batch_matrix(R,N)
    TrainModel(model2, Y2, s2, s_init2, s_init22, Y_val2,s_val2,s_val_init_,s_val_init2_,epochs)
    # ----
    # Generate test data
    s, Y = generate_training_data_so3_with_j_ambiguity(N,Lambda,R)
    s_init = initialize_batch_matrix(R,N)
    s_init2 = initialize_batch_matrix(R,N)
    x_est, loss_nn = EvaluateModel(model, Y, s, s_init, s_init2)
    print('Loss without J-ambiguity in training data')
    x_est_without, loss_nn_without = EvaluateModel(model2, Y, s, s_init, s_init2)


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



    return loss_ppm, loss_pim, None, loss_nn, loss_nn_without

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
                    loss_ppm, loss_pim, loss_amp, loss_nn, loss_nn_without_j_ambiguity = task(N=N, R=R, Lambda=Lambda, DEPTH=d, seed=t, epochs=epochs)
                    df = df.append({'loss_ppm': loss_ppm,
                                    'loss_pim': loss_pim,
                                    'loss_amp': loss_amp,
                                    'loss_nn': loss_nn,
                                    'loss_nn_without_j_ambiguity': loss_nn_without_j_ambiguity,
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
        avg_loss_nn_without_j_ambiguity_vs_d = [np.mean(df[df.DEPTH == x].loss_nn_without_j_ambiguity) for x in depth_range]
        plt.plot(depth_range, avg_loss_ppm_vs_d)
        plt.plot(depth_range, avg_loss_pim_vs_d)
        # plt.plot(depth_range, avg_loss_amp_vs_d)
        plt.plot(depth_range, avg_loss_nn_vs_d, color=u'#d62728')
        plt.plot(depth_range, avg_loss_nn_without_j_ambiguity_vs_d, color=u'#d62728')
        plt.legend(['PPM', 'spectral method', 'unrolled algorithm', 'unrolled algorithm without j ambiguity'],fontsize=16,loc='upper right')
        plt.xlabel('Depth / Iterations',fontsize=16)
        plt.ylabel('Mean Error',fontsize=16)
        plt.savefig(self.get_results_path()+'.eps')
        plt.savefig(self.get_results_path()+'.png')
        plt.clf()


if __name__ == '__main__':
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 1000, 'num_trials': 1, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 200, 'Lambda': 1.2})
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1,3,5,9,15,20,50], 'epochs': 500, 'Lambda': 1.5})
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1,3,5,9,15,20], 'epochs': 500, 'Lambda': 1.2})
    CompareSO3UnrollingJSynchExperiment(params={'N': 20, 'R': 100, 'num_trials': 1, 'depth_range': [9], 'epochs': 100, 'Lambda': 2.})
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1,3,5,9,15,20], 'epochs': 100, 'Lambda': 1.5})
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [9], 'epochs': 20, 'Lambda': 1.5})
    # CompareSO3UnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [9], 'epochs': 10, 'Lambda': 1.2})
    # CompareGaussianUnrollingExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 300, 'Lambda': 1.5, 'L': 10})
