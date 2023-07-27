import numpy as np
import pandas as pd
from data_generation.generate_data import generate_training_data_antipodal_reconstruction
from experiments.base_experiment import Experiment
from models.unrolling_synchronization_antipodal_reconstruction_loss_mlp import BuildModel, TrainModel, EvaluateModel, loss_z_over_2, reconstruction_loss_z_over_2
from synchronization.ppm import ppm_z_over_2
from synchronization.pim import pim_z_over_2
from synchronization.amp import amp_z_over_2
import matplotlib.pyplot as plt
import tensorflow as tf


def task(N, R, Lambda, DEPTH, seed, epochs, L):
    #todo: test with reconstruction_loss_z_over_2
    # Initialize random seed
    np.random.seed(seed)
    v, Y, y, x = generate_training_data_antipodal_reconstruction(N,Lambda,R, L)

    model = BuildModel(L, N,Lambda,DEPTH)
    x_init = 1e-1*np.expand_dims(np.random.rand(R,N),axis=-1)
    x_init2 = 1e-1*np.expand_dims(np.random.rand(R,N),axis=-1)
    v_val, Y_val, y_val, x_val  = generate_training_data_antipodal_reconstruction(N,Lambda,R, L)
    x_val_init = 1e-1 * np.expand_dims(np.random.rand(R, N), axis=-1)
    x_val_init2 = 1e-1 * np.expand_dims(np.random.rand(R, N), axis=-1)
    TrainModel(model, Y,y, x, x_init,x_init2,Y_val,y_val, x_val, x_val_init,x_val_init2, epochs)

    # Generate test data
    v, Y, y, x = generate_training_data_antipodal_reconstruction(N,Lambda,R, L)
    x_init = 1e-1*np.expand_dims(np.random.rand(R,N),axis=-1)
    x_init2 = 1e-1*np.expand_dims(np.random.rand(R,N),axis=-1)
    x_est, loss_nn = EvaluateModel(model, Y,y, x, x_init, x_init2)

    z_total = []
    for r in range(R):
        z, num_iter = ppm_z_over_2(Y[r], x_init[r,:], max_iterations=DEPTH)
        z_total.append(z)
    z1 = np.asarray(z_total)
    x_est = tf.reduce_mean(y * tf.repeat(tf.expand_dims(tf.squeeze(z1, axis=-1), axis=1), L, axis=1), axis=-1)
    loss_ppm = reconstruction_loss_z_over_2(x.astype(np.float32), x_est)
    # loss_ppm = loss_z_over_2(x.astype(np.float32), x_est)

    print('[PPM] loss = %f' % loss_ppm)

    z_total = []
    for r in range(R):
        z, num_iter = pim_z_over_2(Y[r], x_init[r, :], max_iterations=DEPTH)
        z_total.append(z)
    z1 = np.asarray(z_total)
    x_est = tf.reduce_mean(y * tf.repeat(tf.expand_dims(tf.squeeze(z1, axis=-1), axis=1), L, axis=1), axis=-1)
    loss_pim = reconstruction_loss_z_over_2(x.astype(np.float32), x_est)
    # loss_pim = loss_z_over_2(x.astype(np.float32), x_est)
    print('[PIM] loss = %f' % loss_pim)

    z_total = []
    for r in range(R):
        z, num_iter = amp_z_over_2(Y[r], x_init[r,:], x_init2[r,:], Lambda, max_iterations=DEPTH)
        z_total.append(z)
    z1 = np.asarray(z_total)
    x_est = tf.reduce_mean(y * tf.repeat(tf.expand_dims(tf.squeeze(z1, axis=-1), axis=1), L, axis=1), axis=-1)
    loss_amp = reconstruction_loss_z_over_2(x.astype(np.float32), x_est)
    # loss_amp = loss_z_over_2(x.astype(np.float32), x_est)
    print('[AMP] loss = %f' % loss_amp)

    return loss_ppm, loss_pim, loss_amp, loss_nn

class CompareAntipodalUnrollingReconstructionLossMLPExperiment(Experiment):
    def __init__(self, params: dict):
        super().__init__(params=params)

    def run_experiment(self):
        N = self.params['N']
        R = self.params['R'] # Number of training batches
        depth_range = self.params['depth_range']
        Lambda = self.params['Lambda']
        num_trials = self.params['num_trials']
        epochs = self.params['epochs']
        L = self.params['L']
        df = pd.DataFrame()
        for d in depth_range:
            for t in range(num_trials):
                loss_ppm, loss_pim, loss_amp, loss_nn = task(N=N, R=R, Lambda=Lambda, DEPTH=d, seed=t, epochs=epochs, L=L)
                df = df.append({'loss_ppm': loss_ppm,
                                'loss_pim': loss_pim,
                                'loss_amp': loss_amp,
                                'loss_nn': loss_nn,
                                'DEPTH': d,
                                'trial': t,
                                'Lambda': Lambda,
                                'R': R,
                                'N': N,
                                'L': L,
                                'epochs': epochs}, ignore_index=True)
        self.results = df

    def plot_results(self):
        df = self.results
        depth_range = self.params['depth_range']
        avg_loss_ppm_vs_d = [np.mean(df[df.DEPTH == x].loss_ppm) for x in depth_range]
        avg_loss_pim_vs_d = [np.mean(df[df.DEPTH == x].loss_pim) for x in depth_range]
        avg_loss_amp_vs_d = [np.mean(df[df.DEPTH == x].loss_amp) for x in depth_range]
        avg_loss_nn_vs_d = [np.mean(df[df.DEPTH == x].loss_nn) for x in depth_range]
        plt.plot(depth_range, avg_loss_ppm_vs_d, '-o')
        plt.plot(depth_range, avg_loss_pim_vs_d, '-^')
        plt.plot(depth_range, avg_loss_amp_vs_d, '-s')
        plt.plot(depth_range, avg_loss_nn_vs_d, '-*')
        plt.legend(['PPM', 'PM', 'AMP', 'unrolled algorithm'],fontsize=16, loc='upper right')
        plt.xlabel('Depth / Iterations', fontsize=16)
        plt.ylabel('Mean Error', fontsize=16)
        plt.savefig(self.get_results_path()+'.eps')
        plt.savefig(self.get_results_path()+'.png')
        plt.clf()


if __name__ == '__main__':
    # CompareAntipodalUnrollingReconstructionLossExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 300, 'Lambda': 0.4, 'L': 21})
    # CompareAntipodalUnrollingReconstructionLossExperiment(params={'N': 20, 'R': 10000, 'num_trials': 5, 'depth_range': [1, 3, 5, 9, 15, 20, 50], 'epochs': 300, 'Lambda': 0.4, 'L': 21})
    CompareAntipodalUnrollingReconstructionLossMLPExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [5], 'epochs': 2000, 'Lambda': 0.8, 'L': 21})
    # CompareAntipodalUnrollingReconstructionLossExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [9], 'epochs': 20, 'Lambda': 0.4, 'L': 21})
