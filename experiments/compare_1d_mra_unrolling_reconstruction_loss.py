import numpy as np
import pandas as pd

from common.math_utils import align_in_fourier
from data_generation.generate_data_1d_mra import generate_training_data_1d_mra
from experiments.base_experiment import Experiment
from models.unrolling_synchronization_1d_mra_reconstruction_loss import BuildModel, TrainModel, EvaluateModel, reconstruction_loss_u_1
from synchronization.ppm import ppm_u_1
from synchronization.pim import pim_u_1
from synchronization.amp import amp_u_1
import matplotlib.pyplot as plt
import tensorflow as tf


def task(N, R, Lambda, DEPTH, seed, epochs, L):
    # Initialize random seed
    np.random.seed(seed)
    s, H, v, y_fft, x_fft = generate_training_data_1d_mra(N, Lambda, R, L)

    models = []
    training_loss = []
    # for t in range(5):
    for t in range(1):
        model = BuildModel(L, N, Lambda, DEPTH)
        x_init = 1e-2 * (np.expand_dims(np.random.rand(R, N), axis=-1) + 1j * np.expand_dims(np.random.rand(R, N), axis=-1))
        x_init2 = 1e-2 * (
        np.expand_dims(np.random.rand(R, N), axis=-1) + 1j * np.expand_dims(np.random.rand(R, N), axis=-1))
        s_val, Y_val, v_val, y_fft_val, x_fft_val = generate_training_data_1d_mra(N, Lambda, R, L)
        x_val_init = 1e-2 * (
        np.expand_dims(np.random.rand(R, N), axis=-1) + 1j * np.expand_dims(np.random.rand(R, N), axis=-1))
        x_val_init2 = 1e-2 * (
        np.expand_dims(np.random.rand(R, N), axis=-1) + 1j * np.expand_dims(np.random.rand(R, N), axis=-1))
        TrainModel(model, H.real, H.imag, y_fft, x_fft, x_init.real, x_init.imag,
                   x_init2.real, x_init2.imag, Y_val.real, Y_val.imag, y_fft_val, x_fft_val,
                   x_val_init.real, x_val_init.imag, x_val_init2.real, x_val_init2.imag, epochs)
        models.append(model)
        x_est, loss_nn = EvaluateModel(model, H, y_fft, x_fft, x_init, x_init2)
        training_loss.append(loss_nn)

    # choose model with minimal loss
    print('@@')
    print('models loss:')
    print(training_loss)
    print('chosen model: %d' % (np.argmin(training_loss)))

    model = models[np.argmin(training_loss)]

    # Generate test data
    s, H, v, y_fft, x_fft = generate_training_data_1d_mra(N, Lambda, R, L)
    x_init = 1e-2 * (np.expand_dims(np.random.rand(R, N), axis=-1) + 1j * np.expand_dims(np.random.rand(R, N), axis=-1))
    x_init2 = 1e-2 * (
    np.expand_dims(np.random.rand(R, N), axis=-1) + 1j * np.expand_dims(np.random.rand(R, N), axis=-1))
    x_est, loss_nn = EvaluateModel(model, H, y_fft ,x_fft, x_init, x_init2)
    x_fft_true = tf.concat([x_fft.real.astype(np.float32), x_fft.imag.astype(np.float32)], axis=-1)

    z_total = []
    x_total = []
    for r in range(R):
        z, num_iter = ppm_u_1(H[r], x_init[r,:], max_iterations=DEPTH)
        z_total.append(z)
        x_est = align_in_fourier(y_fft[r], z, L)
        x_total.append(x_est)
    x_total = np.asarray(x_total)
    loss_ppm = reconstruction_loss_u_1(x_fft_true, x_total)
    print('[PPM] loss = %f' % loss_ppm)

    z_total = []
    x_total = []

    for r in range(R):
        z, num_iter = pim_u_1(H[r], x_init[r, :], max_iterations=DEPTH)
        z_total.append(z)
        x_est = align_in_fourier(y_fft[r], z, L)
        x_total.append(x_est)
    x_total = np.asarray(x_total)

    loss_pim = reconstruction_loss_u_1(x_fft_true, x_total)
    print('[PIM] loss = %f' % loss_pim)

    z_total = []
    x_total = []

    for r in range(R):
        z, num_iter = amp_u_1(H[r], x_init[r,:], x_init2[r,:], Lambda, max_iterations=DEPTH)
        z_total.append(z)
        x_est = align_in_fourier(y_fft[r], z, L)
        x_total.append(x_est)
    x_total = np.asarray(x_total)

    loss_amp = reconstruction_loss_u_1(x_fft_true, x_total)
    print('[AMP] loss = %f' % loss_amp)

    return loss_ppm, loss_pim, loss_amp, loss_nn


class Compare1dMRAUnrollingReconstructionLossExperiment(Experiment):
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
        plt.plot(depth_range, avg_loss_ppm_vs_d)
        plt.plot(depth_range, avg_loss_pim_vs_d)
        plt.plot(depth_range, avg_loss_amp_vs_d)
        plt.plot(depth_range, avg_loss_nn_vs_d)
        plt.legend(['PPM', 'PIM', 'AMP', 'NN'])
        plt.xlabel('Depth')
        plt.ylabel('Mean Error')
        plt.savefig(self.get_results_path()+'.eps')
        plt.savefig(self.get_results_path()+'.png')
        plt.clf()


if __name__ == '__main__':
    # Compare1dMRAUnrollingReconstructionLossExperiment(params={'N': 20, 'R': 1000, 'num_trials': 1, 'depth_range': [1, 3, 5, 9, 15], 'epochs': 500, 'Lambda': 1, 'L': 21})
    # Compare1dMRAUnrollingReconstructionLossExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [5], 'epochs': 500, 'Lambda': 1, 'L': 21})
    Compare1dMRAUnrollingReconstructionLossExperiment(params={'N': 20, 'R': 10000, 'num_trials': 1, 'depth_range': [5], 'epochs': 200, 'Lambda': 1, 'L': 21})
