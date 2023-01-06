import numpy as np
import pandas as pd
from data_generation.generate_data import generate_training_data_z_over_2
from experiments.base_experiment import Experiment
from models.unrolling_synchronization_z_over_2 import BuildModel, TrainModel, EvaluateModel, loss_z_over_2
from synchronization.ppm import ppm_z_over_2
from synchronization.pim import pim_z_over_2
from synchronization.amp import amp_z_over_2
import matplotlib.pyplot as plt


def task(N, R, Lambda, DEPTH, seed, epochs, num_iterations):
    # Initialize random seed
    np.random.seed(seed)
    x, Y = generate_training_data_z_over_2(N,Lambda,R)

    training_loss = []
    models = []
    for t in range(3):
        model = BuildModel(N,Lambda,DEPTH)
        x_init = 1e-1*np.expand_dims(np.random.rand(R,N),axis=-1)
        x_init2 = 1e-1*np.expand_dims(np.random.rand(R,N),axis=-1)
        x_val, Y_val  = generate_training_data_z_over_2(N, Lambda, R)
        x_val_init = 1e-1 * np.expand_dims(np.random.rand(R, N), axis=-1)
        x_val_init2 = 1e-1 * np.expand_dims(np.random.rand(R, N), axis=-1)
        TrainModel(model, Y, x, x_init,x_init2,Y_val, x_val, x_val_init,x_val_init2, epochs)
        models.append(model)
        x_est, loss_nn = EvaluateModel(model, Y_val, x_val, x_val_init, x_val_init2)
        training_loss.append(loss_nn)

    model = models[np.argmin(training_loss)]

    # Generate test data
    x, Y = generate_training_data_z_over_2(N,Lambda,R)
    x_init = 1e-1*np.expand_dims(np.random.rand(R,N),axis=-1)
    x_init2 = 1e-1*np.expand_dims(np.random.rand(R,N),axis=-1)
    x_est, loss_nn = EvaluateModel(model, Y, x, x_init, x_init2)
    training_loss.append(loss_nn)


    z_total = []
    for r in range(R):
        z,num_iter = ppm_z_over_2(Y[r], x_init[r,:], max_iterations=num_iterations)
        z_total.append(z)
    z1 = np.asarray(z_total)
    print(x.shape)
    print(z1.shape)
    loss_ppm = loss_z_over_2(x.astype(np.float32), z1.astype(np.float32))
    print('[PPM] loss = %f' % loss_ppm)

    z_total = []
    for r in range(R):
        z, num_iter = pim_z_over_2(Y[r], x_init[r, :], max_iterations=num_iterations)
        z_total.append(z)
    z1 = np.asarray(z_total)
    loss_pim = loss_z_over_2(x.astype(np.float32), z1.astype(np.float32))
    print('[PIM] loss = %f' % loss_pim)

    z_total = []
    for r in range(R):
        z, num_iter = amp_z_over_2(Y[r], x_init[r,:], x_init2[r,:], Lambda, max_iterations=num_iterations)
        z_total.append(z)
    z1 = np.asarray(z_total)
    loss_amp = loss_z_over_2(x.astype(np.float32), z1.astype(np.float32))
    print('[AMP] loss = %f' % loss_amp)

    return loss_ppm, loss_pim, loss_amp, loss_nn

class CompareZOver2UnrollingVsSNRExperiment(Experiment):
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
                                'num_iterations': num_iterations,
                                'epochs': epochs}, ignore_index=True)
        self.results = df

    def plot_results(self):
        df = self.results
        Lambda_range = self.params['Lambda_range']
        avg_loss_ppm_vs_d = [np.mean(df[df.Lambda == x].loss_ppm) for x in Lambda_range]
        avg_loss_pim_vs_d = [np.mean(df[df.Lambda == x].loss_pim) for x in Lambda_range]
        avg_loss_amp_vs_d = [np.mean(df[df.Lambda == x].loss_amp) for x in Lambda_range]
        avg_loss_nn_vs_d = [np.mean(df[df.Lambda == x].loss_nn) for x in Lambda_range]
        plt.plot(Lambda_range, avg_loss_ppm_vs_d, '-o')
        plt.plot(Lambda_range, avg_loss_pim_vs_d, '-^')

        plt.plot(Lambda_range, avg_loss_amp_vs_d, '-s')
        plt.plot(Lambda_range, avg_loss_nn_vs_d, '-*')
        plt.legend(['PPM', 'PM', 'AMP', 'unrolled algorithm'],fontsize=14)
        plt.xlabel(r'$\lambda$',fontsize=16)
        plt.ylabel('Mean Error',fontsize=16)
        plt.savefig(self.get_results_path()+'.eps')
        plt.savefig(self.get_results_path()+'.png')
        plt.clf()


if __name__ == '__main__':
    # CompareZOver2UnrollingExperiment(params={'N': 20, 'R': 20000, 'num_trials': 1, 'Lambda_range': np.arange(0.5,2.75,0.25), 'epochs': 300, 'NN_depth': 9, 'num_iterations': 100})
    # CompareZOver2UnrollingVsSNRExperiment(params={'N': 20, 'R': 20000, 'num_trials': 1, 'Lambda_range': np.arange(0.25,3.25,0.25), 'epochs': 300, 'depth': 9, 'num_iterations': 100})
    # CompareZOver2UnrollingVsSNRExperiment(params={'N': 20, 'R': 20000, 'num_trials': 1, 'Lambda_range': np.arange(0.25,3.5,0.25), 'epochs': 100, 'depth': 9, 'num_iterations': 100})
    CompareZOver2UnrollingVsSNRExperiment(params={'N': 40, 'R': 20000, 'num_trials': 1, 'Lambda_range': np.arange(0.25,3.5,0.25), 'epochs': 100, 'depth': 9, 'num_iterations': 100})