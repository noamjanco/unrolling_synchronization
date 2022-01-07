import numpy as np
import pandas as pd
from data_generation.generate_data import generate_data_z_over_2
from experiments.base_experiment import Experiment
from synchronization.ppm import ppm_z_over_2
from synchronization.pim import pim_z_over_2
from synchronization.amp import amp_z_over_2
from common.math_utils import rel_error_z_over_2
import matplotlib.pyplot as plt



class CompareZOver2Experiment(Experiment):
    def __init__(self, params: dict):
        super().__init__(params=params)

    def run_experiment(self):
        np.random.seed(self.params['seed'])
        N = self.params['N']
        R = self.params['R']

        df = pd.DataFrame()
        for Lambda in self.params['Lambda_range']:
            for r in range(R):
                x, Y = generate_data_z_over_2(N, Lambda)
                x_init = 1e-1 * np.expand_dims(np.random.rand(N), axis=-1)
                x_init2 = 1e-1 * np.expand_dims(np.random.rand(N), axis=-1)

                x_est, num_iter = ppm_z_over_2(Y, x_init)
                err_ppm = rel_error_z_over_2(x_est, x)
                num_iter_ppm = num_iter
                print('ppm rel error: %f, num iter: %d' % (err_ppm, num_iter))

                x_est, num_iter = pim_z_over_2(Y, x_init)
                err_pim = rel_error_z_over_2(x_est, x)
                num_iter_pim = num_iter
                print('pim rel error: %f, num iter: %d' % (err_pim, num_iter))

                x_est, num_iter = amp_z_over_2(Y, x_init, x_init2, Lambda)
                err_amp = rel_error_z_over_2(x_est, x)
                num_iter_amp = num_iter

                print('amp rel error: %f, num iter: %d' % (err_amp, num_iter))
                df = pd.concat([df,
                                pd.DataFrame({'err_ppm': err_ppm,
                                              'num_iter_ppm': num_iter_ppm,
                                              'err_pim': err_pim,
                                              'num_iter_pim': num_iter_pim,
                                              'err_amp': err_amp,
                                              'num_iter_amp': num_iter_amp,
                                              'Lambda': Lambda,
                                              'r': [r],
                                              'N': N})
                                ]
                               , ignore_index=True)
        self.results = df

    def plot_results(self):
        df = self.results
        Lambda_range = self.params['Lambda_range']
        avg_err_ppm = np.asarray([np.mean(df[df['Lambda'] == x].err_ppm) for x in Lambda_range])
        avg_err_pim = np.asarray([np.mean(df[df['Lambda'] == x].err_pim) for x in Lambda_range])
        avg_err_amp = np.asarray([np.mean(df[df['Lambda'] == x].err_amp) for x in Lambda_range])
        avg_num_iter_ppm = np.asarray([np.mean(df[df['Lambda'] == x].num_iter_ppm) for x in Lambda_range])
        avg_num_iter_pim = np.asarray([np.mean(df[df['Lambda'] == x].num_iter_pim) for x in Lambda_range])
        avg_num_iter_amp = np.asarray([np.mean(df[df['Lambda'] == x].num_iter_amp) for x in Lambda_range])
        plt.semilogy(Lambda_range, avg_err_ppm)
        plt.semilogy(Lambda_range, avg_err_pim)
        plt.semilogy(Lambda_range, avg_err_amp)
        plt.legend(['PPM', 'PIM', 'AMP'])
        plt.ylabel('Error')
        plt.xlabel('SNR')
        # plt.subplot(212)
        # plt.plot(Lambda_range,avg_num_iter_ppm)
        # plt.plot(Lambda_range,avg_num_iter_pim)
        # plt.plot(Lambda_range,avg_num_iter_amp)
        # plt.legend(['PPM','PIM','AMP'])
        # plt.ylabel('# Iterations')
        # plt.show()
        plt.savefig(self.get_results_path()+'.eps')
        plt.savefig(self.get_results_path()+'.png')


if __name__ == '__main__':
    CompareZOver2Experiment(params={'N': 2000, 'R': 10, 'Lambda_range': np.arange(1.2, 3.2, 0.2), 'seed': 1})
