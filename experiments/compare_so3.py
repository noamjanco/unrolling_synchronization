import numpy as np
import pandas as pd
from data_generation.generate_data_so3 import generate_data_so3
from experiments.base_experiment import Experiment
from synchronization.ppm import ppm_so3
from synchronization.pim import pim_so3
from common.math_utils import rel_error_so3, initialize_matrix
import matplotlib.pyplot as plt



class CompareSO3Experiment(Experiment):
    def __init__(self, params: dict):
        super().__init__(params=params)

    def run_experiment(self):
        np.random.seed(self.params['seed'])
        N = self.params['N']
        R = self.params['R']
        tol = self.params['tol']

        df = pd.DataFrame()
        for Lambda in self.params['Lambda_range']:
            for r in range(R):
                Rot, H = generate_data_so3(N, Lambda)
                Rot_init = initialize_matrix(N)

                Rot_est, num_iter = ppm_so3(H, tol=tol, z_init=Rot_init)
                err_ppm = rel_error_so3(Rot_est, Rot)
                num_iter_ppm = num_iter
                print('ppm rel error: %f, num iter: %d' % (err_ppm, num_iter))

                Rot_est = pim_so3(H)
                err_pim = rel_error_so3(Rot_est, Rot)
                num_iter_pim = 0
                print('pim rel error: %f, num iter: %d' % (err_pim, num_iter))

                # x_est, num_iter = amp_u_1(Y, x_init, x_init2, Lambda , tol=tol)
                # err_amp = rel_error_u_1(x_est, x)
                # num_iter_amp = num_iter
                # print('amp rel error: %f, num iter: %d' % (err_amp, num_iter))

                df = pd.concat([df,
                                pd.DataFrame({'err_ppm': err_ppm,
                                              'num_iter_ppm': num_iter_ppm,
                                              'err_pim': err_pim,
                                              'num_iter_pim': num_iter_pim,
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
        # avg_err_amp = np.asarray([np.mean(df[df['Lambda'] == x].err_amp) for x in Lambda_range])
        avg_num_iter_ppm = np.asarray([np.mean(df[df['Lambda'] == x].num_iter_ppm) for x in Lambda_range])
        avg_num_iter_pim = np.asarray([np.mean(df[df['Lambda'] == x].num_iter_pim) for x in Lambda_range])
        # avg_num_iter_amp = np.asarray([np.mean(df[df['Lambda'] == x].num_iter_amp) for x in Lambda_range])
        plt.plot(Lambda_range, avg_err_ppm)
        plt.plot(Lambda_range, avg_err_pim)
        # plt.semilogy(Lambda_range, avg_err_amp)
        plt.legend(['PPM', 'PIM'])
        plt.ylabel('Error')
        plt.xlabel('SNR')
        plt.show()
        # plt.subplot(212)
        # plt.plot(Lambda_range,avg_num_iter_ppm)
        # plt.plot(Lambda_range,avg_num_iter_pim)
        # plt.plot(Lambda_range,avg_num_iter_amp)
        # plt.legend(['PPM','PIM','AMP'])
        # plt.ylabel('# Iterations')
        # plt.show()
        plt.savefig(self.get_results_path()+'.eps')
        plt.savefig(self.get_results_path()+'.png')
        plt.clf()


if __name__ == '__main__':
    CompareSO3Experiment(params={'N': 100, 'R': 10, 'Lambda_range': np.arange(.4,3.,0.2), 'seed': 1, 'tol': 1e-3})
