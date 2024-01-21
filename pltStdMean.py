import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import pyplot


CLabel = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5',
          'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3',
          'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz',
          'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz',
          'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz',
          'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2',
          'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

chan_index = [CLabel.index('Cz'), CLabel.index('Fz'), CLabel.index('FCz'), CLabel.index('F3'),
              CLabel.index('F4'), CLabel.index('C3'), CLabel.index('C4'), CLabel.index('FC3'),
              CLabel.index('Fp1'), CLabel.index('Fp2'), CLabel.index('FC4'), CLabel.index('F7'),
              CLabel.index('F8'), CLabel.index('T7'), CLabel.index('T8')]

# only plot data on channel FCz
index = [2]

sub = ['1', '2', '3', '4', '5', '6']
ses = ['1', '2']


'''
--------------------------------Data--------------------------------
'''
sub_idx, ses_idx = 2, 1

dmd_file_path = 'D:/dmdSVM/PhaseData/'
dmd_cor_file_name = dmd_file_path + 'sub' + sub[sub_idx] + '_ses' + ses[ses_idx] + '_correct.mat'
dmd_eor_file_name = dmd_file_path + 'sub' + sub[sub_idx] + '_ses' + ses[ses_idx] + '_error.mat'
dmd_cor_data = scipy.io.loadmat(dmd_cor_file_name)['feature'][:, index, :]   # 3D matrix
dmd_eor_data = scipy.io.loadmat(dmd_eor_file_name)['feature'][:, index, :]
dmd_cor_data = np.squeeze(dmd_cor_data)  # 2D matrix
dmd_eor_data = np.squeeze(dmd_eor_data)


'''
--------------------------------Plot--------------------------------
'''
# print(plt.style.available)
with plt.style.context(['classic']):
    plt.style.use('seaborn-whitegrid')
    palette = pyplot.get_cmap('Set1')
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }
    fig = plt.figure()

    dt = 1 / 300
    iters = np.array(list(range(-60, 232)), dtype=float) * dt
    # iters = list(range(dmd_cor_data.shape[-1]))
    for i in range(1):
        color = palette(0)
        ax = fig.add_subplot(1, 1, i + 1)
        avg = np.mean(dmd_cor_data, axis=0)
        std = np.std(dmd_cor_data, axis=0)
        r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
        r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
        ax.plot(iters, avg, color=color, label="No ErrP")
        ax.fill_between(iters, r1, r2, color=color, alpha=0.2)

        color = palette(1)
        avg = np.mean(dmd_eor_data, axis=0)
        std = np.std(dmd_eor_data, axis=0)
        r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
        r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
        ax.plot(iters, avg, color=color, label="ErrP")
        ax.fill_between(iters, r1, r2, color=color, alpha=0.2)

        ax.legend(loc='lower right')
        ax.set_xlabel('Times(s)')
        ax.set_ylabel('PVD')
    plt.title('PVD of 1-12Hz modes at channel FCz')
    plt.show()
    # fontsize=22