import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pydmd import DMD
import seaborn as sns
import math
import mne


CLabel = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5',
          'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3',
          'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz',
          'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz',
          'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz',
          'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2',
          'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

index = [0, 1, 2]

sub = ['1', '2', '3', '4', '5', '6']
ses = ['1', '2']

epoch_file_path = 'D:/eegDMD/dmdSVM/64-epoch-data/'
# 0-5, 0-1
s_idx, se_idx = 5, 1

epoch_eor_file_name = epoch_file_path + 'sub' + sub[s_idx] + '_ses' + ses[se_idx] + '_error.mat'
epoch_eor_data = scipy.io.loadmat(epoch_eor_file_name)['feature']
# print(epoch_eor_data.shape)  # (89, 64, 301)
epoch_cor_file_name = epoch_file_path + 'sub' + sub[s_idx] + '_ses' + ses[se_idx] + '_correct.mat'
epoch_cor_data = scipy.io.loadmat(epoch_cor_file_name)['feature']


def ShiftStack(epochdata):
    '''
    :param epoch data: (num_epochs, channel, timepoints)
    :return: shift-stacked epoch data: (num_epochs, stacks, timepoints)
    '''
    stackedEpochData = []
    for i in range(0, len(epochdata)):
        trial = epochdata[i]
        m = trial.shape[0]
        n = trial.shape[1]
        nstack = math.ceil(n * 2 / (m + 2))
        aug_trial = np.array(trial[:, 0:n - nstack + 0 + 1])
        for st in range(1, nstack):
            aug_trial = np.vstack((aug_trial, trial[:, st:n - nstack + st + 1]))
        stackedEpochData.append(aug_trial)
    return np.array(stackedEpochData)


aug_epoch = ShiftStack(epoch_eor_data)
aug_cor_epoch = ShiftStack(epoch_cor_data)

for i in range(aug_epoch.shape[0]):
# for i in range(77, 78):
    print(i)
    aug_trial = aug_epoch[i, :, :]
    aug_cor_trial = aug_cor_epoch[70, :, :]

    dt = 1 / 300

    dmd = DMD(svd_rank=0, tlsq_rank=100, exact=True, opt=True)
    dmd2 = DMD(svd_rank=0, tlsq_rank=100, exact=True, opt=True)

    dmd.fit(aug_trial)
    dmd_f = dmd.frequency / dt
    dmd_b = dmd.amplitudes
    dmd_phi = dmd.modes
    dmd_dynamic = dmd.dynamics

    dmd2.fit(aug_cor_trial)
    dmd2_f = dmd2.frequency / dt
    dmd2_b = dmd2.amplitudes
    dmd2_phi = dmd2.modes
    dmd2_dynamic = dmd2.dynamics

    recon_trial = dmd.reconstructed_data
    # print(dmd_b.shape)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # abs(aug_trial[0:64, :]-recon_trial.real[0:64, :])
    # plt.figure(figsize=(4, 7))
    # sns.heatmap(data=np.absolute(dmd_phi), xticklabels=False, yticklabels=False, cmap="viridis")  # cmap="RdBu_r" cmap="viridis"
    # plt.ylabel("Stacked Channels")
    # plt.xlabel("Number of Modes")
    # plt.title("Magnitude of DMD Modes")
    # plt.savefig(r'D:\eegDMD\dmdSVM\ResultsFigure\dmd_m.png', format='png', bbox_inches='tight')
    # plt.show()

    idx = [k for k in range(len(dmd_f)) if abs(dmd_f[k]) > 2 and abs(dmd_f[k]) < 12]
    idx2 = [k for k in range(len(dmd2_f)) if abs(dmd2_f[k]) > 2 and abs(dmd2_f[k]) < 12]
    # # print(idx)  75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 89, 90, 93, 94, 95, 96, 97, 98, 99, 100
    # print(dmd_f[idx])
    b = dmd_b[idx]
    phi = dmd_phi[46, idx]
    # phi = np.expand_dims(phi, axis=0)
    dynamic = dmd_dynamic[idx, :]
    dynamic = dynamic.T
    # print(phi.shape, dynamic.shape)
    cmp = dynamic * phi

    b2 = dmd2_b[idx2]
    phi2 = dmd2_phi[46, idx2]
    # phi = np.expand_dims(phi, axis=0)
    dynamic2 = dmd2_dynamic[idx2, :]
    dynamic2 = dynamic2.T
    # print(phi.shape, dynamic.shape)
    cmp2 = dynamic2 * phi2
    # dd = dd.T
    # print(dd.shape)
    # plt.plot(dd.real)
    # plt.show()
    times = np.array(list(range(-60, 232)), dtype=float) * dt
    ang = np.angle(cmp)  # theta
    expang = np.exp(1.0j * ang)  # Euler
    # print(expang.shape)   # (292, 22)
    expang = expang.T

    varvec = np.var(expang, axis=0)

    #
    ang2 = np.angle(cmp2)  # theta
    expang2 = np.exp(1.0j * ang2)  # Euler
    # print(expang.shape)   # (292, 22)
    expang2 = expang2.T

    varvec2 = np.var(expang2, axis=0)

    with plt.style.context(['science', 'no-latex']):
        plt.figure(figsize=(4, 3.5))
        # plt.plot(times, varvec2, label='No ErrP', color='#ff7f0e', linewidth=1.2)
        # plt.plot(times, varvec, label='ErrP', color='#1f77b4', linewidth=1.2)
        plt.plot(times, varvec2, label='No ErrP', color='red', linewidth=1.1)
        plt.plot(times, varvec, label='ErrP', color='black', linewidth=1.1)
        plt.legend()
        plt.xlabel('Time(s)')
        plt.ylabel('PVD')
        plt.title('PVD of 1-12Hz modes at channel FCz')
        # plt.savefig('ErrP.png')
        plt.show()

    # ITPC
    with plt.style.context(['science', 'no-latex']):
        plt.figure(figsize=(4, 3.5))
        itpc = abs(np.mean(expang, axis=0))
        itpc2 = abs(np.mean(expang2, axis=0))

        plt.plot(times, itpc, label='ErrP', color='black', linewidth=1.1)
        plt.plot(times, itpc2, label='No ErrP', color='red', linewidth=1.1)
        plt.legend()
        plt.xlabel('Time(s)')
        plt.ylabel('IMPC')
        plt.title('IMPC of 1-12Hz modes  at channel FCz')
        plt.show()
# b = dmd_b[idx]
# print(abs(dmd_f[idx]))
# phi = dmd_phi[:, idx]
# # print(phi.shape)  (640, 22)
# dynamic = dmd_dynamic[idx, :]
#
# recon_freq = phi.dot(dynamic)
#
# phi_m = np.absolute(phi[0:64, :])
# phi_p = np.angle(phi[0:64, :])
#
# info = mne.create_info(ch_names=CLabel, ch_types='eeg', sfreq=300)
# pm_mne = mne.EvokedArray(phi_m, info)
# montage = mne.channels.make_standard_montage('standard_1020')  # attention
# pm_mne.set_montage(montage, on_missing='raise', verbose=None)
#
# fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(ncols=10)
# im, cm = mne.viz.plot_topomap(pm_mne.data[:, 6], pm_mne.info, axes=ax1, show=False)
# im, cm = mne.viz.plot_topomap(pm_mne.data[:, 14], pm_mne.info, axes=ax2, show=False)
# im, cm = mne.viz.plot_topomap(pm_mne.data[:, 18], pm_mne.info, axes=ax3, show=False)
# im, cm = mne.viz.plot_topomap(pm_mne.data[:, 16], pm_mne.info, axes=ax4, show=False)
# im, cm = mne.viz.plot_topomap(pm_mne.data[:, 12], pm_mne.info, axes=ax5, show=False)
# im, cm = mne.viz.plot_topomap(pm_mne.data[:, 20], pm_mne.info, axes=ax6, show=False)
# im, cm = mne.viz.plot_topomap(pm_mne.data[:, 2], pm_mne.info, axes=ax7, show=False)
# im, cm = mne.viz.plot_topomap(pm_mne.data[:, 4], pm_mne.info, axes=ax8, show=False)
# im, cm = mne.viz.plot_topomap(pm_mne.data[:, 8], pm_mne.info, axes=ax9, show=False)
# im, cm = mne.viz.plot_topomap(pm_mne.data[:, 0], pm_mne.info, axes=ax10, show=False)
# ax_x_start = 0.9
# ax_x_width = 0.04
# ax_y_start = 0.05
# ax_y_height = 0.9
# cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
# clb = fig.colorbar(im, cax=cbar_ax)
# clb.ax.set_title('Magnitude')
# plt.show()
# # mne.viz.plot_topomap(pm_mne.data[:, 10], pm_mne.info, show=True)

