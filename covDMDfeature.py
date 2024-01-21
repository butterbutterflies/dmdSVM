import numpy as np
import scipy.io
import math
from pydmd import DMD


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

# select channel
index = [0, 1, 2]


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


class DMDFeature:

    def __init__(self, sampleRate):
        self.chan_index = chan_index
        self.sampleRate = sampleRate
        self.dt = 1 / self.sampleRate

    def covPhase(self, epoch):
        """
        epoch: (epochs, channels, time_points)
        """
        epoch_phase = list()
        aug_epoch = ShiftStack(epoch)

        nm = 100
        for i in range(len(aug_epoch)):
            aug_trial = aug_epoch[i]
            dmd = DMD(svd_rank=0, tlsq_rank=100, exact=True, opt=True)
            dmd.fit(aug_trial)
            dmd_f = dmd.frequency / self.dt
            dmd_b = dmd.amplitudes
            dmd_phi = dmd.modes
            dmd_dynamic = dmd.dynamics

            idx = [k for k in range(len(dmd_f)) if abs(dmd_f[k]) > 2 and abs(dmd_f[k]) < 12]
            nm = min(nm, len(idx))
            phi = dmd_phi[:, idx]
            dynamic = dmd_dynamic[idx, :]

            trial_phase = []
            for j in range(len(self.chan_index)):
                cmp = np.empty(shape=(len(idx), len(dynamic[0])), dtype=complex)
                for n in range(len(idx)):
                    cmp[n, :] = phi[self.chan_index[j]][n] * dynamic[n, :]
                # angle
                ang = np.angle(cmp)
                trial_phase.append(ang)
                # # Euler's formular
                # expang = np.exp(1.0j * ang)
                # # complex matrix
                # trial_phase.append(expang)
            epoch_phase.append(trial_phase)

        # extract min num modes
        new_epoch_phase = self.alignModes(epoch_phase, nm)
        return new_epoch_phase

    def alignModes(self, epoch_phase, nm):
        """ epoch_phase: 4 dim list """
        align_epoch_phase = []
        for i in range(len(epoch_phase)):
            align_trial_phase = []
            for j in range(len(self.chan_index)):
                phase = epoch_phase[i][j]
                align_phase = [phase[k] for k in range(nm)]
                align_trial_phase.append(align_phase)
            align_epoch_phase.append(align_trial_phase)
        return np.array(align_epoch_phase)


'''--------------------------------------------------------------------------------------'''
epoch_file_path = 'D:/eegDMD/dmdSVM/forDmd-epoch-data/'
sub = ['1', '2', '3', '4', '5', '6']
ses = ['1', '2']

downSample = 300

for i in range(len(sub)):  # len(sub)
    for j in range(len(ses)):  # len(ses)
        print("subject %s session %s" % (i+1, j+1))

        epoch_cor_file_name = epoch_file_path + 'sub' + sub[i] + '_ses' + ses[j] + '_correct.mat'
        epoch_eor_file_name = epoch_file_path + 'sub' + sub[i] + '_ses' + ses[j] + '_error.mat'
        epoch_cor_data = scipy.io.loadmat(epoch_cor_file_name)['feature']   # [:, index, :]
        epoch_eor_data = scipy.io.loadmat(epoch_eor_file_name)['feature']
        print(epoch_cor_data.shape)
        print(epoch_eor_data.shape)
        # dmd = DMDFeature(downSample)
        #
        # dmd_phase_cor = dmd.covPhase(epoch_cor_data)
        # dmd_phase_eor = dmd.covPhase(epoch_eor_data)
        #
        # scipy.io.savemat('./dmd-modes-angle/sub' + str(i + 1) + '_ses' + str(j + 1) + '_error' + '.mat',
        #                  {'feature': dmd_phase_eor})
        # scipy.io.savemat('./dmd-modes-angle/sub' + str(i + 1) + '_ses' + str(j + 1) + '_correct' + '.mat',
        #                  {'feature': dmd_phase_cor})






