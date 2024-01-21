import mne
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


class EEGPreprocess:
    """
    Read and preprocess EEG data for a session of a subject
    Return epoch correct and error EEG data
    """

    def __init__(self, file_path: str, sub: str, ses: str, downSample):
        self.file_path = file_path
        self.sub = sub
        self.ses = ses
        self.file_name = self.file_path + 'Subject0' + self.sub + '_s' + self.ses + '.mat'
        self.downSample = downSample
        self.SampleRate = None

    def _process_epoch_data(self, runs_data, i, low, ts, te):
        run = runs_data[0, i]
        eeg = run[0, 0]['eeg'].T

        header = run[0, 0]['header']
        self.SampleRate = header[0, 0]['SampleRate'][0][0]
        event = header[0, 0]['EVENT']

        # create Event 3dArray
        _pos = event[0, 0]['POS']
        _typ = event[0, 0]['TYP']
        pos = np.array([j[0] for j in _pos]).reshape(len(_pos), -1)
        typ = np.array([j[0] for j in _typ]).reshape(len(_typ), -1)
        events = np.concatenate((pos, np.zeros([len(pos), 1]), typ), axis=1)
        events = events.astype(int)

        # preprocessed epoch data
        error, correct = self._preprocess_epoch(eeg, events, low, ts, te)

        return error, correct

    def read_preprocess_epoch_data(self, low: float, ts: float, te: float):
        """
        :param low: e.g. 30. 20. .etc
        :param ts: the start time of epoch (sec), e.g. -0.2
        :param te: the end time of epoch (sec), e.g. 1.0
        :return: correct_data, error_data in format of 3dArray, (epochs, channels, time_points)
        """
        runs_data = scipy.io.loadmat(self.file_name)['run']

        # preprocess one block each time
        print('---------Subject0' + self.sub + '_session' + self.ses + '---------')

        error, correct = self._process_epoch_data(runs_data, 0, low, ts, te)
        error_data = error
        correct_data = correct
        for i in range(1, len(runs_data[0])):
            error, correct = self._process_epoch_data(runs_data, i, low, ts, te)
            error_data = np.vstack((error_data, error))
            correct_data = np.vstack((correct_data, correct))

        return np.array(error_data), np.array(correct_data)

    def _preprocess_epoch(self, eeg, events, low, ts, te):

        # preprocess and epoch eeg based on mne
        info = mne.create_info(ch_names=CLabel, ch_types='eeg', sfreq=self.SampleRate)
        raw = mne.io.RawArray(eeg, info)
        montage = mne.channels.make_standard_montage('standard_1020')  # attention
        raw = raw.set_montage(montage, on_missing='raise', verbose=None)

        # down-sampling
        raw, events = raw.resample(self.downSample, events=events)

        # re-reference
        '''when you apply the average reference you reduce the rank of the data. CSP requires
           you to pass full rank data to have positive definite matrices. You need to regularize the
           covariance by adding a constant to the diagonal or apply a PCA before CSP'''
        raw = raw.set_eeg_reference(ref_channels='average')

        # low-pass filter
        raw = raw.filter(0.1, low, fir_design='firwin')  # 30.

        # epoch the desired eeg data and remove baseline
        try:
            correct_label = {'Correct1': 5, 'Correct2': 10}
            error_label = {'Error1': 6, 'Error2': 9}
            epochs_correct = mne.Epochs(raw, np.array(events), event_id=correct_label, tmin=ts,
                                        tmax=te, baseline=(ts, 0))
            epochs_error = mne.Epochs(raw, np.array(events), event_id=error_label, tmin=ts,
                                      tmax=te, baseline=(ts, 0))
        except:
            correct_label = {'Correct1': 5, 'Correct2': 10}
            error_label = {'Error1': 6}
            epochs_correct = mne.Epochs(raw, np.array(events), event_id=correct_label, tmin=ts,
                                        tmax=te, baseline=(ts, 0))
            epochs_error = mne.Epochs(raw, np.array(events), event_id=error_label, tmin=ts,
                                      tmax=te, baseline=(ts, 0))
        # extract 3dArray: (epochs, channels, time_points)
        epochs_correct = epochs_correct.load_data()
        epochs_error = epochs_error.load_data()
        correct = epochs_correct.get_data('eeg')
        error = epochs_error.get_data('eeg')

        return error, correct


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
        self.chan_index = [CLabel.index('Cz'), CLabel.index('Fz'), CLabel.index('FCz'), CLabel.index('F3'),
                           CLabel.index('F4'), CLabel.index('C3'), CLabel.index('C4'), CLabel.index('FC3'),
                           CLabel.index('Fp1'), CLabel.index('Fp2'), CLabel.index('FC4'), CLabel.index('F7'),
                           CLabel.index('F8'), CLabel.index('T7'), CLabel.index('T8')]
        self.sampleRate = sampleRate
        self.dt = 1 / self.sampleRate

    def phaseFeature(self, epoch):
        '''
        :param epoch: (epochs, channels, time_points)
        :return: phase feature: (epochs, 3, time_points)
        '''
        epoch_phase = []
        epoch = epoch[:, :, 60::]   # 0.0--0.8s
        aug_epoch = ShiftStack(epoch)
        for i in range(len(aug_epoch)):
            aug_trial = aug_epoch[i]
            dmd = DMD(svd_rank=0, tlsq_rank=100, exact=True, opt=True)
            dmd.fit(aug_trial)
            dmd_f = dmd.frequency / self.dt
            dmd_b = dmd.amplitudes
            dmd_phi = dmd.modes
            dmd_dynamic = dmd.dynamics

            idx = [k for k in range(len(dmd_f)) if abs(dmd_f[k]) > 2 and abs(dmd_f[k]) < 12]
            b = dmd_b[idx]
            phi = dmd_phi[:, idx]
            dynamic = dmd_dynamic[idx, :]

            trial_phase = []
            for j in range(len(self.chan_index)):
                cmp = np.empty(shape=(len(idx), len(dynamic[0])), dtype=complex)
                for n in range(len(idx)):
                    cmp[n, :] = phi[self.chan_index[j]][n] * dynamic[n, :]
                # angle
                ang = np.angle(cmp)
                # Euler's formular
                expang = np.exp(1.0j * ang)
                # variance of complex vector
                varvec = np.var(expang, axis=0)
                trial_phase.append(abs(varvec.T))
            epoch_phase.append(trial_phase)
        return np.array(epoch_phase)

    def bPhiFeature(self, epoch):
        '''
        :param epoch: (epochs, channels, time_points)
        :return: bPhiFeature in FCz channel: (epochs, 4, time_points'')
        '''
        epoch_bPhi = []
        aug_epoch = ShiftStack(epoch)  # -1---2s
        channel = self.chan_index[-1]
        bidx = int(0.8 * self.sampleRate)
        for i in range(len(aug_epoch)):
            aug_trial = aug_epoch[i]

            seg_trial = aug_trial[:, 0:bidx + 0]
            dmd = DMD(svd_rank=0, tlsq_rank=100, exact=True, opt=True)
            dmd.fit(seg_trial)
            dmd_f = dmd.frequency / self.dt
            dmd_b = dmd.amplitudes
            dmd_phi = dmd.modes[channel, :]

            bphi = np.multiply(dmd_b, dmd_phi)
            idx = [k for k in range(len(dmd_f)) if dmd_f[k] >= 1 and dmd_f[k] < 25]
            bphi = bphi[idx]
            f = dmd_f[idx]
            mbphi = self._freqbandMean(f, bphi)
            fbpack = mbphi
            for j in range(1, len(aug_trial[0])-bidx+1, 2):
                seg_trial = aug_trial[:, j:bidx + j]
                dmd = DMD(svd_rank=0, tlsq_rank=100, exact=True, opt=True)
                dmd.fit(seg_trial)
                dmd_f = dmd.frequency / self.dt
                dmd_b = dmd.amplitudes
                dmd_phi = dmd.modes[channel, :]

                bphi = np.multiply(dmd_b, dmd_phi)
                idx = [k for k in range(len(dmd_f)) if dmd_f[k] >= 1 and dmd_f[k] < 25]
                bphi = bphi[idx]
                f = dmd_f[idx]  # 只包含正频率点
                mbphi = self._freqbandMean(f, bphi)
                fbpack = np.vstack((fbpack, mbphi))

            epoch_bPhi.append(fbpack.T)

        return np.array(epoch_bPhi)

    def _freqbandMean(self, ft, bphit):
        idx1 = [i for i in range(len(ft)) if ft[i] >= 1 and ft[i] <= 4]
        idx2 = [i for i in range(len(ft)) if ft[i] > 4 and ft[i] <= 8]
        idx3 = [i for i in range(len(ft)) if ft[i] > 8 and ft[i] <= 12]
        idx4 = [i for i in range(len(ft)) if ft[i] > 12 and ft[i] <= 16]

        mbphi = []
        mbphi.append(np.mean(abs(bphit[idx1])))
        mbphi.append(np.mean(abs(bphit[idx2])))
        mbphi.append(np.mean(abs(bphit[idx3])))
        mbphi.append(np.mean(abs(bphit[idx4])))

        return np.array(mbphi)


if __name__ == '__main__':
    # original data
    file_path = 'D:/ErrpData/theMonitoring/'
    sub = ['1', '2', '3', '4', '5', '6']
    ses = ['1', '2']

    low = 20.
    ts, te = -0.2, 0.8
    downSample = 300

    for i in range(len(sub)):  # len(sub)
        for j in range(len(ses)):  # len(ses)
            print("subject %s session %s" % (i+1, j+1))

            eeg_process = EEGPreprocess(file_path, sub[i], ses[j], downSample)
            error, correct = eeg_process.read_preprocess_epoch_data(low, ts, te)

            # dmd = DMDFeature(downSample)
            # chan_index = dmd.chan_index

            # aug_error_phase = dmd.phaseFeature(error)
            # aug_correct_phase = dmd.phaseFeature(correct)

            error_data = error
            correct_data = correct

            # tps = aug_error_phase.shape[-1]
            #
            # error_data = error[:, chan_index, 0:tps]
            # correct_data = correct[:, chan_index, 0:tps]

            # scipy.io.savemat('./0-8-chan-dmd/sub'+str(i+1)+'_ses'+str(j+1)+'_error'+'.mat', {'feature': aug_error_phase})
            # scipy.io.savemat('./0-8-chan-dmd/sub'+str(i+1)+'_ses'+str(j+1)+'_correct'+'.mat', {'feature': aug_correct_phase})

            scipy.io.savemat('./forDmd-epoch-data/sub' + str(i + 1) + '_ses' + str(j + 1) + '_error' + '.mat',
                             {'feature': error_data})
            scipy.io.savemat('./forDmd-epoch-data/sub' + str(i + 1) + '_ses' + str(j + 1) + '_correct' + '.mat',
                             {'feature': correct_data})

            # tps = aug_error_phase.shape[-1]
            #
            # aug_error_bphi = dmd.bPhiFeature(error)
            # aug_error_bphi = aug_error_bphi[:, :, 0:tps]
            # aug_correct_bphi = dmd.bPhiFeature(correct)
            # aug_correct_bphi = aug_correct_bphi[:, :, 0:tps]

            # aug_error_feature = []
            # for k in range(len(aug_error_phase)):
            #     feature = np.vstack((aug_error_phase[k], aug_error_bphi[k]))
            #     aug_error_feature.append(feature)
            # aug_error_feature = np.array(aug_error_feature)
            #
            # aug_correct_feature = []
            # for k in range(len(aug_correct_phase)):
            #     feature = np.vstack((aug_correct_phase[k], aug_correct_bphi[k]))
            #     aug_correct_feature.append(feature)
            # aug_correct_feature = np.array(aug_correct_feature)














