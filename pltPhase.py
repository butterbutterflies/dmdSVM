import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

file_path = 'E:/BCI_RL/important/ErrP/DMD/code/IncDMD/rawdata/'
sub = ['1', '2', '3', '4', '5', '6']
ses = ['1', '2']
file_name = file_path+'Subject0'+sub[2]+'_s'+ses[1]+'.mat'

runsdata = scipy.io.loadmat(file_name)['run']
print(runsdata.shape)   # 1-10 blocks

# not total data
run = runsdata[0,0]   # 0,0 - 0,9
eeg = run[0,0]['eeg'].T
print(eeg.shape)

header = run[0,0]['header']
SampleRate = header[0,0]['SampleRate'][0][0]   # 2D Array
print(SampleRate)

CLabel = ['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1',
           'C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7','P9',
           'PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2','AF8','AF4',
           'AFz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz',
           'C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2']

event = header[0,0]['EVENT']
POS = event[0,0]['POS']
TYP = event[0,0]['TYP']
pos = np.array([i[0] for i in POS]).reshape(len(POS), -1)
typ = np.array([i[0] for i in TYP]).reshape(len(TYP), -1)
print(pos.shape)

# preprocess EEG data based on mne
# create events ndarray
EventArray = np.concatenate((pos, np.zeros([len(pos),1]), typ), axis=1)
EventArray = EventArray.astype(int)
print(EventArray.dtype)
print(EventArray.shape)
print(np.issubdtype(EventArray.dtype, np.integer))

# creating Raw object
info = mne.create_info(ch_names=CLabel, ch_types='eeg', sfreq=SampleRate)
raw = mne.io.RawArray(eeg, info)
print(len(raw.times))

montage = mne.channels.make_standard_montage('biosemi64')
raw = raw.set_montage(montage, on_missing = 'raise', verbose = None)

# 数据预处理前降采样
raw, EventArray = raw.resample(200, events=EventArray)
print(EventArray.dtype)
print(EventArray.shape)
print(np.issubdtype(EventArray.dtype, np.integer))

# 重参考---50HZ工频陷波滤波---0-10Hz低通滤波---分段
raw = raw.set_eeg_reference(ref_channels='average')

# raw.notch_filter(np.arange(60,241,60))  #  电力线噪音，数据在50Hz、150Hz、200Hz和250Hz存在窄频率峰值
# raw.plot_psd(fmin=0, fmax=40)

# raw = raw.filter(l_freq=locut, h_freq=hicut, filter_length=filtlength*2,l_trans_bandwidth=cutoffArray[0], h_trans_bandwidth=cutoffArray[1], fir_design='firwin')
raw = raw.filter(1,20.,fir_design='firwin')
# raw.plot_psd(fmin=0, fmax=40)

EventDict = {'Correct1':5, 'Correct2':10, 'Error1':6, 'Error2':9,
             'Target located in the right':1, 'Cursor moves to the left':2}
# Events = mne.find_events(raw, stim_channel='STI')
# event_fig = mne.viz.plot_events(EventArray, event_id=EventDict, sfreq=raw.info['sfreq'])

# epoch the desired eeg data and remove baseline
EpochCorrect = {'Correct1':5, 'Correct2':10}
EpochError = {'Error1':6, 'Error2':9}
epochsCorrect = mne.Epochs(raw, np.array(EventArray), event_id=EpochCorrect, tmin=-0.2,
                    tmax=0.8, baseline=(-0.2,0))
print(epochsCorrect)
epochsError = mne.Epochs(raw, np.array(EventArray), event_id=EpochError, tmin=-0.2,
                    tmax=0.8, baseline=(-0.2,0))
print(epochsError)

epochsCorrect = epochsCorrect.load_data()
epochsError = epochsError.load_data()
# epochsCorrect.resample(newSampleRate)
# epochsError.resample(newSampleRate)

correctdata = epochsCorrect.get_data('eeg')
correcttimes = epochsCorrect.times
errordata = epochsError.get_data('eeg')
errortimes = epochsError.times
print(errordata.shape)

chan_index = [CLabel.index('Cz'), CLabel.index('Fz'), CLabel.index('FCz')]
print(chan_index)

# 首先构造一个trial的特征向量
import math

trial = errordata[3]  # nchannel * time points
trialc = correctdata[11]  # nchannel * time points

# EEG stack
m = trial.shape[0]  # 行
n = trial.shape[1]  # 列
nstack = math.ceil(n*2 / (m+2))

aug_trial = np.array(trial[:,0:n-nstack+0+1])
for st in range(1, nstack):
    aug_trial = np.vstack((aug_trial, trial[:,st:n-nstack+st+1]))
print(nstack)
print(aug_trial.shape)

m = trialc.shape[0]  # 行
n = trialc.shape[1]  # 列
nstack = math.ceil(n*2 / (m+2))

aug_trialc = np.array(trialc[:,0:n-nstack+0+1])
for st in range(1, nstack):
    aug_trialc = np.vstack((aug_trialc, trialc[:,st:n-nstack+st+1]))
print(nstack)
print(aug_trialc.shape)

# DMD for feature vector
from pydmd import DMD

dmd = DMD(svd_rank=0, exact=True)
dmdc = DMD(svd_rank=0, exact=True)
# dmd_optb = DMD(svd_rank=0, exact=True, opt=True), 可能opt方法这里有什么要求

dmd.fit(aug_trial)
dmdc.fit(aug_trialc)
# dmd_optb.fit(aug_trial)

# dt不对，导致f显示不对，但其实对重构数据没有影响
# np.log(self.eigs).imag / (2 * np.pi * self.original_time["dt"])
dt = 1 / 200
dmd_f = dmd.frequency / dt
dmd_phi = dmd.modes
dmd_dynamic = dmd.dynamics
dmd_b = dmd.amplitudes

idx = [i for i in range(len(dmd_f)) if abs(dmd_f[i])>2 and abs(dmd_f[i])<12]

b = dmd_b[idx]
phi = dmd_phi[:, idx]
dynamic = dmd_dynamic[idx, :]
print(dmd_dynamic.shape)
print(dmd_b.shape)

recon_freq = phi.dot(dynamic)
print(recon_freq.shape)

# 某一通道处的模态相位分析
# 乘以不乘以b？
cmp = np.empty(shape=(len(idx),len(dynamic[0])), dtype=complex)
for i in range(len(idx)):
    cmp[i, :] = phi[chan_index[2]][i] * dynamic[i, :]
# angle
ang = np.angle(cmp)
print(ang.shape)
# Euler's formular
expang = np.exp(1.0j * ang)
print(expang.shape)
# variance of complex vector
varvec = np.var(expang, axis=0)
# print(varvec.shape)
# print(varvec)
# angvar = np.angle(varvec)
# print(angvar)

dtc = 1 / 200
dmdc_f = dmdc.frequency / dt
dmdc_phi = dmdc.modes
dmdc_dynamic = dmdc.dynamics
dmdc_b = dmdc.amplitudes

idxc = [i for i in range(len(dmdc_f)) if abs(dmdc_f[i])>2 and abs(dmdc_f[i])<12]

bc = dmdc_b[idxc]
phic = dmdc_phi[:, idxc]
dynamicc = dmdc_dynamic[idxc, :]
print(dmdc_dynamic.shape)
print(dmdc_b.shape)

recon_freq = phi.dot(dynamic)
print(recon_freq.shape)

# 某一通道处的模态相位分析
# 乘以不乘以b？
cmpc = np.empty(shape=(len(idxc),len(dynamicc[0])), dtype=complex)
for i in range(len(idxc)):
    cmpc[i, :] = phi[chan_index[2]][i] * dynamicc[i, :]
# angle
angc = np.angle(cmpc)
print(angc.shape)
# Euler's formular
expangc = np.exp(1.0j * angc)
print(expangc.shape)
# variance of complex vector
varvecc = np.var(expangc, axis=0)
# print(varvec.shape)
# print(varvec)
# angvar = np.angle(varvec)
# print(angvar)

plt.plot(errortimes[0:len(aug_trial[0])], varvec, label='ErrP')
plt.plot(errortimes[0:len(aug_trialc[0])], varvecc, label='No ErrP')
plt.legend()
plt.xlabel('Time(s)')
plt.ylabel('Phase variance')
plt.title('Phase variance of modes 1-12Hz at channel FCz')
plt.savefig('Total.png')
plt.show()
