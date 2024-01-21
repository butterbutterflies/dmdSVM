import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt

from pyriemann.estimation import ERPCovariances, Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import normalize

from sklearn.covariance import GraphicalLassoCV, ledoit_wolf, empirical_covariance, OAS


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
index_dmd = [0, 1, 2]
index_epoch = [0, 1, 2]

# subject session
sub = ['1', '2', '3', '4', '5', '6']
ses = ['1', '2']


def calComplexCov(epoch_dmd):
    """ epoch_dmd: complex matrix, (epoch_num, num of modes, time)"""

    epoch_cov = []
    for i in range(epoch_dmd.shape[0]):
        # dmd_cov_cpx = np.cov(epoch_dmd[i, :, :])
        dmd_cov_cpx = empirical_covariance(epoch_dmd[i, :, :].T)
        dmd_cov = np.abs(dmd_cov_cpx)

        # whether positive definite
        if not np.all(np.linalg.eigvals(dmd_cov) > 0):
            vals, vecs = np.linalg.eig(dmd_cov)
            vals[np.where(vals <= 0)] = 0.001
            lam = np.diag(vals)
            inv_vecs = np.linalg.inv(vecs)
            dmd_cov = np.dot(np.dot(vecs, lam), inv_vecs)  # positive definite
            dmd_cov = (dmd_cov + dmd_cov.T) / 2  # symmetrical

        epoch_cov.append(dmd_cov)

    return np.array(epoch_cov)


class dmdRiemannVec:

    def __init__(self, dmd_path):
        self.dmd_file_path = dmd_path
        self.dmd_data = None
        self.label = None
        self.dim = None

        self.dmd_prj1 = None
        self.dmd_prj2 = None
        self.dmd_prj3 = None

    def load_data(self, s_idx, se_idx):
        # dmd feature
        dmd_cor_file_name = self.dmd_file_path + 'sub' + sub[s_idx] + '_ses' + ses[se_idx] + '_correct.mat'
        dmd_eor_file_name = self.dmd_file_path + 'sub' + sub[s_idx] + '_ses' + ses[se_idx] + '_error.mat'
        dmd_cor_data = scipy.io.loadmat(dmd_cor_file_name)['feature'][:, index_dmd, :, :]
        dmd_eor_data = scipy.io.loadmat(dmd_eor_file_name)['feature'][:, index_dmd, :, :]  # [:, index, :]
        # dmd_cor_data = np.squeeze(dmd_cor_data, axis=1)
        # dmd_eor_data = np.squeeze(dmd_eor_data, axis=1)

        self.dmd_data = np.vstack((dmd_cor_data, dmd_eor_data))
        # self.dim = self.dmd_data.shape[1]

        # labels: 0 for correct, 1 for error
        labels = [0] * len(dmd_cor_data) + [1] * len(dmd_eor_data)
        self.label = np.array(labels, dtype=int)

    def get_train_feature(self, train_index):
        # 0 for correct, 1 for error

        self.dmd_prj1 = TangentSpace(metric='riemann')
        self.dmd_prj2 = TangentSpace(metric='riemann')
        self.dmd_prj3 = TangentSpace(metric='riemann')

        dmd_train = self.dmd_data[train_index, :, :, :]
        dmd_train1 = dmd_train[:, 0, :, :]
        dmd_train2 = dmd_train[:, 1, :, :]
        dmd_train3 = dmd_train[:, 2, :, :]

        label_train = self.label[train_index]

        dmd_cov1 = calComplexCov(dmd_train1)
        dmd_vec1 = self.dmd_prj1.fit(dmd_cov1, label_train).transform(dmd_cov1)
        dmd_cov2 = calComplexCov(dmd_train2)
        dmd_vec2 = self.dmd_prj1.fit(dmd_cov2, label_train).transform(dmd_cov2)
        dmd_cov3 = calComplexCov(dmd_train3)
        dmd_vec3 = self.dmd_prj1.fit(dmd_cov3, label_train).transform(dmd_cov3)

        dmd_vec = np.hstack((dmd_vec1, dmd_vec2, dmd_vec3))
        print(dmd_vec.shape)

        return dmd_vec, label_train

    def get_test_feature(self, test_index):
        # 0 for correct, 1 for error
        dmd_test = self.dmd_data[test_index, :, :, :]

        dmd_test1 = dmd_test[:, 0, :, :]
        dmd_test2 = dmd_test[:, 1, :, :]
        dmd_test3 = dmd_test[:, 2, :, :]

        label_test = self.label[test_index]

        dmd_cov1 = calComplexCov(dmd_test1)
        dmd_vec1 = self.dmd_prj1.transform(dmd_cov1)
        dmd_cov2 = calComplexCov(dmd_test2)
        dmd_vec2 = self.dmd_prj2.transform(dmd_cov2)
        dmd_cov3 = calComplexCov(dmd_test3)
        dmd_vec3 = self.dmd_prj3.transform(dmd_cov3)

        dmd_vec = np.hstack((dmd_vec1, dmd_vec2, dmd_vec3))

        return dmd_vec, label_test


class epochRiemannVec:

    def __init__(self, epoch_path):
        self.epoch_file_path = epoch_path
        self.epoch_data = None
        self.label = None

        self.epoch_est = None
        self.epoch_prj = None

    def load_data(self, s_idx, se_idx):
        # dmd feature
        epoch_cor_file_name = self.epoch_file_path + 'sub' + sub[s_idx] + '_ses' + ses[se_idx] + '_correct.mat'
        epoch_eor_file_name = self.epoch_file_path + 'sub' + sub[s_idx] + '_ses' + ses[se_idx] + '_error.mat'
        epoch_cor_data = scipy.io.loadmat(epoch_cor_file_name)['feature'][:, index_epoch, :]
        epoch_eor_data = scipy.io.loadmat(epoch_eor_file_name)['feature'][:, index_epoch, :]  # [:, index, :]
        self.epoch_data = np.vstack((epoch_cor_data, epoch_eor_data))

        # labels: 0 for correct, 1 for error
        labels = [0] * len(epoch_cor_data) + [1] * len(epoch_eor_data)
        self.label = np.array(labels, dtype=int)

    def get_train_feature(self, train_index):
        # 0 for correct, 1 for error
        self.epoch_est = ERPCovariances(classes=[0, 1], estimator='lwf')
        self.epoch_prj = TangentSpace(metric='riemann')

        epoch_train = self.epoch_data[train_index, :, :]

        label_train = self.label[train_index]

        epoch_cov = self.epoch_est.fit(epoch_train, label_train).transform(epoch_train)
        epoch_vec = self.epoch_prj.fit(epoch_cov, label_train).transform(epoch_cov)

        return epoch_vec, label_train

    def get_test_feature(self, test_index):
        # 0 for correct, 1 for error
        epoch_test = self.epoch_data[test_index, :, :]

        label_test = self.label[test_index]

        epoch_cov = self.epoch_est.transform(epoch_test)
        epoch_vec = self.epoch_prj.transform(epoch_cov)

        return epoch_vec, label_test


'''
-----------------------------------Data-------------------------------------------
'''

dmd_file_path = 'D:/eegDMD/dmdSVM/dmd-modes-phase/'
epoch_file_path = 'D:/eegDMD/dmdSVM/EpochData/'

# 0-5, 0-1
sub_idx, ses_idx = 0, 0

dfv = dmdRiemannVec(dmd_file_path)
efv = epochRiemannVec(epoch_file_path)

dfv.load_data(sub_idx, ses_idx)
efv.load_data(sub_idx, ses_idx)

y = dfv.label
n_samples = y.shape[-1]

# Attention!
epoch_emb_dim = len(index_epoch)*3*(len(index_epoch)*3+1) // 2  # 45
# dmd_emb_dim = (dfv.dim * (dfv.dim+1)) // 2   # 171

'''
-----------------------------------Classifier--------------------------------------
'''
clf = make_pipeline(LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'))

'''
------------------------------------Train--------------------------------------------
'''
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)

pr = np.zeros(n_samples)   # predict the results
for train_idx, test_idx in skf.split(np.zeros(n_samples), y):

    ''' Data Preparation '''
    x_dmd_train, y_train = dfv.get_train_feature(train_idx)
    # x_epoch_train, _ = efv.get_train_feature(train_idx)

    x_dmd_test, y_test = dfv.get_test_feature(test_idx)
    # x_epoch_test, _ = efv.get_test_feature(test_idx)

    # x_train = np.hstack((normalize(x_dmd_train), normalize(x_epoch_train)))
    # x_test = np.hstack((normalize(x_dmd_test), normalize(x_epoch_test)))
    x_train = normalize(x_dmd_train)
    x_test = normalize(x_dmd_test)
    # x_train = x_epoch_train
    # x_test = x_epoch_test

    clf.fit(x_train, y_train)
    pr[test_idx] = clf.predict(x_test)

'''
------------------------------------Result--------------------------------------------
'''
print(classification_report(y, pr))
# confusion matrix
names = ["correct", "error"]
cm = confusion_matrix(y, pr)
ConfusionMatrixDisplay(cm, display_labels=names).plot()
plt.show()
