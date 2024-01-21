import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from pyriemann.estimation import ERPCovariances
from pyriemann.tangentspace import TangentSpace

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import normalize


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

index = [0, 1, 2]

sub = ['1', '2', '3', '4', '5', '6']
ses = ['1', '2']


class RiemannFeatureVec:

    def __init__(self, dmd_path, epoch_path):
        self.dmd_file_path = dmd_path
        self.epoch_file_path = epoch_path

        self.dmd_data = None
        self.epoch_data = None
        self.label = None

        self.dmd_est = None
        self.dmd_prj = None
        self.epoch_est = None
        self.epoch_prj = None

    def load_data(self, s_idx, se_idx):
        # dmd feature
        dmd_cor_file_name = self.dmd_file_path + 'sub' + sub[s_idx] + '_ses' + ses[se_idx] + '_correct.mat'
        dmd_eor_file_name = self.dmd_file_path + 'sub' + sub[s_idx] + '_ses' + ses[se_idx] + '_error.mat'
        dmd_cor_data = scipy.io.loadmat(dmd_cor_file_name)['feature'][:, index, :]
        dmd_eor_data = scipy.io.loadmat(dmd_eor_file_name)['feature'][:, index, :]  # [:, index, :]
        self.dmd_data = np.vstack((dmd_cor_data, dmd_eor_data))

        # epoch data
        epoch_cor_file_name = self.epoch_file_path + 'sub' + sub[s_idx] + '_ses' + ses[se_idx] + '_correct.mat'
        epoch_eor_file_name = self.epoch_file_path + 'sub' + sub[s_idx] + '_ses' + ses[se_idx] + '_error.mat'
        epoch_cor_data = scipy.io.loadmat(epoch_cor_file_name)['feature'][:, index, :]
        epoch_eor_data = scipy.io.loadmat(epoch_eor_file_name)['feature'][:, index, :]
        self.epoch_data = np.vstack((epoch_cor_data, epoch_eor_data))

        # labels: 0 for correct, 1 for error
        labels = [0] * len(epoch_cor_data) + [1] * len(epoch_eor_data)
        self.label = np.array(labels, dtype=int)

    def get_train_feature(self, train_index):
        # 0 for correct, 1 for error
        self.dmd_est = ERPCovariances(classes=[1], estimator='lwf')
        self.dmd_prj = TangentSpace(metric='riemann')

        self.epoch_est = ERPCovariances(classes=[0, 1], estimator='lwf')
        self.epoch_prj = TangentSpace(metric='riemann')

        dmd_train = self.dmd_data[train_index, :, :]
        epoch_train = self.epoch_data[train_index, :, :]

        # epoch_vec = np.empty([len(train_index), np.size(epoch_train[0, :, :])])
        # for i in range(len(train_index)):
        #     epoch_vec[i, :] = epoch_train[i, :, :].flatten()

        dmd_vec = np.empty([len(train_index), np.size(dmd_train[0, :, :])])
        for i in range(len(train_index)):
            dmd_vec[i, :] = dmd_train[i, :, :].flatten()

        label_train = self.label[train_index]

        dmd_cov = self.dmd_est.fit(dmd_train, label_train).transform(dmd_train)
        dmd_vec = self.dmd_prj.fit(dmd_cov, label_train).transform(dmd_cov)

        epoch_cov = self.epoch_est.fit(epoch_train, label_train).transform(epoch_train)
        epoch_vec = self.epoch_prj.fit(epoch_cov, label_train).transform(epoch_cov)

        return dmd_vec, epoch_vec, label_train
        # return dmd_vec, label_train

    def get_test_feature(self, test_index):
        dmd_test = self.dmd_data[test_index, :, :]
        epoch_test = self.epoch_data[test_index, :, :]

        # epoch_vec = np.empty([len(test_index), np.size(epoch_test[0, :, :])])
        # for i in range(len(test_index)):
        #     epoch_vec[i, :] = epoch_test[i, :, :].flatten()

        dmd_vec = np.empty([len(test_index), np.size(dmd_test[0, :, :])])
        for i in range(len(test_index)):
            dmd_vec[i, :] = dmd_test[i, :, :].flatten()

        label_test = self.label[test_index]

        dmd_cov = self.dmd_est.transform(dmd_test)
        dmd_vec = self.dmd_prj.transform(dmd_cov)

        epoch_cov = self.epoch_est.transform(epoch_test)
        epoch_vec = self.epoch_prj.transform(epoch_cov)

        return dmd_vec, epoch_vec, label_test
        # return dmd_vec, label_test


'''
-----------------------------------Data-------------------------------------------
'''

dmd_file_path = 'D:/dmdSVM/PhaseData/'
epoch_file_path = 'D:/dmdSVM/EpochData/'
# 0-5, 0-1
sub_idx, ses_idx = 0, 1

rfv = RiemannFeatureVec(dmd_file_path, epoch_file_path)
rfv.load_data(sub_idx, ses_idx)

y = rfv.label
n_samples = y.shape[-1]
# Attention!
emb_dim = len(index)*(len(index)*2+1)

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
    x_dmd_train, x_epoch_train, y_train = rfv.get_train_feature(train_idx)
    x_dmd_test, x_epoch_test, y_test = rfv.get_test_feature(test_idx)
    x_train = np.hstack((normalize(x_dmd_train), normalize(x_epoch_train)))
    x_test = np.hstack((normalize(x_dmd_test), normalize(x_epoch_test)))

    clf.fit(x_train, y_train)
    scores = clf.predict_proba(x_test)
    print(y_test.shape)
    print(scores.shape)
    fpr, tpr, thresholds = roc_curve(y_test, np.array(scores).T[1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--', color='red')
    plt.grid()
    plt.legend()
    plt.show()
    break

