import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mne.decoding import CSP, UnsupervisedSpatialFilter


'''
   Subject session
'''
sub = ['1', '2', '3', '4', '5', '6']   # 123
ses = ['1', '2']


class CSPFeatureVec:

    def __init__(self, epoch_path):
        self.epoch_file_path = epoch_path

        self.epoch_data = None
        self.label = None

        self.csp = None
        self.pca = None

    def load_data(self, s_idx, se_idx):
        # epoch data
        epoch_cor_file_name = self.epoch_file_path + 'sub' + sub[s_idx] + '_ses' + ses[se_idx] + '_correct.mat'
        epoch_eor_file_name = self.epoch_file_path + 'sub' + sub[s_idx] + '_ses' + ses[se_idx] + '_error.mat'
        epoch_cor_data = scipy.io.loadmat(epoch_cor_file_name)['feature'][:, :, 60::]
        epoch_eor_data = scipy.io.loadmat(epoch_eor_file_name)['feature'][:, :, 60::]
        self.epoch_data = np.vstack((epoch_cor_data, epoch_eor_data))

        # labels: 0 for correct, 1 for error
        labels = [0] * len(epoch_cor_data) + [1] * len(epoch_eor_data)
        self.label = np.array(labels, dtype=int)

    def get_train_feature(self, train_index):
        # CSP
        self.pca = UnsupervisedSpatialFilter(PCA(n_components='mle', svd_solver='full'), average=True)
        self.csp = CSP(n_components=30, reg='empirical', log=True, cov_est='epoch', norm_trace=True)
        # adding PCA

        epoch_train = self.epoch_data[train_index, :, :]
        label_train = self.label[train_index]

        epoch_pca = self.pca.fit_transform(epoch_train, label_train)
        epoch_csp = self.csp.fit_transform(epoch_pca, label_train)

        return epoch_csp, label_train

    def get_test_feature(self, test_index):
        epoch_test = self.epoch_data[test_index, :, :]
        label_test = self.label[test_index]

        epoch_pca = self.pca.transform(epoch_test)
        epoch_csp = self.csp.transform(epoch_pca)

        return epoch_csp, label_test


'''
-----------------------------------Data-------------------------------------------
'''

epoch_file_path = 'D:/eegDMD/dmdSVM/300-chan-Epoch/'
# 0-5, 0-1
sub_idx, ses_idx = 5, 1

cfv = CSPFeatureVec(epoch_file_path)
cfv.load_data(sub_idx, ses_idx)

y = cfv.label
n_samples = y.shape[-1]

# data = cfv.epoch_data
# plt.plot(data[-1, 46, :])
# plt.show()

'''
-----------------------------------Classifier--------------------------------------
'''
clf = make_pipeline(LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'))

'''
------------------------------------Train--------------------------------------------
'''
# RepeatedStratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)

pr = np.zeros(n_samples)   # predict the results
for train_idx, test_idx in skf.split(np.zeros(n_samples), y):
    ''' Data Preparation '''
    x_epoch_train, y_train = cfv.get_train_feature(train_idx)
    x_epoch_test, y_test = cfv.get_test_feature(test_idx)
    x_train = x_epoch_train
    x_test = x_epoch_test

    print(x_train.shape)
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
