import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE

from pyriemann.estimation import XdawnCovariances, ERPCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

from mne.decoding import CSP

from dmdFeature import EEGPreprocess


'''
   Subject session
'''
sub = ['1', '2', '3', '4', '5', '6']   # 123
ses = ['1', '2']
sub_idx, ses_idx = 2, 0


'''
   DMD Features
'''
dmd_file_path = 'D:/eegDMD/dmdSVM/FeaturesData/'
cor_file_name = dmd_file_path+'sub'+sub[sub_idx]+'_ses'+ses[ses_idx]+'_correct.mat'
eor_file_name = dmd_file_path+'sub'+sub[sub_idx]+'_ses'+ses[ses_idx]+'_error.mat'

cor_data = scipy.io.loadmat(cor_file_name)['feature']
eor_data = scipy.io.loadmat(eor_file_name)['feature']
# (394, 7, 291) (99, 7, 291)
DmdData = np.vstack((cor_data, eor_data))
# DmdData = DmdData[:, 0:3, :]   # phase feature only
# DmdData = DmdData[:, 3::, :]   # spectrum feature only

# label: 0 for correct, 1 for error
label = [0]*len(cor_data) + [1]*len(eor_data)
label = np.array(label, dtype=int)

'''
  spatial filtered DMD features 
'''
filteredData = XdawnCovariances(nfilter=2, estimator='lwf').fit_transform(DmdData, label)
# XdawnCovariances: (493, 8, 8)
# ERPCovariances: (493, 21, 21)

DmdPrjVector = TangentSpace(metric='riemann').fit_transform(filteredData, label)
# (493, 36)   # N*(N+1)/2
# tsne = TSNE(n_components=2, init='pca', random_state=0)
# prjTsne = tsne.fit_transform(prjVector)
# plt.scatter(prjTsne[:, 0], prjTsne[:, 1], c=label)
# plt.show()


'''
  Original Signals   &  CSP Features
'''
CLabel = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5',
          'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3',
          'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz',
          'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz',
          'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz',
          'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2',
          'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

chan_index = [CLabel.index('Cz'), CLabel.index('Fz'), CLabel.index('FCz')]

file_path = 'D:/ErrpData/theMonitoring/'

low = 20.
ts, te = -1, 2
downSample = 300

eeg_process = EEGPreprocess(file_path, sub[sub_idx], ses[ses_idx], downSample)
error, correct = eeg_process.read_preprocess_epoch_data(low, ts, te)

error, correct = error[:, :, 240:540], correct[:, :, 240:540]
oriData = np.vstack((correct, error))
# CSP
csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
CspFeature = csp.fit_transform(oriData, label)   # (493, 4)


'''
   PrjDmdFeature + CSPFeature
'''
DmdPrjVector, CspFeature = normalize(DmdPrjVector), normalize(CspFeature)
AugFeature = np.hstack((DmdPrjVector, CspFeature))


'''
    LR   or   SVM   or   LDA
'''
totalData = AugFeature

cv = KFold(n_splits=10, shuffle=True, random_state=42)
clf = make_pipeline(LinearDiscriminantAnalysis())    # LogisticRegression()   SVC(kernel='rbf')

pr = np.zeros(len(label))
for train_idx, test_idx in cv.split(totalData):
    y_train, y_test = label[train_idx], label[test_idx]

    clf.fit(totalData[train_idx], y_train)
    pr[test_idx] = clf.predict(totalData[test_idx])

# Printing the results
print(classification_report(label, pr))

acc = np.mean(pr == label)
print("Classification accuracy: %f " % acc)

# confusion matrix
names = ["correct", "error"]
cm = confusion_matrix(label, pr)
ConfusionMatrixDisplay(cm, display_labels=names).plot()
plt.show()


'''
  Classification with XDAWN + MDM
'''
# n_components = 3
# crossV = KFold(n_splits=10, shuffle=True, random_state=42)
# clf = make_pipeline(XdawnCovariances(n_components), MDM())
#
# pr = np.zeros(len(label))   # predict the results
# for train_idx, test_idx in crossV.split(DmdData):
#     y_train, y_test = label[train_idx], label[test_idx]
#
#     clf.fit(DmdData[train_idx], y_train)
#     pr[test_idx] = clf.predict(DmdData[test_idx])
#
# print(classification_report(label, pr))
#
# # confusion matrix
# names = ["correct", "error"]
# cm = confusion_matrix(label, pr)
# ConfusionMatrixDisplay(cm, display_labels=names).plot()
# plt.show()
