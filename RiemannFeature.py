import numpy as np
import random
import scipy.io

import torch
from torch import nn, optim
from torch.autograd import Variable
from pytorchtools import EarlyStopping

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import normalize

from pyriemann.estimation import ERPCovariances
from pyriemann.tangentspace import TangentSpace


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

index = [0, 1, 2, 3, 4, 7, 10, 5, 6]
# index = [0, 1, 2]

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
        dmd_eor_data = scipy.io.loadmat(dmd_eor_file_name)['feature'][:, index, :]
        self.dmd_data = np.vstack((dmd_cor_data, dmd_eor_data))

        # epoch data
        epoch_cor_file_name = self.epoch_file_path + 'sub' + sub[s_idx] + '_ses' + ses[se_idx] + '_correct.mat'
        epoch_eor_file_name = self.epoch_file_path + 'sub' + sub[s_idx] + '_ses' + ses[se_idx] + '_error.mat'
        epoch_cor_data = scipy.io.loadmat(epoch_cor_file_name)['feature'][:, index, :]
        epoch_eor_data = scipy.io.loadmat(epoch_eor_file_name)['feature'][:, index, :]
        self.epoch_data = np.vstack((epoch_cor_data, epoch_eor_data))

        # labels: 0 for correct, 1 for error
        labels = [0] * len(dmd_cor_data) + [1] * len(dmd_eor_data)
        self.label = np.array(labels, dtype=int)

    def get_train_feature(self, train_index):
        # 0 for correct, 1 for error
        self.dmd_est = ERPCovariances(classes=[1], estimator='lwf')
        self.dmd_prj = TangentSpace(metric='riemann')

        self.epoch_est = ERPCovariances(classes=[1], estimator='lwf')
        self.epoch_prj = TangentSpace(metric='riemann')

        dmd_train = self.dmd_data[train_index, :, :]
        epoch_train = self.epoch_data[train_index, :, :]
        label_train = self.label[train_index]

        dmd_cov = self.dmd_est.fit(dmd_train, label_train).transform(dmd_train)
        dmd_vec = self.dmd_prj.fit(dmd_cov, label_train).transform(dmd_cov)

        epoch_cov = self.epoch_est.fit(epoch_train, label_train).transform(epoch_train)
        epoch_vec = self.epoch_prj.fit(epoch_cov, label_train).transform(epoch_cov)

        return dmd_vec, epoch_vec, label_train

    def get_test_feature(self, test_index):
        dmd_test = self.dmd_data[test_index, :, :]
        epoch_test = self.epoch_data[test_index, :, :]
        label_test = self.label[test_index]

        dmd_cov = self.dmd_est.transform(dmd_test)
        dmd_vec = self.dmd_prj.transform(dmd_cov)

        epoch_cov = self.epoch_est.transform(epoch_test)
        epoch_vec = self.epoch_prj.transform(epoch_cov)

        return dmd_vec, epoch_vec, label_test


class GatedSumLR(nn.Module):

    def __init__(self, embedding_dim):
        super(GatedSumLR, self).__init__()

        self.embedding_dim = embedding_dim
        self.input_dim = 2 * self.embedding_dim
        self.hidden_dim = self.embedding_dim
        self.output_dim = 2

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = [dmd_embedding, epoch_embedding]
        dmd_embedding, epoch_embedding = x[0], x[1]
        x = torch.cat(x, dim=1)
        # x = dmd_embedding + epoch_embedding
        # x = epoch_embedding
        g = self.sigmoid(self.fc1(x))
        h = g * dmd_embedding + (1 - g) * epoch_embedding
        o = self.fc2(h)
        return o


'''
-----------------------------------Data-------------------------------------------
'''

dmd_file_path = 'D:/eegDMD/dmdSVM/PhaseData/'
epoch_file_path = 'D:/eegDMD/dmdSVM/EpochData/'
# 0-5, 0-1
sub_idx, ses_idx = 0, 0

rfv = RiemannFeatureVec(dmd_file_path, epoch_file_path)
rfv.load_data(sub_idx, ses_idx)

y = rfv.label
n_samples = y.shape[-1]

# Attention!
emb_dim = len(index)*(len(index)*2+1)


'''
------------------------------------Net-------------------------------------------
'''
seed = 20
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

use_gpu = torch.cuda.is_available()

num_epochs = 500

'''
-----------------------------------Train------------------------------------------
'''
# 3-Folds Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)

i = 0
pr = np.zeros(n_samples)   # predict the results
for train_idx, test_idx in skf.split(np.zeros(n_samples), y):
    # Providing y is sufficient to generate the splits and hence np.zeros(n_samples)
    # may be used as a placeholder for X instead of actual training data.
    i += 1
    print("---------------K-Fold: %d----------------" % i)

    ''' Data Preparation '''
    x_dmd_train, x_epoch_train, y_train = rfv.get_train_feature(train_idx)
    x_dmd_test, x_epoch_test, y_test = rfv.get_test_feature(test_idx)
    # x_dmd_train, x_epoch_train = normalize(x_dmd_train), normalize(x_epoch_train)
    # x_dmd_test, x_epoch_test = normalize(x_dmd_test), normalize(x_epoch_test)
    x_dmd_train = Variable(torch.Tensor(x_dmd_train))
    x_epoch_train = Variable(torch.Tensor(x_epoch_train))
    y_train = Variable(torch.LongTensor(y_train))
    x_dmd_test = Variable(torch.Tensor(x_dmd_test))
    x_epoch_test = Variable(torch.Tensor(x_epoch_test))
    y_test = Variable(torch.LongTensor(y_test))
    if use_gpu:
        x_dmd_train, x_epoch_train, y_train = x_dmd_train.cuda(), x_epoch_train.cuda(), y_train.cuda()
        x_dmd_test, x_epoch_test, y_test = x_dmd_test.cuda(), x_epoch_test.cuda(), y_test.cuda()
    x_train = [x_dmd_train, x_epoch_train]
    x_test = [x_dmd_test, x_epoch_test]

    ''' Model '''
    gsLR = GatedSumLR(emb_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gsLR.parameters(), lr=0.001, weight_decay=0.0001)  # regularization

    if use_gpu:
        gsLR = gsLR.cuda()
        criterion = criterion.cuda()

    ''' epoch loop '''
    patience = 20
    early_stopping = EarlyStopping(patience, verbose=True)
    for epoch in range(num_epochs):
        # backward
        gsLR.train()
        optimizer.zero_grad()
        outputs = gsLR(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if use_gpu:
            loss = loss.cpu()
        if epoch % 10 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        # early stop
        gsLR.eval()
        test_pre = gsLR(x_test)
        if use_gpu:
            test_pre = test_pre.cpu()
            y_test = y_test.cpu()
        o_pre = test_pre.data.numpy()
        o_test = y_test.data.numpy()
        test_loss = criterion(torch.Tensor(o_pre), torch.LongTensor(o_test))
        early_stopping(test_loss, gsLR)
        if early_stopping.early_stop:
            print("Early stopping: Test loss: %1.5f" % test_loss)
            break

    gsLR.eval()
    test_pre = gsLR(x_test)
    if use_gpu:
        test_pre = test_pre.cpu()
        y_test = y_test.cpu()
    o_pre = test_pre.data.numpy()
    o_label = np.argmax(o_pre, axis=1)
    pr[test_idx] = o_label

print("----------------------------------------------------------")
print(classification_report(y, pr))
# confusion matrix
names = ["correct", "error"]
cm = confusion_matrix(y, pr)
ConfusionMatrixDisplay(cm, display_labels=names).plot()
plt.show()
