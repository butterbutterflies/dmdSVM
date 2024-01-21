from sklearn.manifold import TSNE
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


file_path = 'D:/eegDMD/dmdSVM/FeaturesData/'
sub = ['1', '2', '3', '4', '5', '6']
ses = ['1', '2']
cor_file_name = file_path+'sub'+sub[2]+'_ses'+ses[1]+'_correct.mat'
eor_file_name = file_path+'sub'+sub[2]+'_ses'+ses[1]+'_error.mat'
cor_data = scipy.io.loadmat(cor_file_name)['feature']
eor_data = scipy.io.loadmat(eor_file_name)['feature']
# (394, 7, 291) (99, 7, 291)
# print(cor_data.shape, eor_data.shape)

# label: 0 for correct, 1 for error
label = [0]*len(cor_data) + [1]*len(eor_data)
label = np.array(label, dtype=int)

'''
HzData
'''
fourHzData = cor_data[:, 3, :]  # (394, 291)
fourHzData = np.vstack((fourHzData, eor_data[:, 3, :]))  # (493, 291)
# print(fourHzData.shape)
eightHzData = cor_data[:, 4, :]
eightHzData = np.vstack((eightHzData, eor_data[:, 4, :]))
#
twelveHzData = cor_data[:, 5, :]
twelveHzData = np.vstack((twelveHzData, eor_data[:, 5, :]))
#
sixteenHzData = cor_data[:, 6, :]
sixteenHzData = np.vstack((sixteenHzData, eor_data[:, 6, :]))

tsne = TSNE(n_components=2, init='pca', random_state=0)

#
fourTsne = tsne.fit_transform(fourHzData)
plt.scatter(fourTsne[:, 0], fourTsne[:, 1], c=label)
plt.show()

eightTsne = tsne.fit_transform(eightHzData)
plt.scatter(eightTsne[:, 0], eightTsne[:, 1], c=label)
plt.show()

twelveTsne = tsne.fit_transform(twelveHzData)
plt.scatter(twelveTsne[:, 0], twelveTsne[:, 1], c=label)
plt.show()

sixteenTsne = tsne.fit_transform(sixteenHzData)
plt.scatter(sixteenTsne[:, 0], sixteenTsne[:, 1], c=label)
plt.show()

'''
Phase Data
'''
# CzData = cor_data[:, 0, :]  # (394, 291)
# CzData = np.vstack((CzData, eor_data[:, 0, :]))
#
# FzData = cor_data[:, 1, :]
# FzData = np.vstack((FzData, eor_data[:, 1, :]))
#
# FCzData = cor_data[:, 2, :]
# FCzData = np.vstack((FCzData, eor_data[:, 2, :]))
#
# tsne = TSNE(n_components=2, random_state=0)

# CzTsne = tsne.fit_transform(CzData)
# plt.scatter(CzTsne[:, 0], CzTsne[:, 1], c=label)
# plt.show()

# FzTsne = tsne.fit_transform(FzData)
# plt.scatter(FzTsne[:, 0], FzTsne[:, 1], c=label)
# plt.show()

# FCzTsne = tsne.fit_transform(FCzData)
# plt.scatter(FCzTsne[:, 0], FCzTsne[:, 1], c=label)
# plt.show()

