# import numpy as np
# import matplotlib.pyplot as plt

# a = np.arange(5) + 1j*np.arange(6, 11)
# fig, ax = plt.subplots()
# ax.scatter(a.real, a.imag)
# plt.show()

# a1 = [[1]*4 for _ in range(2)]
# a2 = [[2]*4 for _ in range(3)]
# a3 = [[3]*4 for _ in range(4)]
#
# tt = []
#
# ee = []
# for j in range(5):
#     ee.append(a1)
# tt.append(ee)
#
# ee = []
# for j in range(5):
#     ee.append(a2)
# tt.append(ee)
#
# ee = []
# for j in range(5):
#     ee.append(a3)
# tt.append(ee)
#
# print(tt[2][4])
#
# temp = tt[2][4]
#
# print(temp[0])

import numpy as np
import scipy.io
import math

from pyriemann.estimation import ERPCovariances, Covariances

from sklearn.covariance import GraphicalLassoCV, ledoit_wolf, empirical_covariance, OAS

# subject session
sub = ['1', '2', '3', '4', '5', '6']
ses = ['1', '2']

dmd_file_path = 'D:/eegDMD/dmdSVM/dmd-modes-phase/'
dmd_cor_file_name = dmd_file_path + 'sub' + sub[0] + '_ses' + ses[0] + '_error.mat'

dmd_cor_data = scipy.io.loadmat(dmd_cor_file_name)['feature'][:, 2, :, :]
# print(dmd_eor_data.shape)

# for i in range(len(dmd_cor_data)):
#     temp = dmd_cor_data[i, :, :]
#     temp_cov_cpx = empirical_covariance(temp.T)
#     temp_cov = np.abs(temp_cov_cpx)
#     eigs = np.linalg.eigvals(temp_cov)
#     if not np.all(eigs > 0):
#         print(eigs)
#         print(i)

temp = dmd_cor_data[2, :, :]
temp_cov_cpx = empirical_covariance(temp.T)
# print(temp)
# # print(temp.shape)
# #
# # # temp_cov, _ = ledoit_wolf(temp)
# # # print(temp_cov)
# #
# temp_cov_cpx = np.cov(temp)
temp_cov = np.abs(temp_cov_cpx)
# eigs = np.linalg.eigvals(temp_cov)
# idx = np.where(eigs > 0)
if not np.all(np.linalg.eigvals(temp_cov) > 0):
    vals, vecs = np.linalg.eig(temp_cov)
    vals[np.where(vals <= 0)] = 0.0001
    lam = np.diag(vals)
    inv_vecs = np.linalg.inv(vecs)
    temp_cov_new = np.dot(np.dot(vecs, lam), inv_vecs)
    temp_cov_new = (temp_cov_new + temp_cov_new.T) / 2
    print(temp_cov_new)
    print(np.all(np.linalg.eigvals(temp_cov_new) > 0))
print('-------------')
print(temp_cov)


def isSymmetrical(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if not x[i][j] == x[j][i]:
                return False
    else:
        return True


print(isSymmetrical(temp_cov))

# print(eigs)
# print(idx)
# print(np.place(eigs, eigs <= 0, 0.001))
# eigs[np.where(eigs <= 0)] = 0.001
# print(eigs)
# print(temp_cov)
# print(np.abs(temp_cov))
# # print(temp_cov.shape)
#
# # dmd_est = Covariances(estimator='lwf')
