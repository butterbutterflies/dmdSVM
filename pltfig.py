import matplotlib.pyplot as plt
import numpy as np

# 输入统计数据
sub = ('Subject 1', 'Subject 2', 'Subject 3', 'Subject 4', 'Subject 5', 'Subject 6')

csp = [[0.70, 0.22, 0.09], [0.73, 0.38, 0.35], [0.77, 0.23, 0.07],
       [0.80, 0.59, 0.18], [0.74, 0.15, 0.05], [0.78, 0.33, 0.10]]
csp = np.array(csp)
wav = [[0.88, 0.77, 0.74], [0.79, 0.54, 0.48], [0.87, 0.70, 0.62],
       [0.83, 0.58, 0.58], [0.83, 0.62, 0.55], [0.74, 0.36, 0.29]]
wav = np.array(wav)
dmd = [[0.91, 0.87, 0.75], [0.85, 0.74, 0.56], [0.91, 0.86, 0.70],
       [0.87, 0.74, 0.62], [0.88, 0.76, 0.67], [0.77, 0.46, 0.31]]
dmd = np.array(dmd)

dr = [[0.83,0.77,0.85,0.71,0.78,0.71],[0.75,0.57,0.71,0.30,0.60,0.27],[0.49,0.29,0.52,0.09,0.29,0.03]]
dr = np.array(dr)
er=[[0.90, 0.85, 0.92, 0.87, 0.88, 0.78],[0.86, 0.75, 0.88, 0.77, 0.78, 0.55], [0.70, 0.55, 0.71, 0.57, 0.61, 0.28]]
er = np.array(er)


bar_width = 0.2
idx_csp = np.arange(len(sub))
idx_wav = idx_csp + bar_width
idx_dmd = idx_wav + bar_width

# 使用两次 bar 函数画出两组条形图 color='steelbule',
plt.bar(idx_csp, height=dr[2, :], width=bar_width, color='#F0E68C', label='DR')
plt.bar(idx_wav, height=er[2, :], width=bar_width, color='#FFC0CB', label='ER')
plt.bar(idx_dmd, height=dmd[:, 2], width=bar_width, color='#8FBC8F', label='DR+ER')

# plt.bar(idx_csp, height=dr[0, :], width=bar_width, label='DR')
# plt.bar(idx_wav, height=er[0, :], width=bar_width, label='ER')
# plt.bar(idx_dmd, height=dmd[:, 0], width=bar_width, label='DR+ER')

# plt.barh(dr[0, :], idx_csp, label='DR')
# plt.barh(er[0, :], idx_wav, label='ER')
# plt.barh(dmd[:, 0], idx_dmd, label='DR+ER')

loc = "lower left"
plt.legend(loc=loc)
plt.xticks(idx_wav, sub)
plt.ylabel('Weighted Accuracy')
# plt.title('')

plt.show()

# DMD
# 0.91    0.87    0.75
# 0.93    0.88    0.78
# 0.85    0.74    0.56
# 0.82    0.71    0.47
# 0.91    0.86    0.70
# 0.92    0.86    0.67
# 0.87    0.74    0.62
# 0.82    0.62    0.40
# 0.88    0.76    0.67
# 0.85    0.74    0.54
# 0.77    0.46    0.31
# 0.80    0.44    0.24

# wav = [[0.88, 0.77, 0.74], [0.79, 0.54, 0.48], [0.87, 0.70, 0.62],
#        [0.83, 0.58, 0.58], [0.83, 0.62, 0.55], [0.74, 0.36, 0.29]]
# waveform
# 0.88     0.77    0.74
# 0.91     0.85    0.73
# 0.79     0.54    0.48
# 0.78     0.52    0.51
# 0.87     0.70    0.62
# 0.91     0.79    0.66
# 0.83     0.58    0.58
# 0.77     0.42    0.35
# 0.83     0.62    0.55
# 0.77     0.49    0.43
# 0.74     0.36    0.29
# 0.77     0.32    0.27

# csp = [[0.70, 0.22, 0.09], [0.73, 0.38, 0.35], [0.77, 0.23, 0.07],
#        [0.80, 0.59, 0.18], [0.74, 0.15, 0.05], [0.78, 0.33, 0.10]]
# csp
# 0.70    0.22    0.09
# 0.77    0.38    0.14
# 0.73    0.38    0.35
# 0.70    0.33    0.34
# 0.77    0.23    0.07
# 0.78    0.12    0.06
# 0.80    0.59    0.18
# 0.77    0.17    0.05
# 0.74    0.15    0.05
# 0.76    0.39    0.15
# 0.78    0.33    0.10
# 0.80    0.18    0.05

# DR
# dr = [[0.83,0.77,0.85,0.71,0.78,0.71],[0.75,0.57,0.71,0.30,0.60,0.27],[0.49,0.29,0.52,0.09,0.29,0.03]]
# 0.83	0.84	0.77	0.73	0.85	0.86	0.71	0.72	0.78	0.77	0.71	0.76
# 0.75	0.71	0.57	0.46	0.71	0.68	0.30	0.20	0.60	0.54	0.27	0.29
# 0.49	0.47	0.29	0.19	0.52	0.45	0.09	0.05	0.29	0.31	0.03	0.04
# er=[[0.90, 0.85, 0.92, 0.87, 0.88, 0.78],[0.86, 0.75, 0.88, 0.77, 0.78, 0.55], [0.70, 0.55, 0.71, 0.57, 0.61, 0.28]]
# 0.90	0.93	0.85	0.82	0.92	0.92	0.87	0.81	0.88	0.83	0.78	0.82
# 0.86	0.88	0.75	0.69	0.88	0.88	0.77	0.61	0.78	0.72	0.55	0.53
# 0.70	0.76	0.55	0.46	0.71	0.64	0.57	0.34	0.61	0.48	0.28	0.28
