import scipy.io as sio


myMat = sio.loadmat('03_01_c0001_info.mat')

print myMat['pose']