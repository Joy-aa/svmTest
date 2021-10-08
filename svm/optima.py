import time

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from numpy import mat
from sklearn.svm import SVC

data = sio.loadmat("./data_sets/ex6data3.mat")
X,y = data['X'], data['y']
Xval, yval=data['Xval'], data['yval']
def plot_data():
    plt.scatter(X[:,0], X[:,1], c = y.flatten(), cmap = 'jet')
    plt.xlabel('x1')
    plt.ylabel('y1')
    # plt.show()
# plot_data()
def plot_boundary(model):
    x_min, x_max = -1,0.4
    y_min, y_max = -0.7,1
    xx,yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                        np.linspace(y_min, y_max, 500))
    z = model.predict(np.c_[xx.flatten(), yy.flatten()])
    zz = z.reshape(xx.shape)
    plt.contour(xx,yy,zz)
    plot_data()
    plt.show()
Cvalues=[0.01, 0.05, 0.1, 0.3, 1, 3, 10, 30, 100,500]#10组参数
gammas=[0.01, 0.05, 0.1, 0.3, 1, 3, 10, 30, 100,500]#10组参数
best_score=0
best_params = (0,0)
# for c in Cvalues:
#     for gamma in gammas:
#         svc = SVC(C=c, kernel='rbf', gamma=gamma)
#         svc.fit(X,y.flatten())
#         score = svc.score(Xval,yval.flatten())
#         if score > best_score:
#             best_params = (c, gamma)
#             best_score = score
# print("最佳识别率为：",best_score)
# print("参数（c，gamma）为：",best_params)
st = time.time()
svc2 = SVC(C = 0.3, kernel='rbf',gamma=100)
svc2.fit(X,y.flatten())
ed = time.time()
# print(svc2.predict(X))
print("耗时：",ed-st)
print("准确率：",svc2.score(X, y.flatten()))
plot_boundary(svc2)