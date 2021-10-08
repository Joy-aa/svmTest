import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data = sio.loadmat("./data_sets/ex6data2.mat")
X,y = data['X'], data['y']
def plot_data():
    plt.scatter(X[:,0], X[:,1], c = y.flatten(), cmap = 'jet')
    plt.xlabel('x1')
    plt.ylabel('y1')
    # plt.show()
# plot_data()
def plot_boundary(model):
    x_min, x_max = 0,1
    y_min, y_max = 0.4,1
    xx,yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                        np.linspace(y_min, y_max, 500))
    z = model.predict(np.c_[xx.flatten(), yy.flatten()])
    zz = z.reshape(xx.shape)
    plt.contour(xx,yy,zz)
    plot_data()
    plt.show()
svc1 = SVC(C = 1, kernel='sigmoid',gamma=0.1)
st = time.time()
svc1.fit(X,y.flatten())
ed = time.time()
# print(svc1.predict(X))
print("耗时：",ed-st)
print("准确率：",svc1.score(X, y.flatten()))
plot_boundary(svc1)