import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data = sio.loadmat("./data_sets/ex6data1.mat")
X,y = data['X'], data['y']
def plot_data():#数据可视化
    plt.scatter(X[:,0], X[:,1], c = y.flatten(), cmap = 'jet')
    plt.xlabel('x1')
    plt.ylabel('y1')
def plot_boundary(model): #绘制决策边界
    x_min, x_max = -0.5,4.5
    y_min, y_max = 1.3,5
    xx,yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                        np.linspace(y_min, y_max, 500))
    z = model.predict(np.c_[xx.flatten(), yy.flatten()])
    zz = z.reshape(xx.shape)
    plt.contour(xx,yy,zz)
    plot_data()
    plt.show()
svc1 = SVC(C = 1, kernel='rbf') #SVC的属性设置
svc1.fit(X,y.flatten())#flatten将y转变为一维数组
print(svc1.predict(X)) #输出svc的预测结果
print("准确率：",svc1.score(X, y.flatten())) #输出准确率
plot_boundary(svc1)

# svc100 = SVC(C=100, kernel='linear')
# svc100.fit(X,y.flatten())#flatten将y转变为一维数组
# print(svc100.predict(X))
# print(svc100.score(X, y.flatten()))
# plot_boundary(svc100)