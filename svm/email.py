import joblib
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data1 = sio.loadmat("./data_sets/spamTrain.mat")
X,y = data1['X'], data1['y']
data2 = sio.loadmat("./data_sets/spamTest.mat")
Xtest,ytest = data2['Xtest'], data2['ytest'].flatten()
Cvalues=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100,500]#9
gammas=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100,500]#9
best_score=0
best_params = (0,0)
model_path = 'svm.model'
for c in Cvalues:
    for gamma in gammas:
        svc = SVC(C=c, kernel='rbf',gamma=gamma)
        rf = svc.fit(X,y.flatten())
        score = svc.score(Xtest,ytest.flatten())
        if score > best_score:
            best_score = score
            best_params = (c,gamma)
            joblib.dump(rf, model_path)
print("训练准确率：{:.4f}".format(best_score))
print("最优参数(c,gamma)为：",best_params)
clf = joblib.load(model_path)
error = 0
preResult = clf.predict(Xtest).flatten()
for i in range(len(ytest)):
    if ytest[i] != preResult[i]:
        error += 1
accuracy = (1000 - error) / 1000
print("测试准确率：{:.4f}".format(accuracy))