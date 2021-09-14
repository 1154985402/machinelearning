# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
from sklearn import svm
from sklearn.metrics import accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data_read = urllib.request.urlopen(url);
datas = np.loadtxt(data_read,delimiter=",",dtype=str )
datas[datas == 'Iris-setosa'] = 1
datas[datas == 'Iris-versicolor'] = 2
datas[datas == 'Iris-virginica'] = 3
data = datas.astype(np.float64)
data1 = data[data[:,4] == 1]
data2 = data[data[:,4] == 2]
data3 = data[data[:,4] == 3]
print(data1)
plt.figure(figsize=(10,10),dpi=100)
plt.scatter(data1[:,0],data1[:,1],c='r')
plt.scatter(data2[:,0],data2[:,1],c='b')
plt.scatter(data3[:,0],data3[:,1],c='g')
plt.show()
X_train, X_test, y_train, y_test =\
    sklearn.model_selection.train_test_split(datas[:, 0:3], datas[:, 4], test_size=0.5, random_state=0)

gammas = [ i/100 for i in range(1,10,1)]
Cs = [i for i in range(5,20,1)]
maxscore = 0;
bestC = 0;
bestgamma = 0;
scores = []
k = []
i = 0;
for (gamma, c) in zip(gammas,Cs):
        k.append(i)
        i = i + 1
        print(gamma,c)
        model =  svm.SVC(C = c ,gamma = gamma,kernel='rbf')
        model.fit(X_train,y_train)
        y_res = model.predict(X_test)
        score = accuracy_score(y_test,y_res)
        scores.append(score)
        print("score:{%f}"%(score))
        if(score > maxscore):
            maxscore = score
            bestC = c
            bestgamma = gamma

plt.figure(figsize=(30,20))
plt.plot(k,scores,'s-',color = 'r',label="ATT-RLSTM")

plt.show()

print("when gamma = {%s} and C = {%d} , the score is {%s}"%(bestgamma,bestC,maxscore))



# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
