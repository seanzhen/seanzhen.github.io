---
layout: post
title: Ridge Regression
---

## Ridge Regression
### 概念

> * 算法适用范围

岭回归最先用来处理特征数多于样本数的情况，现在也用于在估计中加入偏差，从而得到更好的估计。这里通过引入$\lambda $来限制了所有w之和，通过引入该惩罚项，能够减少不重要的参数，这个技术在统计学中也叫缩减。


> * **岭回归中的岭是什么？**

岭回归使用了单位矩阵乘以常量lambda，我们观察其中的单位矩阵I，可以看到值I贯穿整个对角线，其余元素全是0，形象地，在0构成的平面上有一条1组成的“岭”，这就是岭回归中的“岭”的由来。

缩减方法可以去掉不重要的参数，因此能更好地理解数据。此外，与简单的线性回归相比，缩减法能取得更好的预测效果。

### 正则化与稀疏性 （学习于《百面机器学习》）

稀疏性说白了就是模型的很多参数为0，这相当于对模型进行了一次特征选择，只留下一些比较重要的特征，提高模型的繁华能力，降低过拟合的可能性。

> * L1正则化与L2 正则化

L2正则项约束后的解空间是圆形，而L1正则项约束的解空间是多边形。显然，多边形的解空间更容易在尖角处与等高线碰撞出稀疏解。

> * 为什么L1和L2的解空间是不用的？

事实上，“带正则项”和“带约束条件”是等价的。为了约束w的可能取值空间从而方植过拟合，我们为该最优化问题加上一个约束，就是w的L2范数的平方不能大于m。

L2正则化相当于为参数定义了一个圆形的解空间（因为必须保证L2范数不能大于m), 而L1正则化相当于为参数定义了一个菱形的解空间。如果原问题目标函数的最优解不是恰好落在解空间内，那么约束条件下的最优解一定是在解空间的边界上，而L1“棱角分明”的解空间显然更容易与目标函数等高线在角点碰撞，从而产生稀疏解。

### 公式推导
![（等会儿再写）][1]

```python
#岭回归
def ridgeRegress(xMat,yMat,lam = 0.2) #lambda = 0.2
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1]*lam
    if linalg.set(denom) == 0.0:
        print "This matrix id singular, cannot do inverse"
        return
    ws = denom.I*(xMat.T*yMat)
    return ws
def ridgeTest(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    XMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMeans)/xVar
    #以上步骤为标准化
    numTestPts = 30 #30个lambda
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat,yMat,lam = exp(i-10))
        wMat[i:] = ws.T
    return wMat
    #这样可以看出lambda在取非常小的值时和取非常大的值时分别对结果造成的影响
```
在sklearn中的文档代码：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
# X is the 10x10 Hilbert matrix
#生成数据
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)
# print(X)
# print(y)
n_alphas = 200 #lambda
alphas = np.logspace(-10, -2, n_alphas) #从-10到-2两百个数
clf = linear_model.Ridge(fit_intercept=False)#初定义，不需要截距

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, y)
    coefs.append(clf.coef_)#10个因子
plt.figure(figsize=(10,8))
ax = plt.gca()#gca() returns the current axes
ax.plot(alphas, coefs)
ax.set_xscale('log')#注意这一步，alpha是对数化了的
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')#图边界的设置
plt.show()
```
得到结果
![ridge regression.png-50.8kB][2]
    

   
  [1]: http://static.zybuluo.com/seanzhen/yibkjzhildedj4jpkh6qj7an/ridge%20regression1.png
  [2]: http://static.zybuluo.com/seanzhen/gm39krd5fq8e96389e06i05t/ridge%20regression.png
  [3]: http://static.zybuluo.com/seanzhen/wic70ly5b1qsm5s0qyj8ix7h/%E6%9E%81%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1.png
