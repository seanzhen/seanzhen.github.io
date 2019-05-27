---
layout: post
title: linear regression & logistics regression学习笔记
---

![_config.yml]({{ site.baseurl }}/images/L2.png)



本文是作者学习周志华教授的《机器学习》这本书后整理的简要笔记，内容大部分来源于周志华教授的书籍资料。

------
## Linear Regression
### 1.基本形式

给定有d个属性描述的示例![](http://latex.codecogs.com/gif.latex?\\\x=(x_{1};x_{2};...;x_{d})),其中![](http://latex.codecogs.com/gif.latex?\\\x_{i})是x在第i个属性上的取值，线性模型试图学得一个通过属性的线性组合来进行预测的函数，即
![](http://latex.codecogs.com/gif.latex?\\\f(x)=w_{1}x_{1}+w_{2}x_{2}+...+w_{d}x_{1}+b)
一般用向量形式写成：
![](http://latex.codecogs.com/gif.latex?\\\f(x)=w^{T}x+b)

> * 线性模型形式简单、易于建模，但却稳藏着机器学习中一些重要的基本思想。
> * 许多功能更为强大的非线性模型可在线性模型的基础上通过引入层级结构或高维映射而得。
> * 此外由于w直观表达了各属性在预测中的重要性，因此线性模型有很好的的解释性。

### 2.线性回归

![linear regrssion.png-12.8kB][1]

**我们首先考虑单自变量的线性回归**

![L2.png-268.3kB][2]
![L3.png-132.4kB][3]
![L4.png-22.4kB][4]


更一般的情形如开头的数据集D，样本由d个属性描述，此时我们试图学得
![](http://latex.codecogs.com/gif.latex?\\\f(x_{i})=w^{T}x_{i}+b,使得f(x_{i})\simeq y_{i})
这称为“多元线性回归”
![](https://s21.postimg.org/fyph7d1g7/image.png)
![](https://s28.postimg.org/9vpqj162l/image.png)
![](http://chuantu.biz/t5/60/1491630752x2890149782.png)

### 3.对数线性回归

线性回归虽简单，却又丰富的变化。例如对于样例![](http://latex.codecogs.com/gif.latex?\\\(x,y)，y\in R),当我们希望线性模型的预测值逼近真实标记y时，就得到了线性模型，为便于观察，我们把线性回归模型简写为
![](http://latex.codecogs.com/gif.latex?\\\y = w^{T}x+b)
可否令模型预测值逼近y的衍生物呢？譬如说，假设我们示例多对应的输出标记是在指数尺度上变化，那就可将输出标记的对数作为线性模型逼近的目标，即
![](http://latex.codecogs.com/gif.latex?\\\lny=w^{T}x+b)
这就是“对数线性回归”，它实际上是在试图让$e^{w^{T}x+b}$逼近y，在形式上仍是线性回归，但实质上已是在求取输入空间到输出空间的非线性函数映射，这里的对数函数起到了将线性回归模型的预测值与真实标记联系起来的作用。
![](http://chuantu.biz/t5/60/1491631003x2890149782.png)

更一般地，考虑单调可微函数$g(·)$，令
![](http://latex.codecogs.com/gif.latex?\\\y=g^{-1}(w^{T}x+b))
这样得到的模型称为“广义线性模型”，其中函数$g(·)$称为“联系函数”（link function）。显然，对数线性回归是广义线性模型在![](http://latex.codecogs.com/gif.latex?\\\g(·)=ln(·))时的特例。

------

## Logistics Regression

上面讨论了如何使用线性模型进行学习，但若要做的是分类任务该怎么办？可用广义线性模型：只需找一个单调可微函数将分类任务的真实标记y与线性回归的预测值联系起来
考虑二分类任务，其输出标记![](http://latex.codecogs.com/gif.latex?\\y \in {0,1})，而线性回归模型产生的预测值![](http://latex.codecogs.com/gif.latex?\\z=w^{T}x+b)是实值，于是，我们需将实值z转换成0/1的值，最理想的即是“单位阶跃函数”
![](http://chuantu.biz/t5/60/1491631463x2890149782.png)
上图的y可看出，单位阶跃函数不连续，因此不能直接用作广义线性模型中的![](http://latex.codecogs.com/gif.latex?\\g^{-}(·))，于是我们希望找到能在一定程度上近似单位阶跃函数的“替代函数”，并希望它单调可微，logistics正式这样一个常用的替代函数：
![](http://latex.codecogs.com/gif.latex?\\\y = \frac{1}{1+e^{-z}})
可以看出，logistics函数是一种“Sigmoid函数”，它将z值转化为一个接近0或者1的y值，并且其输出值在z=0附近变化很陡，将logistics函数作为$g^{-}(·)$带入可得，得到
![](http://latex.codecogs.com/gif.latex?\\y = \frac{1}{1+e^{-(w^{T}x+b)}})
可变化为
![](http://latex.codecogs.com/gif.latex?\\frac{y}{1-y}=w^{T}x+b)
若将y视为样本x作为正例的可能性，则1-y是其反例可能性，两者的比值![](http://latex.codecogs.com/gif.latex?\\frac{y}{1-y})称为odds(几率），反映了x作为正例的相对可能性，对几率取对数则得到“对数几率”（logit）
![](http://latex.codecogs.com/gif.latex?\\\ln\frac{y}{1-y})
由此可看出，上式实际上是在用线性回归模型的预测结果去逼近真实标记的logit，因此，其对应的模型就称为“logistics regression”.特别需要注意到，虽然它的名字是“回归”，但实际却是一种分类学习方法。这种方法有很多优点：

- [x] 例如它是直接对分类可能性进行建模，无需事先假设数据分布，这样就避免了假设分布不准确多带来的问题
- [x] 它不是仅预测出“类别”，而是可得到近似概率预测，这对许多需利用概率辅助决策的任务很有用；
- [x] 此外，logit函数是任意阶可导的凸函数，有恨到的数学性质，现有的许多数值优化算法都可直接用于求取最优解

![](https://s8.postimg.org/5gxcqlrzp/L10.png)
## 极大似然估计
估计类条件概率的一种常用策略是先假定其具有某种确定的概率分布形式，再基于训练样本对概率分布的参数进行估计。
事实上，概率模型的训练过程就是参数估计过程。对于参数估计，统计学界的两个学派分别提供了不同的解决方案：

> * 频率主义学派认为参数虽然未知，但却是客观存在的固定值，因此，可通过优化似然函数等准则来确定参数值。
> * 贝叶斯学派则认为参数是未观察到的随机变量，其本身也可有分布，因此，可假定参数服从一个先验分布，然后观测到的数据来计算参数的后验分布。

这里是源自于频率主义学派的极大似然估计，这是根据数据采样来估计概率分布参数的经典方法。
![L15.png-63.1kB][5]
**需注意的是：** 用过这种参数化的方法虽能是类条件概率估计变得相对简单，但估计结果的准备性严重依赖于所假设的概率分布形式是否符合潜在的真实数据分布。在现实应用中，欲做出能较好地接近潜在真是分布的假设，往往需在一定程度上利用关于应用任务本身的经验知识，否则若凭“猜测”来假设概率分布形式，很可能产生误导性的结果。

![](http://chuantu.biz/t5/60/1491632592x2890149782.png)
![](http://chuantu.biz/t5/60/1491632607x2890149782.png)

## 梯度下降
梯度下降法（gradient descent）是一种常用的一阶（first-order）优化方法，是求解无约束问题最简单、最经典的方法之一。
考虑无约束优化问题$min_{x}f(x)$,其中f(x)为连续可微函数。若能构造一个序列$x^{0},x^{1},x^{2},...$,满足
![](http://latex.codecogs.com/gif.latex?\\f(x^{t+1}<f(x^{t}),t=0,1,2,..)
则不断执行该过程即可收敛到局部极小点，欲满足上式，根据泰勒展示有
![](http://latex.codecogs.com/gif.latex?\\f(x+\Delta x)\simeq f(x)+\Delta x^{T}\bigtriangledown f(x))
于是，欲满足$f(x+\Delta x)<f(x)$,可选择
![](http://latex.codecogs.com/gif.latex?\\Delta x = -\gamma \bigtriangledown f(x))
其中步长$\gamma$是一个小常数，这就是梯度下降法
![](http://chuantu.biz/t5/60/1491633221x2890149782.png)

**注意：** logistics回归求解的损失函数是似然函数，需要最大化似然函数，所以我们要用的是梯度上升算法。但因为其和梯度下降的原理是一样的，只是一个是找最大值，一个是找最小值。找最大值的方向就是梯度的方向，最小值的方向就是梯度的负方向。

当目标函数f（x）二阶连续可微是，就可以替换为更精确的二阶泰勒展示，这样就得到了牛顿法。牛顿法是典型的二阶方法，其迭代轮数远小于梯度下降法。但牛顿法使用了二阶导数$\bigtriangledown ^{2}f(x)$,其每轮迭代中设计到海森矩阵的求逆，计算复杂度相当高，尤其在高维问题中几乎不可行。若能以较低的计算大家寻找海森矩阵的近似逆矩阵，则可显著降低计算开销，这就衍生了**拟牛顿法**（quasi-Newton method）
![](http://chuantu.biz/t5/60/1491633683x2890149782.png)

## Logistic Regression 代码

```python
from numpy import *  
import matplotlib.pyplot as plt  
import time  

# 计算sigmoid 函数  
def sigmoid(inX):  
    return 1.0 / (1 + exp(-inX)) 
# 输入:  train_x is a mat datatype, each row stands for one sample 每一行代表一个样本  
#        train_y is mat datatype too, each row is the corresponding label 每一行是相应的标签
#        opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'gradDescent'} 输入初始参数  
def trainLogRegres(train_x, train_y, opts):  #训练参数的函数
    # calculate training time  
    startTime = time.time()  #计算时间
    numSamples, numFeatures = shape(train_x)  
    alpha = opts['alpha'];  #梯度上算法中的gamma
    maxIter = opts['maxIter'] 
    weights = ones((numFeatures, 1)) 初始化权重值
    for k in range(maxIter):  
        if opts['optimizeType'] == 'gradDescent': # 梯度上升算法
            output = sigmoid(train_x * weights)  
            error = train_y - output  
            weights = weights + alpha * train_x.transpose() * error  
        else:  
            raise NameError('Not support optimize method type!')  
            
    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)  
    return weights  
# 训练好参数就开始测试，看正确率
# test your trained Logistic Regression model given test set  
def testLogRegres(weights, test_x, test_y):  
    numSamples, numFeatures = shape(test_x)  
    matchCount = 0  #统计正确个数
    for i in xrange(numSamples):  
        predict = sigmoid(test_x[i, :] * weights)[0,0] > 0.5  #判断是T OR F
        if predict == bool(test_y[i, 0]):  
            matchCount += 1  
    accuracy = float(matchCount) / numSamples  
    return accuracy  
```

## 参考文献
周志华《机器学习》


  [1]: http://static.zybuluo.com/curiousbull/yxsqrjnx65nluvl5bl2rpcli/linear%20regrssion.png
  [2]: http://static.zybuluo.com/curiousbull/reg14ifq15zugfq7h55qyzfb/L2.png
  [3]: http://static.zybuluo.com/curiousbull/e9q8fm3lbq8qlgp2s2kb1r17/L3.png
  [4]: http://static.zybuluo.com/curiousbull/5dreirmel1u387x47xlmhwji/L4.png
  [5]: http://static.zybuluo.com/curiousbull/0t1xxmzyxu68dq8nwxaw21sw/L15.png



  [1]: http://python.jobbole.com/87562/
  [2]: http://blog.csdn.net/monsion/article/details/20631737
  [3]: http://www.jb51.net/article/73450.htm
  [4]: http://blog.csdn.net/cjhc666/article/details/54953723
  [5]: http://blog.csdn.net/djd1234567/article/details/45009895
  
  
  
