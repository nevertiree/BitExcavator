#Kaggle数据挖掘&机器学习比赛经验笔记

## 前言

记得[Andrew Ng的公开课CS229](http://open.163.com/special/opencourse/machinelearning.html)中说过，他在理解新知识的生活，都会把公司逐步推导再实践一遍。这是一种很好的学习方法。在机器学习的学习过程中，如果学了几个月或者半年而没有实践的话，这一大段的时间等于是白费了。所以我挑选[Kaggle](https://www.kaggle.com/)来作为练习。

希望可以把Kaggle的每个赛题用锤炼不同的模型，加深对各个算法或者工具的理解，有更加具体的概念。下面是一些时间点的感悟和笔记。

## 目录

1.[Titanic幸存者预测](https://github.com/nevertiree/BitExcavator/tree/master/1.Titanic)

2.[手写数字识别](https://github.com/nevertiree/BitExcavator/tree/master/2.DigitRecognizer)

3.[面部识别](https://github.com/nevertiree/BitExcavator/tree/master/3.FacialDetection)

## Titanic幸存者预测

2016.10.14

>用[Random Forest](http://scikit-learn.org/stable/modules/ensemble.html#forest)算法完成了入门比赛Titanic，实现的方式是调用了Python的[sklearn](http://scikit-learn.org/)包。这个过程有点暴力、有点弱智。最大的收获是学习了sklearn包，以及NumPy,Pandas等等包的使用。

## 手写数字识别

2016.10.18

>用Random Forest算法完成了手写数字的预测，准确率达到94% 。我们认为不能再继续使用Random Forest了。Random Forest有一种黑箱的感觉。以后只能用Random Forest测试重要性，而模型本身我们决定自己造轮子。

## 面部识别

2016.10.19

>面部识别居然要使用deep learn。 有点小激动。