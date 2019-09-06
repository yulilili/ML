import numpy as np

def loadDataSet(fileName):
    """加载数据集的函数，在文件中读取数据，
    并将特征数据存在dataMat数组中
    标签数据存在labelMat数组中"""
    # 每行表示一个样本，除最后一个外都是特征 最后一个数值是标签
    # 每列标识的都是同一个特征属性
    # 每行有numFeat列
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []  # 用于存储特征数据
    labelMat = []  # 用于存储标签数据

    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        if float(curLine[-1]) == 0.0:
            labelMat.append(-1.0)
        else:
            labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    :param dataMatrix:特征数据
    :param dimen:考虑的维度，哪一列；对应的就是一个属性
    :param threshVal: 给出的阈值
    :param threshIneq: 区分的方向，是小于等于还是大于
    :return: 返回的结果，是以阈值和方向来划分，各个数据的分类结果。以一个[m,1]的数组来返回结果
            若符合阈值和方向，则置为0，反之则置为1
    """
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    # 返回预测值，原数据均为1，若预测为真则赋为0
    return retArray

def buildStump(dataArr, classLabels, D):
    """
    以该数据和权重来进行分类，递增的选择阈值来获得分类器，返回误差率最小的分类器
    选择在当前的数据集权重D下最合适（误分类率最小）的维度和阈值来作为分类器
    :param dataArr: 特征数据
    :param classLabels: 标签数据
    :param D: 权重矩阵
    :return: 返回误差率最小的分类器的各种数值（用以分类的维度，阈值，方向；误差率；在训练集上分类的结果）

    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 3.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        # 考虑第i列的值，也就是第i个属性（因为这里有多个特征属性来标识一个样本）
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(0, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                # 'lt'表示小于或等于 / 'gt'表示大于
                # 阈值逐步增加的来取得，每步增加一个stepSize
                threshVal = (rangeMin+float(j)*stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                #print("predictVals:",predictedVals)
                # 在该属性上的预测结果（用errArr来存储）：如果结果和label一样，则置为0，反之置为1
                # 其实相当于就是I(G(x)!=y)的结果
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                # 将errArr乘以权重，得到的就是该分类在训练数据集上的分类误差率（相当于给训练数据加上了权重）
                weightedError = D.T * errArr
                #print("split:dim %d,thresh %.2f, thresh inequal:%s,the weighted error is %.3f"%(i, threshVal, inequal, weightedError))
                # 记录到目前为止在训练数据集上表现最好的分类器
                if weightedError < minError:
                    minError = weightedError  # 最小的分类误差
                    bestClassEst = np.copy(predictedVals)  # 最好的分类效果 np.copy深复制，两个变量不会互相影响  /  b=a[:]的这种是浅复制，b会重新创建对象，但是b的数据完全由a保管，两者的变化完全同步，互相影响
                    bestStump['dim'] = i   # 相应的维度，对应的也就是用于分类的特征属性
                    bestStump['thresh'] = threshVal  # 分类阈值
                    bestStump['ineq'] = inequal  # 分类方向
    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    """
    训练过程
    :param dataArr: 训练集特征数据
    :param classLabels: 训练集的标签数据
    :param numIt: 最大循环训练的次数
    :return: 返回的是每次训练得到的弱分类器（各分类器以字典形式存储，其中包括各种属性）
    以及分类器集合在训练数据集上的表现；

    """
    # 存储所有的弱分类器
    weakClassArr = []
    m = np.shape(dataArr)[0]
    # D为权重 初始化为1/m
    D = np.mat(np.ones((m, 1))/m)
    # 到目前为止所有的分类器预测的结果的综合
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 循环的最大次数
    for i in range(numIt):
        # 当前权重下得到的分类器
        # 并得到相应的弱分类器的一些特征（用以分类的属性、阈值、方向）/分类误差率/分类结果
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("error:", error)
        #print("D:", D.T)
        # 弱分类器的系数
        # （根据其分类误差率来决定的分类器的系数，也就是该分类器在最后的分类器中所占的权重）
        alpha = float(0.5*np.log((1.0-error)/max(error, 1e-16)))
        # 再在用于存储分类器的字典中添加一个数据：该分类器的系数
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print("classEst:", classEst.T)
        # 更新权重
        # w(m+1)=(w(m)*exp(-alpha*y*G(x)))/z
        # z为分子在所有样本上的数值之和
        # 如果在上一个分类器上分类正确则权重会降低，反之分类错误的权重会上升
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()

        # 目前得到的弱分类器的集合的表现
        aggClassEst += alpha*classEst
        #print("aggClassEst:", aggClassEst.T)
        # 现在得到的误分类点的个数
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        #print("aggErrors:", aggErrors, '\n')
        errorRate = aggErrors.sum()/m
        print("total error:",errorRate, "\n")
        # 如果现在得到的分类器在训练集上全部分类正确，或者分类错误的点很少；则可以结束循环，不再继续训练
        if errorRate == 0.0:
            break

    return weakClassArr, aggClassEst

def adaClassify(datToClass, classifierArr):
    """

    :param datToClass: 用以分类的数据
    :param classifierArr: 训练得到的所有的弱分类器
    :return:
    """
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        # 在各弱分类器上的表现
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst

    return np.sign(aggClassEst)



# 加载训练数据
dataMat, classLabels = loadDataSet('./horse_dataset/horseColicTraining.txt')
# 训练得到的分类器
print(classLabels)
classifierArr, _ = adaBoostTrainDS(dataMat, classLabels, 9)
# # 加载测试数据
testArr, testLabelArr = loadDataSet('./horse_dataset/horseColicTest.txt')
# 用训练得到的分类器来跑测试数据集，得到相应的预测结果
predict = adaClassify(testArr, classifierArr)
c=0
for i in range(len(testLabelArr)):
    if predict[i]==testLabelArr[i]:
        c+=1
    print("predict is %d,label is %d" % (predict[i], testLabelArr[i]))
print (c/len(testLabelArr))










