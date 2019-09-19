from math import log
import operator

"""
以信息增益作为选择特征的标准，优先选择信息增益较大的特征作为决策的节点
直到每个叶子节点上都只有同样标签的样本或者所有的特征全部已经考虑过了为止

信息增益是利用熵来确定的，原始数据集的熵-利用了某一特征进行划分之后的数据集的熵（就是利用该特征信息得到的增益）
"""

def Ent(dataset):
    """
    计算数据集的熵
    :param dataset:训练数据集
    :return: 熵（表示随机变量不确定性的度量），
            这里用于衡量数据的纯度；熵越大表示数据越杂乱，反之数据越单一规律
    """
    numEntries = len(dataset)
    labelCounts = {}

    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Ent = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        Ent -= (prob * log(prob, 2))
    return Ent


def splitDataSet(dataSet, axis, value):
    """
    依据特征维度axis上取值是否为value来对原始数据集进行分割
    得到在axis上取值为value的数据

    :param dataSet: 原始数据集
    :param axis: 特征维度
    :param value: 用于划分的特征取值
    :return: 返回在axis维度上取值为value的数据集
    """
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reduceFeatVex=featVec[:axis]
            reduceFeatVex.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVex)
    return retDataSet


def info_Gain(dataset, fea):
    """
    计算根据特征fea得到的信息增益

    :param dataset: 数据集
    :param fea: 依据的特征
    :return: 返回根据特征fea划分数据集得到的信息增益
    """
    # 计算原本数据集的熵
    ori_Ent=Ent(dataset)

    numEntries = len(dataset)
    feaDB={}
    for i in range(numEntries):
        if dataset[i][fea] not in feaDB.keys():
            feaDB[dataset[i][fea]] = 0
        feaDB[dataset[i][fea]]+=1

    # 计算根据特征fea进行划分之后的数据集的熵
    fea_Ent = 0
    for k in feaDB.keys():
        l = feaDB[k]
        data_k=splitDataSet(dataset,fea,k)
        current_Ent = float(l)/numEntries*Ent(data_k)
        fea_Ent += current_Ent
    # 返回信息增益
    return ori_Ent-fea_Ent


def chooseBestFeaToSplit(dataset):
    """
    在当前的数据集的特征中选择信息增益最大的特征

    :param dataset: 当前用于选择特征的数据集
    :return: 返回信息增大最大的特征
    """
    best_InfoGain = 0
    best_Feature = -1
    for i in range(len(dataset[0])-1):
        curInfoGain = info_Gain(dataset,i)
        if best_InfoGain<curInfoGain:
            best_InfoGain=curInfoGain
            best_Feature=i
    return best_Feature


def majorityCnt(classList):
    """
    少数服从多数  选择该分类列表中（体现在决策树中就是叶子节点上的样本）
    数量最多的类别代表该分类列表，作为决策的结果

    :param classList: 分类列表（决策树构造过程中的叶子节点）
    :return: 返回数量最多的类别（代表该分类列表，作为决策的结果）
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def creatTree(dataset,labels):
    # 当前数据集中的类别标签情况
    classList = [example[-1] for example in dataset]

    # 如果仅包含一个类别，则不必再继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 如果该数据集只剩下一个特征未被用于分类
    # 则将当前数据集中最多的类别代表当前的数据集
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    # 选择最优的特征
    best_Feature = chooseBestFeaToSplit(dataset)
    bestFeatLabel = labels[best_Feature]

    # 分类结果以字典形式保存
    myTree = {bestFeatLabel: {}}
    del(labels[best_Feature])
    # 所选择的特征在数据集的每一个样本中的取值
    featValues = [example[best_Feature] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = creatTree(
            splitDataSet(dataset, best_Feature, value), subLabels)
    return myTree


if __name__ == '__main__':
    # 读取数据
    # dataset就是所有的初始的数据样本
    # labels就是数据样本中特征各维度对应的具体的特征名
    dataSet = [['青年', '否', '否', '一般',  '否'],
               ['青年', '否', '否', '好',   '否'],
               ['青年', '是', '否', '好',   '是'],
               ['青年', '是', '是', '一般',  '是'],
               ['青年', '否', '否', '一般',  '否'],
               ['中年', '否', '否', '一般',  '否'],
               ['中年', '否', '否', '好',    '否'],
               ['中年', '是', '是', '好',    '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好',    '是'],
               ['老年', '是', '否', '好',    '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般',   '否']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    # 输出决策树的模型
    # {'有自己的房子': {'是': '是', '否': {'有工作': {'是': '是', '否': '否'}}}}
    print(creatTree(dataSet,labels))