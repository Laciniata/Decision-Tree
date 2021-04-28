from collections import Counter
import pandas as pd
import random
import math
import copy

splitSign = ' ≤ '


# tested
def dataInit():
    """
    初始化数据和属性集。

    Returns: (数据, 属性集)
    """

    Data = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, '好瓜'],
            ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, '好瓜'],
            ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, '好瓜'],
            ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, '好瓜'],
            ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, '好瓜'],
            ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, '好瓜'],
            ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, '好瓜'],
            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, '好瓜'],
            ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, '坏瓜'],
            ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, '坏瓜'],
            ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, '坏瓜'],
            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, '坏瓜'],
            ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, '坏瓜'],
            ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, '坏瓜'],
            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, '坏瓜'],
            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, '坏瓜']
            # ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, '坏瓜']
            ]
    Attribute = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度']
    IsDiscrete = [True, True, True, True, True, True, False]
    return Data, Attribute, IsDiscrete


# tested
def generateDataFrame(Data, Attribute):
    """
    将数据集转化为DataFrame类型。

    Args:
        Data (list): 数据集
        Attribute (list): 属性集

    Returns: DataFrame
    """
    column = Attribute.copy()
    column.append('类型')
    frame = pd.DataFrame(Data, columns=column)
    return frame


# tested
def getAttributeValue(Data, Attribute, IsDiscrete):
    """
    统计所有离散属性的所有取值。

    Args:
        Data (): 数据集
        Attribute (): 属性集
        IsDiscrete (): 是否离散

    Returns: 字典，所有离散属性的所有取值。例如：
    {'色泽': {'乌黑', '青绿', '浅白'}, '根蒂': {'硬挺', '稍蜷', '蜷缩'}, '敲声': {'浊响', '沉闷', '清脆'},
    '纹理': {'模糊', '清晰', '稍糊'}, '脐部': {'凹陷', '稍凹', '平坦'}, '触感': {'硬滑', '软粘'}}
    """
    AttributeValue = {}
    for i in range(len(IsDiscrete)):
        if not IsDiscrete[i]:  # 连续属性不统计取值
            continue
        tempSet = set()
        for sample in Data:
            tempSet.add(sample[i])
        AttributeValue[Attribute[i]] = tempSet
    return AttributeValue


# tested
def sameCategory(Data):
    """
    判断是否所有数据均属于相同类型。

    Args:
        Data (list): 数据集
    Returns: (类型, 是否相同)
    """
    categories = [y[-1] for y in Data]  # 数据集中的y向量（所有数据的标签）
    return categories[0], len(set(categories)) == 1


# tested
def sameValue(Data):
    """
    判断是否所有数据的所有属性值完全相同。

    Args:
        Data (list): 数据集
    Returns: 是否相同
    """
    xSet = set()
    for sample in Data:
        *x, y = sample
        xSet.add(tuple(x))
    return len(xSet) <= 1


# tested
def mostCategory(Data):
    """
    获取数据中最主要的类别。

    Args:
        Data (list): 数据集
    Returns: 最主要的类别(例如：“好瓜”)
    """
    categories = [y[-1] for y in Data]  # 数据集中的y向量（所有数据的标签）
    return Counter(categories).most_common(1)[0][0]


# tested
def entropy(Data, Attribute):
    frame = generateDataFrame(Data, Attribute)
    pctofCategory = dict(frame['类型'].value_counts() / frame['类型'].value_counts().sum())
    entropy = 0.0
    for _, number in pctofCategory.items():
        entropy -= number * math.log2(number)
    return entropy


# tested
def informationGain(Data, Attribute, IsDiscrete, serlNum, dividingPoint=None):
    """
    计算信息增益。

    Args:
        Data (): 数据集
        Attribute (): 属性集
        IsDiscrete (): 属性是否离散
        serlNum (): 要计算信息增益的属性的下标

    Returns: 信息增益
    """
    frame = generateDataFrame(Data, Attribute)
    ent = entropy(Data, Attribute)
    gain = ent
    if IsDiscrete[serlNum]:
        AttrValues = set([sample[serlNum] for sample in Data])
        dvSeries = frame[Attribute[serlNum]].value_counts()
        for v in AttrValues:
            dvNumber = dvSeries[v]
            dvData = [sample for sample in Data if sample[serlNum] == v]  # 取属性的某个值v的子数据集
            gain -= dvNumber / len(Data) * entropy(dvData, Attribute)
    else:
        dvNumbers = [0, 0]  # 第一项为小于划分点的数量，第二项为大于划分点的数量
        for i in range(len(Data)):
            dvNumbers[0 if Data[i][serlNum] < dividingPoint else 1] += 1
        dvData1 = [sample for sample in Data if sample[serlNum] < dividingPoint]
        dvData2 = [sample for sample in Data if sample[serlNum] >= dividingPoint]
        gain = gain - dvNumbers[0] / len(Data) * entropy(dvData1, Attribute) - dvNumbers[1] / len(Data) * entropy(
            dvData2, Attribute)

    return gain


# tested
def getDividingPointList(continuousValues: pd.Series):
    """
    获得所有候选划分点。

    Args:
        continuousValues (): 待划分的连续值序列

    Returns:
        list: 所有候选划分点
    """

    sortedValues = continuousValues.sort_values().values.tolist()
    DividingPointList = []
    for i in range(continuousValues.size - 1):
        DividingPointList.append((sortedValues[i] + sortedValues[i + 1]) / 2)
    return DividingPointList


# tested
def bestAttribute(Data, Attribute, IsDiscrete):
    """
    使用信息增益，获取最优化分属性和其在属性集中的下标。

    Args:
        Data (): 数据集
        Attribute (): 属性集
    Returns: (最优化分属性, 在属性集中的下标)
    离散属性的最优划分属性是返回字符串
    连续属性的最优划分属性是返回(属性名称, 划分点)
    """
    gain = {}  # {属性：信息增益}，存储所有信息增益计算结果，连续属性取最大
    frame = generateDataFrame(Data, Attribute)
    dividingPoint = None  # 连续值最佳划分点
    for i, attr in enumerate(Attribute):
        if IsDiscrete[i]:
            gain[attr] = informationGain(Data, Attribute, IsDiscrete, i)
        else:
            gain[attr] = 0  # 连续值最佳信息增益
            DividingPointList = getDividingPointList(frame[attr])
            for dvdpt in DividingPointList:
                tempGain = informationGain(Data, Attribute, IsDiscrete, i, dvdpt)
                if tempGain > gain[attr]:
                    dividingPoint = dvdpt
                    gain[attr] = tempGain
    gainlist = sorted(gain.items(), key=lambda x: x[1], reverse=True)  # 降序排列所有信息增益
    bestAttribute = gainlist[0][0]
    # 在Attribute中查找bestAttribute的下标，再用下标在IsDiscrete中查找是否为离散属性
    bestSerlNum = Attribute.index(bestAttribute)
    if IsDiscrete[bestSerlNum]:  # tested
        return bestAttribute, bestSerlNum
    else:
        return (bestAttribute, dividingPoint), bestSerlNum


# tested
def simplifyData(DataofBAV, bestAttrSerl):
    """
    将数据中对应最优划分属性的离散分量删除。

    Args:
        DataofBAV (): 数据
        bestAttrSerl (): 要删除的下标
    Returns: 列表，简化后的数据
    """
    simplifiedData = DataofBAV.copy()
    for sample in simplifiedData:
        del sample[bestAttrSerl]
    return simplifiedData


# tested
def addColumn(Data, serlNum, splitPoint):
    """
    在数据集的末尾，为连续值的划分增加一列属性，True(小于等于)或False(大于)

    Args:
        serlNum (int): 属性的下标
        splitPoint (float): 划分点
        Data (list): 数据集

    Returns:
    返回增加了属性的数据集
    """
    DataAdded = Data.copy()
    for i, sample in enumerate(DataAdded):
        DataAdded[i].append(sample[serlNum] <= splitPoint)
    return DataAdded


# tested
def treeGenerate(Data, Attribute, IsDiscrete, AttributeValue):
    """
    生成决策树。

    Args:
        IsDiscrete (list[bool]): 属性是否离散
        AttributeValue (dict): 所有离散属性的所有取值
        Data (list): 数据集
        Attribute (list): 属性集
    Returns: 决策树（字典）
    """
    category, same = sameCategory(Data)
    if same:
        return category
    if len(Attribute) == 0 or sameValue(Data):
        return mostCategory(Data)
    bestAttr, bestAttrSerl = bestAttribute(Data, Attribute, IsDiscrete)
    # 递归过程中不修改原始数据集，属性集
    DataCopy = copy.deepcopy(Data)  # 包含子列表的完全复制
    AttributeCopy = Attribute.copy()
    IsDiscreteCopy = IsDiscrete.copy()
    if isinstance(bestAttr, tuple):  # 如果是连续属性，不需要增删AttributeCopy
        attr, splitPoint = bestAttr
        addColumn(DataCopy, bestAttrSerl, splitPoint)  # 根据划分点统计，在数据末尾添加一列，True(<=)和False(>)
        bestAttr = str(attr) + splitSign + str(splitPoint)  # 整理成字符串形式（'密度 ≤ 0.3815'）
        DecisionTree = {bestAttr: {}}
        bestAttrSerl = -1  # 最佳属性为最后一列属性
        BestAttrValues = [True, False]
        for bestAttrValue in BestAttrValues:
            DataofBAV = [sample for sample in DataCopy if sample[bestAttrSerl] == bestAttrValue]
            if len(DataofBAV) == 0:
                return mostCategory(DataCopy)  # 返回D中样本最多的类
            else:
                DecisionTree[bestAttr][bestAttrValue] = treeGenerate(simplifyData(DataofBAV, bestAttrSerl),
                                                                     AttributeCopy, IsDiscreteCopy, AttributeValue)
    else:
        if IsDiscreteCopy[bestAttrSerl]:  # 离散属性选择之后即删除该属性，连续则不删
            del AttributeCopy[bestAttrSerl]
            del IsDiscreteCopy[bestAttrSerl]
        DecisionTree = {bestAttr: {}}
        BestAttrValues = AttributeValue[bestAttr]  # 更换为一开始就生成的离散属性的所有取值，否则可能出现某一子树中缺取值，以至于无法分类的情况
        for bestAttrValue in BestAttrValues:
            DataofBAV = [sample for sample in DataCopy if sample[bestAttrSerl] == bestAttrValue]
            if len(DataofBAV) == 0:
                return mostCategory(DataCopy)  # 返回D中样本最多的类
            else:
                DecisionTree[bestAttr][bestAttrValue] = treeGenerate(simplifyData(DataofBAV, bestAttrSerl),
                                                                     AttributeCopy, IsDiscreteCopy, AttributeValue)
    return DecisionTree


# tested
def classify(sample, Attribute, DecisionTree: dict):
    """
    用决策树对单个数据分类。

    Args:
        sample (list): 单个数据
        Attribute (): 属性集
        DecisionTree (): 决策树

    Returns:
    分类结果
    """
    if not isinstance(DecisionTree, dict):  # 此时的DecisionTree即为分类结果
        return DecisionTree
    judgementPrinciple = tuple(DecisionTree.keys())[0]
    try:
        index = Attribute.index(judgementPrinciple)
    except ValueError:  # 找不到属性，说明是连续值（key中含有划分点信息）
        judgementCategory, splitValue = judgementPrinciple.split(splitSign)
        index = Attribute.index(judgementCategory)
        value = sample[index]
        # print(DecisionTree[judgementPrinciple][value <= float(splitValue)])
        category = classify(sample, Attribute, DecisionTree[judgementPrinciple][value <= float(splitValue)])
    else:
        value = sample[index]
        # print(DecisionTree[judgementPrinciple][value])
        category = classify(sample, Attribute, DecisionTree[judgementPrinciple][value])
    return category


# tested, abandoned
def removeListItem(originList, itemToRemove):
    """
    去除列表中的特定项
    Args:
        originList (): 待删除的列表
        itemToRemove (): 待删除的项

    Returns:
        删除后的列表
    """
    for i in itemToRemove:
        while i in originList:
            originList.remove(i)
    return originList


# tested
def dictToList(NestedDict: dict, itemToRemove=None):
    """
    按深度由浅入深地将key记录在列表中，并返回列表。列表形式为['纹理', ('纹理', '稍糊', '触感'), ('纹理', '清晰', '密度 ≤ 0.3815')]。

    Args:
        itemToRemove (): 不希望放入list的key
        NestedDict (dict): 待搜索的字典

    Returns:
    按深度由浅入深顺序的key的列表
    """
    keyList = []
    # 此时是根结点
    for k1, v1 in NestedDict.items():  # 纹理，{稍糊:...,...}
        keyList.append(k1)
        for k2, v2 in v1.items():  # 稍糊，{敲声:...}
            if v2 not in itemToRemove:
                keyList.append((k1, k2, tuple(v2)[0]))  # （纹理，稍糊，敲声），只取到前3层。
    pointer = 1
    while pointer <= len(keyList) - 1:  # 每次层数深度＋2
        first = True
        DisassembledDict = {}  # 拆到当前层的字典，{沉闷：坏瓜，浊响：{脐部：...}，...}
        for k in keyList[pointer]:
            if first:
                DisassembledDict = NestedDict[k]
                first = False
            else:
                DisassembledDict = DisassembledDict[k]
        for k1, v1 in DisassembledDict.items():  # 沉闷，坏瓜；浊响，{脐部：...}
            if isinstance(v1, dict):  # 不需要else处理
                for k2, v2 in v1.items():  # 脐部，{凹陷：坏瓜，...}
                    keyList.append(keyList[pointer] + (k1, k2))
        pointer += 1
    return keyList


# tested
def categoryCount(Data):
    """
    计算数据集中各个类别的样本数。

    Args:
        Data (list): 数据集

    Returns:
        各个类别的样本数
    """
    CategoryList = [y[-1] for y in Data]
    return dict(Counter(CategoryList))


# tested
def shuffleSplit(Data, proportion):
    """
    随机划分训练集 和 验证集或测试集。
    Args:
        proportion (): 训练集占数据集的比例
        Data (): 数据集
    Returns:
        [训练集，验证集或测试集]
    """
    # 按类别将Data分成一个字典，其key是类型，value是一个list，list的元素是该类的sample
    CategoryList = [y[-1] for y in Data]  # 数据集中的y向量（所有数据的标签）
    CategorySet = set(CategoryList)
    classifiedData = {c: [] for c in CategorySet}
    for sample in Data:
        classifiedData[sample[-1]].append(sample)
    # 计算每个类别的数量，进而计算每个类别的要选取的数量（按proportion）
    categoryNum = dict(Counter(CategoryList))  # {'好瓜': 8, '坏瓜': 8}
    selectTrainNum = {c: int(n * proportion) for c, n in categoryNum.items()}
    # 随机排序，顺序选择
    TrainSet = []
    TestingSet = []
    for category, samples in classifiedData.items():
        random.shuffle(samples)
        TrainSet.extend(samples[0:selectTrainNum[category]])
        TestingSet.extend(samples[selectTrainNum[category]:])
    return TrainSet, TestingSet


# tested
def precision(TestingSet, Attribute, DecisionTree):
    """
    计算分类精度。

    Args:
        TestingSet (): 测试集
        Attribute (): 属性集
        DecisionTree (): 决策树

    Returns:
        分类精度
    """
    right = 0
    for sample in TestingSet:
        if classify(sample, Attribute, DecisionTree) == sample[-1]:
            right += 1
    return right / len(TestingSet)


def postpruning(TrainingSet, ValidationSet, Attribute, DecisionTree):
    """
    对决策树后剪枝。

    Args:
        Attribute (): 属性集
        TrainingSet (): 训练集
        ValidationSet (): 验证集
        DecisionTree (): 决策树

    Returns:
    是否剪枝了，完成后剪枝的决策树，剪枝后在验证集的精度。
    """
    '''按深度由浅入深将key记录在列表中，形式为['纹理', ('纹理', '稍糊', '触感'), ('纹理', '清晰', '密度 ≤ 0.3815')]'''
    DecisionList = dictToList(DecisionTree, list(set(y[-1] for y in Data)))  #
    '''建立一个字典TrainingNodeDict，其key为DecisionList中的值，value为到这个节点的各类型数量。
       字典形式为{('纹理', '稍糊', '触感'):{'好瓜':2,'坏瓜':3}}'''
    TrainingNodeDict = {}
    for i, d in enumerate(DecisionList):
        if i == 0:
            categoryNum = categoryCount(TrainingSet)
            TrainingNodeDict[d] = categoryNum
        else:
            TrainingSetCopy = copy.deepcopy(TrainingSet)
            TrainingSetChosen = []
            for j in range(int(len(d) / 2)):
                if d[j * 2] in Attribute:
                    attrSerlNum = Attribute.index(d[j * 2])
                    TrainingSetChosen = [sample for sample in TrainingSetCopy if sample[attrSerlNum] == d[j * 2 + 1]]
                else:
                    judgementCategory, splitValue = d[j * 2].split(splitSign)
                    splitValue = float(splitValue)
                    attrSerlNum = Attribute.index(judgementCategory)
                    TrainingSetChosen = [sample for sample in TrainingSetCopy if sample[attrSerlNum] <= splitValue]
                TrainingSetCopy = TrainingSetChosen
            categoryNum = categoryCount(TrainingSetChosen)
            TrainingNodeDict[d] = categoryNum
    '''倒序遍历，从最深的叶节点开始尝试剪枝。'''
    pruned = False
    finalPrecision = precision(ValidationSet, Attribute, DecisionTree)  # 剪枝后在验证集的精度，初始值为剪枝前在验证集的精度
    for d in reversed(DecisionList):
        maxCategory = max(TrainingNodeDict[d], key=TrainingNodeDict[d].get)  # 找到数量最多的类型
        prunedTree = copy.deepcopy(DecisionTree)
        ROI = prunedTree  # 感兴趣部分，要修剪的部分。迭代后变成{'软粘': '好瓜', '硬滑': '坏瓜'}的形式。
        if isinstance(d, tuple):
            for k in d:  # 例如：纹理，稍糊，触感
                if k == d[-2]:
                    break
                ROI = ROI[k]
            ROI[d[-2]] = maxCategory  # 不能一路走到想要的值，而是在最后一步使用直接改变的方式，否则无法起到指针的效果，而是类似复制的效果
        else:
            prunedTree = maxCategory
        beforePrecision = precision(ValidationSet, Attribute, DecisionTree)
        afterPrecision = precision(ValidationSet, Attribute, prunedTree)
        if afterPrecision > beforePrecision:
            DecisionTree = prunedTree
            finalPrecision = afterPrecision
            pruned = True
    return pruned, DecisionTree, finalPrecision


def testCode():
    # print(informationGain(Data, Attribute, IsDiscrete, 5))

    # frame = generateDataFrame(Data, Attribute)
    # print(frame.values.tolist())

    # frame = generateDataFrame(Data, Attribute)
    # DividingPointList = getDividingPointList(frame['密度'])
    # print(DividingPointList)
    # print(informationGain(Data, Attribute, IsDiscrete, 6, 0.3815))

    # print(mostCategory(Data))

    # print(sameCategory([[1, 2, 1], [7, 7, 1], [0, 8, 1], [8, 9, 's']]))

    # print(bestAttribute(Data, Attribute, IsDiscrete))

    # DataofBAV = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, '好瓜'],
    #              ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, '好瓜'],
    #              ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, '好瓜'],
    #              ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, '好瓜'],
    #              ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, '好瓜'],
    #              ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, '坏瓜'],
    #              ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, '坏瓜']
    #              ]
    # bestAttrSerl = 4
    # print(simplifyData(DataofBAV, bestAttrSerl))

    # print(addColumn(Data, 6, 0.6))

    # print(classify(['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719], Attribute, DecisionTree))
    # print(classify(['青绿', '蜷缩', '沉闷', '清晰', '稍凹', '硬滑', 0.3815], Attribute, DecisionTree))
    # print(classify(['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634], Attribute, DecisionTree))
    # print(classify(['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481], Attribute, DecisionTree))

    # print(removeListItem(['好瓜', '坏瓜', {'key1': 'value1', 'key2': {'key1': 'value1'}}, '其它', '坏瓜'], ['好瓜', '坏瓜']))
    # print(dictToList(DecisionTree, ['好瓜', '坏瓜']))

    # TrainingSet = [['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, '坏瓜'], ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.36, '坏瓜'], ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, '坏瓜'], ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, '坏瓜'], ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, '坏瓜'], ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, '坏瓜'], ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, '好瓜'], ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, '好瓜'], ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, '好瓜'], ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, '好瓜'], ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, '好瓜'], ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, '好瓜']]
    # TestingSet = [['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, '坏瓜'], ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, '坏瓜'], ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, '好瓜'], ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, '好瓜']]
    # DecisionTree = {'密度 ≤ 0.3815': {True: '坏瓜', False: {'密度 ≤ 0.6455': {True: '好瓜', False: '坏瓜'}}}}

    pass


if __name__ == '__main__':
    Data, Attribute, IsDiscrete = dataInit()
    AttributeValue = getAttributeValue(Data, Attribute, IsDiscrete)
    # DecisionTree = treeGenerate(Data, Attribute, IsDiscrete, AttributeValue)
    # print(DecisionTree)
    TrainingSet, TestingSet = shuffleSplit(Data, 0.75)
    print('训练集：{0}\n验证集：{1}'.format(TrainingSet, TestingSet))
    DecisionTree = treeGenerate(TrainingSet, Attribute, IsDiscrete, AttributeValue)
    print('决策树：', DecisionTree)
    print('训练集上精度：', precision(TrainingSet, Attribute, DecisionTree))
    print('验证集上精度：', precision(TestingSet, Attribute, DecisionTree))
    pruned, DecisionTree, finalPrecision = postpruning(TrainingSet, TestingSet, Attribute, DecisionTree)
    if pruned:
        print('后剪枝后的决策树：', DecisionTree)
        print('后剪枝后在训练集上的精度：', precision(TrainingSet, Attribute, DecisionTree))
        print('后剪枝后在验证集上的精度：', finalPrecision)
