import matplotlib.pyplot as plt
import numpy as np

# разнести реализацию в файлики metrics.py 
# между классами и функциями 2 строчки
#[:] a copy of the whole array
# true positive
def calculateTruePositive(predict, groundTruth, threshold):
    return np.add.reduce((predict[ : ] >= threshold) & (groundTruth == 1))

    
# false positive
def calculateFalsePositive(predict, groundTruth, threshold):
    return np.add.reduce((predict[ : ] >= threshold) & (groundTruth == 0))


# true negative
def calculateTrueNegative(predict, groundTruth, threshold):
    return np.add.reduce((predict[ : ] <= threshold) & (groundTruth == 0))


# false negative
def calculateFalseNegative(predict, groundTruth, threshold):
    return np.add.reduce((predict[ : ] <= threshold) & (groundTruth == 1))


# sensitivity, recall, hit rate, or true positive rate (TPR)
def calculateTruePositiveRate(truePositive, falseNegative):
    return truePositive / (truePositive + falseNegative)


def calculateTruePositiveRate2(truePositive, groundTruth):
    return truePositive / np.add.reduce(groundTruth == 1)


def calculateTruePositiveRate3(predict, groundTruth, threshold):
    tp = calculateTruePositive(predict, groundTruth, threshold)
    fn = calculateFalseNegative(predict, groundTruth, threshold)
    return tp / (tp + fn)


# fall-out or false positive rate (FPR)
def calculateFalsePositiveRate(falsePositive, trueNegative):
    return falsePositive / (falsePositive + trueNegative)


def calculateFalsePositiveRate2(truePositive, ground_truth):
    return truePositive / np.add.reduce(ground_truth == 0)


#  specificity, selectivity or true negative rate (TNR)
def calculateTrueNegativeRate(trueNegative, falsePositive):
    return trueNegative / (trueNegative + falsePositive)


# precision or positive predictive value (PPV)
def calculatePrecision(truePositive, falsePositive):
    return truePositive / (truePositive + falsePositive)


def calculatePrecision2(predict, groundTruth, threshold):
    tp = calculateTruePositive(predict, groundTruth, threshold)
    fp = calculateFalsePositive(predict, groundTruth, threshold)
    if tp + fp == 0:
        return 1
    return tp / (tp + fp)


# negative predictive value (NPV)
def calculateNegativePredictiveValue(trueNegative, falseNegative):
    return trueNegative / (trueNegative + falseNegative)


# false negative rate (FNR)
def calculateFalseNegativeRate(falseNegative, truePositive):
    return falseNegative / (falseNegative + truePositive)


# false positive rate (FPR)
def calculateFalsePositiveRate(falsePositive, trueNegative):
    return falsePositive / (falsePositive + trueNegative)


# false discovery rate (FDR)
def calculateFalseDiscoveryRate(falsePositive, truePositive):
    return falsePositive / (falsePositive + truePositive)


# false omission rate (FOR)
def calculateFalseOmissionRate(falseNegative, trueNegative):
    return falseNegative / (falseNegative + trueNegative)


# draw ROC-curve plot
def ROCCurve(truePositiveRate, falsePositiveRate): 
    fig = plt.figure()
    plt.plot(truePositiveRate, falsePositiveRate)
    plt.show()


# draw PR-curve plot
def PRCurve(precision, recall):
    fig = plt.figure()
    plt.plot(precision, recall)
    plt.show()


def getTP(ZSorted, y_testSorted, thresholds):
    return np.array(list(map(lambda item: calculateTruePositive(ZSorted, y_testSorted, item), thresholds)))


def getFP(ZSorted, y_testSorted, thresholds):
    return np.array(list(map(lambda item: calculateFalsePositive(ZSorted, y_testSorted, item), thresholds)))


def getTN(ZSorted, y_testSorted, thresholds):
    return np.array(list(map(lambda item: calculateTrueNegative(ZSorted, y_testSorted, item), thresholds)))


def getFN(ZSorted, y_testSorted, thresholds):
    return np.array(list(map(lambda item: calculateFalseNegative(ZSorted, y_testSorted, item), thresholds)))


def getPrecision(ZSorted, y_testSorted, thresholds):
    return  np.array(list(map(lambda item:calculatePrecision2(ZSorted, y_testSorted, item), thresholds)))


def getRecall(ZSorted, y_testSorted, thresholds):
    return np.array(list(map(lambda item:calculateTruePositiveRate3(ZSorted, y_testSorted, item), thresholds)))


def getZ(classifier, X_test):
    if hasattr(classifier, "decision_function"):
        return classifier.decision_function(X_test)
    else:
        return classifier.predict_proba(X_test)[:, 1]


def getROCCurve(ZSorted, y_testSorted, thresholds):
    return calculateFalsePositiveRate(getFP(ZSorted, y_testSorted, thresholds), \
                                      getTN(ZSorted, y_testSorted, thresholds)), \
           calculateTruePositiveRate(getTP(ZSorted, y_testSorted, thresholds), \
                                     getFN(ZSorted, y_testSorted, thresholds))  


def getPRCurve(ZSorted, y_testSorted, thresholds):
    return getPrecision(ZSorted, y_testSorted, thresholds), \
           getRecall(ZSorted, y_testSorted, thresholds)


def drawROCCurve(ZSorted, y_testSorted):
    thresholds = np.insert(np.append(ZSorted, 0), 0, 1)
    fpr, tpr = getROCCurve(ZSorted, y_testSorted, thresholds)  
    print('integral ROC_curve = ' + str(np.trapz(tpr, fpr)))
    ROCCurve(fpr, tpr)

    
def drawPRCurve(ZSorted, y_testSorted):
    thresholds = ZSorted
    precision, recall = getPRCurve(ZSorted, y_testSorted, thresholds)
    print('integral PR_curve = ' + str(np.trapz(precision, recall)))
    PRCurve(recall, precision)  