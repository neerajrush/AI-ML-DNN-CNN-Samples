from numpy import *
import operator

def dataSet():
    numDataArray = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['label1','label1','label2','label2']
    return numDataArray, labels

def classifier(unknown, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(unknown, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__ == "__main__":
	dset, dlabels = dataSet()
	x = classifier([0.7,1.0], dset, dlabels, 3)
	print(x)
