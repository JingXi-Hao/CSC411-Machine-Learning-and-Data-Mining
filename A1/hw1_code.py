import numpy as np
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier

# question 2a 
def load_data():
    # read the clean_real.txt
    cleanRealFile = open("./clean_real.txt", "r")
    # read the whole file into an array
    realHeadlines = cleanRealFile.readlines()
    # close the file
    cleanRealFile.close()
    
    # read the fake_real.txt
    cleanFakeFile = open("./clean_fake.txt", "r")
    # read the whole file into an array
    fakeHeadlines = cleanFakeFile.readlines()
    # close the file
    cleanFakeFile.close()
    
    # combine two files together into one array
    totalHeadlines = realHeadlines + fakeHeadlines
    
    # vectorize the collection of raw documents
    vectorizer = TfidfVectorizer()
    vectorizedHeadlines = vectorizer.fit_transform(totalHeadlines)
    featureList = vectorizer.get_feature_names()
    
    # add classifier to each healine, let real headline be 0 and fake headline
    # be 1
    realHeadlinesLength = len(realHeadlines)
    fakeHeadlinesLength = len(fakeHeadlines)
    totalHeadlinesLength = len(totalHeadlines)
    realHeadlineFlag = 1
    fakeHeadlineFlag = 0
    totalDataSet = []
    
    for i in range(totalHeadlinesLength):
        row = vectorizedHeadlines[i]
        if i < realHeadlinesLength:
            totalDataSet.append([row, realHeadlineFlag])
        else:
            totalDataSet.append([row, fakeHeadlineFlag])
    
    # randomize the total data set
    np.random.shuffle(totalDataSet)
    
    # splits the entire dataset randomly into 70% training, 15% validation, 
    # and 15% test examples
    numberOfDataInTraining = int((0.7) * (totalHeadlinesLength))
    numberOfDataInValidation = int((0.15) * (totalHeadlinesLength))
    numberOfDataInTest = totalHeadlinesLength - numberOfDataInTraining - numberOfDataInValidation
    
    trainingSet = totalDataSet[0 : numberOfDataInTraining]
    
    highIndex = numberOfDataInTraining + numberOfDataInValidation
    validationSet = totalDataSet[numberOfDataInTraining : highIndex]
    
    lowIndex = highIndex
    testSet = totalDataSet[lowIndex : totalHeadlinesLength]
    
    return (trainingSet, validationSet, testSet, featureList)

# question 2b
def computeAccuracy(validationData, validationY, clf):
    numberOfSuccess = 0
    
    for i in range(len(validationData)):
        predictedY = clf.predict([validationData[i]])
        if predictedY[0] == validationY[i]:
            numberOfSuccess = numberOfSuccess + 1
             
    return numberOfSuccess / float(len(validationData))

def select_model(trainingSet, validationSet):
    # first we develop 10 decision tree classiers, 5 for each criteria (entropy
    # and gini)
    clfGiniList = []
    clfEntropyList = []
    depthList = [5, 25, 45, 55, 65]
    for j in range(0, len(depthList)):
        clfGini = DecisionTreeClassifier(criterion="gini", max_depth=depthList[j])
        clfEntropy = DecisionTreeClassifier(criterion="entropy", max_depth=depthList[j])
        clfGiniList.append(clfGini)
        clfEntropyList.append(clfEntropy)
    
    # get the training data
    trainingData = []
    trainingY = []
    
    for k in range(len(trainingSet)):
        dataWithY = trainingSet[k]
        data = dataWithY[0].toarray()[0]
        y = dataWithY[1]
        trainingData.append(data)
        trainingY.append(y)
        
    # get the validation data
    validationData = []
    validationY = []
    
    for s in range(len(validationSet)):
        dataWithY = validationSet[s]
        data = dataWithY[0].toarray()[0]
        y = dataWithY[1]
        validationData.append(data)
        validationY.append(y)
        
    # train the model and compute accuracy
    for l in range(0, len(depthList)):
        clfGini = clfGiniList[l].fit(trainingData, trainingY)
        clfEntropy = clfEntropyList[l].fit(trainingData, trainingY)
        accuracyGini = computeAccuracy(validationData, validationY, clfGini)
        accuracyEntropy = computeAccuracy(validationData, validationY, clfEntropy)
        print ("The accuracy for tree decision classifier with criterion Gini and max_depth", depthList[l], "is", accuracyGini)
        print ("The accuracy for tree decision classifier with criterion Entropy and max_depth", depthList[l], "is", accuracyEntropy, "\n")
        
    return (trainingData, trainingY)
    
# question 2d
def compute_information_gain(trainingData, trainingY, keyword, featureList):
    # find index of the keyword in featureList
    index = featureList.index(keyword)
    
    # find total number of real headlines and fake headlines in training set
    totalReal = 0
    totalFake = 0
    total = len(trainingY)
    for j in range(len(trainingY)):
        if trainingY[j] == 0:
            totalFake += 1
        else:
            totalReal += 1
    
    # compute entropy for root
    h_Y = -((float(totalFake) / total) * np.log2(float(totalFake) / total)) - ((float(totalReal) / total) * np.log2(float(totalReal) / total))
    
    realInLeft = 0
    fakeInLeft = 0
    realInRight = 0
    fakeInRight = 0
    # split training set by using keyword, not containing the keyword would be
    # left side of the tree; otherwise, would be in right side of the tree
    for i in range(len(trainingData)):
        headline = trainingData[i]
        # headline contain keyword, then it should be in right side of the tree
        if headline[index] > 0:
            # headline is fake
            if trainingY[i] == 0:
                fakeInRight = fakeInRight + 1
            else:
                realInRight = realInRight + 1
        # headline not contain keywork, and will be in left side of the tree
        else:
            if trainingY[i] == 0:
                fakeInLeft = fakeInLeft + 1
            else:
                realInLeft  = realInLeft + 1
    
    # compute left conditional entropy
    leftTotal = realInLeft + fakeInLeft
    h_L = -((float(fakeInLeft) / leftTotal) * np.log2(float(fakeInLeft) / leftTotal)) - ((float(realInLeft) / leftTotal) * np.log2(float(realInLeft) / leftTotal))
    
    # compute right conditional entropy
    rightTotal = realInRight + fakeInRight
    h_R = -((float(fakeInRight) / rightTotal) * np.log2(float(fakeInRight) / rightTotal)) - ((float(realInRight) / rightTotal) * np.log2(float(realInRight) / rightTotal))
    
    # compute information gain
    result = h_Y - ((float(leftTotal) / total) * h_L) - ((float(rightTotal) / total) * h_R)
    
    print("The information gain for keyword", keyword, "is", result)
    
    return result    
        
# main function is defined here
if __name__ == "__main__":
    #question 2a
    trainingSet, validationSet, testSet, featureList = load_data()
    
    #question 2b
    trainingData, trainingY = select_model(trainingSet, validationSet)
    
    # question 2d
    keywordsList = ['trump', 'trumps', 'donald', 'military', 'obama']
    for value in keywordsList:
        result = compute_information_gain(trainingData, trainingY, value, featureList)
    