from sklearn import svm
from sklearn.svm import LinearSVC
import math
import array
import timeit


def getTrainData():
    data=[]
    
    with open('traindata','r') as infile:
        for line in infile:
            temp=line.split()
            row=array.array('I',[])
            for i in temp:
                row.append(int(i))
            data.append(row)               
    return data

def getTrueLabels():
    labels=[]
    
    with open('trueclass.txt','r') as infile:
        for line in infile:
            temp=line.split()[0]
            datalabel=array.array('I',[int(temp)])
            labels.append(datalabel)               
    return labels

def getTestData():
    testdata=[]
    
    with open('testdata','r') as infile:
        for line in infile:
            temp=line.split()
            row=array.array('I',[])
            for i in temp:
                row.append(int(i))
            testdata.append(row)               
    return testdata


def pearsonvalue(datacol,labels,col):                   #tested
    num=den1=den2=meanX=meanY=0
    totaldata=len(datacol)

    meanY=sum(labels)/len(labels)                                 

    meanX=sum(datacol)/totaldata
    for i in range(totaldata):
        a=datacol[i]-meanX
        b=labels[i]-meanY
        num+=a*b
        den1+=a**2
        den2+=b**2
    if den1!=0 and den2!=0:
        r=abs(num/math.sqrt(den1*den2))
    else:
        r=0
    return r

def getBestFeatures(data,labels,k):                          #tested
    pearson_coefficients=[]
    correspondingfeatures=[i for i in range(len(data[0]))]
    
    start_time = timeit.default_timer()

    featurematrix=list(zip(*data))
    labelcol=list(zip(*labels))[0]

    for j in range(len(data[0])):
        pvalue=pearsonvalue(featurematrix[j],labelcol,j)
        pearson_coefficients.append(pvalue)
        
    #print('total time for pearson function:',timeit.default_timer() - start_time)
    

    best_features=[]
    for j in range(k):
        featureidx=pearson_coefficients.index(max(pearson_coefficients))
        best_features.append(correspondingfeatures.pop(featureidx))
        del pearson_coefficients[featureidx]
        
    return best_features
    

def transform(data,bestFeatures):                                       #tested
    newdata=[]
    for datapoint in data:
        row=[]
        for f in bestFeatures:
            row.append(datapoint[f])
        newdata.append(row)
    return newdata
    

def crossvalidation(data,labels):                           #data and labels are lists of arrays      #tested
    cvfolds=10

    labelslist=list(zip(*labels))[0]
    classzerolength=labelslist.count(0)
    classonelength=labelslist.count(1)                         
    
    startzero=labelslist.index(0)
    startone=labelslist.index(1)
    
    cvprediction=[]
    accuracies=[]
    
    '''
    spliting train and testdata
    '''

    validation_traindata=data[startzero:startzero+int(0.9*classzerolength)]+data[startone:startone+int(0.9*classonelength)]
    validation_testdata=data[startzero+int(0.9*classzerolength):startone]+data[startone+int(0.9*classonelength):classzerolength+classonelength]

    validation_labels=labels[startzero:startzero+int(0.9*classzerolength)]+labels[startone:startone+int(0.9*classonelength)]
    validation_testlabels=labels[startzero+int(0.9*classzerolength):startone]+labels[startone+int(0.9*classonelength):classzerolength+classonelength]


    classzero_step=int((0.9*classzerolength)/10)
    classone_step=int((0.9*classonelength)/10)

    '''
    feature selection on validation traindata
    '''
    
    bestfeatures=getBestFeatures(validation_traindata,validation_labels,15)        

    transformed_data=transform(validation_traindata,bestfeatures)

    transformed_testdata=transform(validation_testdata,bestfeatures)

    

    '''
    creating 10 folds and training of model on each training data
    '''
    
    validation_labelslist=list(zip(*validation_labels))[0]
    for i in range(cvfolds):
        traindata=[]
        testdata=[]
        trainlabels=[]
        testlabels=[]

        startzero=validation_labelslist.index(0)
        startone=validation_labelslist.index(1)

        for j in range(cvfolds):
            if j!=i:
                traindata+=transformed_data[startzero:startzero+classzero_step]+transformed_data[startone:startone+classone_step]
                trainlabels+=validation_labels[startzero:startzero+classzero_step]+validation_labels[startone:startone+classone_step]
            else:
                testdata+=transformed_data[startzero:startzero+classzero_step]+transformed_data[startone:startone+classone_step]
                testlabels+=validation_labels[startzero:startzero+classzero_step]+validation_labels[startone:startone+classone_step]
            startzero+=classzero_step
            startone+=classone_step
    
        
        clf=svm.LinearSVC(C=0.01)
        clf.fit(traindata,trainlabels)
        cvpredictions=clf.predict(testdata)

        match=0
        for x in range(len(cvpredictions)):
            if cvpredictions[x]==testlabels[x][0]:         #cvpredictions type--np array
                match+=1
        accuracies.append(match/len(testlabels))    
    '''
    predicting on previously 10% split test data - validation_testdata
    '''
    
    testdata_predictions=clf.predict(transformed_testdata)
    test_match=0
    for x in range(len(testdata_predictions)):
        if testdata_predictions[x]==validation_testlabels[x][0]:        
            test_match+=1
    final_accuracy=test_match/len(validation_testlabels)
    
    print('10-fold cross validation accuracies:',accuracies)
    print('Mean of 10-fold cross validation accuracy',sum(accuracies)/len(accuracies))

    print('\n\nCross Validation Final Test Accuracy:',final_accuracy)

    return (accuracies,bestfeatures,transformed_data,transformed_testdata,validation_labels,validation_testlabels,clf)         


def classify(data,truelabels,testdata):
    #start_time = timeit.default_timer()
    
    '''
    cross validation function gives trained all the accuracies measured during evaluation, 
    best features selected,featured data and trained classifier
    '''
##    results=crossvalidation(data,truelabels)
##
##    scores=results[0]
##    bestfeatures=results[1]
##    reduceddata=results[2]+results[3]
##    corresponding_labels=results[4]+results[5]
##    
##    trained_classifier=results[6]

    bestfeatures=getBestFeatures(data,truelabels,15)

    newdata=transform(data,bestfeatures)
    newtestdata=transform(testdata,bestfeatures)
    
    
    print('\nNumber of features used:',len(bestfeatures))

    print('\nColumn numbers selected:')
    for f in bestfeatures:
        print(f,' ')
    #print('total time for cross validation function:',timeit.default_timer() - start_time)

    clf=LinearSVC(C=0.01)
    clf.fit(newdata,truelabels)

    final_predictions=clf.predict(newtestdata)

    print('predictions for test data:')
    for i in final_predictions:
        print(i)
    


    with open('testpredictions.txt','w') as outfile:
        for i,each in enumerate(final_predictions):
            outfile.write(str(each)+' '+str(i)+'\n')
    print('\n\n-------File of Final TestData Predictions written:test_predictions.txt------')
    #print('\n\ntotal time for classify function:',timeit.default_timer() - start_time)


    

    
if __name__=='__main__':
    start_time = timeit.default_timer()
    data=getTrainData()
    truelabels=getTrueLabels()
    testdata=getTestData()
    print('loaded all data')
    print('total time for loading:',timeit.default_timer() - start_time)


    classify(data,truelabels,testdata)
    #testfunction(data,truelabels)
    print('\n\nDone !!')
    print('---------------------------------------------------------------------')
    
    
    

