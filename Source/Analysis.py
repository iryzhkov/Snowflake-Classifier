import numpy as np
import image
import os

from random import shuffle

def getHistData (filename,nBins=32,data=None,count=0):
    k = 256/nBins
    
    if (filename is None):
        return data, count
    
    if (data is None):
        data = np.zeros(nBins,dtype=float)

    im = imread(filename)
    l,w = im.shape

for r in xrange(l):
    for c in xrange(w):
        data[im[r,c] // k] += 1
    
    return data, count+l*w

def vectorAngle (v1, v2):
    return np.arccos(np.sum(v1*v2)/((np.sum(v1**2)**0.5)*np.sum(v2**2)**0.5))

def getVectorAngles (SetOfVectors, v):
    result = np.array([0.0]*len(SetOfVectors))
    
    for i in range(len(SetOfVectors)):
        result[i] = vectorAngle(v, SetOfVectors[i])
    return result

def getFilenames (path, testTrainRatio=0.5):
    filenames = os.listdir(path)
    
    for i in range(len(filenames)):
        filenames[i] = os.path.join(path,filenames[i])

    shuffle(filenames)
    trainNames = filenames[:int(len(filenames)*testTrainRatio)]
    testNames = filenames[int(len(filenames)*testTrainRatio):]

return trainNames, testNames


def getClassHist (filenames, nBins=32, testTrainRatio=0.5):
    hist = np.zeros(nBins)
    count = 0
    
    for filename in filenames:
        hist, count = getHistData(filename,nBins,hist,count)
    
    return hist, count

class Classifier:
    def __init__(self, classes=['DrySnow', 'Graupel'], nBins=32):
        self.classes = classes
        self.hists = [None]*len(classes)
        self.counts = [0]*len(classes)
        self.nBins = 32
        self.trained = False
        
        self.trainFileNames = [None]*len(classes)
        self.testFileNames = [None]*len(classes)
        
        self.nClasses = len(self.classes)
        
        self.coeffs = None
    
    def train(self, testTrainRatio=0.5):
        for i in range(self.nClasses):
            self.trainFileNames[i], self.testFileNames[i] = getFilenames('Train_' + self.classes[i],testTrainRatio)
            self.hists[i], self.counts[i] = getClassHist(self.trainFileNames[i],self.nBins)
            
            ''' start doing linear regression classification (using numpy linear algebra package)
                so we have ax = b
                a is a matrix
                b is the results.
                we want b = [0,0,0,...,0,1,0,0,...], where 1 is in the position of the selected class
                '''
        
        A = []
        B = []
        
        plt.figure(figsize=(14,7))
        plt.subplot(121)
        color=iter(cm.rainbow(np.linspace(0,1,self.nClasses)))
        plt.title ('Train Data and linear classifier')
        
        #fix the problem with legend coloring
        for i,c in zip(range(self.nClasses), color):
            for filename in self.trainFileNames[i]:
                v, count = getHistData(filename, self.nBins)
                
                
                x = np.append(getVectorAngles(self.hists, v), [1])
                A.append(x.T)
                b = np.zeros(len(self.classes))
                
                plt.scatter(x[0], x[1], c=c)
                
                b[i] = 1
                B.append(b.T)
        
        
        self.coef = np.linalg.lstsq(A, B)[0]
        errorMatrix = np.zeros((len(self.classes), len(self.classes)), dtype=float)
        
        x = np.linspace(-0.5,2,500)
        v = np.array([x[0],0,1])
        r = np.dot(v,self.coef)
        
        xMin, xMax = 0,0
        yMin, yMax = 0,1
        
        before = np.argmax(r)
        
        for i in x:
            v = np.array([i,0,1])
            r = np.dot(v,self.coef)
            if (np.argmax(r) != before):
                xMin = i
                break
        
        
        v = np.array([x[0],1,1])
        r = np.dot(v,self.coef)
        before = np.argmax(r)
        
        for i in x:
            v = np.array([i,1,1])
            r = np.dot(v,self.coef)
            if (np.argmax(r) != before):
                xMax = i
                break
                
                x = [xMin, xMax]
                y = [yMin, yMax]
                
                plt.plot(x,y,'r--')
                
                plt.subplot(122)
                plt.title('Test Data and linear classifier')
                plt.plot(x,y,'r--')
                count = 0
                color=iter(cm.rainbow(np.linspace(0,1,self.nClasses)))
                for i, c in zip(range(self.nClasses),color):
                    for filename in self.testFileNames[i]:
                        r, x = self.classify(filename)
                            plt.scatter(x[0], x[1], c=c)
                                errorMatrix[i][r] += 1
                                    count += 1
                                
                                self.trained = True
                                    return errorMatrix

    def classify (self, filename):
        v, count = v, count = getHistData(filename, self.nBins)
        x = np.append(getVectorAngles(self.hists, v), [1])
        results = np.dot(x,self.coef)
        return np.argmax(results), x