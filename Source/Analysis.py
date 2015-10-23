import image
import os

def angleBetweenVectors (self, v1, v2):
    return np.arccos(np.sum(v1*v2)/((np.sum(v1**2)**0.5)*np.sum(v2**2)**0.5))

def getHistData (self, filename,nBins=32,data=None, count=0):
    k = 256/nBins
        
    if (filename is None):
        return data, count
    
    if (data is None):
        data = np.zeros(nBins,dtype=float)
    
    im = image.imread(filename)
    l,w = im.shape
        
    for r in xrange(l):
        for c in xrange(w):
            data[im[r,c] // k] += 1

    return data, count+l*w

def getClassHist (path, nBins=32):
    hist = np.zeros(nBins)
    count = 0
    
    for filename in os.listdir(path):
        hist, count = getHistData(os.path.join(path,filename),nBins,hist,count)
        
    return hist, count

class Classifier:
    def init(self, classes, nBins=32):
        self.classes = classes
        self.hists = [None]*len(classes)
        self.counts = [0]*len(classes)
        self.nBins = 32
        self.trained = False
        
    def train(self):
        try:
            for i in range(len(classes)):
                self.hists[i], self.counts[i] = getClassHist('Train_'+classes[i],self.nBins)
            self.trained = True
        except e:
            print ('Error during trainig: ' + str(e))