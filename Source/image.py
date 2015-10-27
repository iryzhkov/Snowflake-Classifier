from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def imread (filename):
    im = Image.open(filename).convert('LA')
    l,w = im.size
    pix = im.load()
    result = np.zeros((l,w), dtype=int)
        
    for r in xrange(l):
        for c in xrange(w):
            d,x = pix[r,c]
            result[r,c] = d

    return result
    
def imsave(img, filename):
    t = Image.fromarray(img)
    t.save(filename)
    return
    
    
def bfs (data, mask, i, j):
    iMax, iMin = i,i
    jMax, jMin = j,j
    
    q = [(i,j)]
    mask[i,j] = 1
    
    h, l = data.shape

    while (len(q) > 0):
        i, j = q.pop(0)
        if (i > iMax):
            iMax = i
        if (i < iMin):
            iMin = i
        if (j > jMax):
            jMax = j
        if (j < jMin):
            jMin = j
        
        if (i+1 < h): 
            if (data[i+1,j] == 255 and mask[i+1,j] == 0):
                mask[i+1,j] = 1
                q.append((i+1,j))
        if (j+1 < l): 
            if (data[i,j+1] == 255 and mask[i,j+1] == 0):
                mask[i,j+1] = 1
                q.append((i,j+1))
        if (i-1 >= 0): 
            if (data[i-1,j] == 255 and mask[i-1,j] == 0):
                mask[i-1,j] = 1
                q.append((i-1,j))
        if (j-1 >= 0): 
            if (data[i,j-1] == 255 and mask[i,j-1] == 0):
                mask[i,j-1] = 1
                q.append((i,j-1))
    
    box = (iMax, iMin, jMax, jMin)
    
    return mask, box

def boxImage (filename, minDim = 30, t=24):
    
    img = imread(filename)
    helpImg = np.zeros(img.shape)
    
    h, l = img.shape
    
    hm, lm = int(h*0.95),int(l*0.85)
    hM, lM = int(h*0.05),int(l*0.15)
    
    im, jm, iM, jM = hm, lm, hM, lm
    
    for r in range(hM,hm):
        for c in range(lM,lm):
            if img[r,c] > t:
                if (r+2 > iM):
                    iM = r+2
                if (r-2 < im):
                    im = r-2
                if (c+2 > jM):
                    jM = c+2
                if (c-2 < jm):
                    jm = c-2
                helpImg[r-2:r+2,c-2:c+2] = 255

    mask = np.zeros(img.shape)
    boxes = []
    
    for i in range(im,iM,5):
        for j in range(jm,jM,5):
            if (helpImg[i,j] == 255 and mask[i,j] == 0):
                mask, box = bfs(helpImg, mask, i, j)
                boxes.append(box)
    
    images = []

    
    for box in boxes:
        yM, ym, xM, xm = box
        if (yM - ym >= minDim and xM - xm >= minDim):
            images.append(img[ym:yM, xm:xM])

    return images