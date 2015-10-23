from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def open (self, filename):
    im = Image.open(filename).convert('LA')
    l,w = im.size
    pix = im.load()
    result = np.zeros((l,w), dtype=int)
        
    for r in xrange(l):
        for c in xrange(w):
            d,x = pix[r,c]
            result[r,c] = d

    return result