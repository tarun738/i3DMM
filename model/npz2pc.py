import numpy as np


npzfile = np.load("./data/SdfSamples/dataset/heads/xxxxx_exx.npz")

posSamples = npzfile['pos']
negSamples = npzfile['neg']
points = np.append(posSamples,negSamples,axis=0)
if 'pospoi' in npzfile:
    pospoiSamples = npzfile['pospoi']
    points = np.append(points,pospoiSamples,axis=0)
if 'negpoi' in npzfile:
    negpoiSamples = npzfile['negpoi']
    points = np.append(points,negpoiSamples,axis=0)
print(points.shape[0])
f= open("sample.obj","w+")
if points.shape[1] == 4:
    for i in range(points.shape[0]):
        f.write( "v " + str(points[i,0]) + " " + str(points[i,1]) + " " + str(points[i,2]) + "\n")
else:
    for i in range(points.shape[0]):
        color = ((points[i,4:7] + 0.5)*255.0).astype('uint8')
        f.write( "v " + str(points[i,0]) + " " + str(points[i,1]) + " " + str(points[i,2]) + " " + str(color[0]) + " " + str(color[1]) + " " + str(color[2]) + "\n")
f.close()
