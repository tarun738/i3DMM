import numpy as np
import argparse
import json
import os
import cv2
from multiprocessing import Process
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import trimesh as tm
import sys
sys.path.append('../utils')
import obj_utils as ou
from meshToSDF import sampleSDFsFromMesh
from multiprocessing import Pool
def read_centroids(file):
    centroids = []
    f = open(file,"r+")
    for line in f:
        subline = line.split(",")
        centroids.append([float(subline[0]),float(subline[1]),float(subline[2])])
    if os.path.isfile(file[:-4]+"Ears.txt"):
        f = open(file[:-4]+"Ears.txt","r+")
        for line in f:
            subline = line.split(",")
            centroids.append([float(subline[0]),float(subline[1]),float(subline[2])])
        f.close()
    centroids = np.array(centroids)
    # Eyes, nose, lips, (ignore chin)
    if centroids.shape[0] == 16:
        centroids = np.array([[(centroids[0,:]+centroids[1,:])/2.0],[(centroids[2,:]+centroids[3,:])/2.0],[centroids[4,:]],[(centroids[5,:]+centroids[6,:])/2.0],[np.mean(centroids[8:12],axis=0)],[np.mean(centroids[12:16],axis=0)]])
    else:
        centroids = np.array([[(centroids[0,:]+centroids[1,:])/2.0],[(centroids[2,:]+centroids[3,:])/2.0],[centroids[4,:]],[(centroids[5,:]+centroids[6,:])/2.0]])
    print(centroids)
    print(centroids.shape)
    return centroids
def splitPoi(samples,centroids):
    norms = np.zeros((samples.shape[0],centroids.shape[0]))
    for i in range(centroids.shape[0]):
        norms[:,i] = np.linalg.norm(samples[:,0:3]-centroids[i,:],2,axis=1)
    flags = np.sum(norms<=0.15,axis=1)
    idxPoi = []
    idx = []
    for i in range(flags.shape[0]):
        if flags[i] > 0:
            idxPoi.append(i)
        else:
            idx.append(i)
    print("Sanity: " + str(len(idxPoi)) + " " + str(len(idx)) + " " + str(samples.shape[0]))
    return samples[idxPoi,:],samples[idx,:]
def meshToColorSDF(tupFiles):
    outPath = tupFiles[0]
    mesh = tupFiles[1]
    texture = tupFiles[2]
    # Read the points of interest, centroids of eyes, nose and lips
    poi_centroids_file = os.path.join(Path(mesh).parent,"centroids.txt")
    poi_centroids = read_centroids(poi_centroids_file)
    # poi_centroids = poi_centroids.squeeze()

    # get vertices and vertex color indices from mesh
    vertices, vertex_color_uv, _, faces, facesText, _, _ = ou.read_all_obj(mesh)
    print(texture)
    texture = cv2.imread(texture)
    numPerFace = 12
    a = np.linspace(0,1,numPerFace,endpoint=True)
    index = []
    for i in range(0,numPerFace):
        temp = 1-a[i]
        b = np.linspace(0,temp,numPerFace,endpoint=True)
        c = temp-b
        for j in range(0,numPerFace):
            newIdx = [a[i],b[j].tolist(),c[j].tolist()]
            if newIdx not in index:
                index.append(newIdx)
    index = np.array(index)
    N = index.shape[0]
    print(N)
    points = np.zeros((faces.shape[0]*N,3))
    uvs = np.zeros((points.shape[0],2))
    print("Generating Samples")
    for i in range(0,faces.shape[0]):
        a = np.expand_dims(index[:,0],axis=1)
        b = np.expand_dims(index[:,1],axis=1)
        c = 1-a-b
        i0 = faces[i,0]
        i1 = faces[i,1]
        i2 = faces[i,2]
        it0 = facesText[i,0]
        it1 = facesText[i,1]
        it2 = facesText[i,2]
        points[i*N:(i+1)*N,:] = a * vertices[i0] + b * vertices[i1] + c * vertices[i2]
        uvs[i*N:(i+1)*N,:] = a * vertex_color_uv[it0] + b * vertex_color_uv[it1] + c * vertex_color_uv[it2]

    print("Getting Colors")
    colors = ou.fetch_colors(texture,uvs)
    colors = (colors/255.0) - 0.5
    # ou.obj_write("samples.obj",np.append(points,colors,axis=1))
    samples = sampleSDFsFromMesh(mesh, numSamples=500000 , nsDev=0.05, nsPer=0.94, lms=poi_centroids, lmProb=0.8)
    pos = samples[samples[:,3]>=0,:]
    neg = samples[samples[:,3]<0,:]
    # load sdf samples

    # get the nearest points on the point cloud for each sdf sample of pos and neg
    search_tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points)
    distances, indices = search_tree.kneighbors(pos[:,0:3])
    indices = indices.squeeze()
    posColors = np.append(pos[:,0:4],colors[indices,:],axis=1)
    pospoi,pos = splitPoi(posColors,poi_centroids)
    distances, indices = search_tree.kneighbors(neg[:,0:3])
    indices = indices.squeeze()
    negColors = np.append(neg[:,0:4],colors[indices,:],axis=1)
    negpoi,neg = splitPoi(negColors,poi_centroids)

    np.savez(outPath,pospoi=pospoi.astype('float32'),pos=pos.astype('float32'),negpoi=negpoi.astype('float32'),neg=neg.astype('float32'))
    return
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Preprocess Meshes for SDF samples with color")
    arg_parser.add_argument(
        "--samples_directory",
        "-i",
        dest="sample_dir",
        required=True,
        help="Directory containing sdf samples",
    )
    arg_parser.add_argument(
        "--input_meshes_directory",
        "-m",
        dest="mesh_dir",
        required=True,
        help="Directory containing meshes",
    )

    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment",
        required=True,
        help="experiment folder",
    )

    arg_parser.add_argument(
        "--splits",
        "-s",
        dest="splits",
        required=True,
        help="split train/test",
    )

    args = arg_parser.parse_args()

    # load the json file to get the input files for the current experiment
    with open(args.experiment+"/specs.json",'r') as specsFile:
        contents = json.load(specsFile)
        if args.splits == "train" or args.splits == "Train":
            splitsFile = contents["TrainSplit"]
        else:
            splitsFile = contents["TestSplit"]
    print(splitsFile)
    # get samples from the input directory and the source meshes and textures
    with open(splitsFile,'r') as split:
        splitsFile = json.load(split)
    procs = Pool(8)
    files_list = []
    for dataset in splitsFile:
        for folder in splitsFile[dataset]:
            outFolder = os.path.join(args.sample_dir,"SdfSamples",dataset,folder)
            if not os.path.isdir(outFolder):
                os.makedirs(outFolder)
            for file in splitsFile[dataset][folder]:
                write_samples_file = os.path.join(outFolder,file+".npz")
                inFolder = os.path.join(args.mesh_dir,folder,file,"models")
                mesh_file = os.path.join(inFolder,"model_normalized.obj")
                texture_file = os.path.join(inFolder,file[:9]+".jpg")
                # update the samples file to have texture
                if not os.path.isfile(write_samples_file):
                    files_list.append([write_samples_file,mesh_file,texture_file])
    files_list = tuple(files_list)
    procs.map(meshToColorSDF,files_list)
