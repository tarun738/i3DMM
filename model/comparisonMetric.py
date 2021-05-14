import json
import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh
import sys
sys.path.append('../utils')
import obj_utils as ou
import os
from plyfile import PlyData, PlyElement
def read_centroids(file):
    centroids = []
    f = open(file,"r+")
    for line in f:
        subline = line.split(",")
        centroids.append([float(subline[0]),float(subline[1]),float(subline[2])])
    f.close()
    centroids = np.array(centroids)
    # Eyes, nose, lips, (ignore chin)
    return centroids

def splitPoi(samples,centroids):
    norms = np.zeros((samples.shape[0],centroids.shape[0]))
    for i in range(centroids.shape[0]):
        norms[:,i] = np.linalg.norm(samples[:,0:3]-centroids[i,:],2,axis=1)
    flags = np.sum(norms<=0.2,axis=1)
    # flags = norms[:,2]<=0.3
    idxPoi = []
    idx = []
    for i in range(flags.shape[0]):
        if flags[i] > 0:
            idxPoi.append(i)
        else:
            idx.append(i)
    print("Sanity: " + str(len(idxPoi)) + " " + str(len(idx)) + " " + str(samples.shape[0]))
    return samples[idxPoi,:],samples[idx,:]

def readMeshCropAndSample(path,poi,centThresh=0.35,readTexture=False,num_mesh_samples=150000,maskFile=None, applyMask=False, noNeck=False):
    if readTexture == False:
        textFile = None
        if path[-3:] == "obj":
            verts,color,faces,_,_ = ou.obj_read(path,textFile)
        else:
            plydata = PlyData.read(path)
            plyFaces = plydata['face'].data
            plyVerts = plydata['vertex'].data
            print(len(plyVerts))
            verts = np.zeros((len(plyVerts),3))
            faces = np.zeros((len(plyFaces),3),dtype='int')
            color = np.zeros((len(plyVerts),3))
            for i in range(0,verts.shape[0]):
                verts[i,:] = np.array([plyVerts['x'][i],plyVerts['y'][i],plyVerts['z'][i]])
                color[i,:] = np.array([plyVerts['red'][i],plyVerts['green'][i],plyVerts['blue'][i]])
            for i in range(0,faces.shape[0]):
                faces[i,:] = np.array([plyFaces['vertex_indices'][i][0],plyFaces['vertex_indices'][i][1],plyFaces['vertex_indices'][i][2]],dtype='int')
    else:
        textFile = os.path.join(path[:-20],os.path.basename(os.path.dirname(os.path.dirname(path)))+".jpg")
        verts,_,faces,_,color = ou.obj_read(path,textFile)
    #crop mesh to one region
    if applyMask == True:
        if maskFile == None:
            # mask = np.linalg.norm(verts-poi,axis=1) <= centThresh
            mask = np.ones((verts.shape[0],),dtype='bool')
        else:
            maskVerts,maskCols,_,_,_ = ou.obj_read(maskFile,None)
            if maskVerts.shape[0] != verts.shape[0]:
                print("*********************************************************************")
                print("Transferring mask")
                print("*********************************************************************")
                maskTree = KDTree(maskVerts)
                _, maskIdx = maskTree.query(verts)
                maskCols = maskCols[maskIdx,:]
            mask = np.linalg.norm(maskCols-np.array([[1.0,0.0,0.0]]),axis=1) == 0.0
        num = np.sum(mask)
        update_idx = np.zeros((verts.shape[0],1)) - 1
        new_vert = np.zeros((num,verts.shape[1]+color.shape[1]))
        idx = 0
        for i in range(verts.shape[0]):
            if mask[i] == 1:
                new_vert[idx,:] = np.append(verts[i,:],color[i,:])
                update_idx[i] = idx
                idx = idx + 1

        new_faces = faces
        face_mask = []
        for i in range(new_faces.shape[0]):
            for j in range(new_faces.shape[1]):
                if update_idx[new_faces[i][j]] == -1:
                    face_mask.append(i)
                new_faces[i][j] = update_idx[new_faces[i][j]]
        new_faces = np.delete(new_faces,face_mask,0)
    else:
        if noNeck == False:
            new_faces = faces
            new_vert = np.append(verts,color,axis=1)
        else:
            # crop neck based on centroids as described in preprocessing

            #'REyeR','REyeL','LEyeR','LEyeL','Nose','LipsR','LipsL','Chin'
            eyesCentroid = np.mean(poi[0:4,:],axis=0)

            nose = poi[4,:]
            ythreshold = poi[7,1] - (np.linalg.norm(eyesCentroid-nose,2))/2

            mask = verts[:,1] >= ythreshold*0.95
            num = np.sum(mask)
            update_idx = np.zeros((verts.shape[0],1)) - 1
            new_vert = np.zeros((num,verts.shape[1]+color.shape[1]))
            idx = 0
            for i in range(verts.shape[0]):
                if mask[i] == 1:
                    new_vert[idx,:] = np.append(verts[i,:],color[i,:])
                    update_idx[i] = idx
                    idx = idx + 1

            new_faces = faces
            face_mask = []
            for i in range(new_faces.shape[0]):
                for j in range(new_faces.shape[1]):
                    if update_idx[new_faces[i][j]] == -1:
                        face_mask.append(i)
                    new_faces[i][j] = update_idx[new_faces[i][j]]
            new_faces = np.delete(new_faces,face_mask,0)
            print("*********************************************************************")
            print("Cropped Neck faces")
            print("*********************************************************************")
    ou.obj_write(os.path.basename(path)[:-4]+"_cropped.obj", new_vert,faces=new_faces)

    mesh = trimesh.Trimesh(vertices=new_vert[:,:3],faces=new_faces)
    sampledPoints = trimesh.sample.sample_surface(mesh, num_mesh_samples)[0]

    searchTree = KDTree(new_vert[:,:3])
    _, idxs = searchTree.query(sampledPoints)
    sampledcolors = new_vert[idxs,3:]
    sampledcolors = sampledcolors/np.max(np.max(sampledcolors))

    return sampledPoints, sampledcolors


def compute_metrics(gtMeshPath, evalMeshPath, num_mesh_samples=150000, evaluateColor=True, maskFile=None,full=False,noNeck=False):

    #giving a maskfile ensures that only masked region is compared with
    centroidPath = gtMeshPath[:-20]+"centroids.txt"
    centroids = read_centroids(centroidPath)

    if full == False:
        gtMaskPath = gtMeshPath[:-4]+"_mask.obj"
        applyMask=True
    else:
        gtMaskPath = None
        applyMask=False
    #one way computation

    #get masked GT points.
    masked_gt_points_sampled,masked_gt_points_color = readMeshCropAndSample(gtMeshPath,centroids,readTexture=True, maskFile=gtMaskPath, applyMask=applyMask, noNeck=noNeck)
    #get unmasked eval mesh points
    unmasked_gen_points_sampled,unmasked_gen_points_colors = readMeshCropAndSample(evalMeshPath,centroids,readTexture=False, applyMask=False)

    #get masked eval points
    unmasked_gt_points_sampled,unmasked_gt_points_color = readMeshCropAndSample(gtMeshPath,centroids,readTexture=True, applyMask=False)
    masked_gen_points_sampled,masked_gen_points_colors = readMeshCropAndSample(evalMeshPath,centroids,readTexture=False,maskFile=maskFile, applyMask=True)


    # one direction
    unmasked_gen_points_kd_tree = KDTree(unmasked_gen_points_sampled)
    one_distances, one_vertex_ids = unmasked_gen_points_kd_tree.query(masked_gt_points_sampled)
    gt_to_gen_chamfer = np.mean(one_distances)

    # other direction
    unmasked_gt_points_kd_tree = KDTree(unmasked_gt_points_sampled)
    two_distances, two_vertex_ids = unmasked_gt_points_kd_tree.query(masked_gen_points_sampled)
    gen_to_gt_chamfer = np.mean(two_distances)

    completeness = one_distances
    accuracy = two_distances
    # max_side_length = np.max(bb_max - bb_min)
    f_score_threshold = 0.01 # deep structured implicit functions sets tau = 0.01
    # L2 chamfer
    l2_chamfer = ((completeness).mean() + (accuracy).mean())/2
    # F-score
    f_completeness = np.mean(completeness <= f_score_threshold)
    f_accuracy = np.mean(accuracy <= f_score_threshold)
    f_score = 100*2 * f_completeness * f_accuracy / (f_completeness + f_accuracy) # harmonic mean

    color_error = 0
    if evaluateColor == True:
        colot_gt_to_gen_chamfer = np.mean(np.linalg.norm(masked_gt_points_color-unmasked_gen_points_colors[one_vertex_ids,:],axis=1))
        colot_gen_to_gt_chamfer = np.mean(np.linalg.norm(unmasked_gt_points_color[two_vertex_ids,:]-masked_gen_points_colors,axis=1))
        color_error = (colot_gt_to_gen_chamfer + colot_gen_to_gt_chamfer)/2
    print("Symmetric Chamfer L2: " + str(l2_chamfer))
    print("f_score: " + str(f_score))
    print("f_score: " + str(f_score))
    print("gt to eval Chamfer: " + str(gt_to_gen_chamfer))
    print("eval to gt Chamfer: " + str(gen_to_gt_chamfer))
    print("color_error symmetric: " + str(color_error))
    print("color_error gt to eval: " + str(colot_gt_to_gen_chamfer))
    print("color_error eval to gt: " + str(colot_gen_to_gt_chamfer))
    return gt_to_gen_chamfer, gen_to_gt_chamfer, l2_chamfer, f_score, colot_gt_to_gen_chamfer, colot_gen_to_gt_chamfer, color_error


# example call
#compute_trimesh_chamfer(gt mesh path, evaluation mesh path, num_mesh_samples=150000, otherModels=comparisons, maskFile=gt mesh mask path)
