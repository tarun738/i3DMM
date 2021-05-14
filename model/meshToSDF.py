## changing opengl to egl to ensure that the code works with ssh
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import trimesh as tm
from mesh_to_sdf.surface_point_cloud import sample_from_mesh
#
# ## mesh to sdf library
# import mesh_to_sdf
#
# print("Import works")



#read mesh
def sampleSDFsFromMesh(meshFile, numSamples=500000 , nsDev=0.0001, nsPer=0.94, lms=None, lmProb=0.8, normalize=False):
   # compute the number of points to sample near the surface and away from the surface
   NSurface=int(numSamples*nsPer/2)
   NBox = numSamples-NSurface*2
   nsDev2 = nsDev/10
   #load mesh using trimesh
   mesh = tm.load(meshFile, force='mesh', process=False)
   if normalize:
       cent = np.mean(mesh.vertices)
       mesh.vertices = mesh.vertices - cent
       norm = np.max(np.linalg.norm(mesh.vertices,axis=1))
       mesh.vertices = mesh.vertices/norm


   #sample points on the surface
   if lms is None:
       samples, _ =tm.sample.sample_surface(mesh,NSurface)
   else:
       NSurfacePoi = int(NSurface*lmProb)
       NSurfaceOther = NSurface - NSurfacePoi
       countPoi = 0
       samplesPoi = np.zeros((0,3))
       samplesOther = np.zeros((0,3))
       print(lms)
       while samplesPoi.shape[0] < NSurfacePoi:
           tempSamples, _ =tm.sample.sample_surface(mesh,NSurface)
           norms = np.zeros((tempSamples.shape[0],lms.shape[0]))
           for i in range(lms.shape[0]):
               norms[:,i] = np.linalg.norm(tempSamples[:,0:3]-lms[i,:],2,axis=1)
           flags = np.sum(norms<=0.15,axis=1)
           idxPoi = []
           idx = []
           for i in range(flags.shape[0]):
               if flags[i] > 0:
                   idxPoi.append(i)
               else:
                   idx.append(i)
           samplesPoi = np.append(samplesPoi,tempSamples[idxPoi,:],axis=0)
           samplesOther = np.append(samplesOther,tempSamples[idx,:],axis=0)
           print(samplesPoi.shape)
           print(samplesOther.shape)
       np.random.shuffle(samplesPoi)
       samplesPoi = samplesPoi[:NSurfacePoi,:]
       np.random.shuffle(samplesOther)
       samplesOther = samplesOther[:NSurfaceOther,:]
       samples = np.append(samplesPoi,samplesOther,axis=0)
       np.random.shuffle(samples)

   # add gaussian noise with nsDev standard deviation such that the samples are not on the mesh but near the surface
   gaussNoiseSurface = np.random.normal(loc=0.0,scale=nsDev,size=(samples.shape[0],samples.shape[1]))
   nearSurfaceSamples1=samples+gaussNoiseSurface
   gaussNoiseSurface = np.random.normal(loc=0.0,scale=nsDev2,size=(samples.shape[0],samples.shape[1]))
   nearSurfaceSamples2=samples+gaussNoiseSurface

   #generate random samples around the object
   samplesBox = tm.sample.volume_rectangular((2,2,2),NBox)

   #trimesh sdf
   # #find the signed distance from the near surface points to the mesh
   # sdfsSurface1 = tm.proximity.signed_distance(mesh,nearSurfaceSamples1)
   # sdfsSurface2 = tm.proximity.signed_distance(mesh,nearSurfaceSamples2)
   #
   # # find the signed distance from the samples in the box to the mesh
   # sdfsBox = tm.proximity.signed_distance(mesh,samplesBox)
   #mesh-to-sdf
   sdfSampler = sample_from_mesh(mesh)
   sdfsSurface1 = sdfSampler.get_sdf(nearSurfaceSamples1)
   sdfsSurface2 = sdfSampler.get_sdf(nearSurfaceSamples2)
   sdfsBox = sdfSampler.get_sdf(samplesBox)


   # append both the samples and sdfs and return
   samplesAll = np.append(nearSurfaceSamples1,nearSurfaceSamples2,axis=0)
   samplesAll = np.append(samplesAll,samplesBox,axis=0)
   sdfs = np.append(sdfsSurface1,sdfsSurface2)
   sdfs = np.append(sdfs,sdfsBox)
   sdfs = np.expand_dims(sdfs,axis=1)
   sdfSamples = np.append(samplesAll,sdfs,axis=1)
   return sdfSamples

if __name__ == "__main__":
    # some test mesh file
    meshFile = "any.ply"
    sdfsamples = sampleSDFsFromMesh(meshFile, numSamples=500000 , nsDev=0.0001, nsPer=0.94)


    # with open('samples.obj','w+') as f:
    #     for i in range(0,sdfsamples.shape[0]):
    #         f.write("v " + str(sdfsamples[i,0]) + " " + str(sdfsamples[i,1]) + " " + str(sdfsamples[i,2]) + "\n")
    #     f.close()
    # print(sdfsamples.shape)

    posSamples = sdfsamples[sdfsamples[:,3]>=0,:]
    negSamples = sdfsamples[sdfsamples[:,3]<0,:]

    print(posSamples.shape)
    print(negSamples.shape)
    np.savez("samples.npz",posSamples=posSamples,negSamples=negSamples)
