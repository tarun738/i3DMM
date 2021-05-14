#image rendering utilities

import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np
import i3DMM
from sphereTracer import Camera, render

class i3DMM_model():
    def __init__(self, decoder_deform, decoder_ref, decoder_col, latent_geom, latent_col, max_batch=2**20):
        self.decoder_deform = decoder_deform
        self.decoder_ref = decoder_ref
        self.decoder_col = decoder_col
        self.latent_vec_geom = latent_geom
        self.latent_vec_color = latent_col
        self.max_batch = max_batch
def reconstructAndSaveMesh(mesh_filename,decoder_deform,decoder_ref,decoder_col,latent_geom,latent_col):
    decoder_deform.eval()
    decoder_ref.eval()
    decoder_col.eval()
    if not os.path.exists(os.path.dirname(mesh_filename)):
        os.makedirs(os.path.dirname(mesh_filename))
    start = time.time()
    with torch.no_grad():
        i3DMM.mesh.create_mesh(
            decoder_deform, decoder_ref, decoder_col, latent_geom, latent_col, [], 0, [], mesh_filename, False, N=192, max_batch=int(2 ** 20),
            correspondencesDebug=False,
            refMeshDebug=False,
        )
    logging.info("total time: {}".format(time.time() - start))

def renderAndSaveImage(fileName, f_width, f_height, decoder_deform, decoder_ref, decoder_col, latent_geom, latent_col, rot=0, both=False, skip=True, vid=False):
    decoder_deform.eval()
    decoder_ref.eval()
    decoder_col.eval()
    decoder = i3DMM_model(decoder_deform, decoder_ref, decoder_col, latent_geom, latent_col, max_batch=int(2 ** 20))
    start = time.time()
    dist = 2.5
    if not vid:
        if os.path.isfile(os.path.join( fileName[0], "gammaCorrected", "color" , fileName[1]+".png")) and skip==True:
            return
        cam = Camera(np.array([0,0,dist]),np.array([0,0,0]),np.array([0,1,0]),0.5,[f_width,f_height])
        render(cam,fileName,decoder)

        if rot == np.pi/4 or both:
            rot = np.pi/4
            cam = Camera(np.array([dist/np.sqrt(2),0,dist/np.sqrt(2)]),np.array([0,0,0]),np.array([1/np.sqrt(2),1,1/np.sqrt(2)]),0.5,[f_width,f_height])
            fileName[0] = fileName[0] + "_sideview"
            render(cam,fileName,decoder)
        logging.info("Time to complete: {}".format(time.time()-start))
    else:
        id = 0
        dist = 3
        for r in range(0,362,5):
            file = fileName.copy()
            file[1] = fileName[1]+"_"+str(int(id)).zfill(3)
            id +=1
            th = r*np.pi/180.0
            a = np.cos(th)
            b = np.sin(th)
            R = np.array([[a,0,b],[0,1,0],[-b,0,a]])
            cOrg = np.dot(R,np.array([[0],[0],[dist]])).T.squeeze()
            up = np.dot(R,np.array([[0],[1],[0]])).T.squeeze()
            cam = Camera(cOrg,np.array([0,0,0]),up,0.5,[f_width,f_height])
            render(cam,file,decoder)

        logging.info("Time to complete: {}".format(time.time()-start))
    return
