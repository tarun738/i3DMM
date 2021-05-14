# a naive sphere tracer

import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np
import i3DMM
import i3DMM.workspace as ws
import math
from PIL import Image
import cv2
import sys
from sklearn.neighbors import NearestNeighbors
infinity = sys.float_info.max
max_iter = 200
eps = 1e-2
eps_n = 1e-4

def normalize(v):
    return v/np.linalg.norm(v)


class fImage:
    def __init__(self,width, height):
        self.data = np.zeros((height,width,4), 'float')
        self.height = height
        self.width = width
    def set(self,i,j, col):
        self.data[self.height -1 -j,i,0] = col[0]
        self.data[self.height -1 -j,i,1] = col[1]
        self.data[self.height -1 -j,i,2] = col[2]
        self.data[self.height -1 -j,i,3] = col[3]

    def get(self,i,j):
        return [self.data[self.height-1,i,0],self.data[self.height-1,i,1],self.data[self.height-1,i,2],self.data[self.height-1,i,3]]

    def write(self,filename, ext):
        print("Saving image as",filename)
        rgbCp = self.data*256
        rgbCp[rgbCp<0]=0
        rgbCp[rgbCp>255] = 255
        rgbArray = np.zeros((self.height,self.width,4), 'uint8')
        rgbArray[...] = rgbCp
        img = Image.fromarray(rgbArray, 'RGBA')

        file = os.path.join(filename[0],"normal",ext,filename[1]+".png")
        if not os.path.isdir(os.path.join(filename[0],"normal",ext)):
            os.makedirs(os.path.join(filename[0],"normal",ext))
        img.save(file)


        rgbArraynew = rgbArray
        rgbArraynew[:,:,0:3] = 255*np.power(rgbArray[:,:,0:3]/255.0,(1/0.65))
        img = Image.fromarray(rgbArraynew, 'RGBA')

        file = os.path.join(filename[0],"gammaCorrected",ext,filename[1]+".png")
        if not os.path.isdir(os.path.join(filename[0],"gammaCorrected",ext)):
            os.makedirs(os.path.join(filename[0],"gammaCorrected",ext))
        img.save(file)


class Ray:
    def __init__(self,o,d,t = float('inf')):
        self.o = o
        self.d = d
        self.t = t



class SurfaceInteraction:
    def __init__(self, hitPoint,normal):
        self.hitPoint = hitPoint
        self.normal = normal

def render(camera,out_f,decoder):
    width = camera.res[0]
    height = camera.res[1]
    im = fImage(width,height)
    rows = np.linspace(0,width,width,endpoint=False)
    cols = np.linspace(0,height,height,endpoint=False)
    x = ((rows +0.5)*2)/width - 1
    y = ((cols + 0.5)*2)/height - 1
    xy = np.zeros((width,height),dtype=(float,2))
    for i in range(0,width):
        for j in range(0,height):
            xy[i,j] = (x[i],y[j])
    c1,c2,c3 = raymarch(xy,camera,decoder)
    img1 = np.zeros((width,height,4),dtype='uint8')
    img2 = np.zeros((width,height,4),dtype='uint8')
    img3 = np.zeros((width,height,4),dtype='uint8')
    for i in range(0,width):
        for j in range(0,height):
            im.set(i,j,c1[i,j])
            img1[i,j] = (255*c1[i,j]).astype('uint8')
    im.write(out_f, "geometry")
    for i in range(0,width):
        for j in range(0,height):
            im.set(i,j,c2[i,j])
            img2[i,j] = (255*c2[i,j]).astype('uint8')
    im.write(out_f,"color")
    for i in range(0,width):
        for j in range(0,height):
            im.set(i,j,c3[i,j])
            img3[i,j] = (255*c3[i,j]).astype('uint8')
    im.write(out_f, "correspondences")
    return img1

class Camera:
    def __init__(self, o, at, up, fov, res, transform=None, takeDirectFOV=False,invZ=True):
        self.aspect = float(res[0])/res[1]
        self.origin = o
        v = at-o
        if invZ==True:
            z = normalize(-v)
        else:
            z = normalize(v)
        x = normalize(np.cross(up,z))
        y = normalize(np.cross(z,x))
        self.transform = np.array([x,y,z]).transpose()
        print(self.transform)
        self.tanFOV = math.tan(fov/2)
        if takeDirectFOV == True:
            self.tanFOV = fov
        self.res = res
        if not transform is None:
            self.transform = transform

    def generateRay(self,x,y):
        o = self.origin
        d = np.array([x*self.aspect*self.tanFOV,y*self.tanFOV,-1])
        d = normalize(self.transform.dot(d))
        return Ray(o,d)
    def generateRays(self,xy):
        d = np.zeros((xy.shape[0],xy.shape[1],3),dtype='float')
        for i in range(xy.shape[0]):
            for j in range(xy.shape[1]):
                x = xy[i,j][0]
                y = xy[i,j][1]
                ray = np.array([x*self.aspect*self.tanFOV,y*self.tanFOV,-1])
                ray = normalize(self.transform.dot(ray))
                d[i,j] = ray
        return d
    def getOrigin(self):
        o = self.origin
        return o

def computeNormal(x,sdf_val,decoder):
    return normalize([sdf(x+[eps_n,0,0],decoder)-sdf_val, sdf(x+[0,eps_n,0],decoder)-sdf_val,sdf(x+[0,0,eps_n],decoder)-sdf_val])

def shade(lightPos, inNormal, surfacePoint, diffuseColor, org, lightPower=4.0):
    normalInterp = inNormal
    vertPos = surfacePoint

    lightColor = np.array([1.0, 1.0, 1.0]);
    ambientColor = np.array([0.01, 0.01, 0.01]);
    # diffuseColor = np.array([0.6, 0.6, 0.6]);
    specColor = np.array([1.0, 1.0, 1.0]);
    shininess = 200.0;
    screenGamma = 2.2; # Assume the monitor is calibrated to the sRGB color space
    mode = 1

    lightDir = lightPos - vertPos;
    if np.linalg.norm(normalInterp) < 1e-7:
        normalInterp = lightDir

    normal = normalize(normalInterp);
    distance = np.linalg.norm(lightDir);
    distance = distance * distance;
    lightDir = normalize(lightDir);
    if np.dot(lightDir, normal) < 0:
        normal = -normal
    lambertian = np.max([np.dot(lightDir, normal), 0.0]);
    if lambertian < 1e-2:
        lambertian = 1.0
    specular = 0.0;

    if lambertian > 0.0:

        viewDir = normalize(org-vertPos);

        halfDir = normalize(lightDir + viewDir);
        specAngle = np.max([np.dot(halfDir, normal), 0.0]);
        specular = np.power(specAngle, shininess);

        # this is phong (for comparison)
        if mode == 2:
          reflectDir = -lightDir - 2*np.dot(-lightDir,normal)*normal;
          specAngle = np.max([np.dot(reflectDir, viewDir), 0.0]);
          # note that the exponent is different here
          specular = np.power(specAngle, shininess/4.0)
    colorLinear = ambientColor + (diffuseColor * lambertian * lightColor * lightPower / distance) #+ (specColor * specular * lightColor * lightPower / distance)
    # apply gamma correction (assume ambientColor, diffuseColor and specColor
    # have been linearized, i.e. have no gamma correction in them)
    if colorLinear[0] > 1.0:
        colorLinear[0] = 0.9
    if colorLinear[1] > 1.0:
        colorLinear[1] = 0.9
    if colorLinear[2] > 1.0:
        colorLinear[2] = 0.9
    colorGammaCorrected = np.power(colorLinear, (1.0 / screenGamma));
    # use the gamma corrected color in the fragment
    return colorGammaCorrected
def raymarch(xy,camera,decoder):
    lightPos = camera.getOrigin()
    lightDirection = normalize(np.array([0,0,0])-lightPos)
    iw = xy.shape[0]
    ih = xy.shape[1]
    img1 = np.zeros((iw,ih,4))
    img2 = np.zeros((iw,ih,4))
    img3 = np.zeros((iw,ih,4))
    rayorigin = camera.getOrigin()
    print("Generating Rays")
    rays = camera.generateRays(xy)
    print(rays.shape)
    x = np.zeros((xy.shape[0],xy.shape[1],3),dtype='float')
    x[:,:] = rayorigin
    #get init sdf
    print("Getting Initial SDF")
    t = 1.5*np.ones((iw,ih),dtype='float')
    #background color
    bg = np.array([0.2,0.2,0.2,0.0])
    surface = np.zeros((xy.shape[0],xy.shape[1],3))
    surfaceSDF = np.zeros((xy.shape[0],xy.shape[1]))
    flag = np.zeros((iw,ih))
    print("Ray Marching")
    for i in range(0,max_iter):
        logging.info("Step : " + str(i))
        T = np.zeros((iw,ih,3),dtype='float')
        T[:,:,0] = t
        T[:,:,1] = t
        T[:,:,2] = t
        x = x + 0.9*rays*T;
        t = sdf(x,decoder);
        for imgi in range(0,iw):
            for imgj in range(0,ih):
                if (t[imgi,imgj] < eps) and (not (flag[imgi,imgj] == 1)):
                    surface[imgi,imgj] = x[imgi,imgj]
                    surfaceSDF[imgi,imgj] = t[imgi,imgj]
                    flag[imgi,imgj] = 1
    normals, colors, ccolors = sdf(surface,decoder,computeNormals=True)
    normals = cv2.GaussianBlur(normals,(3,3),0)
    for imgi in range(0,iw):
        for imgj in range(0,ih):
            if flag[imgi,imgj] == 1:
                img1[imgi,imgj] = np.append(shade(lightPos, normals[imgi,imgj], surface[imgi,imgj],np.array([0.75,0.75,0.75],dtype='float'),camera.getOrigin()),1.0)
            else:
                img1[imgi,imgj] = bg
    for imgi in range(0,iw):
        for imgj in range(0,ih):
            if flag[imgi,imgj] == 1:
                img2[imgi,imgj] = np.append(colors[imgi,imgj],1.0)
            else:
                img2[imgi,imgj] = bg
    for imgi in range(0,iw):
        for imgj in range(0,ih):
            if flag[imgi,imgj] == 1:
                img3[imgi,imgj] = np.append(ccolors[imgi,imgj,:],1.0)
            else:
                img3[imgi,imgj] = bg
    return img1,img2, img3

def getCColors(xyz):
    N = 30
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    samples = samples.numpy()
    colors = samples*0
    colors = np.load('renderColorsFin.npy')
    colors = colors[None][0]
    searchTree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(samples)
    _, indices = searchTree.kneighbors(xyz)
    return torch.clamp(torch.tensor(colors[indices,:]),0,1).squeeze().numpy()

def sdf(x,decoder,computeNormals=False):

    length = x.shape[0]*x.shape[1]
    xyz = np.reshape(x,(length,x.shape[2])).astype('float')
    max_batch = int(decoder.max_batch)
    if computeNormals==False:
        pred_sdf = np.zeros((length,))
        head = 0
        with torch.no_grad():
            while head < length:
                sample_subset = torch.tensor(xyz[head : min(head + max_batch, length), 0:3],requires_grad=False,dtype=torch.float32).cuda()
                temp,_ = i3DMM.utils.decode_sdf(decoder.decoder_deform, decoder.decoder_ref, decoder.decoder_col, decoder.latent_vec_geom, decoder.latent_vec_color, sample_subset, onlySDFs=True)
                temp = torch.clamp(temp, -0.1, 0.1)
                pred_sdf[head : min(head + max_batch, length)] = temp.squeeze(1).detach().cpu().numpy()
                head += max_batch
        sdf = np.reshape(pred_sdf,(x.shape[0],x.shape[1]))
        check = np.linalg.norm(x,axis=2)-1
        for i in range(0,x.shape[0]):
            for j in range(0,x.shape[1]):
                if check[i,j] >=0.0:
                    sdf[i,j] = 0.1
        return sdf
    else:
        loss_l1 = torch.nn.L1Loss()
        normals = np.zeros((length,3))
        colors = np.zeros((length,3))
        ccolors = np.zeros((length,3))
        pred_sdf = np.zeros((length,))
        head = 0
        while head < length:
            sample_subset = torch.tensor(xyz[head : min(head + max_batch, length), 0:3],dtype=torch.float32,requires_grad=True).cuda()
            temp,temp_colors,deltas = i3DMM.utils.decode_sdf(decoder.decoder_deform, decoder.decoder_ref, decoder.decoder_col, decoder.latent_vec_geom, decoder.latent_vec_color, sample_subset)
            temp = torch.clamp(temp, -0.1, 0.1)
            pred_sdf[head : min(head + max_batch, length)] = temp.squeeze(1).detach().cpu().numpy()
            sample_subset.retain_grad()
            loss = temp.mean()
            loss.backward()
            normals[head : min(head + max_batch, length)] = sample_subset.grad.detach().cpu().numpy()
            colors[head : min(head + max_batch, length)] = temp_colors.detach().cpu().numpy()
            ccolors[head : min(head + max_batch, length)] = getCColors((deltas.detach().cpu().numpy() + xyz[head : min(head + max_batch, length), 0:3]))

            head += max_batch
        sdf = np.reshape(pred_sdf,(x.shape[0],x.shape[1]))
        normals = np.reshape(normals,(x.shape[0],x.shape[1],3))
        colors = np.reshape(colors,(x.shape[0],x.shape[1],3))
        ccolors = np.reshape(ccolors,(x.shape[0],x.shape[1],3))
        return normals,colors,ccolors
