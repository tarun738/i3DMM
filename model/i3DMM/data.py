import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

import i3DMM.workspace as ws


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        return NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"]).float()
    neg_tensor = torch.from_numpy(npz["neg"]).float()
    if "pospoi" in npz and "negpoi" in npz:
        pospoi_tensor = torch.from_numpy(npz["pospoi"]).float()
        negpoi_tensor = torch.from_numpy(npz["negpoi"]).float()
        return [pos_tensor, neg_tensor, pospoi_tensor, negpoi_tensor]
    else:
        return [pos_tensor, neg_tensor]

    # pos_tensor = torch.cat([pos_tensor, pospoi_tensor],0)
    # neg_tensor = torch.cat([neg_tensor, negpoi_tensor],0)


def unpack_sdf_samples(filename, centroids, num_centroids, subsample=None, uniformSampling=False):
    npz = np.load(filename)
    print(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"])).float()
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"])).float()
    pospoi_tensor = remove_nans(torch.from_numpy(npz["pospoi"])).float()
    negpoi_tensor = remove_nans(torch.from_numpy(npz["negpoi"])).float()

    # split the sample into half
    half = int(subsample / 2)
    fourth = int(half/4)
    threeFourth = int(3*half/4)
    if uniformSampling == False:
        #### Remove 4 points from the 3/4 to accomodate the centroids
        random_pos = (torch.rand(fourth) * pos_tensor.shape[0]).long()
        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        random_neg = (torch.rand(fourth) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)


        random_pospoi = (torch.rand(threeFourth) * pospoi_tensor.shape[0]).long()
        sample_pospoi = torch.index_select(pospoi_tensor, 0, random_pospoi[:-4])
        random_negpoi = (torch.rand(threeFourth) * negpoi_tensor.shape[0]).long()
        sample_negpoi = torch.index_select(negpoi_tensor, 0, random_negpoi[:-4])
        sample_pos = torch.cat([sample_pos, sample_pospoi],0)
        sample_neg = torch.cat([sample_neg, sample_negpoi],0)

    else:
        posSamples_tensor = torch.cat([pos_tensor,pospoi_tensor],0)
        random_pos = (torch.rand(half) * posSamples_tensor.shape[0]).long()
        sample_pos = torch.index_select(posSamples_tensor, 0, random_pos[:-4])


        negSamples_tensor = torch.cat([neg_tensor,negpoi_tensor],0)
        random_neg = (torch.rand(half) * negSamples_tensor.shape[0]).long()
        sample_neg = torch.index_select(negSamples_tensor, 0, random_neg[:-4])

    temp_centrd = torch.cat([centroids,torch.zeros([8,4])])
    samples = torch.cat([sample_pos, sample_neg, temp_centrd], 0)

    return samples


def unpack_sdf_samples_from_ram_completion(data, subsample=None, test=False):
    if subsample is None:
        return data
    pospoi_tensor = data[2]
    negpoi_tensor = data[3]

    # split the sample into half
    half = int(subsample / 2)

    random_pospoi = (torch.rand(half) * pospoi_tensor.shape[0]).long()
    sample_pospoi = torch.index_select(pospoi_tensor, 0, random_pospoi)
    random_negpoi = (torch.rand(half) * negpoi_tensor.shape[0]).long()
    sample_negpoi = torch.index_select(negpoi_tensor, 0, random_negpoi)

    samples = torch.cat([sample_pospoi, sample_negpoi], 0)
    return samples.float()

def unpack_sdf_samples_from_ram(data, centroids, num_centroids, subsample=None, test=False, uniformSampling=False):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]
    if len(data)>2:
        pospoi_tensor = data[2]
        negpoi_tensor = data[3]
    # split the sample into half
    half = int(subsample / 2)
    if test == False:
        fourth = int(half/4)
        threeFourth = int(3*half/4)
    else:
        fourth = int(half/2)
        threeFourth = int(half/2)
    if uniformSampling == False:
        #### Remove 4 points from the 3/4 to accomodate the centroids
        random_pos = (torch.rand(fourth) * pos_tensor.shape[0]).long()
        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        random_neg = (torch.rand(fourth) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

        random_pospoi = (torch.rand(threeFourth) * pospoi_tensor.shape[0]).long()
        random_negpoi = (torch.rand(threeFourth) * negpoi_tensor.shape[0]).long()
        if num_centroids == 0:
            sample_pospoi = torch.index_select(pospoi_tensor, 0, random_pospoi)
            sample_negpoi = torch.index_select(negpoi_tensor, 0, random_negpoi)
        else:
            sample_pospoi = torch.index_select(pospoi_tensor, 0, random_pospoi[:-int(num_centroids/2)])
            sample_negpoi = torch.index_select(negpoi_tensor, 0, random_negpoi[:-int(num_centroids/2)])


        sample_pos = torch.cat([sample_pos, sample_pospoi],0)
        sample_neg = torch.cat([sample_neg, sample_negpoi],0)
    else:

        if len(data)>2:
            posSamples_tensor = torch.cat([pos_tensor,pospoi_tensor],0)
            negSamples_tensor = torch.cat([neg_tensor,negpoi_tensor],0)
        else:
            posSamples_tensor = pos_tensor
            negSamples_tensor = neg_tensor

        random_pos = (torch.rand(half) * posSamples_tensor.shape[0]).long()

        random_neg = (torch.rand(half) * negSamples_tensor.shape[0]).long()
        if num_centroids == 0:
            sample_pos = torch.index_select(posSamples_tensor, 0, random_pos)
            sample_neg = torch.index_select(negSamples_tensor, 0, random_neg)
        else:
            sample_pos = torch.index_select(posSamples_tensor, 0, random_pos[:-int(num_centroids/2)])
            sample_neg = torch.index_select(negSamples_tensor, 0, random_neg[:-int(num_centroids/2)])

    if num_centroids == 0:
        samples = torch.cat([sample_pos, sample_neg], 0)
    else:
        temp_centrd = torch.cat([centroids,torch.zeros([num_centroids,4])],1)
        samples = torch.cat([sample_pos, sample_neg, temp_centrd], 0)
    return samples.float()


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        centroids,
        num_centroids,
        load_ram=False,
        uniformSamplingBatches=False,
        print_filename=False,
        num_files=1000000,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram
        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                pospoi_tensor = remove_nans(torch.from_numpy(npz["pospoi"]))
                negpoi_tensor = remove_nans(torch.from_numpy(npz["negpoi"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                        pospoi_tensor[torch.randperm(pospoi_tensor.shape[0])],
                        negpoi_tensor[torch.randperm(negpoi_tensor.shape[0])],
                    ]
                )
        self.centroids = centroids;
        self.num_centroids = num_centroids
        self.uniformSamplingBatches = uniformSamplingBatches

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        _, fileId = os.path.split(self.npyfiles[idx])
        fileId = fileId[:-4]
        centroids = torch.tensor(self.centroids[fileId],dtype=torch.float)
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], centroids, self.num_centroids, self.subsample, uniformSampling=self.uniformSamplingBatches),
                idx
            )
        else:
            return unpack_sdf_samples(filename, centroids, self.num_centroids, self.subsample, uniformSampling=self.uniformSamplingBatches), idx
