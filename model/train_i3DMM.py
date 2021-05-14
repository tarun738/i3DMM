#adapted from deepSDF code base

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time
import numpy as np
import i3DMM
import i3DMM.workspace as ws

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length

def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules
# assign a combination {identity, expression, hairstyle} of geometry latent
# codes for each mesh in the dataset
def defineLatentSpace(filesList):
    identities = []
    expressions = []
    hairstyles = []
    for i in range(len(filesList)):
        # get filename
        file = filesList[i]
        #strip the file extension
        fileName = os.path.basename(file)[:-4]
        #we name each file as <5 char identity code>_<exx> where e01 - e10 are
        # different expressions, and e11-e13 are hairstyles in neutral expression.
        # e07 is the neutral expression in short hairstyle
        temp = fileName.split('_')
        identity = temp[0]
        tempHolder = temp[1]
        if int(tempHolder[1:]) >= 11:
            expression = "e07"
        else:
            expression = tempHolder
        # captured hairstyle typess
        if int(tempHolder[1:]) == 13:
            hairstyle = "long"
        elif int(tempHolder[1:]) == 11:
            hairstyle = "cap1"
        elif int(tempHolder[1:]) == 12:
            hairstyle = "cap2"
        else:
            hairstyle = "short"

        if identity not in identities:
            identities.append(identity)
        if expression not in expressions:
            expressions.append(expression)
        if hairstyle not in hairstyles:
            hairstyles.append(hairstyle)

    identities = sorted(identities)
    expressions = sorted(expressions)
    hairstyles = sorted(hairstyles)

    #create a tuple of identity, expression, hairstyle indices for a given index
    # in dataset fileslists
    mapIdExp = torch.zeros([len(filesList), 3], dtype=torch.int64)
    for i in range(len(filesList)):
        file = filesList[i]
        fileName = os.path.basename(file)[:-4]
        temp = fileName.split('_')
        identity = temp[0]
        tempHolder = temp[1]

        if int(tempHolder[1:]) >= 11:
            expression = "e07"
        else:
            expression = tempHolder

        if int(tempHolder[1:]) == 13:
            hairstyle = "long"
        elif int(tempHolder[1:]) == 11:
            hairstyle = "cap1"
        elif int(tempHolder[1:]) == 12:
            hairstyle = "cap2"
        else:
            hairstyle = "short"

        #get indices
        idx_id = identities.index(identity)
        idx_exp = expressions.index(expression)
        idx_hairstyle = hairstyles.index(hairstyle)
        #save the indices for a given file
        mapIdExp[i][0] = idx_id
        mapIdExp[i][1] = idx_exp+len(identities)
        mapIdExp[i][2] = idx_hairstyle+len(identities)+len(expressions)
    return len(identities)+len(expressions)+len(hairstyles), identities, expressions, hairstyles, mapIdExp

#same as geometry but no expression spaces
def defineLatentSpaceColor(filesList):
    identities = []
    expressions = []
    hairstyles = []
    for i in range(len(filesList)):
        file = filesList[i]
        fileName = os.path.basename(file)[:-4]
        temp = fileName.split('_')
        identity = temp[0]

        tempHolder = temp[1]

        if int(tempHolder[1:]) == 11:
            hairstyle = "cap1"
        elif int(tempHolder[1:]) == 12:
            hairstyle = "cap2"
        else:
            # assigning both long and tied hairs to "short" color
            # as hair color doesn't change when tied, we checked :D
            hairstyle = "short"

        if identity not in identities:
            identities.append(identity)
        if hairstyle not in hairstyles:
            hairstyles.append(hairstyle)

    identities = sorted(identities)
    hairstyles = sorted(hairstyles)

    #no expression space
    mapIdExp = torch.zeros([len(filesList), 2], dtype=torch.int64)
    for i in range(len(filesList)):
        file = filesList[i]
        fileName = os.path.basename(file)[:-4]
        temp = fileName.split('_')
        identity = temp[0]
        tempHolder = temp[1]

        if int(tempHolder[1:]) == 11:
            hairstyle = "cap1"
        elif int(tempHolder[1:]) == 12:
            hairstyle = "cap2"
        else:
            hairstyle = "short"

        idx_id = identities.index(identity)
        idx_hairstyle = hairstyles.index(hairstyle)

        mapIdExp[i][0] = idx_id
        mapIdExp[i][1] = idx_hairstyle+len(identities)

    return len(identities)+len(hairstyles), identities, hairstyles, mapIdExp

def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename,     lat_vecs, ref_col = False):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        if not ref_col == True:
            lat_vecs.load_state_dict(data["latent_codes"])
        else:
            print(data["latent_codes"])
            lat_vecs.weight.data[0, :].copy_(data["latent_codes"]['weight'][0, :])
            lat_vecs.weight.data[-1, :].copy_(data["latent_codes"]['weight'][-1, :])

    return data["epoch"]

def save_logs(
    experiment_directory,
    loss_geom,
    loss_text,
    timing_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss_geom": loss_geom,
            "loss_text": loss_text,
            "timing": timing_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    print(full_filename)
    data = torch.load(full_filename)

    return (
        data["loss_geom"],
        data["loss_text"],
        data["timing"],
        data["epoch"]
    )


def clip_logs(loss_geom, loss_text, timing_log, epoch):

    iters_per_epoch = len(loss_log) // len(loss_geom)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    loss_geom = loss_geom[: (iters_per_epoch * epoch)]
    loss_text = loss_text[: (iters_per_epoch * epoch)]
    timing_log = timing_log[:epoch]

    return (loss_geom, loss_text, timing_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())



def main_function(experiment_directory, continue_from, batch_split, train_color, train_geometry, train_reference_only, train_with_reference_from, ref_experiment_directory):

    logging.debug("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + str(specs["Description"]))

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    ref_split_file = specs["RefSplit"]

    arch_deform = __import__("networks." + specs["NetworkArchDeform"], fromlist=["DeformNet"])
    arch_ref = __import__("networks." + specs["NetworkArchRef"], fromlist=["RefNet"])
    arch_col = __import__("networks." + specs["NetworkArchCol"], fromlist=["ColorDecoder"])

    logging.debug(specs["NetworkSpecsDeform"])
    logging.debug(specs["NetworkSpecsRef"])
    logging.debug(specs["NetworkSpecsCol"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch, train_color, train_deform, train_reference):
        if train_deform == 1:
            save_model(experiment_directory, "latest_deform.pth", decoder_deform, epoch)
            save_optimizer(experiment_directory, "latest_deform.pth", optimizer_deform, epoch)
            save_latent_vectors(experiment_directory, "latest_geom.pth", lat_vecs_geom, epoch)
        if train_color == 1:
            save_model(experiment_directory, "latest_col.pth", decoder_col, epoch)
            save_optimizer(experiment_directory, "latest_col.pth", optimizer_col, epoch)
            save_latent_vectors(experiment_directory, "latest_col.pth", lat_vecs_col, epoch)
        if train_reference == 1:
            save_model(experiment_directory, "latest_ref.pth", decoder_ref, epoch)
            save_optimizer(experiment_directory, "latest_ref.pth", optimizer_ref, epoch)

    def save_checkpoints(epoch, train_color, train_deformation, train_reference):

        if train_color == 1:
            save_model(experiment_directory, str(epoch) + "_col.pth", decoder_col, epoch)
            save_optimizer(experiment_directory, str(epoch) + "_col.pth", optimizer_col, epoch)
            save_latent_vectors(experiment_directory, str(epoch) + "_col.pth", lat_vecs_col, epoch)
        if train_deformation == 1:
            save_model(experiment_directory, str(epoch) + "_deform.pth", decoder_deform, epoch)
            save_optimizer(experiment_directory, str(epoch) + "_deform.pth", optimizer_deform, epoch)
            save_latent_vectors(experiment_directory, str(epoch) + "_geom.pth", lat_vecs_geom, epoch)
        if train_reference == 1:
            save_model(experiment_directory, str(epoch) + "_ref.pth", decoder_ref, epoch)
            save_optimizer(experiment_directory, str(epoch) + "_ref.pth", optimizer_ref, epoch)


    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    #default options for codes
    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)
    cent_supv_reg = get_spec_with_default(specs, "centroidSupervisionRegularizer", 0)
    xyzmap_reg_lambda = get_spec_with_default(specs, "XYZMapRegularizationLambda", 1e-4)
    latentXYZGradReg = get_spec_with_default(specs, "latentXYZGradientRegularizer", 0)
    referenceShapeEpoch = get_spec_with_default(specs, "referenceShapeEpoch", 101)
    colorLossWeight = get_spec_with_default(specs, "colorLossWeight", 1)
    uniformSamplingBatches = get_spec_with_default(specs, "uniformSampling", False)
    code_bound = get_spec_with_default(specs, "CodeBound", None)
    newWeightCentSupVis = get_spec_with_default(specs, "newWeightingCentSupVision", False)
    latentXYZGradEntireDeform = get_spec_with_default(specs, "latentXYZGradOnEntireDeform", False)
    addEarCorrespondences = get_spec_with_default(specs, "addEarCorrespondences", False)
    faceLandmarksFile = get_spec_with_default(specs, "faceLandmarksFile", 'eightCentroids.npy')
    earLandmarksFile = get_spec_with_default(specs, "earLandmarksFile", 'gtEarCentroids.npy')

    #we use latent_size/3 for each spaces
    decoder_deform = arch_deform.DeformNet(latent_size, **specs["NetworkSpecsDeform"]).cuda()
    decoder_ref = arch_ref.RefNet(**specs["NetworkSpecsRef"]).cuda()

    #since no expression space for color, 2/3 *latent size
    decoder_col = arch_col.ColorDecoder(int(latent_size*2/3), **specs["NetworkSpecsCol"]).cuda()

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    decoder_deform = torch.nn.DataParallel(decoder_deform)
    decoder_ref = torch.nn.DataParallel(decoder_ref)
    decoder_col = torch.nn.DataParallel(decoder_col)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    # with open(test_split_file, "r") as f:
    #     test_split = json.load(f)
    if train_reference_only == 1:
        with open(ref_split_file, "r") as f:
            train_split = json.load(f)

    # load centroids
    if cent_supv_reg == 0.0:
        num_centroids = 0
    else:
        if addEarCorrespondences:
            num_centroids = 16
        else:
            num_centroids = 8
    # 8 facial landmarks file
    # these files must contain a dict ex: { filename in dataset : [[x,y,z],[],[]]}
    datasetCentroids = np.load(faceLandmarksFile,allow_pickle=True)
    datasetCentroids = datasetCentroids[None][0]
    if addEarCorrespondences == True:
        # 8 landmarks on both ears
        addCentroids = np.load(earLandmarksFile,allow_pickle=True)
        addCentroids = addCentroids[None][0]
        for key in datasetCentroids:
           if key in addCentroids:
               datasetCentroids[key] = np.append(datasetCentroids[key],addCentroids[key],0)

    sdf_dataset = i3DMM.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, datasetCentroids, num_centroids, load_ram=True, uniformSamplingBatches=uniformSamplingBatches,
    )
    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )
    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    if addEarCorrespondences:
        num_centroids = 16
        #create two points for each scan, the lazy way
        earCentroids = np.zeros((num_scenes,2,3))
        for i in range(0,num_scenes):
            file = sdf_dataset.npyfiles[i]
            id = os.path.basename(file)
            id = id[:-4]
            earCentroids[i,0,:] = np.mean(datasetCentroids[id][8:12,:])
            earCentroids[i,1,:] = np.mean(datasetCentroids[id][12:16,:])
        earCentroids = torch.FloatTensor(earCentroids).cuda()
        earCentroids.requires_grad=True
    # for geometry
    num_latentcodes, identities, expressions, hairstyles, mapIdExp = defineLatentSpace(sdf_dataset.npyfiles)

    # when long hair, we can't see ears, we checked this as well :D
    # this is to collapse the ear centroids to one point
    hairStyleCheck = torch.zeros((num_scenes,))
    for i in range(0,hairStyleCheck.shape[0]):
        hairstyle = mapIdExp[i][2]
        expression = mapIdExp[i][1]
        identity = mapIdExp[i][0]

        if hairstyles[hairstyle-len(identities)-len(expressions)] == "long":
            hairStyleCheck[i] = 1
    print(num_latentcodes)
    print(identities)
    print(expressions)
    print(hairstyles)
    for i in range(0,len(sdf_dataset)):
        hairstyle = mapIdExp[i][2]
        expression = mapIdExp[i][1]
        identity = mapIdExp[i][0]
        print(sdf_dataset.npyfiles[i] + " " + identities[identity] + " " + expressions[expression-len(identities)] + " " + hairstyles[hairstyle-len(identities)-len(expressions)])

    # for color
    num_latentcodes_col, identities_col, hairstyles_col, mapIdExpCol = defineLatentSpaceColor(sdf_dataset.npyfiles)
    # mapIdExp = torch.IntTensor(mapIdExp)
    print(num_latentcodes_col)
    print(identities_col)
    print(hairstyles_col)
    for i in range(0,len(sdf_dataset)):
        hairstyle = mapIdExpCol[i][1]
        identity = mapIdExpCol[i][0]
        print(sdf_dataset.npyfiles[i] + " " + identities_col[identity] + " " + hairstyles_col[hairstyle-len(identities_col)])
    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder_deform)
    logging.debug(decoder_ref)
    logging.debug(decoder_col)

    # create latent vectors of size int(latent_size/3)
    lat_vecs_geom = torch.nn.Embedding(num_latentcodes, int(latent_size/3), max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs_geom.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )
    lat_vecs_col = torch.nn.Embedding(num_latentcodes_col, int(latent_size/3), max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs_col.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    logging.debug(
        "initialized geometric latent codes with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs_geom)
        )
    )
    logging.debug(
        "initialized color latent codes with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs_col)
        )
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    if addEarCorrespondences:
        optimizer_deform = torch.optim.Adam(
            [
                {
                    "params": decoder_deform.parameters(),
                    "lr": lr_schedules[0].get_learning_rate(0),
                },
                {
                    "params": lat_vecs_geom.parameters(),
                    "lr": lr_schedules[1].get_learning_rate(0),
                },
                {
                    "params": earCentroids,
                    "lr": lr_schedules[1].get_learning_rate(0),
                },
            ]
        )
    else:
        optimizer_deform = torch.optim.Adam(
            [
                {
                    "params": decoder_deform.parameters(),
                    "lr": lr_schedules[0].get_learning_rate(0),
                },
                {
                    "params": lat_vecs_geom.parameters(),
                    "lr": lr_schedules[1].get_learning_rate(0),
                },
            ]
        )

    if train_reference_only:
        optimizer_ref = torch.optim.Adam(
            [
                {
                    "params": decoder_ref.parameters(),
                    "lr": lr_schedules[0].get_learning_rate(0),
                },
            ]
        )
    else:
        optimizer_ref = torch.optim.Adam(
            [
                {
                    "params": decoder_ref.parameters(),
                    "lr": lr_schedules[2].get_learning_rate(0),
                },
            ]
        )

    optimizer_col = torch.optim.Adam(
        [
            {
                "params": decoder_col.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs_col.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    loss_log = []
    loss_log_geom = []
    loss_log_text = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}
    start_epoch = 1


    if train_with_reference_from is not None:
        model_epoch_ref = ws.load_model_parameters(
            ref_experiment_directory, train_with_reference_from, decoder_ref, False, False, True
        )
        if os.path.isfile(os.path.join(ref_experiment_directory, "ModelParameters", train_with_reference_from + "_col.pth")):
            logging.info("Loading color of the reference shape")
            lat_epoch_col = load_latent_vectors(
                ref_experiment_directory, train_with_reference_from + "_col.pth", lat_vecs_col, ref_col=True
            )
            model_epoch_col = ws.load_model_parameters(
                ref_experiment_directory, train_with_reference_from, decoder_col, True, False, False
            )

#different conditions to train only for color or only for train_geometry or only
# the reference shape

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        if train_geometry == 1:
            lat_epoch_geom = load_latent_vectors(
                experiment_directory, continue_from + "_geom.pth", lat_vecs_geom
            )
            model_epoch_deform = ws.load_model_parameters(
                experiment_directory, continue_from, decoder_deform, False, True, False
            )
            optimizer_epoch_deform = load_optimizer(
                experiment_directory, continue_from + "_deform.pth", optimizer_deform
            )
            model_epoch_ref = ws.load_model_parameters(
                experiment_directory, continue_from, decoder_ref, False, False, True
            )
            optimizer_epoch_geom = load_optimizer(
                experiment_directory, continue_from + "_ref.pth", optimizer_ref
            )
            if not (model_epoch_deform == optimizer_epoch_deform and model_epoch_deform == lat_epoch_geom):
                raise RuntimeError(
                    "epoch mismatch geometry: {} vs {} vs {}".format(
                        model_epoch_deform, optimizer_epoch_deform, lat_epoch_geom
                    )
                )
            start_epoch = model_epoch_deform + 1

        if train_color == 1:
            lat_epoch_col = load_latent_vectors(
                experiment_directory, continue_from + "_col.pth", lat_vecs_col
            )

            model_epoch_col = ws.load_model_parameters(
                experiment_directory, continue_from, decoder_col, True, False, False
            )

            optimizer_epoch_col = load_optimizer(
                experiment_directory, continue_from + "_col.pth", optimizer_col
            )

            loss_log_geom, loss_log_text, timing_log, log_epoch = load_logs(
                experiment_directory
            )
            if not (model_epoch_col == optimizer_epoch_col and model_epoch_col == lat_epoch_col):
                raise RuntimeError(
                    "epoch mismatch geometry: {} vs {} vs {} vs {}".format(
                        model_epoch_col, optimizer_epoch_col, lat_epoch_col, log_epoch
                    )
                )

            start_epoch = model_epoch_col + 1
        logging.debug("loaded")
    else:
        start_epoch = 1

    if train_color == 1 and train_geometry == 0 and not train_reference_only == 1:
        continue_from = "latest"
        logging.info('Loading geometry from latest')

        lat_epoch_geom = load_latent_vectors(
            experiment_directory, continue_from + "_geom.pth", lat_vecs_geom
        )
        model_epoch_deform = ws.load_model_parameters(
            experiment_directory, continue_from, decoder_deform, False, True, False
        )
        optimizer_epoch_deform = load_optimizer(
            experiment_directory, continue_from + "_deform.pth", optimizer_deform
        )
        model_epoch_ref = ws.load_model_parameters(
            experiment_directory, continue_from, decoder_ref, False, False, True
        )
        optimizer_epoch_geom = load_optimizer(
            experiment_directory, continue_from + "_ref.pth", optimizer_ref
        )
        loss_log_geom, loss_log_text, timing_log, log_epoch = load_logs(
            experiment_directory
        )

        if not (model_epoch_deform == optimizer_epoch_deform and model_epoch_deform == lat_epoch_geom):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch_deform, optimizer_epoch_deform, lat_epoch_geom, log_epoch
                )
            )

        logging.debug("loaded geometry")
    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of deromation network parameters: {}".format(
            sum(p.data.nelement() for p in decoder_deform.parameters())
        )
    )
    logging.info(
        "Number of reference network parameters: {}".format(
            sum(p.data.nelement() for p in decoder_ref.parameters())
        )
    )
    logging.info(
        "Number of color network parameters: {}".format(
            sum(p.data.nelement() for p in decoder_col.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs_geom.num_embeddings * lat_vecs_geom.embedding_dim,
            lat_vecs_geom.num_embeddings,
            lat_vecs_geom.embedding_dim,
        )
    )
    logging.info(
        "Number of color code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs_col.num_embeddings * lat_vecs_col.embedding_dim,
            lat_vecs_col.num_embeddings,
            lat_vecs_col.embedding_dim,
        )
    )
    #helper functions
    def reshapeForCentroids(inputStructure, num_samp_per_scene, num_centroids):
        if num_centroids == 0:
            return inputStructure, 1
        input_dim1 = inputStructure.shape[0]
        input_dim2 = inputStructure.shape[1]
        input_perSceneDim = inputStructure.reshape(int(input_dim1 / num_samp_per_scene), num_samp_per_scene,
                                                    input_dim2)
        input_realData = input_perSceneDim[:, :-num_centroids, :]
        input_centroids = input_perSceneDim[:, -num_centroids:, :]
        centDim = int(input_dim1 / num_samp_per_scene) * num_centroids
        input_reshaped = torch.cat(
            [input_realData.reshape(int(input_dim1 / num_samp_per_scene) * (num_samp_per_scene - num_centroids),
                                     input_dim2),
             input_centroids.reshape(centDim, input_dim2)])
        return input_reshaped, centDim
    def reshapeForCentroidsEars(inputStructure, num_samp_per_scene, num_centroids,sidx,earCentroids, longHairIdxs):
        if num_centroids == 0:
            return inputStructure, 1
        input_dim1 = inputStructure.shape[0]
        input_dim2 = inputStructure.shape[1]
        input_perSceneDim = inputStructure.reshape(int(input_dim1 / num_samp_per_scene), num_samp_per_scene,
                                                    input_dim2)
        input_realData = input_perSceneDim[:, :-num_centroids, :]
        input_centroids = input_perSceneDim[:, -num_centroids:, :]
        print(input_centroids.shape)
        print(earCentroids.shape)
        for i in range(0,len(sidx)):
            if longHairIdxs[sidx[i]] == 1:
                input_centroids[i,-8:-4,-3:] = earCentroids[sidx[i],0,:].repeat(4,1)
                input_centroids[i,-4:,-3:] = earCentroids[sidx[i],1,:].repeat(4,1)
        centDim = int(input_dim1 / num_samp_per_scene) * num_centroids
        input_reshaped = torch.cat(
            [input_realData.reshape(int(input_dim1 / num_samp_per_scene) * (num_samp_per_scene - num_centroids),
                                     input_dim2),
             input_centroids.reshape(centDim, input_dim2)])
        return input_reshaped, centDim

    for epoch in range(start_epoch, num_epochs + 1):
        if epoch >= referenceShapeEpoch:
            referenceSetToEval = True
            logging.info('Fixing reference shape')
        else:
            referenceSetToEval = False
        start = time.time()

        logging.info("epoch {}...".format(epoch))

        if train_color == 1:
            decoder_col.train()
            adjust_learning_rate(lr_schedules, optimizer_col, epoch)
        else:
            decoder_col.eval()

        if train_geometry == 1:
            decoder_deform.train()
            adjust_learning_rate(lr_schedules, optimizer_deform, epoch)
            if not referenceSetToEval == True:
                decoder_ref.train()
                adjust_learning_rate(lr_schedules, optimizer_ref, epoch)
            else:
                decoder_ref.eval()
        else:
            decoder_deform.eval()
            decoder_ref.eval()
        if train_reference_only == 1:
            decoder_ref.train()
            adjust_learning_rate(lr_schedules, optimizer_ref, epoch)

        for sdf_data, indices in sdf_loader:
            # Process the input data
            sdf_data = sdf_data.reshape(-1, 7).float()

            num_sdf_samples = sdf_data.shape[0]
            logging.info("Training samples: {}".format(num_sdf_samples))

            sdf_data.requires_grad = False

            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3].unsqueeze(1)
            color_gt = 0.5 + sdf_data[:, 4:7].squeeze()
            #expect the color to be from 0-1

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            xyz = torch.chunk(xyz, batch_split)
            sceneIDX = torch.chunk(indices, batch_split)
            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )

            sdf_gt = torch.chunk(sdf_gt, batch_split)
            color_gt = torch.chunk(color_gt, batch_split)
            batch_loss = 0.0
            batch_loss_geom = 0.0
            batch_loss_text = 0.0


            if train_color == 1:
                optimizer_col.zero_grad()
            if train_geometry == 1:
                optimizer_deform.zero_grad()
                optimizer_ref.zero_grad()
            if train_reference_only == 1:
                optimizer_ref.zero_grad()
            for i in range(batch_split):

                # geometry forward pass
                batch_vecs_geom = torch.cat([lat_vecs_geom(mapIdExp[indices[i],0]),lat_vecs_geom(mapIdExp[indices[i],1]),lat_vecs_geom(mapIdExp[indices[i],2])], dim=1)
                input_geom = torch.cat([batch_vecs_geom, xyz[i]], dim=1)


                input_geom_dim1 = input_geom.shape[0]
                #add centroids to the inputs for each mesh in the batch
                if addEarCorrespondences:
                    input_geom_reshaped,centDim = reshapeForCentroidsEars(input_geom,num_samp_per_scene,num_centroids,sceneIDX[i],earCentroids,hairStyleCheck)
                else:
                    input_geom_reshaped,centDim = reshapeForCentroids(input_geom,num_samp_per_scene,num_centroids)
                latent_xyz = decoder_deform(input_geom_reshaped)

                if not train_reference_only == 1:
                    pred_sdf = decoder_ref(latent_xyz+input_geom_reshaped[:, -3:].cuda())
                    if enforce_minmax:
                        pred_sdf = torch.clamp(pred_sdf, minT, maxT)
                else:
                    pred_sdf = decoder_ref(input_geom_reshaped[:, -3:].cuda())
                    if enforce_minmax:
                        pred_sdf = torch.clamp(pred_sdf, minT, maxT)
                    #since we add centroids to the end of xyzs of each batch,
                    # we need to get the right sdf supervision
                    sdf_gt_supervision,_ = reshapeForCentroids(sdf_gt[i].unsqueeze(1),num_samp_per_scene,num_centroids)
                    chunk_loss_reference = loss_l1(pred_sdf.squeeze().cuda(), sdf_gt_supervision.squeeze().cuda())/ num_sdf_samples
                    logging.info("reference_training_loss = {}".format(chunk_loss_reference.item()))
                    chunk_loss_reference.backward()

                # color forward pass
                batch_vecs_col = torch.cat([lat_vecs_col(mapIdExpCol[indices[i],0]),lat_vecs_col(mapIdExpCol[indices[i],1])], dim=1)

                input_col = torch.cat([batch_vecs_col.cuda(), xyz[i].cuda()], dim=1)

                #remove centroids from color loss
                input_col_reshaped,_ = reshapeForCentroids(input_col,num_samp_per_scene,num_centroids)

                # deformation
                if not train_reference_only == 1:
                    input_col_reshaped[:,-3:] = input_col_reshaped[:,-3:] + latent_xyz

                if train_color == 1:
                    pred_color = decoder_col(input_col_reshaped)
                    color_supervision,centDim = reshapeForCentroids(color_gt[i],num_samp_per_scene,num_centroids)
                    if num_centroids == 0:
                        chunk_loss_col = colorLossWeight * loss_l1(pred_color.squeeze().cuda(), color_supervision.squeeze().cuda()) / num_sdf_samples
                    else:
                        chunk_loss_col = colorLossWeight * loss_l1(pred_color[:-centDim, :].squeeze().cuda(),
                                                  color_supervision[:-centDim, :].squeeze().cuda()) / num_sdf_samples
                if train_geometry == 1:
                    sdf_gt_supervision,_ = reshapeForCentroids(sdf_gt[i].unsqueeze(1),num_samp_per_scene,num_centroids)
                    chunk_loss_geom = (loss_l1(pred_sdf.squeeze().cuda(), sdf_gt_supervision.squeeze().cuda()) + xyzmap_reg_lambda*torch.sum(torch.norm(latent_xyz,dim=1))) / num_sdf_samples

                    if not latentXYZGradReg == 0.0:
                        logging.info("Regularizing deformations with {} weight".format(latentXYZGradReg))
                        if latentXYZGradEntireDeform == True:
                            latent_gradients = torch.autograd.grad(outputs=(input_col_reshaped[:,-3:] + latent_xyz), inputs=input_geom, grad_outputs=torch.ones(latent_xyz.size()).cuda(), retain_graph=True, only_inputs=True)[0]
                        else:
                            latent_gradients = torch.autograd.grad(outputs=(latent_xyz), inputs=input_geom, grad_outputs=torch.ones(latent_xyz.size()).cuda(), retain_graph=True, only_inputs=True)[0]
                        chunk_loss_geom += latentXYZGradReg * torch.sum(
                            torch.norm(latent_gradients[:, -3:].squeeze().cuda(), dim=1)) / num_sdf_samples
                    else:
                        logging.info("No delta XYZ regularizer")
                    if not cent_supv_reg == 0.0:
                        indicesPairwiseLoss = latent_xyz[-centDim:, :] + input_geom_reshaped[-centDim:, -3:].cuda()
                        indicesPairwiseLoss = indicesPairwiseLoss.reshape(int(input_geom_dim1 / num_samp_per_scene), num_centroids, 3)
                        if newWeightCentSupVis == True:
                            for indI in range(0, num_centroids):
                                chunk_loss_geom += cent_supv_reg * torch.mean(
                                    torch.nn.functional.pdist(indicesPairwiseLoss[:, indI, :].squeeze().cuda(), p=2))/num_centroids
                        else:
                            for indI in range(0, num_centroids):
                                chunk_loss_geom += cent_supv_reg * torch.sum(
                                    torch.nn.functional.pdist(indicesPairwiseLoss[:, indI, :].squeeze().cuda(), p=2)) / (
                                                  num_sdf_samples)
                    else:
                        logging.info("No centroid supervision")



                if do_code_regularization:
                    if train_geometry == 1:
                        l2_size_loss_geom = torch.sum(torch.norm(batch_vecs_geom, dim=1))
                        reg_loss_geom = (
                                                code_reg_lambda * min(1, epoch / 100) * l2_size_loss_geom
                                        ) / num_sdf_samples
                        chunk_loss_geom = chunk_loss_geom + reg_loss_geom.cuda()

                    if train_color == 1:
                        l2_size_loss_col = torch.sum(torch.norm(batch_vecs_col, dim=1))
                        reg_loss_col = (
                                               code_reg_lambda * min(1, epoch / 100) * l2_size_loss_col
                                       ) / num_sdf_samples
                        chunk_loss_col = chunk_loss_col + reg_loss_col.cuda()

                if train_geometry == 1 and train_color == 1:
                    chunk_loss = chunk_loss_geom + chunk_loss_col
                    chunk_loss.backward()
                    logging.info("col_loss = {}".format(chunk_loss_col.item()))
                    logging.info("geom_loss = {}".format(chunk_loss_geom.item()))
                # geometry backpropagation
                elif train_geometry == 1:
                    chunk_loss_geom.backward()
                    logging.info("geom_loss = {}".format(chunk_loss_geom.item()))
                # color backpropagation
                elif train_color == 1:
                    chunk_loss_col.backward()
                    logging.info("col_loss = {}".format(chunk_loss_col.item()))

                if train_geometry == 1:
                    batch_loss_geom += chunk_loss_geom.detach()
                    logging.info("Batch_geom_loss = {}".format(batch_loss_geom))
                    loss_log_geom.append(batch_loss_geom)
                if train_color == 1:
                    batch_loss_text += chunk_loss_col.detach()
                    logging.info("Batch_col_loss = {}".format(batch_loss_text))
                    loss_log_text.append(batch_loss_text)

            if train_geometry == 1:
                optimizer_deform.step()
                if referenceSetToEval == False:
                    optimizer_ref.step()
            if train_color == 1:
                optimizer_col.step()
            if train_reference_only:
                optimizer_ref.step()


        end = time.time()

        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)
        logging.info("Time taken for this epoch: {}".format(seconds_elapsed))

        if epoch in checkpoints:
            save_checkpoints(epoch, train_color, train_geometry, train_geometry or train_reference_only)

        if epoch % log_frequency == 0:
            save_latest(epoch, train_color, train_geometry, train_geometry or train_reference_only)
            save_logs(
                experiment_directory,
                loss_log_geom,
                loss_log_text,
                timing_log,
                epoch,
            )

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train the i3DMM model")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )
    arg_parser.add_argument(
        "--train_color",
        "-t",
        dest="train_color",
        default=0,
        help="Setting this flag will only train color using the deformations from the latest geometric epoch",
    )
    arg_parser.add_argument(
        "--train_geometry",
        "-g",
        dest="train_geometry",
        default=1,
        help="Setting this flag will only train geometry",
    )
    arg_parser.add_argument(
        "--train_reference",
        "-r",
        dest="train_reference",
        default=0,
        help="Setting this flag will only train geometry",
    )

    arg_parser.add_argument(
        "--train_with_reference_from",
        "--cref",
        dest="train_with_reference_from",
        default=None,
        help="Setting this flag will only train geometry",
    )
    arg_parser.add_argument(
        "--ref_experiment_directory",
        "--refDir",
        dest="ref_experiment_directory",
        default=None,
        help="Get Reference experiment directory",
    )

    i3DMM.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    i3DMM.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split), int(args.train_color), int(args.train_geometry), int(args.train_reference), args.train_with_reference_from, args.ref_experiment_directory)
