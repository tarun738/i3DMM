# this code base has been adapted from deepSDF
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
import train_i3DMM as train
from image_utils import renderAndSaveImage

def reconstruct(
    decoderDeform,
    decoderRef,
    decoderCol,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latentGeom = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
        latentCol = torch.ones(1, int(latent_size*2/3)).normal_(mean=0, std=stat).cuda()
    else:
        latentGeom = torch.normal(stat[0].detach(), stat[1].detach()).cuda()
        latentCol = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latentGeom.requires_grad = True
    latentCol.requires_grad = True

    optimizerGeom = torch.optim.Adam([latentGeom], lr=lr)
    optimizerColor = torch.optim.Adam([latentCol], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss() # gives the mean of L1 differences

    for e in range(num_iterations):

        decoderDeform.eval()
        decoderRef.eval()
        decoderCol.eval()

        sdf_data = i3DMM.data.unpack_sdf_samples_from_ram(
            test_sdf, [], 0, num_samples).cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)
        color_gt = sdf_data[:,4:7]
        color_gt = color_gt + 0.5
        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizerGeom, e, decreased_by, adjust_lr_every)
        adjust_learning_rate(lr, optimizerColor, e, decreased_by, adjust_lr_every)

        optimizerGeom.zero_grad()
        optimizerColor.zero_grad()

        latentGeomInputs = latentGeom.expand(num_samples, -1)
        latentColInputs = latentCol.expand(num_samples, -1)
        inputsGeom = torch.cat([latentGeomInputs, xyz], 1).cuda()

        deltaXYZ = decoderDeform(inputsGeom)
        pred_sdf = decoderRef(xyz+deltaXYZ)
        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        inputsCol = torch.cat([latentColInputs, xyz+deltaXYZ], 1).cuda()
        pred_color = decoderCol(inputsCol)

        loss = (loss_l1(pred_sdf, sdf_gt) + 0.25*loss_l1(pred_color, color_gt))
        if l2reg:
            loss += 1e-4 * torch.mean(latentGeom.pow(2))
            loss += 1e-4 * torch.mean(latentCol.pow(2))
        loss += (1e-4)*torch.mean(torch.norm(deltaXYZ,dim=1))
        loss.backward()
        optimizerGeom.step()
        optimizerColor.step()

        if e % 50 == 0:
            logging.info("Epoch {}".format(e))
            logging.info(loss.item())
            logging.info(latentGeom.norm())
            logging.info(latentCol.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, latentGeom.detach(), latentCol.detach()

def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Fit to i3DMM model given preprocessed SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    arg_parser.add_argument(
        "--reconstruct_only_ref",
        "-r",
        dest="reconstruct_only_ref",
        default=0,
        help="Reconstruct only the reference shape.",
    )
    arg_parser.add_argument(
        "--reconstruct_full_model",
        "-f",
        dest="reconstruct_full_model",
        default=0,
        help="Reconstruct from full model.",
    )
    arg_parser.add_argument(
        "--numTextureTranfer",
        "--nTT",
        dest="numTextureTransfer",
        default=0,
        help="Number of texture transfers.",
    )
    arg_parser.add_argument(
        "--imageNMesh",
        "--imNM",
        dest="imageNMesh",
        default=False,
        help="Reconstruct images not meshes.",
    )
    arg_parser.add_argument(
        "--imgHeight",
        "--imH",
        dest="imgHeight",
        default=256,
        help="Rendered image height.",
    )
    arg_parser.add_argument(
        "--imgWidth",
        "--imW",
        dest="imgWidth",
        default=256,
        help="Rendered image height.",
    )
    arg_parser.add_argument(
        "--reconstructMean",
        "--RMean",
        dest="reconstructMean",
        default=False,
        help="Reconstruct the mean shape.",
    )
    arg_parser.add_argument(
        "--textureTransferFromTest",
        "--TTT",
        dest="textureTransferFromTest",
        default=False,
        help="Reconstruct the mean shape.",
    )
    arg_parser.add_argument(
        "--loadAllTestLatent",
        dest="loadAllTestLatent",
        default=False,
        help="Reconstruct the mean shape.",
    )
    arg_parser.add_argument(
        "--save360Video",
        dest="save360Video",
        default=False,
        help="Reconstruct the mean shape.",
    )
    rot = 0

    i3DMM.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    i3DMM.configure_logging(args)
    loadAllTestLatent = bool(args.loadAllTestLatent)
    logging.info("loadAllTestLatent {}".format(loadAllTestLatent))
    reconstructMean = bool(args.reconstructMean)
    imageNMesh = bool(args.imageNMesh)
    imgHeight = int(args.imgHeight)
    imgWidth = int(args.imgWidth)
    save360Video = bool(args.save360Video)
    textureTransferFromTest = bool(args.textureTransferFromTest)
    numTextureTransfer = int(args.numTextureTransfer)
    reconstruct_ref_only = int(args.reconstruct_only_ref)
    reconstruct_full_model = int(args.reconstruct_full_model)
    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var
    idListTest=[]
   # for exdir in ["heads"]: #os.listdir("./examples"):
    if 1:
        exp_directory = args.experiment_directory
        #exp_directory = os.path.join("./examples",exdir)

        specs_filename = os.path.join(exp_directory, "specs.json")
        if not os.path.isfile(specs_filename):
            raise Exception(
                'The experiment directory does not include specifications file "specs.json"'
            )

        specs = json.load(open(specs_filename))

        arch_deform = __import__("networks." + specs["NetworkArchDeform"], fromlist=["DeformNet"])
        arch_ref = __import__("networks." + specs["NetworkArchRef"], fromlist=["RefNet"])
        arch_col = __import__("networks." + specs["NetworkArchCol"], fromlist=["ColorDecoder"])

        condNoiseColDims = get_spec_with_default(specs, "colorNoiseConditioningDimensions", 0)
        condNoiseColStd  = get_spec_with_default(specs, "colorNoiseConditioningStdDev", 1e-4)
        colorExpressionSpace  = get_spec_with_default(specs, "haveColorExpressionSpace", False)
        reducedColorHairSpace = get_spec_with_default(specs, "reducedColorHairSpace", False)

        latent_size = specs["CodeLength"]

        decoder_deform = arch_deform.DeformNet(latent_size, **specs["NetworkSpecsDeform"])
        decoder_ref = arch_ref.RefNet(**specs["NetworkSpecsRef"])

        if colorExpressionSpace == True:
            decoder_col = arch_col.ColorDecoder(latent_size+condNoiseColDims, **specs["NetworkSpecsCol"])
        else:
            decoder_col = arch_col.ColorDecoder(int(latent_size*2/3)+condNoiseColDims, **specs["NetworkSpecsCol"])

        decoder_deform = torch.nn.DataParallel(decoder_deform)
        decoder_ref = torch.nn.DataParallel(decoder_ref)
        decoder_col = torch.nn.DataParallel(decoder_col)

        for name, param in decoder_deform.module.named_parameters():
            print(name)
        for name, param in decoder_ref.module.named_parameters():
            print(name)
        for name, param in decoder_col.module.named_parameters():
            print(name)


        saved_model_state_ref = torch.load(
            os.path.join(
                exp_directory, ws.model_params_subdir, args.checkpoint + "_ref.pth"
            )
        )
        decoder_ref.load_state_dict(saved_model_state_ref["model_state_dict"])
        if not reconstruct_ref_only == 1:
            saved_model_state_deform = torch.load(
                os.path.join(
                    exp_directory, ws.model_params_subdir, args.checkpoint + "_deform.pth"
                )
            )
            saved_model_state_col = torch.load(
                os.path.join(
                    exp_directory, ws.model_params_subdir, args.checkpoint + "_col.pth"
                )
            )
            saved_model_epoch = saved_model_state_col["epoch"]
            decoder_deform.load_state_dict(saved_model_state_deform["model_state_dict"])
            decoder_col.load_state_dict(saved_model_state_col["model_state_dict"])
        else:
            saved_model_epoch = saved_model_state_ref["epoch"]
            saved_model_state_col = torch.load(
                os.path.join(
                    exp_directory, ws.model_params_subdir, args.checkpoint + "_col.pth"
                )
            )
            decoder_col.load_state_dict(saved_model_state_col["model_state_dict"])


        decoder_deform = decoder_deform.module.cuda()
        decoder_ref = decoder_ref.module.cuda()
        decoder_col = decoder_col.module.cuda()

        with open(args.split_filename, "r") as f:
            split = json.load(f)

        npz_filenames = i3DMM.data.get_instance_filenames(args.data_source, split)


        train_split_file = specs["TrainSplit"]
        with open(train_split_file, "r") as f:
            trainingSplit = json.load(f)
        training_npz_filenames = i3DMM.data.get_instance_filenames(args.data_source, trainingSplit)
        _, IDsList, _, _, latentMap = train.defineLatentSpace(training_npz_filenames)
        _, _, _, latentMapColor = train.defineLatentSpaceColor(training_npz_filenames)

        # random.shuffle(npz_filenames)

        logging.debug(decoder_deform)
        logging.debug(decoder_ref)
        logging.debug(decoder_col)

        err_sum = 0.0
        repeat = 1
        save_latvec_only = False
        rerun = 0

        reconstruction_dir = os.path.join(
            exp_directory, ws.reconstructions_subdir, str(saved_model_epoch)
        )

        if not os.path.isdir(reconstruction_dir):
            os.makedirs(reconstruction_dir)

        if imageNMesh == True:
            if not save360Video:
                reconstruction_meshes_dir = os.path.join(
                    reconstruction_dir, ws.reconstruction_images_subdir
                )
            else:
                reconstruction_meshes_dir = os.path.join(
                    reconstruction_dir, "vid360"
                )

        else:
            reconstruction_meshes_dir = os.path.join(
            reconstruction_dir, ws.reconstruction_meshes_subdir
        )
        if not os.path.isdir(reconstruction_meshes_dir):
            os.makedirs(reconstruction_meshes_dir)
        reconstruction_codes_dir = os.path.join(
            reconstruction_dir, ws.reconstruction_codes_subdir
        )
        if not os.path.isdir(reconstruction_codes_dir):
            os.makedirs(reconstruction_codes_dir)

        reconstructedLatentGeom = torch.empty(1,int(latent_size))
        reconstructedLatentCol = torch.empty(1,int(latent_size*2/3))

        if loadAllTestLatent == True:
            for ii, npz in enumerate(npz_filenames):
                npzFile = os.path.basename(npz)
                logging.info("Current file: {}".format(npzFile))
                latent_filename_col = os.path.join(
                    reconstruction_codes_dir, npzFile[:-4] + "_col.pth")
                latent_filename_geom = os.path.join(
                    reconstruction_codes_dir, npzFile[:-4] + "_geom.pth"
                )
                if not npzFile[:9] in idListTest:
                    idListTest.append(npzFile[:9])
                    if ii == 0:
                        reconstructedLatentGeom =  torch.load(latent_filename_geom).detach().squeeze(0)
                        reconstructedLatentCol = torch.load(latent_filename_col).detach().squeeze(0)
                    else:
                        reconstructedLatentGeom = torch.cat([reconstructedLatentGeom.cuda(),torch.load(latent_filename_geom).detach().squeeze(0)],dim=0)
                        reconstructedLatentCol = torch.cat([reconstructedLatentCol.cuda(),torch.load(latent_filename_col).detach().squeeze(0)],dim=0)
                    logging.info("shape of geom shape {}".format(reconstructedLatentGeom.shape))
                    logging.info("shape of col shape {}".format(reconstructedLatentCol.shape))
        for ii, npz in enumerate(npz_filenames):

            if "npz" not in npz:
                continue

            full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)

            logging.debug("loading {}".format(npz))
            print(npz)
            npzFile = os.path.basename(npz)
            print(npzFile)
            data_sdf = i3DMM.data.read_sdf_samples_into_ram(full_filename)

            for k in range(repeat):

                if rerun > 1:
                    mesh_filename = os.path.join(
                        reconstruction_meshes_dir, npzFile[:-4] + "-" + str(k + rerun)
                    )
                    latent_filename = os.path.join(
                        reconstruction_codes_dir, npzFile[:-4] + "-" + str(k + rerun) + ".pth"
                    )
                else:
                    mesh_filename = os.path.join(reconstruction_meshes_dir, npzFile[:-4])
                    latent_filename_col = os.path.join(
                        reconstruction_codes_dir, npzFile[:-4] + "_col.pth")
                    latent_filename_geom = os.path.join(
                        reconstruction_codes_dir, npzFile[:-4] + "_geom.pth"
                    )

                if (
                    args.skip
                    and os.path.isfile(mesh_filename + ".ply")
                    and os.path.isfile(latent_filename_geom)
                    and os.path.isfile(latent_filename_col)
                ):
                    continue


                logging.info("Meshfile name {}".format(mesh_filename))
                refMeshFile = os.path.join(mesh_filename[:-9]+"mean")
                if not os.path.exists(os.path.dirname(refMeshFile)):
                    os.makedirs(os.path.dirname(refMeshFile))
                if (reconstructMean or reconstruct_ref_only == 1) and not os.path.isfile(refMeshFile+".ply"):
                    logging.info("Saving mean mesh")
                    start = time.time()
                    latent_mean_geom = torch.ones(1, int(latent_size)).normal_(mean=0, std=0.1).cuda()
                    latent_mean_geom.requires_grad = True
                    latentLoadedColor = ws.load_latent_vectors_color(args.experiment_directory,args.checkpoint);

                    latent_mean_col = torch.cat([torch.ones(1, int(latent_size*2/3)).normal_(mean=0, std=0.1).cuda()])
                    latent_mean_col = torch.cat([latentLoadedColor[0,:],latentLoadedColor[-1,:]])
                    with torch.no_grad():
                        i3DMM.mesh.create_mesh(
                            decoder_deform, decoder_ref, decoder_col, latent_mean_geom, latent_mean_col, [], 0, [], refMeshFile, True, N=256, max_batch=int(2 ** 19)
                        )
                    logging.debug("total time: {}".format(time.time() - start))

                logging.info("reconstructing {}".format(npz))

                data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
                data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]
                data_sdf[2] = data_sdf[2][torch.randperm(data_sdf[2].shape[0])]
                data_sdf[3] = data_sdf[3][torch.randperm(data_sdf[3].shape[0])]


                if not reconstruct_ref_only == 1:
                    
                    logging.info("npz {}".format(npz))
                    logging.info("npz idx {}".format(npz_filenames.index(npz)))

                    if os.path.isfile(latent_filename_geom) and os.path.isfile(latent_filename_col):
                        latent_geom = torch.load(latent_filename_geom).detach().squeeze(0)
                        latent_col = torch.load(latent_filename_col).detach().squeeze(0)
                        loadedLatent = 2
                    else:
                        start = time.time()
                        err = 0;
                        err, latent_geom, latent_col = reconstruct(
                          decoder_deform,
                          decoder_ref,
                          decoder_col,
                          int(args.iterations),
                          latent_size,
                          data_sdf,
                          0.01,  # [emp_mean,emp_var],
                          0.1,
                          num_samples=16384,
                          lr=5e-3,
                          l2reg=True,
                         )
                        logging.info("reconstruct time: {}".format(time.time() - start))
                        err_sum += err
                        logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
                        logging.debug(ii)
                        loadedLatent = 0
                    ############################ Getting texture transfer indices################################################################
                    textureTransferLatents = []

                    ttIdxs = []
                    ttIdxNames = []

                    if textureTransferFromTest == True:
                        if (len(idListTest) > numTextureTransfer) or loadAllTestLatent:
                            for idxs in range(0,numTextureTransfer):
                                tempRand = torch.randint(0,len(idListTest),(1,))
                                temp = tempRand.item()
                                while (idListTest[temp][:5] == npzFile[:5]) or (idListTest[temp] in ttIdxNames):
                                    tempRand = torch.randint(0,len(idListTest),(1,))
                                    temp = tempRand.item()
                                ttIdxs.append(temp)
                                ttIdxNames.append(idListTest[temp])
                            latentColorFrom = reconstructedLatentCol
                        for idx in range(0,len(ttIdxs)):
                            textureTransferLatents.append(latentColorFrom[ttIdxs[idx],:])

                    ###############################################################################################################################

                    decoder_deform.eval()
                    decoder_ref.eval()
                    decoder_col.eval()

                    if not os.path.exists(os.path.dirname(mesh_filename)):
                        os.makedirs(os.path.dirname(mesh_filename))
                    if not save_latvec_only :
                        start = time.time()
                        if imageNMesh == True:
                            renderAndSaveImage([reconstruction_meshes_dir, npzFile[:-4]], imgWidth, imgHeight, decoder_deform, decoder_ref, decoder_col, latent_geom, latent_col, rot=rot, skip=False, vid=save360Video)
                            for idx in range(0,len(ttIdxs)):
                                renderAndSaveImage([reconstruction_meshes_dir, npzFile[:-4]+"_TT_"+ttIdxNames[idx]], imgWidth, imgHeight, decoder_deform, decoder_ref, decoder_col, latent_geom, textureTransferLatents[idx], rot=rot, skip=False, vid=save360Video)
                        else:
                            with torch.no_grad():
                                i3DMM.mesh.create_mesh(
                                    decoder_deform, decoder_ref, decoder_col, latent_geom, latent_col, textureTransferLatents, len(ttIdxs), ttIdxNames, mesh_filename, False, N=256, max_batch=int(2 ** 19)
                                )
                        logging.debug("total time: {}".format(time.time() - start))

                    if not os.path.exists(os.path.dirname(latent_filename_col)):
                        os.makedirs(os.path.dirname(latent_filename_col))
                    if not os.path.isfile(latent_filename_geom) and loadedLatent == 0:
                        torch.save(latent_geom.unsqueeze(0), latent_filename_geom)
                    if not os.path.isfile(latent_filename_col) and loadedLatent == 0:
                        torch.save(latent_col.unsqueeze(0), latent_filename_col)
                    if textureTransferFromTest == True:
                        if not npzFile[:9] in idListTest:
                            idListTest.append(npzFile[:9])
                            if ii == 0:
                                reconstructedLatentGeom = latent_geom
                                reconstructedLatentCol = latent_col
                            else:
                                reconstructedLatentGeom = torch.cat([reconstructedLatentGeom.cuda(),latent_geom],dim=0)
                                reconstructedLatentCol = torch.cat([reconstructedLatentCol.cuda(),latent_col],dim=0)
                            logging.info("shape of geom shape {}".format(reconstructedLatentGeom.shape))
                            logging.info("shape of col shape {}".format(reconstructedLatentCol.shape))
