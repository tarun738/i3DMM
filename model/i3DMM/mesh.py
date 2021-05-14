import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import numpy as np
import i3DMM.utils


def create_mesh(
    decoder_deform, decoder_ref, decoder_col, latent_vec_geom, latent_vec_color, latent_vec_colortt, numTextureTransfer, ttIdxNames, filename, meanMesh, N=256, max_batch=32 ** 3, offset=None, scale=None, correspondencesDebug=True, refMeshDebug=True
):
    start = time.time()
    ply_filename = filename

    decoder_deform.eval()
    decoder_ref.eval()
    decoder_col.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0
    with torch.no_grad():
        while head < num_samples:
            sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

            temp,_ = i3DMM.utils.decode_sdf(decoder_deform, decoder_ref, decoder_col, latent_vec_geom, latent_vec_color, sample_subset,meanMesh,onlySDFs=True)
            samples[head : min(head + max_batch, num_samples), 3] = temp.squeeze(1).detach().cpu()
            head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    logging.info("sampling takes: %f" % (end - start))

    start = time.time()
    numpy_3d_sdf_tensor = sdf_values.data.cpu().numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )
    end = time.time()
    logging.info("Marching Cubes takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        verts, faces, normals, values,
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        decoder_deform,
        decoder_ref,
        decoder_col,
        latent_vec_geom,
        latent_vec_color,
        offset,
        scale,
        correspondencesDebug=correspondencesDebug,
        refMeshDebug=refMeshDebug,
        max_batch=max_batch,
    )
    for idx in range(0,numTextureTransfer):
        convert_sdf_samples_to_ply(
            verts, faces, normals, values,
            voxel_origin,
            voxel_size,
            ply_filename + "_TT_" + ttIdxNames[idx] + ".ply",
            decoder_deform,
            decoder_ref,
            decoder_col,
            latent_vec_geom,
            latent_vec_colortt[idx],
            offset,
            scale,
            correspondencesDebug=False,
            refMeshDebug=False,
            max_batch=max_batch,
        )

def getColor(x):
    # assuming each dimension scales from -1 to 1
    # red for x, green for y, blue for z
    # red = np.array([255,0,0])
    # blue = np.array([0,255,0])
    # red = np.array([255,0,0])
    x = (x+1)/2 # shift x to lie from 0 to 1
    color = np.array([int(x[0] * 255) , int(x[1] * 255) , int(x[2] * 255)]);
    return color

def writeCorrespondencesMesh(num_verts,num_faces,mesh_points,faces,sdf_color,deltas,ply_filename_out):
    correspondence_debug_verts = np.zeros((num_verts * 3,),
                                          dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")] + [('red', 'u1'),
                                                                                           ('green', 'u1'),
                                                                                           ('blue', 'u1')])

    correspondence_debug_faces_building = []
    for i in range(0, num_faces):
        correspondence_debug_faces_building.append(((faces[i,:].tolist(),)))
    for i in range(0, num_faces):
        tempFaces = num_verts+2*faces[i,:]
        correspondence_debug_faces_building.append(((tempFaces.tolist(),)))
    idx = 0
    R = np.array([[0,0,1],[0,1,0],[-1,0,0]])
    for i in range(0, num_verts):
        col = np.array([int(sdf_color[i, 0]), int(sdf_color[i, 1]), int(sdf_color[i, 2])])
        temp = mesh_points[i,:].dot(R)
        curr_vert = temp + deltas[i, :].dot(R) + np.array([-1, 0, 0])
        correspondence_debug_verts[i]  = tuple(np.append(temp, col))
        correspondence_debug_verts[idx + num_verts] = tuple(np.append(curr_vert, col))
        correspondence_debug_verts[idx + 1 + num_verts] = tuple(np.append(curr_vert, col))
        factor = int(num_verts/3000)
        if i%factor == 0:
            correspondence_debug_faces_building.append((([i, idx + num_verts, idx + 1 + num_verts],)))
        idx = idx + 2
    correspondence_debug_faces = np.array(correspondence_debug_faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(correspondence_debug_verts, "vertex")
    el_faces = plyfile.PlyElement.describe(correspondence_debug_faces, "face")
    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.info("saving mesh to %s" % (ply_filename_out))
    correspondence_debug_ply_filename_out = ply_filename_out[:-4] + "_correspondences" + ".ply"
    ply_data.write(correspondence_debug_ply_filename_out)


def writeMesh(num_verts,num_faces,mesh_points,faces_building,sdf_color,ply_filename_out):
    new_verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]+[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i in range(0,num_verts):
        col = np.array([max(min(255,int(sdf_color[i,0])),0),max(min(255,int(sdf_color[i,1])),0),max(min(255,int(sdf_color[i,2])),0)])
        new_verts_tuple[i] = tuple(np.append(mesh_points[i, :],col))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])
    el_verts = plyfile.PlyElement.describe(new_verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)


def convert_sdf_samples_to_ply(
    verts, faces, normals, values,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    decoder_deform,
    decoder_ref,
    decoder_col,
    latent_vec_geom,
    latent_vec_color,
    offset=None,
    scale=None,
    correspondencesDebug=True,
    refMeshDebug=True,
    max_batch=int(2**19),
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    if ply_filename_out[-8:-4] == "mean":
        meanMesh = True
    else:
        meanMesh = False

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    with torch.no_grad():
        head = 0
        num_samples = mesh_points.shape[0]
        sdf_color = torch.zeros((num_samples,3),dtype=torch.float)
        sdf_color.requires_grad = False
        deltas = torch.zeros((num_samples,3),dtype=torch.float)
        sdf_color.requires_grad = False
        while head < num_samples:
            sample_subset = torch.tensor(mesh_points[head : min(head + max_batch, num_samples), 0:3],dtype=torch.float).cuda()
            _, temp_sdf_color, tempDeltas = i3DMM.utils.decode_sdf(decoder_deform, decoder_ref, decoder_col, latent_vec_geom, latent_vec_color, sample_subset,meanMesh)
            sdf_color[head : min(head + max_batch, num_samples), :] = temp_sdf_color.squeeze(1).detach().cpu()
            deltas[head : min(head + max_batch, num_samples), :] = tempDeltas.squeeze(1).detach().cpu()
            head += max_batch
    sdf_color = sdf_color*255
    deltas = deltas.detach().cpu().numpy()

    if correspondencesDebug == True and meanMesh == False:
        writeCorrespondencesMesh(num_verts, num_faces, mesh_points, faces, sdf_color, deltas, ply_filename_out)



    logging.info("saving mesh to %s" % (ply_filename_out))

    writeMesh(num_verts,num_faces,mesh_points,faces_building,sdf_color,ply_filename_out)


    if refMeshDebug == True and meanMesh == False:
        ply_filename_out = ply_filename_out[:-4] + "_ref" + ".ply"
        logging.info("saving reference mesh to %s" % (ply_filename_out))
        writeMesh(num_verts,num_faces,mesh_points+deltas,faces_building,sdf_color,ply_filename_out)
