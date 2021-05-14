import numpy as np
import cv2
def obj_read(mesh,texture_normal=None):
    obj_file = open(mesh,'r')
    vertices = []
    vertex_colors = []
    vertex_color_uv = []
    faces = []
    vertex_text_map = []
    new_vertex_color_uv = []
    vertex_colors_normal = []
    vertex_no = 0;
    vertex_text_no = 0;
    faces_no = 0;
    for line in obj_file:
        if line[0] == "#" or line == "":
            continue;
        subline = line.split()
        if subline == []:
            continue;
        if subline[0] == "v":
            vertices.append([float(subline[1]), float(subline[2]), float(subline[3])])
            if len(subline)>4:
                vertex_colors.append([float(subline[4]), float(subline[5]), float(subline[6])])
            vertex_no = vertex_no+1
        elif subline[0] == "vt":
            vertex_color_uv.append([float(subline[1]), float(subline[2])])
            vertex_text_no = vertex_text_no + 1
        elif subline[0] == "f":
            sub1 = subline[1].split('/')
            sub2 = subline[2].split('/')
            sub3 = subline[3].split('/')
            faces.append([int(float(sub1[0]))-1,int(float(sub2[0]))-1,int(float(sub3[0]))-1])
            if len(sub1) > 1:
                if not sub1[1] == "": 
                    if len(sub1)>1:
                        vertex_text_map.append([int(sub1[0])-1,int(sub1[1])-1])
                    if len(sub2)>1:
                        vertex_text_map.append([int(sub2[0])-1,int(sub2[1])-1])
                    if len(sub3)>1:
                        vertex_text_map.append([int(sub3[0])-1,int(sub3[1])-1])
            faces_no = faces_no + 1
    obj_file.close()
    vertices = np.array(vertices)
    faces = np.array(faces)
    vertex_colors = np.array(vertex_colors)
    if len(vertex_color_uv)>0:
        vertex_color_uv = np.array(vertex_color_uv)
        new_vertex_color_uv = np.zeros((vertex_color_uv.shape[0],vertex_color_uv.shape[1]))
    if len(vertex_text_map)>0:
        vertex_text_map = np.array(vertex_text_map)
        for i in range(vertex_text_map.shape[0]):
            new_vertex_color_uv[vertex_text_map[i,0],:] = vertex_color_uv[vertex_text_map[i,1],:]
    if texture_normal is not None:
        vertex_colors_normal = fetch_colors(cv2.imread(texture_normal),new_vertex_color_uv)
        vertex_colors_normal = vertex_colors_normal[0:vertices.shape[0],:]
    return vertices, vertex_colors, faces, new_vertex_color_uv, vertex_colors_normal
def read_all_obj(mesh,texture_normal=None):
    obj_file = open(mesh,'r')
    vertices = []
    vertex_colors = []
    vertex_color_uv = []
    vertex_normal = []
    faces = []
    facesText = []
    facesNorm = []
    otherLines = []
    vertex_no = 0;
    vertex_norm_no = 0;
    vertex_text_no = 0;
    faces_no = 0;
    for line in obj_file:
        if not len(line) == 1:
            subline = line.split()
        else:
            subline = line
            continue
        if subline[0] == "v":
            vertices.append([float(subline[1]), float(subline[2]), float(subline[3])])
            if len(subline)>4:
                vertex_colors.append([float(subline[4]), float(subline[5]), float(subline[6])])
            vertex_no = vertex_no+1
        elif subline[0] == "vn":
            vertex_normal.append([float(subline[1]), float(subline[2]), float(subline[3])])
            vertex_norm_no = vertex_norm_no + 1
        elif subline[0] == "vt":
            vertex_color_uv.append([float(subline[1]), float(subline[2])])
            vertex_text_no = vertex_text_no + 1
        elif subline[0] == "f":
            sub1 = subline[1].split('/')
            sub2 = subline[2].split('/')
            sub3 = subline[3].split('/')
            faces.append([int(sub1[0])-1,int(sub2[0])-1,int(sub3[0])-1])
            facesText.append([int(sub1[1])-1,int(sub2[1])-1,int(sub3[1])-1])
            facesNorm.append([int(sub1[2])-1,int(sub2[2])-1,int(sub3[2])-1])
            faces_no = faces_no + 1
        else:
            otherLines.append(line)
    obj_file.close()
    vertices = np.array(vertices)
    vertex_colors = np.array(vertex_colors)
    vertex_normal = np.array(vertex_normal)
    vertex_color_uv = np.array(vertex_color_uv)
    faces = np.array(faces)
    facesText = np.array(facesText)
    facesNorm = np.array(facesNorm)
    
    
    if vertex_color_uv.shape[0] == 0:
        print(vertex_color_uv.shape)
        raise Exception("Error Reading Obj")
    new_vertex_color_uv = vertex_color_uv
    
    return vertices, vertex_color_uv, vertex_normal, faces, facesText, facesNorm, otherLines
def obj_write(filename, vertices, uvs=None, normals=None, faces=None, facesText=None, facesNorm=None,  otherLines=None):
    meshfile = open(filename,'w+')
    if otherLines is not None:
        for i in range(len(otherLines)):
            writestr = otherLines[i]
            writestr = writestr + "\n"
            meshfile.write(writestr)
    for i in range(vertices.shape[0]):
        writestr = "v"
        for j in range(vertices.shape[1]):
            writestr = writestr + " " + str(vertices[i,j])
        writestr = writestr + "\n"
        meshfile.write(writestr)
    if normals is not None:
        for i in range(normals.shape[0]):
            writestr = "vn"
            for j in range(normals.shape[1]):
                writestr = writestr + " " + str(normals[i,j])
            writestr = writestr + "\n"
            meshfile.write(writestr)
    if uvs is not None:
        for i in range(uvs.shape[0]):
            writestr = "vt"
            for j in range(uvs.shape[1]):
                writestr = writestr + " " + str(uvs[i,j])
            writestr = writestr + "\n"
            meshfile.write(writestr)
    if faces is not None:
        for i in range(faces.shape[0]):
            writestr = "f"
            flag = 0
            for j in range(faces.shape[1]):
                if faces[i,j] == -1:
                    flag = 1
                if (facesText is not None) and (facesNorm is not None):
                    writestr = writestr + " " + str(faces[i,j] + 1)+ "/" + str(facesText[i,j] + 1) + "/" + str(facesNorm[i,j] + 1)
                else:
                    writestr = writestr + " " + str(faces[i,j] + 1)
            writestr = writestr + "\n"
            if flag == 0:
                meshfile.write(writestr)
    meshfile.close()
def fetch_colors(texture, uvs):
    rows = texture.shape[0]
    cols = texture.shape[1]
    #get the colors of each vertices
    # uv starts from bottom row first column
    # rows is y/v and columns is x/u
    u = np.clip(np.round_(rows*(uvs[:,0])).astype('int'),0,rows-1)
    v = np.clip(np.round_(cols*(1-uvs[:,1])).astype('int'),0,cols-1)
    # if u >= cols:
        # print("Bad U")
        # u = cols-1
    # if v >= rows:
        # print("Bad V")
        # v = rows-1
    colors = np.flip(texture[v,u],axis=1)
    # rows = texture.shape[0]
    # cols = texture.shape[1]
    # colors = np.zeros((uvs.shape[0],3))
    # #get the colors of each vertices
    # for i in range(uvs.shape[0]):
        # # uv starts from bottom row first column
        # # rows is y/v and columns is x/u
        # u = int(rows*(uvs[i,0]))
        # v = int(cols*(1-uvs[i,1]))
        # if u >= cols:
            # print("Bad U")
            # u = cols-1
        # if v >= rows:
            # print("Bad V")
            # v = rows-1
        # colors[i,:] = np.flip(texture[v,u])
    return colors