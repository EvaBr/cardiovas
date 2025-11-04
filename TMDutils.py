import numpy as np 
import cc3d
import nrrd
import nibabel as nib
import time
from skimage.morphology import skeletonize, medial_axis, thin, cube, ball, disk, octahedron, binary_dilation
from skimage.segmentation import watershed
from pathlib import Path
from skimage.filters import frangi, sato
import scipy.ndimage as ndi
from skimage.segmentation import expand_labels


def read_img(imgpath):
    """
    Reads an image as a nifti file from a given path, supporting both NRRD and NIfTI formats.
    Fixes also affine and header for later saving to nifti. 
    """
    imgpath = str(imgpath)
    if imgpath.endswith('.nrrd'):
        img, header = nrrd.read(imgpath)
        nifti_im = nib.Nifti1Image(img, None)  # create a NIfTI image with identity affine
        # Update the NIfTI header with necessary information
        tmp = np.eye(4)
        if 'space directions' in header:
            nifti_im.header.set_zooms(np.diag(header['space directions']))
            tmp[:-1,:-1] = header['space directions']
        if 'space origin' in header:
            tmp[:-1,-1] = header['space origin']
        tmp[:2,:]*= -1  # flip x and y axes
        nifti_im.header.set_sform(tmp)
        nifti_im.header.set_data_dtype(img.dtype)
        return nifti_im
    
    elif imgpath.endswith('.nii') or imgpath.endswith('.nii.gz'):
        img = nib.load(imgpath)
        return img
    else:
        raise ValueError("Unsupported image format.")



#get all branching points and leaves
def get_bifur_and_leaves(skel):
    leaves, bifur = [], []
    skeleton = np.nonzero(skel)
    for x,y,z in zip(*skeleton):
            num_ngbhs = skel[x-1:x+2, y-1:y+2, z-1:z+2].sum()
            if num_ngbhs==2:
                leaves.append((x,y,z))
            elif num_ngbhs>3:
                bifur.append((x,y,z))

    cluster_tmp = np.zeros_like(skel, dtype=np.uint8)
    cluster_tmp[*np.stack(leaves).T] = 1
    cluster_tmp[*np.stack(bifur).T] = 1

    clusters, Ncl = cc3d.connected_components(cluster_tmp, connectivity=26, return_N=True)
   # print("found", Ncl, "potential bifur clusters")
    cluster_stats = cc3d.statistics(clusters)

    #we'll need this for cleanup, to see how they connect:
    allclusters = cc3d.connected_components(skel*1-cluster_tmp, connectivity=26)
    #shift by nr of indeces in clusters, to make sure they dont get swapped
    allclusters = np.where(allclusters, allclusters+Ncl, 0)
    edges = cc3d.region_graph(allclusters+clusters, connectivity=26)
    #for each connected component, find if it is a bifur or leaf. 
    bifur_points = []
    not_bifur_leaf = []
    leaf_points = []
    
    for i in range(1, Ncl+1):
        connected_via = [c for c in np.unique(np.array([e for e in edges if ((e[0]==i) or (e[1]==i))])) if c!=i]
        #sanity check: ne bi smel bit povezan z drugimi clusterji - ker pol bi mogli bit v eni in isti CC
        assert all([cv>Ncl for cv in connected_via]), f"Cluster {i} connects clusters.. {connected_via}"
        #CHANGED BECAUSE:
        #need to clean up here: could be that a voxel neighboring a bifur is a leaf; 
        # but it should be deleted and the bifur should remain a bifur (ie not casted to leaf).
        # This can happen eg if theres a bump in the vessel that causes skeletonization to 
        # include a single voxel in that direction... 
        #so do cleanup: if the leaf holds to bifur, It can be deleted, and you only consider 
        # the bifur: check if it connects more than two things. If not, it shouldnt be
        #neither bifur nor leaf. If it connects to only one thing, it should eb a leaf. 
        #if it connects to more than two things, it should be a bifurcation point.
        if len(connected_via)==1:
            #this should be a root.. or sth is veeery weird with the skeleton. If it end up 
            # connecting (via this edge) to two other bifur/edges, then it should neither 
            # be an edge nor a bifur, just leave it. 
            # Otherwise it's a root. 
            #cleanup: you see it borders on one edge only. But what does that edge connect to?
            #if it connects the root to multiple other nodes/clusters, then this CANT be a root, 
            # but rather a degeneration of skeleton, and should be removed. 
            in_edges_bo = np.array([e for e in edges if ((e[0]==connected_via[0]) or (e[1]==connected_via[0]))])
            in_edges_bo = np.unique(in_edges_bo)
            if len(in_edges_bo) > 3: #i should be in this list by construction, And only 1 other one. 
                not_bifur_leaf.append(i) #remove it, its sth weird
            else: #ok, only connected to i and one other one.
                leaf_points.append(i)

        elif len(connected_via)==2:
            # If it connects to two edges, it was again some issue of the skeleton.
            # leave it, it's neither a bifur nor a leaf. 
            # note that the two edges it connects are actually the same edge, for further processing.
            not_bifur_leaf.append(i) #this is sth weird also.. does it need to be thinned further?
        
        else: #len(borders_on)>2 since 0 cant happen by default
            # If it connects to more than two edges, it should be a bifurcation point.
            bifur_points.append(i)

    #check that really all are accounted for:
    assert set(leaf_points+not_bifur_leaf+bifur_points) == set(np.arange(1,Ncl+1)), "some nodes missing: \n "+str(set(leaf_points+not_bifur_leaf+bifur_points)) + "\n vs \n"+ str(set(np.arange(1,Ncl+1)))

    for nei in not_bifur_leaf:
        clusters[clusters==nei] = 0  # just remove it

    #now for the further analyses we will need the actual (x,y,z) coordinates of 
    #pixels belonging to each bifur/leaf. Get them.
    bifur_comp_xyz = {bp: np.argwhere(clusters==bp) for bp in bifur_points}
    leaf_comp_xyz = {lp: np.argwhere(clusters==lp) for lp in leaf_points}
    return bifur_comp_xyz, leaf_comp_xyz, clusters, Ncl #np.where(allclusters,allclusters+Ncl,clusters)




def transform_dict(edges, comp_to_cluster):
    new_edges = {}
    for edge in edges:
        if edge[0] in comp_to_cluster: #its a bifur/leaf point
            assert edge[1] not in comp_to_cluster, (edge, comp_to_cluster) #sanity check that you dont have cluster-cluster edge
            if edge[1] in new_edges:
                new_edges[edge[1]].append(edge[0])
            else:
                new_edges[edge[1]] = [edge[0]]
        elif edge[1] in comp_to_cluster: #its a bifur/leaf point
            assert edge[0] not in comp_to_cluster, (edge, comp_to_cluster) #sanity check that you dont have cluster-cluster edge
            if edge[0] in new_edges:
                new_edges[edge[0]].append(edge[1])
            else:
                new_edges[edge[0]] = [edge[1]]
   
    return new_edges


def recover_vessels_byMIB(recovered_skeleton, labelwidth, voxelsize):
    recovered_vessels = np.zeros_like(recovered_skeleton, dtype=np.uint8)
    shp = recovered_skeleton.shape
    for vox in np.argwhere(recovered_skeleton):
        r = labelwidth[*tuple(vox)]
        rr = int(np.ceil(r/voxelsize[0]))
        cc = int(np.ceil(r/voxelsize[1]))
        zz = int(np.ceil(r/voxelsize[2]))
        recovered_vessels[max(0,vox[0]-rr):min(shp[0],vox[0]+rr+1), 
                        max(0,vox[1]-cc):min(shp[1],vox[1]+cc+1), 
                        max(0,vox[2]-zz):min(shp[2],vox[2]+zz+1)] = ndi.binary_dilation(
                                    recovered_skeleton[max(0,vox[0]-rr):min(shp[0],vox[0]+rr+1), 
                                                    max(0,vox[1]-cc):min(shp[1],vox[1]+cc+1), 
                                                    max(0,vox[2]-zz):min(shp[2],vox[2]+zz+1)], 
                                                    structure=ball(max(rr,cc,zz)))
    return recovered_vessels


def recover_by_expansion(skeleton, label, allowed_parts, maxwidth):
    #the skeleton should now consiste of the full skeleton of the allowed
    #parts, but also självstående komponenter that are the rest of the nodes
    #so basically, mulitple ccs. 
    expanded = expand_labels(skeleton, distance=maxwidth*2)
    #nib.save(nib.Nifti1Image(expanded.astype(np.uint8), np.eye(4)), "/Users/evabreznik/Desktop/MAIASTUFF/expanded1.nii.gz")
    expanded = np.where(label,expanded,0) #only keep whats inside the original label
    #nib.save(nib.Nifti1Image(skeleton.astype(np.uint8), np.eye(4)), "/Users/evabreznik/Desktop/MAIASTUFF/skelet.nii.gz")
    expanded = np.where(np.isin(expanded, np.asarray(list(allowed_parts), dtype=expanded.dtype)), expanded, 0) #but here the allowed parts shoul be only the 1 CC
    return expanded #could also do sanity check: now you should have a single CC


def get_aorta(ctimg, ccs):
    aorta = cc3d.connected_components((ctimg>300)&(ctimg<800)&(ccs==0), connectivity=6)
    aortastat = cc3d.statistics(aorta)
    aortaidx = np.argsort(aortastat['voxel_counts'])
    biggest_three = aortaidx[-4:-1]

    from_center = []
    roundness = []
    from_center_y = []
    for big in biggest_three: #check that it exists at all in this slice
        newcc = cc3d.connected_components(aorta[:,:,-3]==big, connectivity=8)
        statsbig = cc3d.statistics(newcc)
        #check if it exists in the slice at all:
        if len(statsbig['voxel_counts'])==1: #ie only bckg component
            from_center.append(np.inf)
            roundness.append(np.inf)
            from_center_y.append(np.inf)
            continue
        use = np.argsort(statsbig['voxel_counts'])[-2]
        bbx = statsbig['bounding_boxes'][use]
        centr = statsbig['centroids'][use]
        from_center.append(np.square(centr-np.array(aorta.shape[:-1])/2).sum())
        roundness.append(np.abs((bbx[0].stop-bbx[0].start)/(bbx[1].stop-bbx[1].start)-1))
        from_center_y.append(np.abs(centr[1]-np.array(aorta.shape[1])/2))

    #best_by_center = biggest_three[np.argmin(from_center)]
    #best_by_roundness = biggest_three[np.argmin(roundness)]
    #best_by_center_y = biggest_three[np.argmin(from_center_y)]

    # first choose two best ones by center, discard the third one
    discard = np.argsort(from_center)[-1]
    roundness[discard] = np.inf
    from_center_y[discard] = np.inf
    #now among the two best by center choose the one that's also best by roundness or center_y
    best_by_center_y = np.argmin(from_center_y)
    best_by_roundness = np.argmin(roundness)
    if best_by_center_y != best_by_roundness:
        print(".      oh boy...")
        #IDEA: take that one that is more homogeneous; multiply bi disk of bbx size, count how many of those are bigger than..300HU

    bst = biggest_three[best_by_center_y]

    #print(best_by_roundness, best_by_center_y, bst)
    return aorta==bst