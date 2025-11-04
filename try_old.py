
import scipy.ndimage as ndi
from skimage.morphology import  ball
import numpy as np
import nibabel as nib
import nrrd
import cc3d
from skimage.morphology import skeletonize, thin
from TMDutils import get_bifur_and_leaves, transform_dict, recover_vessels_byMIB, recover_by_expansion, get_aorta
from pathlib import Path

allimgs = list(Path("/Users/evabreznik/Desktop/CODING/test_predicted/resampled").glob("*"))
cts = sorted([i for i in allimgs if i.name.endswith("_0000.nii.gz")])
segs = sorted([i for i in allimgs if i not in cts])

def _unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / (n + eps)

def get_next(mask, start):
    next1 = np.argwhere(mask[start[0]-1:start[0]+2, start[1]-1:start[1]+2, start[2]-1:start[2]+2])
    next = np.array([n for n in next1 if tuple(n)!=(1,1,1)]) #remove the start voxel
    print(next1, next)
    return next + np.array([[start[0]-1, start[1]-1, start[2]-1]])


for subject in cts[:1]:
    subjname = subject.name[:-12]
    print(f"Processing {subjname}")
    if subjname!="Diseased_18":
        continue
    ctimg = nib.load(subject).get_fdata()
    #re.split("_|\.", i.name)[:2]
    for method in [i for i in segs if subjname=="_".join(i.name.split("_")[:2]) and "UNCD" in i.name]:
        print(f"  . Processing {method.name[len(subjname)+1:-5]}")
        img = nib.load(method)
        assert img.shape == ctimg.shape, (img.shape, ctimg.shape, method.name)
        
        #we work with each CC separately.
        voxelsize = img.header.get_zooms()[:3]
        mean_voxsize = np.mean(voxelsize)

        ccs, ncomp = cc3d.connected_components(img.get_fdata(), connectivity=26, return_N=True)
        assert ncomp == 2, f"Expected 2 components, got {ncomp}."
        stats = cc3d.statistics(ccs)

        # IDEA
        # COMPARE RESULTS - after using multiple mesh correction methods/tools. To see if our segmentation correction really does anything


        # GET AORTA
        aorta = get_aorta(ctimg, ccs)
        aorta2d_center = [int(i) for i in ndi.center_of_mass(aorta[:,:,-3])] + [None]  #z not needed
        #nib.save(nib.Nifti1Image(aorta.astype(np.uint8)*3 + ccs, img.affine, img.header), "/Users/evabreznik/Desktop/aorta.nii.gz")

        ########### LEFT AND RIGHT MASKS AND VOXEL COUNTS
        mask1 = ccs==1 
        bb1orig = stats['bounding_boxes'][1]  # get bounding box of the second component/first foreground

        mask2 = ccs==2 #do 2 for the other one.
        bb2orig = stats['bounding_boxes'][2]  # get bounding box of the second component/first foreground
        ################


        ##### BOUNDING BOXES !!!!    
        # #instead of doing iteratively, nja bo deterministic; poglej bb od aorte v 2D. 
        # v z smeri dodas 10, v xy pa mora prit vsaj do sredine bb od aorte 
        bbsize = 5
        aorta2d_center[2] = max(bb2orig[2].start, bb1orig[2].start) + (min(bb2orig[2].stop, bb1orig[2].stop)-max(bb2orig[2].start, bb1orig[2].start))//2
        bb1 = tuple(slice(max(0, min(c2d, ind.start)-bbsize),
                          min(s, max(c2d, ind.stop)+bbsize+1),
                          None) for ind, s, c2d in zip(bb1orig, mask1.shape, aorta2d_center))
        bb2 = tuple(slice(max(0, min(c2d, ind.start)-bbsize),
                          min(s, max(c2d, ind.stop)+bbsize+1),
                          None) for ind, s, c2d in zip(bb2orig, mask2.shape, aorta2d_center))
        

        #crop to bounding box
        label1 = mask1[bb1[0], bb1[1], bb1[2]] 
        label2 = mask2[bb2[0], bb2[1], bb2[2]] 

        # now find INFLOW points
        #confine aorta to bounding boxes, for inflow finding. 
        #now the label will be dilated (within appropriate HU window) until it bumps into aorta. The overlap is
        #the inflow point. 
        ct_bool1 = ctimg[*bb1]>250
        ct_bool2 = ctimg[*bb2]>250

        ballsize=1
        #aortatmp1 = ndi.binary_dilation(aorta[*bb1], structure=ball(ballsize))&ct_bool1
        #aortatmp2 = ndi.binary_dilation(aorta[*bb2], structure=ball(ballsize))&ct_bool2
        #inflow1 = np.nonzero(aortatmp1&label1)
        labeltmp1 = ndi.binary_dilation(label1, structure=ball(ballsize))&ct_bool1
        labeltmp2 = ndi.binary_dilation(label2, structure=ball(ballsize))&ct_bool2
        inflow1 = np.nonzero(labeltmp1&aorta[*bb1])
        while len(inflow1[0])==0 and ballsize<5:
            ballsize += 1
            #aortatmp1 = ndi.binary_dilation(aorta[bb1[0], bb1[1], bb1[2]], structure=ball(ballsize))&ct_bool1
            #inflow1 = np.nonzero(aortatmp1&label1)
            labeltmp1 = ndi.binary_dilation(label1, structure=ball(ballsize))&ct_bool1
            inflow1 = np.nonzero(labeltmp1&aorta[*bb1])
        if ballsize>3:
            print("    OBS ballsize1 ", ballsize)
        inflow_centroid1 = np.array([x.mean() for x in inflow1]) #average vsake koordinate
        
        ballsize=1
        #inflow2 = np.nonzero(aortatmp2&label2)
        inflow2 = np.nonzero(labeltmp2&aorta[*bb2])
        while len(inflow2[0])==0 and ballsize<5:
            ballsize += 1
            #aortatmp2 = ndi.binary_dilation(aorta[*bb2], structure=ball(ballsize))&ct_bool2
            #inflow2 = np.nonzero(aortatmp2&label2)
            labeltmp2 = ndi.binary_dilation(label2, structure=ball(ballsize))&ct_bool2
            inflow2 = np.nonzero(labeltmp2&aorta[*bb2])
        if ballsize>3:
            print("    OBS ballsize2 ", ballsize)
        inflow_centroid2 = np.array([x.mean() for x in inflow2]) #average vsake koordinate

        #get a more proper extent of the vessel at inflow, instead of taking just w=5
        inflow1_extent = [(x.min()+bb1[xi].start, x.max()+bb1[xi].start) for xi, x in enumerate(inflow1)]
        inflow2_extent = [(x.min()+bb2[xi].start, x.max()+bb2[xi].start) for xi, x in enumerate(inflow2)]
        #these extents are now in world coordinates, so directly comparable
        boxD = np.minimum(np.array(inflow1_extent), np.array(inflow2_extent))[:,0]
        boxU = np.maximum(np.array(inflow1_extent), np.array(inflow2_extent))[:,1]

        #to make sure the skeleton of the vessels will be ok even at the top, and that it will extend all the way to the top (ie to 
        #coronary ostia) in the first place, lets add some part of the aorta to the coronary seg that we will skeletonize.
        #inflow_centroid1_world_coord = inflow_centroid1 + np.array([bb1[0].start, bb1[1].start, bb1[2].start])
        #inflow_centroid2_world_coord = inflow_centroid2 + np.array([bb2[0].start, bb2[1].start, bb2[2].start])
        #boxD = np.minimum(inflow_centroid1_world_coord, inflow_centroid2_world_coord).astype(int)
        #boxU = np.maximum(inflow_centroid1_world_coord, inflow_centroid2_world_coord).astype(int)
        aorta_to_keep = np.zeros_like(aorta, dtype=np.uint8)
        w = 10
        aorta_to_keep[boxD[0]-w:boxU[0]+w, 
                      boxD[1]-w:boxU[1]+w, 
                      boxD[2]-w:boxU[2]+w] = aorta[boxD[0]-w:boxU[0]+w, 
                                               boxD[1]-w:boxU[1]+w, 
                                               boxD[2]-w:boxU[2]+w]
        from scipy.ndimage import binary_fill_holes, laplace
        from scipy import spatial
        #aorta_to_keep = binary_fill_holes(aorta_to_keep)
        def flood_fill_hull(image):    
            points = np.transpose(np.where(image))
            hull = spatial.ConvexHull(points)
            deln = spatial.Delaunay(points[hull.vertices]) 
            idx = np.stack(np.indices(image.shape), axis = -1)
            out_idx = np.nonzero(deln.find_simplex(idx) + 1)
            out_img = np.zeros(image.shape)
            out_img[out_idx] = 1
            return out_img, hull
        aorta_to_keep, _ = flood_fill_hull(laplace(aorta_to_keep, mode='constant'))
        #label==1 je aorta, label==2 pa inflows!

        #skeletonize to width 1vox; only needed for connectivity analysis, and to see which branches are to be discarded
        #now we skeletonize the vessel+cutout of aorta, to make sure the skeleton reaches out
        nib.save(nib.Nifti1Image(img.get_fdata()+aorta_to_keep, affine=img.affine, header=img.header), "/Users/evabreznik/Desktop/labelWaorta.nii.gz")
        skel1 = skeletonize(label1+aorta_to_keep[*bb1])&label1
        skel2 = skeletonize(label2+aorta_to_keep[*bb2])&label2
       ########################################

        # GET BIFURCATIONS AND LEAVES
        comp_to_bifur1, comp_to_leaf1, clusters1, NCl1 = get_bifur_and_leaves(skel1)
        comp_to_bifur2, comp_to_leaf2, clusters2, NCl2 = get_bifur_and_leaves(skel2) #TODO: now clusterd already contaisn both nodes AND edges!!
        
        #now do a cleanup: Though maybe thats better within get_bifur.
        #the cleanup should: do a full connectivity analysis, like line 148 and on, and resolve potential and neither...
        #or no: fastest to implement at the moment (though less efficient):
        # within the function, dont return both nodes and edges in cluster matrix; only clusters, but remove those that you alreayd know are neither.
        #Then you can rerun the analysis below (check only if you need sth else for the potential roots)..

        #find the root leafs
        rootcomp1 = min([(np.power(np.array(lfp[0])-inflow_centroid1,2).sum(), lvc) for lvc,lfp in comp_to_leaf1.items()], key=lambda x:x[0])[1]
        closest_leaf_point1 = comp_to_leaf1[rootcomp1][0]
        rootcomp2 = min([(np.power(np.array(lfp[0])-inflow_centroid2,2).sum(), lvc) for lvc,lfp in comp_to_leaf2.items()], key=lambda x:x[0])[1]
        closest_leaf_point2 = comp_to_leaf2[rootcomp2][0]


        #get connectivity of the vessel graph and lengths of different parts 
        broken1 = np.where(clusters1, 0, skel1) #remove clusters to do cc3d only on the edges
        broken2 = np.where(clusters2, 0, skel2)
        parts1, nr_parts1 = cc3d.connected_components(broken1, connectivity=26, return_N=True)
        parts2, nr_parts2 = cc3d.connected_components(broken2, connectivity=26, return_N=True)
        assert nr_parts1+NCl1>len(comp_to_bifur1)*2+len(comp_to_leaf1), (nr_parts1, len(comp_to_bifur1), len(comp_to_leaf1)) #sanity check that the clusters really somewhat appropriately break the skeleton.
        assert nr_parts2+NCl2>len(comp_to_bifur2)*2+len(comp_to_leaf2), (nr_parts2, len(comp_to_bifur2), len(comp_to_leaf2)) #sanity check that the clusters really somewhat appropriately break the skeleton.


        #but for the statistics, we actually need to consider cluster as well. But their idxs should
        #stay the same, as we already reference them via `comp_to_cluster`.
        parts1 = np.where(parts1, parts1+NCl1, clusters1) #add offset to parts, so they dont overlap with cluster ids
        parts2 = np.where(parts2, parts2+NCl2, clusters2) #add offset to parts, so they dont overlap with cluster ids
        lengths1 = cc3d.statistics(parts1)['voxel_counts']
        lengths2 = cc3d.statistics(parts2)['voxel_counts']
        edges1 = cc3d.region_graph(parts1, connectivity=26) #a set of (i1,i2) where i1 and i2 components are connected
        edges2 = cc3d.region_graph(parts2, connectivity=26) #a set of (i1,i2) where i1 and i2 components are connected
        #in edges1/2 the "components" in the tuples can be anything - a bifur/leaf OR the actual edge

        # these comp_to_cluster below on the other hand only contain those components (as keys) that correspond to bifur/leaf
        comp_to_cluster1 = comp_to_bifur1 | comp_to_leaf1
        comp_to_cluster2 = comp_to_bifur2 | comp_to_leaf2


        new_edges1 = transform_dict(edges1, comp_to_cluster1) # edge_component:[c1,c2,c3...] if edge_comp spans between components c1-3. in principle any newedges[l]should be 2 long!
        new_edges2 = transform_dict(edges2, comp_to_cluster2) # edge_component:[c1,c2,c3...] if edge_comp spans between components c1-3. in principle any newedges[l]should be 2 long!

       # assert np.all([len(v)==2 for v in new_edges1.values()]), new_edges1
       # assert np.all([len(v)==2 for v in new_edges2.values()]), new_edges2
        # the only time when the above assertion could fail is when you have
        #  a cycle: edge connecting one cluster to itself. Remove that edge!
        candidates = [k for k,v in new_edges1.items() if len(v)==1]
        #remove this edge from the dictionary, and also from skeleton.
        for c in candidates:
            new_edges1.pop(c)
            parts1[parts1==c] = 0
        candidates = [k for k,v in new_edges2.items() if len(v)==1]
        #remove this edge from the dictionary, and also from skeleton.
        for c in candidates:
            new_edges2.pop(c)
            parts2[parts2==c] = 0
        assert set(comp_to_cluster1.keys())==set(np.array(list(new_edges1.values())).flatten()), (comp_to_cluster1.keys(), set(np.array(list(new_edges1.values())).flatten())) #sanity check that all clusters are accounted for
        assert set(comp_to_cluster2.keys())==set(np.array(list(new_edges2.values())).flatten()), (comp_to_cluster2.keys(), set(np.array(list(new_edges2.values())).flatten())) #sanity check that all clusters are accounted for

        new_edges1 = {tuple(v): (k,lengths1[k]) for k,v in new_edges1.items()} #now the items are (c1,c2):length
        new_edges2 = {tuple(v): (k,lengths2[k]) for k,v in new_edges2.items()} #now the items are (c1,c2):length

        #only do removal/filtering by length of those branches that end in leaf, but not in root
        minlen=5
        keep_by_len_1 = {k:v for k,v in new_edges1.items() if ((rootcomp1 in k) or ((k[0] not in comp_to_leaf1) and (k[1] not in comp_to_leaf1)) or (v[1]>=minlen))}
        keep_by_len_2 = {k:v for k,v in new_edges2.items() if ((rootcomp2 in k) or ((k[0] not in comp_to_leaf2) and (k[1] not in comp_to_leaf2)) or (v[1]>=minlen))}

        # get also width of different parts
        labelwidth1 = ndi.distance_transform_edt(label1, sampling=voxelsize) #distance from the background, so you get radius
        widths1 = {k:labelwidth1[parts1==v[0]].mean() for k,v in new_edges1.items()}
        labelwidth2 = ndi.distance_transform_edt(label2, sampling=voxelsize) #distance from the background, so you get radius
        widths2 = {k:labelwidth2[parts2==v[0]].mean() for k,v in new_edges2.items()}
        maxwidth1 = labelwidth1.max()
        maxwidth2 = labelwidth2.max()

        #filter according to the length. Allow edge if at least 5 voxels long for example
        #filter according to the width. Allow edge if at least mean voxel size in radius (on average across the edge)
        keep_by_width1 = {k:v for k,v in keep_by_len_1.items() if widths1[k]>=mean_voxsize*np.sqrt(2)} | {k:v for k,v in new_edges1.items() if rootcomp1 in k} #always keep the root
        keep_by_width2 = {k:v for k,v in keep_by_len_2.items() if widths2[k]>=mean_voxsize*np.sqrt(2)} | {k:v for k,v in new_edges2.items() if rootcomp2 in k}


        #recreate volume by inflating MIB
        #IDEJA: recreation with MIB is problematic. The original is smoother and better. 
        #  try to cut away the problematic branches  directly from the original, so you dont need MIB
        allowed_parts_main1 = set([k[0] for k in keep_by_width1.keys()]+[k[1] for k in keep_by_width1.keys()]+[v[0] for v in keep_by_width1.values()])
        #to do recovery by expansion we instead need to keep only good edges but ALL nodes.
        allowed_parts1 = set(list(comp_to_cluster1.keys())+[v[0] for v in keep_by_width1.values()])
        recovered_skeleton1 = np.isin(parts1,list(allowed_parts1))

        allowed_parts_main2 = set([k[0] for k in keep_by_width2.keys()]+[k[1] for k in keep_by_width2.keys()]+[v[0] for v in keep_by_width2.values()])
        allowed_parts2 = set(list(comp_to_cluster2.keys())+[v[0] for v in keep_by_width2.values()])
        recovered_skeleton2 = np.isin(parts2,list(allowed_parts2))

        assert allowed_parts_main1.issubset(allowed_parts1), (allowed_parts_main1, allowed_parts1)
        assert allowed_parts_main2.issubset(allowed_parts2), (allowed_parts_main2, allowed_parts2)

        #this below recovers vessels, returning a labelled image. do >0 for a mask.
        recovered_vessels1 = recover_by_expansion(parts1*recovered_skeleton1, label1, allowed_parts_main1, maxwidth1)
        recovered_vessels2 = recover_by_expansion(parts2*recovered_skeleton2, label2, allowed_parts_main2, maxwidth2)

        #recovered_vessels1 = recover_vessels_byMIB(recovered_skeleton1, labelwidth1, voxelsize)
        #recovered_vessels2 = recover_vessels_byMIB(recovered_skeleton2, labelwidth2, voxelsize)
        

        #we dont want too big offset; egentligen 0 would be best. But to make sure there even is enough seg to cut, 
        #lets first extend the skeleton in the direction of the inflow, by a smal (10pix width?) cilinder with 
        # radius r=width of the vessel at the inflow.. But that remains ot be done if needed some other time. 
        offset_dist = 1.2 #nudge down along "centerline" a bit

        #NEW: for normals,to avoid potential +- change, use the vector from one root node to the other. 
        # NEW NEW: not good enough. But using again difference between root node and inflow is fine, as 
        # inflow is now outside the segmentation by definition.
        #but taking immediate neighbors for normal computation is not very good...take the second one or so..
        #maybe you can forget the whole centroid shit. and instead just follow the skeleton. Now that your skeleton is 
        #through the entire vessel (up to bordering with aorta), you can use the skeleton points directly.
        neigh1st = get_next(recovered_skeleton1, closest_leaf_point1)
        neigh2nd = [get_next(recovered_skeleton1, neigh) for neigh in neigh1st]

        normal1  = _unit(np.array(closest_leaf_point1) - inflow_centroid1)
        normal2  = _unit(np.array(closest_leaf_point2) - inflow_centroid2)
        cutoff_point1 = np.array(closest_leaf_point1) + offset_dist * normal1
        cutoff_point2 = np.array(closest_leaf_point2) + offset_dist * normal2
        #now check what is on the right side of the plane to be kept, and remove the rest.

        #if it takes too long, make a bounding box close around the overlap("inflow")
        off = max(int(offset_dist*2), 10)
        inflow_bb1 = tuple(slice(x.min()-off, x.max()+off) for x in inflow1)
        local_cutoff_point1 = cutoff_point1 - np.array([x.min()-off for x in inflow1])
        inflow_bb2 = tuple(slice(x.min()-off, x.max()+off) for x in inflow2)
        local_cutoff_point2 = cutoff_point2 - np.array([x.min()-off for x in inflow2])

        #Cut away coronary voxels by the half-space (x - cutoff_pt)·n >= 0 keeps distal side
        tmp_cutout = recovered_vessels1[*inflow_bb1].copy()
        coro_local = np.argwhere(tmp_cutout)
        signed = (coro_local - local_cutoff_point1) @ normal1   
        remove = signed >= 0.0      # distal side
        voxels_to_remove = coro_local[remove]
        #now update the segmentation by the cut:
        tmp_cutout[voxels_to_remove[:,0],voxels_to_remove[:,1],voxels_to_remove[:,2]] = 0 
        recovered_vessels1[*inflow_bb1] = tmp_cutout

        tmp_cutout = recovered_vessels2[*inflow_bb2].copy()
        coro_local = np.argwhere(tmp_cutout)
        signed = (coro_local - local_cutoff_point2) @ normal2   
        remove = signed >= 0.0      # distal side
        voxels_to_remove = coro_local[remove]
        #now update the segmentation by the cut:
        tmp_cutout[voxels_to_remove[:,0],voxels_to_remove[:,1],voxels_to_remove[:,2]] = 0 
        recovered_vessels2[*inflow_bb2] = tmp_cutout


        #DONE. Now fix vessel image size, as youve been working 
        # within a bounding box. Then save.
        recovered_vessels = np.zeros_like(ccs, dtype=np.uint8)
        recovered_vessels[*bb1] = recovered_vessels1>0
        recovered_vessels[*bb2] += recovered_vessels2>0



        #As the final cleanup step check which final branches (ie the ones that 
        # end in a leaf) among the ones we've kept
        # have a too small radius somewhere along the way (not just on average).
        # and on the last pixel before it becomes too thin (measured from the bifurcation 
        # side towards the root) make a perpendicular cut again. 
        to_cut1 = [v[0] for k,v in keep_by_width1.items() if (k[0] in comp_to_leaf1 or k[1] in comp_to_leaf1) and 
                   (rootcomp1 not in k) and (labelwidth1[parts1==v[0]].min()<mean_voxsize*np.sqrt(2))]

        to_cut2 = [v[0] for k,v in keep_by_width2.items() if (k[0] in comp_to_leaf2 or k[1] in comp_to_leaf2) and 
                   (rootcomp2 not in k) and (labelwidth2[parts2==v[0]].min()<mean_voxsize*np.sqrt(2))]

        for branchid in to_cut1:
            branch = parts1==branchid
            #find where to cut. Go from where the branch borders on a bifurcation
            voxs = [tuple(t) for t in np.argwhere(branch)]
            neighbors = [parts1[vox[0]-1:vox[0]+2, vox[1]-1:vox[1]+2, vox[2]-1:vox[2]+2] for vox in voxs]
            neighbors = [np.unique(n[(n!=0)&(n!=branchid)]) for n in neighbors]
            bifurvox = None
            for idn,n in enumerate(neighbors):
                if (len(n)>0) and (n[0] in comp_to_bifur1):
                    bifurvox = voxs[idn] #we assume the bifurcation voxel is only one anyway
                    break
            assert bifurvox is not None, "Bifurcation voxel not found?? Edge between two leafs! Wtf"
            #now find the last voxel before it becomes too thin
            prevs = [bifurvox] 
            cut = True
            while True:
                next = [i for i in voxs if ((i not in prevs) and (max(np.abs(np.array(i)-np.array(prevs[-1])))==1))]
                if len(next)==0:
                    cut = False
                    break #we've come to the end without finding the spot - weird... shouldnt really happen
                else: #there shouldnt be more than 1. But just in case, take the first one
                    next = next[0]
                    if labelwidth1[next]<mean_voxsize*np.sqrt(2):
                        break
                    prevs.append(next)

            #now we found the place to cut. 
            if cut:
                normal = _unit(np.array(prevs[-2])-np.array(prevs[-1]))
                cutoff_point = np.array(prevs[-1]) + 0.5 * normal
                #now check what is on the right side of the plane to be kept, and remove
                points = np.argwhere(recovered_vessels1==branchid)
                signed = (points - cutoff_point) @ normal

                #remove all points that are on the right side of the plane
                remove = points[signed>=0]
                recovered_vessels[remove[:,0],remove[:,1],remove[:,2]] = 0

        for branchid in to_cut2:
            branch = parts2==branchid
            #find where to cut. Go from where the branch borders on a bifurcation
            voxs = [tuple(t) for t in np.argwhere(branch)]
            neighbors = [parts2[vox[0]-1:vox[0]+2, vox[1]-1:vox[1]+2, vox[2]-1:vox[2]+2] for vox in voxs]
            neighbors = [np.unique(n[(n!=0)&(n!=branchid)]) for n in neighbors]
            bifurvox = None
            for idn,n in enumerate(neighbors):
                if (len(n)>0) and (n[0] in comp_to_bifur2):
                    bifurvox = voxs[idn] #we assume the bifurcation voxel is only one anyway
                    break
            assert bifurvox is not None, "Bifurcation voxel not found?? Edge between two leafs! Wtf"
            #now find the last voxel before it becomes too thin
            prevs = [bifurvox] 
            cut = True
            while True:
                next = [i for i in voxs if ((i not in prevs) and (max(np.abs(np.array(i)-np.array(prevs[-1])))==1))]
                if len(next)==0:
                    cut = False
                    break #we've come to the end without finding the spot - weird... shouldnt really happen
                else: #there shouldnt be more than 1. But just in case, take the first one
                    next = next[0]
                    if labelwidth2[next]<mean_voxsize*np.sqrt(2):
                        break
                    prevs.append(next)

            #now we found the place to cut. 
            if cut:
                normal = _unit(np.array(prevs[-2])-np.array(prevs[-1]))
                cutoff_point = np.array(prevs[-1]) + 0.5 * normal
                #now check what is on the right side of the plane to be kept, and remove
                points = np.argwhere(recovered_vessels2==branchid)
                signed = (points - cutoff_point) @ normal

                #remove all points that are on the right side of the plane
                remove = points[signed>=0]
                recovered_vessels[remove[:,0],remove[:,1],remove[:,2]] = 0

        #just to make sure we only take what we want, before saving do another sweep and get only the two main CCs
        labelled = cc3d.connected_components(recovered_vessels, connectivity=26)
        stats = np.argsort(cc3d.statistics(labelled)['voxel_counts'])
        largest_components = stats[-3:-1]
        recovered_vessels = np.isin(labelled, largest_components).astype(np.uint8)

        nib.save(nib.Nifti1Image(recovered_vessels, img.affine, img.header), 
                f"/Users/evabreznik/Desktop/CODING/test_predicted/postprocessed2/{method.name}")