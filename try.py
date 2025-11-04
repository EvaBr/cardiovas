
import scipy.ndimage as ndi
from skimage.morphology import  ball
import numpy as np
import nibabel as nib
import nrrd
import cc3d
from skimage.morphology import skeletonize, thin
from TMDutils import get_bifur_and_leaves, transform_dict, recover_vessels_byMIB, recover_by_expansion, get_aorta
from pathlib import Path
from scipy.ndimage import binary_fill_holes, laplace
from scipy import spatial
from skimage.segmentation import expand_labels

allimgs = list(Path("/Users/evabreznik/Desktop/CODING/test_predicted/resampled").glob("*"))
cts = sorted([i for i in allimgs if i.name.endswith("_0000.nii.gz")])
segs = sorted([i for i in allimgs if i not in cts])

def joinbb(bb1, bb2):
    return tuple(slice(min(bb1[i].start, bb2[i].start), max(bb1[i].stop, bb2[i].stop), None) for i in range(3))

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

def _unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / (n + eps)

def get_next(mask, start):
    next1 = np.argwhere(mask[start[0]-1:start[0]+2, start[1]-1:start[1]+2, start[2]-1:start[2]+2])
    next = np.array([n for n in next1 if tuple(n)!=(1,1,1)]) #remove the start voxel
    return next + np.array([[start[0]-1, start[1]-1, start[2]-1]])

def cut_branch(branch_mask, cutoff_pt, normal, nudge=0.5):
    #the normal should always point distally; inwards to the main body. 
    #cutoff_pt should be in the same coordinates as the branch_mask
    branch_coords = np.argwhere(branch_mask)
    cutoff_pt = np.array(cutoff_pt)+nudge*normal
    # cut away everything beyond the plane with normal "normal" at "cutoff_pt".
    # the function returns the coordinates of all voxels to be "cut away"
    d = np.dot(branch_coords - cutoff_pt, normal)
    to_remove = branch_coords[d < nudge]
    return to_remove

def find_cutoff(vessel, branchid, bifurid, labelwidth, minwidth):
    #vessel is a label map, containg (boundary box bounded) branch and bifur.
    #labelwidth is the edt map, wihtin the same bbox.
    #minwidth= what is not wide enough. Usually 1.5*np.mean(voxelsize) or sth
    #this function is run when we already know there is at least 1 voxel in skeleton with widths<minwidth
    ok = np.where(vessel==branchid, labelwidth, 0)>minwidth # if theres a point where vessel is to small in radius, this will split the vessel in two
    #now check which component of those created borders on the bifur, and keep only that one
    comps, Nc = cc3d.connected_components(ok, connectivity=26, return_N=True)
    #assert Nc>1, "Only {} CC: no split occured when minwidth={}".format(Nc, minwidth)
    if not Nc>1:
        print(".      Only {} CC: no split occured when minwidth={}".format(Nc, minwidth))
        return None, None
    else:
        print(".     prunning...")
    #add the bifur node before getting edges
    bifur_mask = vessel==bifurid
    bifur = tuple(i.mean() for i in np.nonzero(bifur_mask))
    comps = np.where(bifur_mask, Nc+1, comps)
    connections = cc3d.region_graph(comps, connectivity=26)
    #connections now contains a set of (i1,i2) where i1 and i2 components are connected
    #find the component connected to bifu (should be only one)
    c = [con[0] if con[0]!=Nc+1 else con[1] for con in connections if Nc+1 in con]
    assert len(c) == 1, "nr conn to bifur {} is  {}(1). branch={}, bifurs{}, branchs{}, in vessel={}".format(bifurid, len(c), branchid, bifur_mask.sum(), (vessel==branchid).sum(), np.unique(vessel))
    keep = np.where(vessel==branchid, skeletonize(np.isin(comps, [c[0], Nc+1]))*1, 0)
    roots = [v for v in np.argwhere(keep) if keep[v[0]-1:v[0]+2, v[1]-1:v[1]+2, v[2]-1:v[2]+2].sum()==2]
   
    if keep.sum()<5:
        #nothing to prune, just remove entirely.
        return None, branchid
    assert len(roots)==2, f"Found so many roots: {len(roots)}, len of skel {keep.sum()}"
    #get the two rootnodes of the skeleton and see which one is closer ot the bifur. the other one is cutoff
    cutoff = np.argmax([np.linalg.norm(v - bifur) for v in roots])

    return roots[cutoff], keep

def get_normal(cutoff, skel, reached_in=3):
    #cutoff is starting point, assumed to be on the skel. It is also 
    # assumed to be a leaf node of the skeleton!
    allvisited = [tuple(cutoff)]
    visited = {0:[tuple(cutoff)]}
    for varv in range(reached_in):
        prev = visited[varv]
        for prv in prev:
            next = [tuple(i) for i in get_next(skel, prv) if (tuple(i) not in allvisited)]
            allvisited.extend(next)
            if varv+1 not in visited:
                visited[varv+1] = next
            else:
                visited[varv+1].extend(next)
   # print(allvisited)
   # print(visited)
    reachable_in_3 = visited[reached_in]   # a perfect world with a perfect skeleton, there should be exactly one
    if len(reachable_in_3)==0:
        if reached_in>=3:
            if len(visited[reached_in-1])>0:
                reachable_in_3 = visited[reached_in-1]
            else:
                #cant do this, remove branch entirely
                print("branch too short now, removing completely")
                return None
    normal = _unit(reachable_in_3[0] - cutoff)
    return normal


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
        closest_leaf_point1 = np.mean(comp_to_leaf1[rootcomp1], axis=0)
        rootcomp2 = min([(np.power(np.array(lfp[0])-inflow_centroid2,2).sum(), lvc) for lvc,lfp in comp_to_leaf2.items()], key=lambda x:x[0])[1]
        closest_leaf_point2 = np.mean(comp_to_leaf2[rootcomp2], axis=0)


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


        edge_to_comp1 = transform_dict(edges1, comp_to_cluster1) # edge_component:[c1,c2,c3...] if edge_comp spans between components c1-3. in principle any newedges[l]should be 2 long!
        edge_to_comp2 = transform_dict(edges2, comp_to_cluster2) # edge_component:[c1,c2,c3...] if edge_comp spans between components c1-3. in principle any newedges[l]should be 2 long!

       # assert np.all([len(v)==2 for v in edge_to_comp1.values()]), edge_to_comp1
       # assert np.all([len(v)==2 for v in edge_to_comp2.values()]), edge_to_comp2
        # the only time when the above assertion could fail is when you have
        #  a cycle: edge connecting one cluster to itself. Remove that edge!
        cycle_candidates = [k for k,v in edge_to_comp1.items() if len(v)==1]
     #   print("removed cycles" , cycle_candidates)
        #remove this edge from the dictionary, and also from skeleton.
        for c in cycle_candidates:
            edge_to_comp1.pop(c)
            parts1[parts1==c] = 0
        cycle_candidates = [k for k,v in edge_to_comp2.items() if len(v)==1]
     #   print("removed cycles" , cycle_candidates)
        #remove this edge from the dictionary, and also from skeleton.
        for c in cycle_candidates:
            edge_to_comp2.pop(c)
            parts2[parts2==c] = 0
        assert set(comp_to_cluster1.keys())==set(np.array(list(edge_to_comp1.values())).flatten()), (comp_to_cluster1.keys(), set(np.array(list(edge_to_comp1.values())).flatten())) #sanity check that all clusters are accounted for
        assert set(comp_to_cluster2.keys())==set(np.array(list(edge_to_comp2.values())).flatten()), (comp_to_cluster2.keys(), set(np.array(list(edge_to_comp2.values())).flatten())) #sanity check that all clusters are accounted for

        new_edges1 = {tuple(v): (k,lengths1[k]) for k,v in edge_to_comp1.items()} #now the items are (c1,c2):length
        new_edges2 = {tuple(v): (k,lengths2[k]) for k,v in edge_to_comp2.items()} #now the items are (c1,c2):length

        #only do removal/filtering by length of those branches that end in leaf, but not in root
        minlen=5
        keep_by_len_1 = {k:v for k,v in new_edges1.items() if ((rootcomp1 in k) or ((k[0] not in comp_to_leaf1) and (k[1] not in comp_to_leaf1)) or (v[1]>=minlen))}
        keep_by_len_2 = {k:v for k,v in new_edges2.items() if ((rootcomp2 in k) or ((k[0] not in comp_to_leaf2) and (k[1] not in comp_to_leaf2)) or (v[1]>=minlen))}

        # get also width of different parts
        labelwidth1 = ndi.distance_transform_edt(label1, sampling=voxelsize) #distance from the background, so you get radius
        widths1 = {k:labelwidth1[parts1==v[0]].mean() for k,v in new_edges1.items()}
        labelwidth2 = ndi.distance_transform_edt(label2, sampling=voxelsize) #distance from the background, so you get radius
        widths2 = {k:labelwidth2[parts2==v[0]].mean() for k,v in new_edges2.items()}
        maxwidth1 = labelwidth1.max()/min(voxelsize)
        maxwidth2 = labelwidth2.max()/min(voxelsize)

        #filter according to the width. Allow edge if at least mean voxel size in radius (on average across the edge)
        keep_by_width1 = {k:v for k,v in keep_by_len_1.items() if widths1[k]>=mean_voxsize*np.sqrt(2)} | {k:v for k,v in new_edges1.items() if rootcomp1 in k} #always keep the root
        keep_by_width2 = {k:v for k,v in keep_by_len_2.items() if widths2[k]>=mean_voxsize*np.sqrt(2)} | {k:v for k,v in new_edges2.items() if rootcomp2 in k}


        #recreate volume by inflating MIB
        #IDEJA: recreation with MIB is problematic. The original is smoother and better. 
        #  try to cut away the problematic branches  directly from the original, so you dont need MIB
        allowed_parts_main1 = set([k[0] for k in keep_by_width1.keys()]+[k[1] for k in keep_by_width1.keys()]+[v[0] for v in keep_by_width1.values()])
        #to do recovery by expansion we instead need to keep only good edges but ALL nodes.
        allowed_parts1 = set(list(comp_to_cluster1.keys())+[v[0] for v in keep_by_width1.values()])
        recovered_skeleton1 = np.where(np.isin(parts1,np.array(list(allowed_parts1), dtype=parts1.dtype)), parts1, 0)

        allowed_parts_main2 = set([k[0] for k in keep_by_width2.keys()]+[k[1] for k in keep_by_width2.keys()]+[v[0] for v in keep_by_width2.values()])
        allowed_parts2 = set(list(comp_to_cluster2.keys())+[v[0] for v in keep_by_width2.values()])
        recovered_skeleton2 = np.where(np.isin(parts2,np.array(list(allowed_parts2), dtype=parts2.dtype)), parts2, 0)

        assert allowed_parts_main1.issubset(allowed_parts1), (allowed_parts_main1, allowed_parts1)
        assert allowed_parts_main2.issubset(allowed_parts2), (allowed_parts_main2, allowed_parts2)

        #this below recovers vessels, returning a labelled image. do >0 for a mask.
        recovered_vessels1 = recover_by_expansion(recovered_skeleton1, label1, allowed_parts_main1, maxwidth1)
        recovered_vessels2 = recover_by_expansion(recovered_skeleton2, label2, allowed_parts_main2, maxwidth2)
 

        #cut the inflow surface at aorta
        offset_dist = 1.2 #nudge down along "centerline" a bit
        normal1  = _unit(np.array(closest_leaf_point1) - inflow_centroid1)
        normal2  = _unit(np.array(closest_leaf_point2) - inflow_centroid2)
        cutoff_point1 = np.array(closest_leaf_point1) + offset_dist * normal1
        cutoff_point2 = np.array(closest_leaf_point2) + offset_dist * normal2
        #now check what is on the right side of the plane to be kept, and remove the rest.
        #but removal should be done only on the branch that connects aortaroot. 
        aortabranch1 = [e for e in edge_to_comp1 if rootcomp1 in edge_to_comp1[e]]
        aortabranch2 = [e for e in edge_to_comp2 if rootcomp2 in edge_to_comp2[e]]
        assert len(aortabranch1)==1, f"Expected 1 aorta1 branch, found {len(aortabranch1)}"
        assert len(aortabranch2)==1, f"Expected 1 aorta2 branch, found {len(aortabranch2)}"
        #maybe not too slow, to avoid issues leave it at bb1/bb2 cords
        #bbab1 = cc3d.statistics(recovered_vessels1)['bounding_boxes'][aortabranch1[0]]
        #bbab2 = cc3d.statistics(recovered_vessels2)['bounding_boxes'][aortabranch2[0]]

        #Cut away coronary voxels by the half-space (x - cutoff_pt)·n >= 0 keeps distal side
        coro_local = np.argwhere(recovered_vessels1==aortabranch1[0])
        signed = (coro_local - cutoff_point1) @ normal1
        remove = signed < 0.0      # distal side
        voxels_to_remove = coro_local[remove]
        #now update the segmentation by the cut:
        recovered_vessels1[voxels_to_remove[:,0],voxels_to_remove[:,1],voxels_to_remove[:,2]] = 0 

        coro_local = np.argwhere(recovered_vessels2==aortabranch2[0])
        signed = (coro_local - cutoff_point2) @ normal2
        remove = signed < 0.0      # distal side
        voxels_to_remove = coro_local[remove]
        #now update the segmentation by the cut:
        recovered_vessels2[voxels_to_remove[:,0],voxels_to_remove[:,1],voxels_to_remove[:,2]] = 0 

        
        #As the final cleanup step check which final branches (ie the ones that 
        # end in a leaf) among the ones we've kept
        # have a too small radius somewhere along the way (not just on average).
        # and on the last pixel before it becomes too thin (measured from the bifurcation 
        # side towards the root) make a perpendicular cut again. 
        to_prune1 = [(v[0],k[0],k[1]) for k,v in keep_by_width1.items() if (k[0] in comp_to_leaf1 or k[1] in comp_to_leaf1) and 
                   (rootcomp1 not in k) and (labelwidth1[recovered_skeleton1==v[0]].min()<mean_voxsize*np.sqrt(2))]

        to_prune2 = [(v[0],k[0],k[1]) for k,v in keep_by_width2.items() if (k[0] in comp_to_leaf2 or k[1] in comp_to_leaf2) and 
                   (rootcomp2 not in k) and (labelwidth2[recovered_skeleton2==v[0]].min()<mean_voxsize*np.sqrt(2))]


        bounding_boxes_all1 = cc3d.statistics(recovered_vessels1)['bounding_boxes']
        bounding_boxes_all2 = cc3d.statistics(recovered_vessels2)['bounding_boxes']
        remove_branches1 = []
        remove_branches2 = []
        remove_voxels1 = []
        remove_voxels2 = []
        print("to prune 1", to_prune1)
        for candidate in to_prune1:#samo branches to roots, if widthlabel[branch_skel].min()<minwidth
            branchid = candidate[0]
            bifurid = candidate[1] if candidate[1] in comp_to_bifur1 else candidate[2]
            bb_branch = joinbb(bounding_boxes_all1[branchid], bounding_boxes_all1[bifurid])
            branch_mask = np.where(recovered_vessels1[*bb_branch]==branchid, branchid, 0)
            bifur_mask = np.where(recovered_vessels1[*bb_branch]==bifurid, bifurid, 0)
            if bifurid==3:
                print("bifur=3;",bifur_mask.sum(), branch_mask.sum(), np.unique(recovered_vessels1[*bb_branch]))
                print("bb=", bounding_boxes_all1[branchid])
                print("in full vessel", (recovered_vessels1==branch).sum())
            cutoff_pt, branchskel = find_cutoff(branch_mask+bifur_mask, 
                                                branchid, bifurid, labelwidth1[*bb_branch], mean_voxsize*np.sqrt(2))
            if cutoff_pt is None:
                print("nothing to prune...")
                if branchskel is not None:
                    remove_branches1.append(branchid)
                continue
            normal = get_normal(cutoff_pt, branchskel)
            if normal is None:
                #need to entirely remove the branch, as its too short
                remove_branches1.append(branchid)
                continue
            to_remove = cut_branch(branch_mask, cutoff_pt, normal, nudge=-0.3)
            #fix the to_remove to world coords, from bb
            to_remove = to_remove + [bb_branch[0].start, bb_branch[1].start, bb_branch[2].start]
            #then remove them from the final segm. 
            #recovered_vessels1[to_remove] = 0
            remove_voxels1.append(to_remove)

        print("to prune 2", to_prune2)
        for candidate in to_prune2:#samo branches to roots, if widthlabel[branch_skel].min()<minwidth
            branchid = candidate[0]
            bifurid = candidate[1] if candidate[1] in comp_to_bifur2 else candidate[2]
            bb_branch = joinbb(bounding_boxes_all2[branchid], bounding_boxes_all2[bifurid])
            branch_mask = np.where(recovered_vessels2[*bb_branch]==branchid, branchid, 0)
            bifur_mask = np.where(recovered_vessels2[*bb_branch]==bifurid, bifurid, 0)
            if bifurid==3:
                print("bifur=3;", branch_mask.sum(), np.unique(recovered_vessels2[*bb_branch]))
            cutoff_pt, branchskel = find_cutoff(branch_mask+bifur_mask, 
                                                branchid, bifurid, labelwidth2[*bb_branch], mean_voxsize*np.sqrt(2))
            if cutoff_pt is None:
                print("nothing to prune...")
                if branchskel is not None:
                    remove_branches2.append(branchid)
                continue
            normal = get_normal(cutoff_pt, branchskel)
            if normal is None:
                #need to entirely remove the branch, as its too short
                remove_branches2.append(branchid)
                continue
            to_remove = cut_branch(branch_mask, cutoff_pt, normal, nudge=-0.3)
            #fix the to_remove to world coords, from bb
            to_remove = to_remove + [bb_branch[0].start, bb_branch[1].start, bb_branch[2].start]
            #then remove them from the final segm. 
            #recovered_vessels2[to_remove] = 0
            remove_voxels2.append(to_remove)
            

        #DONE. Now fix vessel image size, as youve been working 
        # within a bounding box. Then save.
        recovered_vessels = np.zeros_like(ccs, dtype=np.uint8)
        recovered_vessels[*bb1] = recovered_vessels1>0
        recovered_vessels[*bb2] += recovered_vessels2>0



        #just to make sure we only take what we want, before saving do another sweep and get only the two main CCs
        labelled = cc3d.connected_components(recovered_vessels, connectivity=26)
        stats = np.argsort(cc3d.statistics(labelled)['voxel_counts'])
        largest_components = stats[-3:-1]
        recovered_vessels = np.isin(labelled, largest_components).astype(np.uint8)

        nib.save(nib.Nifti1Image(recovered_vessels, img.affine, img.header), 
                f"/Users/evabreznik/Desktop/CODING/test_predicted/postprocessed2/{method.name}")