#%%
import scipy.ndimage as ndi
from skimage.morphology import  ball
import numpy as np
import nibabel as nib
import nrrd
import cc3d
from skimage.morphology import skeletonize, thin
from TMDutils import get_bifur_and_leaves, transform_dict, recover_vessels_byMIB, recover_by_expansion, get_aorta
from pathlib import Path
#%%

subjname = "Diseased_18"
p = Path("/Users/evabreznik/Desktop/CODING/test_predicted/resampled")
cts = list(p.glob(subjname+"*_0000.nii.gz"))[0]
method = list(p.glob(subjname+"_SM.nii.gz"))[0]

def _unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / (n + eps)



ctimg = nib.load(cts).get_fdata()
img = nib.load(method)
assert img.shape == ctimg.shape, (img.shape, ctimg.shape, method.name)



#%%
#we work with each CC separately.
voxelsize = img.header.get_zooms()[:3]

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

#to make sure the skeleton of the vessels will be ok even at the top, and that it will extend all the way to the top (ie to 
#coronary ostia) in the first place, lets add some part of the aorta to the coronary seg that we will skeletonize.
inflow_centroid1_world_coord = inflow_centroid1 + np.array([bb1[0].start, bb1[1].start, bb1[2].start])
inflow_centroid2_world_coord = inflow_centroid2 + np.array([bb2[0].start, bb2[1].start, bb2[2].start])
boxD = np.minimum(inflow_centroid1_world_coord, inflow_centroid2_world_coord).astype(int)
boxU = np.maximum(inflow_centroid1_world_coord, inflow_centroid2_world_coord).astype(int)
aorta_to_keep = np.zeros_like(aorta, dtype=np.uint8)
w = 5
aorta_to_keep[boxD[0]-w:boxU[0]+w, 
                boxD[1]-w:boxU[1]+w, 
                boxD[2]-w:boxU[2]+w] = aorta[boxD[0]-w:boxU[0]+w, 
                                        boxD[1]-w:boxU[1]+w, 
                                        boxD[2]-w:boxU[2]+w]


#skeletonize to width 1vox; only needed for connectivity analysis, and to see which branches are to be discarded
#now we skeletonize the vessel+cutout of aorta, to make sure the skeleton reaches out
#nib.save(nib.Nifti1Image(img.get_fdata()+aorta_to_keep, affine=img.affine, header=img.header), "/Users/evabreznik/Desktop/CODING/test_predicted/labelWaorta.nii.gz")
skel1 = skeletonize(label1+aorta_to_keep[*bb1])&label1
skel2 = skeletonize(label2+aorta_to_keep[*bb2])&label2
########################################





#%%
# GET BIFURCATIONS AND LEAVES
leaf1, bifur1, clusters1 = get_bifur_and_leaves(skel1)
leaf2, bifur2, clusters2 = get_bifur_and_leaves(skel2)
#zblj = np.zeros_like(ccs)
#zblj[*bb1] = skel1.astype(np.uint8)+clusters1
#nib.save(nib.Nifti1Image(zblj, img.affine, img.header), "/Users/evabreznik/Desktop/left_skel.nii.gz")
#nib.save(nib.Nifti1Image(skel1.astype(np.uint8)+clusters1, np.eye(4)), "/Users/evabreznik/Desktop/left_skel.nii.gz")
#nib.save(nib.Nifti1Image(skel2.astype(np.uint8)+clusters2, np.eye(4)), "/Users/evabreznik/Desktop/right_skel.nii.gz")
#zblj = np.zeros_like(ccs)
#zblj[*bb2] = skel2.astype(np.uint8)+clusters2
#nib.save(nib.Nifti1Image(zblj, img.affine, img.header), "/Users/evabreznik/Desktop/right_skel.nii.gz")

#find the root leafs
closest_leaf_point1 = np.argmin([np.power(np.array(lfp[0])-inflow_centroid1,2).sum() for lfp in leaf1])
closest_leaf_point1 = leaf1[closest_leaf_point1][0]
closest_leaf_point2 = np.argmin([np.power(np.array(lfp[0])-inflow_centroid2,2).sum() for lfp in leaf2])
closest_leaf_point2 = leaf2[closest_leaf_point2][0]

#%%

#get connectivity of the vessel graph and lengths of different parts 
broken1 = skel1 + clusters1
broken2 = skel2 + clusters2
parts1, nr_parts1 = cc3d.connected_components(broken1, connectivity=26, return_N=True)
parts2, nr_parts2 = cc3d.connected_components(broken2, connectivity=26, return_N=True)
assert nr_parts1>len(bifur1)*2+len(leaf1), (nr_parts1, len(bifur1), len(leaf1)) #sanity check that the clusters really somewhat appropriately break the skeleton.
assert nr_parts2>len(bifur2)*2+len(leaf2), (nr_parts2, len(bifur2), len(leaf2)) #sanity check that the clusters really somewhat appropriately break the skeleton.

lengths1 = cc3d.statistics(parts1)['voxel_counts']
lengths2 = cc3d.statistics(parts2)['voxel_counts']
edges1 = cc3d.region_graph(parts1, connectivity=26) #a set of (i1,i2) where i1 and i2 components are connected
edges2 = cc3d.region_graph(parts2, connectivity=26) #a set of (i1,i2) where i1 and i2 components are connected
#in edges1/2 the "components" in the tuples can be anything - a bifur/leaf OR the actual edge

# these comp_to_cluster below on the other hand only contain those components (as keys) that correspond to bifur/leaf
comp_to_cluster1 = {}
comp_to_cluster2 = {}
for bifur in bifur1+leaf1:
    comp_id = parts1[bifur[0]]
    if comp_id in comp_to_cluster1:
        print("issue; component ", comp_id, "mapped to two clusters:", comp_to_cluster1[comp_id], bifur)
    comp_to_cluster1[comp_id] = bifur
    #find out which component is the root component (ie coprresponds to the leaf that's closest to inflow)
    if bifur==closest_leaf_point1:
        rootcomp1 = comp_id
for bifur in bifur2+leaf2:
    comp_id = parts2[bifur[0]]
    if comp_id in comp_to_cluster2:
        print("issue; component ", comp_id, "mapped to two clusters:", comp_to_cluster2[comp_id], bifur)
    comp_to_cluster2[comp_id] = bifur
    #find out which component is the root component (ie coprresponds to the leaf that's closest to inflow)
    if bifur==closest_leaf_point2:
        rootcomp2 = comp_id


new_edges1 = transform_dict(edges1, comp_to_cluster1) # edge_component:[c1,c2,c3...] if edge_comp spans between components c1-3. in principle any newedges[l]should be 2 long!
new_edges2 = transform_dict(edges2, comp_to_cluster2) # edge_component:[c1,c2,c3...] if edge_comp spans between components c1-3. in principle any newedges[l]should be 2 long!

assert np.all([len(v)==2 for v in new_edges1.values()]), new_edges1
assert np.all([len(v)==2 for v in new_edges2.values()]), new_edges2
assert set(comp_to_cluster1.keys())==set(np.array(list(new_edges1.values())).flatten()), (comp_to_cluster1.keys(), set(np.array(list(new_edges1.values())).flatten())) #sanity check that all clusters are accounted for
assert set(comp_to_cluster2.keys())==set(np.array(list(new_edges2.values())).flatten()), (comp_to_cluster2.keys(), set(np.array(list(new_edges2.values())).flatten())) #sanity check that all clusters are accounted for

new_edges1 = {tuple(v): (k,lengths1[k]) for k,v in new_edges1.items()} #now the items are (c1,c2):length
new_edges2 = {tuple(v): (k,lengths2[k]) for k,v in new_edges2.items()} #now the items are (c1,c2):length

# get also width of different parts 
labelwidth1 = ndi.distance_transform_edt(label1, sampling=voxelsize) #distance from the background, so you get radius
widths1 = {k:labelwidth1[parts1==v[0]].mean() for k,v in new_edges1.items()}
labelwidth2 = ndi.distance_transform_edt(label2, sampling=voxelsize) #distance from the background, so you get radius
widths2 = {k:labelwidth2[parts2==v[0]].mean() for k,v in new_edges2.items()}
maxwidth1 = labelwidth1.max()
maxwidth2 = labelwidth2.max()

#filter according to the length. Allow edge if at least 5 voxels long for example
#filter according to the width. Allow edge if at least mean voxel size in radius (on average across the edge)
new_edges1 = {k:v for k,v in new_edges1.items() if v[1]>=5 and widths1[k]>=np.mean(voxelsize)} | {k:v for k,v in new_edges1.items() if rootcomp1 in k}
new_edges2 = {k:v for k,v in new_edges2.items() if v[1]>=5 and widths2[k]>=np.mean(voxelsize)} | {k:v for k,v in new_edges2.items() if rootcomp2 in k}


#recreate volume by inflating MIB
#IDEJA: recreation with MIB is problematic. The original is smoother and better. 
#  try to cut away the problematic branches  directly from the original, so you dont need MIB
allowed_parts_main1 = set([k[0] for k in new_edges1.keys()]+[k[1] for k in new_edges1.keys()]+[v[0] for v in new_edges1.values()])
#to do recovery by expansion we instead need to keep only good edges but ALL nodes.
allowed_parts1 = set(list(comp_to_cluster1.keys())+[v[0] for v in new_edges1.values()])
recovered_skeleton1 = np.isin(parts1,list(allowed_parts1))
# TODO: in this filtering, regardless of how short and small it is you shouldnt remove the first branch out of aorta!
# basically, change the code to only consider (while filtering) branches that end in 
# leaves, but not in aorta leaf. 

allowed_parts_main2 = set([k[0] for k in new_edges2.keys()]+[k[1] for k in new_edges2.keys()]+[v[0] for v in new_edges2.values()])
allowed_parts2 = set(list(comp_to_cluster2.keys())+[v[0] for v in new_edges2.values()])
recovered_skeleton2 = np.isin(parts2,list(allowed_parts2))

assert allowed_parts_main1.issubset(allowed_parts1), (allowed_parts_main1, allowed_parts1)
assert allowed_parts_main2.issubset(allowed_parts2), (allowed_parts_main2, allowed_parts2)

recovered_vessels1 = recover_by_expansion(parts1*recovered_skeleton1, label1, allowed_parts_main1, maxwidth1)
recovered_vessels2 = recover_by_expansion(parts2*recovered_skeleton2, label2, allowed_parts_main2, maxwidth2)

#recovered_vessels1 = recover_vessels_byMIB(recovered_skeleton1, labelwidth1, voxelsize)
#recovered_vessels2 = recover_vessels_byMIB(recovered_skeleton2, labelwidth2, voxelsize)


#TODO: from each leaf walk upwards towards bifur, and at each step check the width of the vessel. if to small, cut.
#or rather . makes more sense if you walk from the bifru to the root... just dont walk betweein bifur nodes!


#we dont want too big offset; egentligen 0 would be best. But to make sure there even is enough seg to cut, 
#lets first extend the skeleton in the direction of the inflow, by a smal (10pix width?) cilinder with 
# radius r=width of the vessel at the inflow.. But that remains ot be done if needed some other time. 
offset_dist = 1.2 #nudge down along "centerline" a bit

#NEW: for normals,to avoid potential +- change, use the vector from one root node to the other. 
# NEW NEW: not good enough. But using again difference between root node and inflow is fine, as 
# inflow is now outside the segmentation by definition.
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
recovered_vessels[*bb1] = recovered_vessels1
recovered_vessels[*bb2] += recovered_vessels2
nib.save(nib.Nifti1Image(recovered_vessels, img.affine, img.header), 
        f"/Users/evabreznik/Desktop/CODING/test_predicted/postprocessed/{method.name}")





#%%
# now i did the seg cuts by hand; load and save back only the two largest CCs.
# calc also metrics?
import re
slike = Path("/Users/evabreznik/Desktop/CODING/test_predicted/resampled")
slike2 = Path("/Users/evabreznik/Desktop/")

for sl in slike2.glob("*.nii.gz"):
    loaded = nib.load(str(sl))
    subj = re.split(r"_|\.", sl.name, maxsplit=2)
    subj_id = "_".join(subj[:2])
    sep="_"
    if len(subj[-1])==6:
        sep="."
    img = loaded.get_fdata()
    ccs, Ncl = cc3d.connected_components(img, connectivity=18, return_N=True)
    assert Ncl>=2, f"Expected at least 2 components, got {Ncl} for {sl.name}."
    stats = cc3d.statistics(ccs)
    mainCCs = stats['voxel_counts'].argsort()[-3:-1]
    #get which is left and which is right
    cc1 = stats['bounding_boxes'][mainCCs[0]]
    cc2 = stats['bounding_boxes'][mainCCs[1]]
    if cc1[0].start>cc2[0].start:
        L, R = mainCCs[0], mainCCs[1]
    else:
        L, R = mainCCs[1], mainCCs[0]

    mask = (ccs==mainCCs[0]) + (ccs==mainCCs[1])
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), loaded.affine, loaded.header),
                 f"/Users/evabreznik/Desktop/tostl/{subj_id}_full{sep}{subj[-1]}")

#%%
    for cc in [(L,"L"), (R,"R")]:
        # Create a binary mask for the current CC
        mask = ccs == cc[0]
        # Save the masked image
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), loaded.affine, loaded.header),
                 f"/Users/evabreznik/Desktop/tostl/{subj_id}_{cc[1]}{sep}{subj[-1]}")

# %%
