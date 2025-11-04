import numpy as np
import nrrd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.morphology import skeletonize, medial_axis

#todo: add typing, also: move to class?
#all checks to return (True/False, true/false), with first denoting if it's ok by prior knowledge, and second wrt GT
#Perhaps better to return raw values, covert to bool for analysis later; since you might want to see the raw numbers at some point. 
#eval conn.comps
def check_CC(out, gt):
    _, nr_labs = np.label(out)
    _, nr_labs_gt = np.label(gt)
    return nr_labs, nr_labs_gt #(nr_labs==2, nr_labs==nr_labs_gt)

#eval the diameter deviations
def check_dia(out, gt, X=4):
    # Compute the medial axis (skeleton) and the distance transform
    skel_out, distance_out = medial_axis(out, return_distance=True)
    skel_gt, distance_gt = medial_axis(gt, return_distance=True)
    #check 1) we hope/need to (for reliable CFD) detect all vessels with at least thickness X.
    #absolute size:
    epsilon = 1
    mo, mg = (distance_out*skel_out).min(), (distance_gt*skel_gt).min()
    sajz_devs = (mo*2-X, mo-mg) #(mo*2>X-epsilon, abs(mo-mg)<epsilon)
    
    #check 2) 
    #overal/cummulative diameter deviation score
    agree = skel_out==skel_gt==1
    diffs = distance_out[agree]-distance_gt[agree]
    diffs_stats = (diffs.min(), diffs.median(), diffs.mean(), diffs.max())
    
    #check 3)
    #agreement of skeletons; skeleton recall
    #general: what length of all length agrees?
    full_skel = skel_gt.sum() #do a more detailed one, via mms?
    #agreement = skel_out[skel_gt>0].sum()/full_skel #perhaps use a dilated version of skel here in x-y? Dilate with diamond, like in kirchoff
    skel_recall = out[skel_gt>0].sum()/full_skel
    
    #detailed: for each leaf branch, how much length agrees?
    #first find all leaf branches. find all thos epoints in th eimage of skel that have at most one neighbor. Then you've found all leafs and all starting points.
    
    
#eval the centerline? As in, angles of bifurcation, bifurcation pattern (how many times it bifurcates - we have an expected nr, right?)


#Betti stuff: betti number and betti matching error

#Normalized surface distance (or hdd?)

#Adapted Rand Index (ARI)

#Variation of information (VOI)

#B-DoU loss

#Wasserstein distance

#OPT-Junction F1

# FROM PAPERS:
#  - clDSC, cbDSC, skeletonRecall
#  - araujo similarity index








