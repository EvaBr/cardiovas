#!/usr/bin/bash

cd
source .bashrc
conda activate kranskarl
#first set the vars
export nnUNet_raw="/home/maia-user/Desktop/nnUNet_raw"
export nnUNet_preprocessed="/home/maia-user/Desktop/nnUNet_preprocessed"
export nnUNet_results="/home/maia-user/Desktop/nnUNet_results"


#conda activate /home/maia-user/.conda/envs/kranskarl

#move my trainers where they should be
#sudo cp cldice.py /opt/code/nnunet/nnunetv2/training/loss/.
#sudo cp EvasTrainers.py /opt/code/nnunet/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainersEva.py
#sudo cp nnUNetTrainer.py /opt/code/nnunet/nnunetv2/training/nnUNetTrainer/.
#sudo cp nnunet_logger.py /opt/code/nnunet/nnunetv2/training/logging/.
#add self.enable_deep_supervision = True in line 154 of .../training/nnUNetTrainer/nnUNetTrainer.py

#run the 3D full res and cascade nnUNet, 5Fold and find best model.  #0 1 2 3 4

#nnUNetv2_train --npz -tr nnUNetTrainerClDSC_9_1 666 3d_fullres 6 
#nnUNetv2_train --npz -tr nnUNetTrainerClDSC_8_2 666 3d_fullres 6 
#nnUNetv2_train --npz -tr nnUNetTrainerClDSC_7_3 666 3d_fullres 6 
#nnUNetv2_train --npz -tr nnUNetTrainerClDSC_6_4 666 3d_fullres 6 
#nnUNetv2_train --npz -tr nnUNetTrainerClDSC_1_1 666 3d_fullres 6 
#echo "All done!"

#might need to rerun val on fold 0
#nnUNetv2_train 666 3d_fullres 0 --val --npz

for fold in 0 1
do 
    #printf "\n\n Doing CASCADE (but first lowres), fold $fold... \n\n"
    #nnUNetv2_train --npz -tr nnUNetTrainer_250epochs 666 3d_lowres $fold
	nnUNetv2_train --npz -tr nnUNetTrainer_250epochs 666 3d_cascade_fullres $fold
done

#find best model, can ensemble;
#nnUNetv2_find_best_configuration 666 -c 3d_cascade_fullres -tr nnUNetTrainer_150epochs



#AHH, did batchsize=1 in nnUNet plans, maybe rewvert and make net smaller?
#now train with clDice, no scheduling, 150epochs, weight 0.7cl and 0.3DSC
#for fold in 0 1 2 3 4
#do 
#    printf "\n\n Doing CL-DSC fold $fold... \n\n"
	#nnUNetv2_train --npz -tr nnUNetTrainerClDSC_150epochs 666 3d_fullres $fold
#    nnUNetv2_train --npz -tr nnUNetTrainerClDSC_250epochs_lossequal 666 3d_fullres $fold
#done

#nnUNetv2_find_best_configuration 666 -c 3d_fullres -tr nnUNetTrainerClDSC_250epochs_lossequal #_150epochs
