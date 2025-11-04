# Project: Uncertainty-Aware Coronary Artery Segmentation using nnU-Net
# Objective: Quantify and visualize uncertainty in coronary artery segmentation using Monte Carlo Dropout

# Step 1: Data Preparation (NOT CODED HERE)
# - Use your dataset of 3000 CCTA images
# - Preprocess images (normalization, resizing) if needed
# - Organize data in the nnU-Net format (imagesTr, labelsTr, imagesTs, etc.)
# - Use nnU-Net's preprocessing scripts to prepare dataset

# Step 2: Model Training with nnU-Net
# - Train nnU-Net normally using nnUNet_train command line tool:
#   nnUNetv2_plan_and_preprocess -d <dataset_id>
#   nnUNetv2_train -d <dataset_id> -c 3d_fullres -f 0

#%%
# Step 3: Inference with Uncertainty Estimation using Monte Carlo Dropout
# Import necessary libraries

import os
import numpy as np
import matplotlib.pyplot as plt
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import load_pickle
import torch

#%%


def enable_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


#%%
# Step 4: Predict with multiple forward passes to estimate uncertainty

def mc_dropout_predict(predictor, image_path, output_dir, n_iter=20):
    all_preds = []

    # Ensure dropout is enabled
    enable_dropout(predictor.network)

    for i in range(n_iter):
        iter_out = os.path.join(output_dir, f"iter_{i}")
        os.makedirs(iter_out, exist_ok=True)

        with torch.no_grad():
            predictor.predict_from_files(
                [[image_path]],  # list of list (modalities)
                iter_out,
                save_probabilities=True,  # needed to get softmax output
                num_processes_preprocessing=2,
                num_processes_segmentation_export=2,
            )

        pred_prob_path = os.path.join(iter_out, os.path.basename(image_path).replace('.nrrd', '.npz'))
        softmax = np.load(pred_prob_path)['softmax']
        all_preds.append(softmax)

    all_preds = np.stack(all_preds, axis=0)  # shape: (n_iter, C, H, W, D)
    mean_pred = np.mean(all_preds, axis=0)
    var_pred = np.var(all_preds, axis=0)
    return mean_pred, var_pred


# Step 5: Visualization

def visualize_uncertainty(image, mean_pred, var_pred):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image.squeeze(), cmap='gray')
    axs[0].set_title('Input Image')
    axs[1].imshow(np.argmax(mean_pred, axis=0), cmap='jet')
    axs[1].set_title('Mean Prediction')
    axs[2].imshow(np.mean(var_pred, axis=0), cmap='hot')
    axs[2].set_title('Uncertainty Map')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    return fig





#%%
# Step 6: Evaluation
# Example: Dice coefficient using majority vote and comparing to ground truth

def dice_coefficient(pred, target, class_index):
    pred_class = (pred == class_index).astype(np.float32)
    target_class = (target == class_index).astype(np.float32)
    intersection = (pred_class * target_class).sum()
    return 2. * intersection / (pred_class.sum() + target_class.sum() + 1e-8)

# Step 7: Reporting
# - Plot and save uncertainty maps for representative samples
# - Correlate uncertainty with segmentation errors
# - Include visualizations and numerical results in your paper/report



def main():
    os.environ["nnUNet_raw"] = "/Users/evabreznik/Desktop/MAIASTUFF/nnUNet_raw"
    os.environ["nnUNet_preprocessed"] = "/Users/evabreznik/Desktop/MAIASTUFF/nnUNet_preprocessed"
    os.environ["nnUNet_results"] = "/Users/evabreznik/Desktop/MAIASTUFF/nnUNet_results"

    # Setup predictor, enable dropout, run mc_dropout_predict etc
    # Initialize the predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir="/Users/evabreznik/Desktop/MAIASTUFF/nnUNet_results/Dataset666_ASOCA/nnUNetTrainerClDSC_7_3__nnUNetResEncUNetMPlans__3d_fullres",
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth"
    )
    # and call
    eval_slike = "/Users/evabreznik/Desktop/MAIASTUFF/nnUNet_raw/Dataset666_ASOCA/imagesTr/"
    test_images = ["Normal_11_0000.nrrd", "Diseased_6_0000.nrrd"]
    for img_path in test_images:
        mean_pred, var_pred = mc_dropout_predict(predictor, eval_slike+img_path, f"MC_outputs/{img_path[:-10]}")
        np.save(f"MC_outputs/{img_path[:-10]}/mean_pred.npy", mean_pred)
        np.save(f"MC_outputs/{img_path[:-10]}/var_pred.npy", var_pred)
        fig = visualize_uncertainty(img_path, mean_pred, var_pred)  #
        fig.savefig(f"MC_outputs/{img_path[:-10]}_visual.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
