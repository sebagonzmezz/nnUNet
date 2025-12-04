Necessary custom files:

- .../nnUNet_raw/DatasetXXX_NAME/dataset.json
- .../nnUNet_preprocessed/DatasetXXX_NAME/nnUNetPlans.json
- .../nnUNet_preprocessed/DatasetXXX_NAME/splits_final.json

# Installation

```bash
cd nnUNet
pip install -e .
```

---

# Dataset folder structure

Datasets must be located in the `nnUNet_raw` folder (which you either define when installing nnU-Net or export/set every time you intend to run nnU-Net commands!). Each segmentation dataset is stored as a separate 'Dataset'. Datasets are associated with a dataset ID, a three digit integer, and a dataset name (which you can freely choose): For example, Dataset005_Prostate has 'Prostate' as dataset name and the dataset id is 5. Datasets are stored in the `nnUNet_raw` folder like this:

```
nnUNet_raw/
├── Dataset001_BrainTumour
├── Dataset002_Heart
├── Dataset003_Liver
├── Dataset004_Hippocampus
├── Dataset005_Prostate
├── ...
```

Within each dataset folder, the following structure is expected:

```
Dataset001_BrainTumour/
├── dataset.json
├── imagesTr
├── imagesTs  # optional
└── labelsTr
```

---

# Experiment planning and preprocessing

Given a new dataset, nnU-Net will extract a dataset fingerprint (a set of dataset-specific properties such as image sizes, voxel spacings, intensity information etc). This information is used to design three U-Net configurations. Each of these pipelines operates on its own preprocessed version of the dataset.

The easiest way to run fingerprint extraction, experiment planning and preprocessing is to use:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

Where `DATASET_ID` is the dataset id (duh). We recommend `--verify_dataset_integrity` whenever it's the first time you run this command. This will check for some of the most common error sources!

---

# Model training

You pick which configurations (2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres) should be trained! If you have no idea what performs best on your data, just run all of them and let nnU-Net identify the best one. It's up to you!

nnU-Net trains all configurations in a 5-fold cross-validation over the training cases.

You can influence the splits nnU-Net uses for 5-fold cross-validation.

Training models is done with the `nnUNetv2_train` command. The general structure of the command is:

```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
```

UNET_CONFIGURATION is a string that identifies the requested U-Net configuration found in DatasetXXX_NAME/nnUNetPlans.json.

For example, to train a model as defined in the `G2_up` trainer class on **fold 0** of the **001 dataset**, using the `G2_up-all` configuration found in `nnUNetPlans.json` and with **pretrained weights for the encoder**, the command would be:

```bash
nnUNetv2_train 001 G2_up-all 0 -tr G2_up -pretrained_encoder /path/to/encoder/weights
```

Training can only be performed on a single fold at a time, so to train all folds sequentially:

```bash
nnUNetv2_train 001 3d_fullres 0 && \
nnUNetv2_train 001 3d_fullres 1 && \
nnUNetv2_train 001 3d_fullres 2 && \
nnUNetv2_train 001 3d_fullres 3 && \
nnUNetv2_train 001 3d_fullres 4
```

---

# Run Prediction

For each of the desired configurations, run:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
```

* `-i` → Folder containing the images on which to run inference.

* `-o` → Output folder in which to save the generated segmentation masks.

* `-d` → Dataset name or ID in which the model was originally trained.

* `-c` → Configuration used during training.

* `-tr` → Trainer class used during training.

* `-f` → Fold to use (default is **all 5 folds**).

* Only specify `--save_probabilities` if you intend to use **ensembling**.
  This option will save the predicted probabilities alongside the predicted segmentation masks, which can require a lot of disk space.

* Please select a **separate `OUTPUT_FOLDER`** for each configuration.

**Note:** By default, inference will be done with all 5 folds from the cross-validation as an ensemble. We strongly recommend using all 5 folds.
Thus, all 5 folds **must have been trained** prior to running inference.
