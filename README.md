# Fine Grained Classification 
This repository offers a PyTorch implementation of the fine-grained classification system originally found in [aioz-ai/sac](https://github.com/aioz-ai/sac). It has been adapted to work with our dataset and comes with a distinct embedding process.
## Getting Started
## Prerequisites

Ensure you have all the required packages installed:

```bash
pip install -e .
```

## download required external models 

Initial model for base model (InceptionV3): 
* s3://internal-model-sharing/vtcv-fine-grained-classification/ext_models/InceptionV3/
* place in `ext_models/InceptionV3/best_f1.pth`

CMUnet mask
* s3://internal-model-sharing/mosmask-unet/experiments/CMUNet/n_channels-3-2_blocks-n_classes-2-1673672754-dataset_size-15326-epochs-100/best_weights.pt
* place in `ext_models/mosquitoes_cmunet448/best_weights.pt`


## Running the Code

For training: 

Execute the following command to run the system:
```bash
python -m vtcv_fine_grained_classification --config batch/TrainConfig.json
```

For testing :
```bash
python -m vtcv_fine_grained_classification --exp-dir experiments/example --mode test
```

## vizualize accuracy and loss
```bash
#runs is a directory that stores the event files
tensorboard --logdir=runs
```

## Key Features

   - Built on top of PyTorch, inspired by [aioz-ai/sac](https://github.com/aioz-ai/sac).
   - Customized to work with our specific dataset.
   - Uses a unique embedding process for fine-grained classification.

## Contributions

