# Fine Grained Classification 

## Getting Started
## Prerequisites

Ensure you have all the required packages installed:

```bash
pip install -e .
```


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

