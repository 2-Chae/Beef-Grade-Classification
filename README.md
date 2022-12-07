# Ordinal-Regression-for-Beef-Grade-Classification
PyTorch Code for the paper:  
[ICCE2023](https://icce.org/2023/Home.html) accepted. (will be posted soon)  


## Installation
### prerequisites
- Python 3.7+
- PyTorch 1.10+
- TorchVision 0.11.2+  

Details are specified in [requirements.txt](requirements.txt).

## Training
See [TRAINING.md](TRAINING.MD) for training.

## Evaluation
Due to the limitation of GPU resources, we needed to store the predicted vectors in memory and then use them in ensemble learning.

### 1. Save outputs
```
python3 save_outputs.py \
        --data_set image_folder \
        --data_path [path_to_train_dataset] \
        --eval_data_path [path_to_test_dataset] \
        --use_amp true \
        --batch_size 8 \
        --input_size 224 \
        --eval true
```

### 2. Ensemble learning
```
python3 ensemble.py
```

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) repositories.  
Based on the [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), we implemented the ordinal regression for the beef grade classification.

