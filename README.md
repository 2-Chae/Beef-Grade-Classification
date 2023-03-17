# Ordinal-Regression-for-Beef-Grade-Classification
PyTorch Code for the paper:  
"[Ordinal Regression for Beef Grade Classification](https://ieeexplore.ieee.org/abstract/document/10043530)", ICCE 2023.  
Chaehyeon Lee, Jiuk Hong, Jonghyuck Lee, Taehoon Choi and Heechul Jung. 


## Installation
### prerequisites
- Python 3.7+
- PyTorch 1.10+
- TorchVision 0.11.2+  

Details are specified in [requirements.txt](requirements.txt).

## Training

We provide ordinal regression learning, hard label learning, and Gaussian-based label distribution learning.  
You can change the learning method by changing `--criterion` that has `['CE', 'GLD', 'OR']`.

The code below is an example of training using ordinal regression.   
```
CUDA_VISIBLE_DEVICES=0 python3 main_reverse.py --model convnext_base_in22ft1k \
                                              --input_size 224 \
                                              --data_set image_folder \
                                              --data_path [path_to_train_dataset]   \
                                              --eval_data_path [path_to_test_dataset]    \
                                              --epochs 20 \
                                              --warmup_epochs 0 \
                                              --save_ckpt true \
                                              --cutmix 0 \
                                              --mixup 0 \
                                              --smoothing 0.1 \
                                              --project beef \
                                              --color_jitter 0.1 \
                                              --use_amp True \
                                              --batch_size 256 \
                                              --enable_wandb True \
                                              --drop_path 0.2 \
                                              --update_freq 2 \
                                              --criterion OR
```

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

