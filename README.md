# EviSeg
## Calculation of the conflict
The further insights and implementation details are provided in these python files.The following will illustrate how to compute the conflict value using a semantic segmentation model as an example.
## Installation 

    # Get Semantic Segmentation source code
    git clone https://github.com/tkyoung13/Calculation-of-the-conflict.git




Before running the code, we recommend  install the following requirements: 

* An NVIDIA GPU and CUDA 9.0 or higher. Some operations only have gpu implementation.
* PyTorch (>= 0.5.1)
* numpy
* sklearn
* scikit-image
* pillow
* tqdm
* tensorboardX
* opencv-python
* apex


## Network architectures

Here we use DeepLabV3+ architecture with `WideResNet38` backbones. There are other options including `SEResNeXt(50, 101)` and `ResNet(50,101)`. 

  
## Pre-trained models
Pre-trained models have been provided. Please download the checkpoints to a designated folder `pretrained_models`. 

* [pretrained_models/camvid_best.pth](https://drive.google.com/file/d/1OzUCbFdXulB2P80Qxm7C3iNTeTP0Mvb_/view?usp=sharing)[1071MB, WideResNet38 backbone]


## Conflict calculation demo for a folder of images

If you want to try trained model on Camvid datasets, simply use

```
bash scripts/eval_camvid.sh pretrained_models/camvid_best.pth results/ --save-dir YOUR_SAVE_DIR
```
This snapshot is trained on CamVid dataset, with `DeepLabV3+` architecture and `WideResNet38` backbone. The predicted conflict images will be saved to `YOUR_SAVE_DIR`. Check it out. 


## Acknowledgments

Parts of the code were heavily derived from [Improving Semantic Segmentation via Video Prediction and Label Relaxation](https://github.com/YeLyuUT/SSeg.git).
