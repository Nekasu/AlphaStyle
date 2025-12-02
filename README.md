# Alpha-Aware Neural Style Transfer with Dual-Gamma Soft Partial Convolution

Official PyTorch implementation of our alpha-aware neural style transfer framework
based on **Dual-Gamma Soft Partial Convolution (DualGammaSoftParConv)**.

This repository contains the cleaned and minimal code used in our Pattern Recognition submission **"Transparency-Aware Style Transfer via Soft Alpha-Guided Feature Propagation"**.
The model extends a recent arbitrary style transfer backbone to the **RGBA** domain and explicitly models the **alpha channel as a visibility prior** throughout training and inference.

- Code repo: <a href='https://github.com/Nekasu/AlphaStyle'>https://github.com/Nekasu/AlphaStyle</a>
- Paper: Coming soon

You can get codes from github using the command bellow:

```bash
git clone https://github.com/Nekasu/AlphaStyle.git
```

Then use the command below to get into the repo:

```bash
cd ./AlphaStyle
```

---

## 1. Introduction

Conventional neural style transfer (NST) methods operate purely in **RGB space** and
implicitly assume that all input pixels are fully opaque. This assumption fails for
RGBA images used in modern graphics, UI design, matting, and layered compositing:
ignoring the alpha channel often leads to **halo artifacts, style leakage into
transparent regions, and inconsistent background compositing**.

This project introduces an **alpha-aware NST framework** that:

- Extends arbitrary style transfer from **RGB** to **RGBA** images.
- Treats the **alpha channel as a continuous visibility prior**, guiding feature
  propagation and supervision at every stage of the pipeline.
- Uses a **Dual-Gamma Soft Partial Convolution (DualGammaSoftParConv)** module
  to perform visibility-aware convolution while preserving the semantics of
  completely transparent regions.
- Supports **content / style images with transparency**, enabling cleaner boundaries
  and better alignment between stylization intensity and visibility.

The implementation builds on a lightweight, encoder–decoder NST backbone and reuses
its contrastive aesthetic loss while inserting alpha-aware modules into the feature
extraction path.

---

## 2. Environment

The code is written in PyTorch and was originally developed with the following setup:

- Python 3.7
- PyTorch 1.13.1
- CUDA-capable GPU (e.g., RTX 4090 24GB)
- Additional Python packages:
  - `torchvision`
  - `tensorboardX`
  - `numpy`
  - `Pillow`
  - `thop`

We recommend creating a dedicated virtual environment:

```bash
conda create -n alpha-nst python=3.7 # create conda environment
conda activate alpha-nst # activate new conda environment

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia # install pytorch(for linux and windows)
conda install pillow numpy thop path # install the needed packages
```

or you can use pip to install the recommended torch verision. 

``` bash
pip install torch==1.13.1 torchvision==0.14.1
pip install pillow numpy thop path
```

To get other pytoch versions, please visit <a href='https://pytorch.org/get-started/previous-versions/'>https://pytorch.org/get-started/previous-versions</a>

## 3. Train

We basically follow the training methods of the AesFA project.

- Download the pre-trained [vgg_normalised.pth](https://drive.google.com/file/d/12D1feMRBWDvi1_3jIbx8vgLJTR3EA0VV/view?usp=drive_link).
- Change the training options in Config.py file.
- The 'phase' must be 'train'.
- The 'train_continue' should be 'on' if you train continuously with the previous model file.     
```python train.py```

## 4. Test

We modify the test files from AesFA. Now they can generate images with the size same as input images'.

- Download pre-trained model [model_iter_160000_epoch_22.pth](https://drive.google.com/file/d/14ncA0xgRDozI1IY5QiWa2T8LyXDG3Ms6/view?usp=drive_link)
- Change options about testing in the Config.py file.
- Change phase into 'test' and other options (ex) data info (num, dir), image load and crop size.
- Also, you can choose whether you want to translate using multi_to_multi or only translate content images using each style image.        
```python test.py```

