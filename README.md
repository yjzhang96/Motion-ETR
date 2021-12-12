# Motion-ETR (official pytorch implementation)
This repository provides the official PyTorch implementation of the [paper](https://ieeexplore.ieee.org/abstract/document/9551756/) accepted in TPAMI:

>Exposure Trajectory Recovery from Motion Blur 
> 
>Youjian Zhang, Chaoyue Wang, Dacheng Tao
>
>Abstract: Motion blur in dynamic scenes is an important yet challenging research topic. Recently, deep learning methods have achieved impressive performance for dynamic scene deblurring. However, the motion information contained in a blurry image has yet to be fully explored and accurately formulated because: (i) the ground truth of dynamic motion is difficult to obtain; (ii) the temporal ordering is destroyed during the exposure; and (iii) the motion estimation from a blurry image is highly ill-posed. By revisiting the principle of camera exposure, motion blur can be described by the relative motions of sharp content with respect to each exposed position. In this paper, we define exposure trajectories, which represent the motion information contained in a blurry image and explain the causes of motion blur. A novel motion offset estimation framework is proposed to model pixel-wise displacements of the latent sharp image at multiple timepoints. Under mild constraints, our method can recover dense, (non-)linear exposure trajectories, which significantly reduce temporal disorder and ill-posed problems. Finally, experiments demonstrate that the recovered exposure trajectories not only capture accurate and interpretable motion information from a blurry image, but also benefit motion-aware image deblurring and warping-based video extraction tasks.

---
## Contents

The contents of this repository are as follows:

1. [Prerequisites](#Prerequisites)
2. [Dataset](#Dataset)
3. [Train](#Train)
4. [Test](#Test)
5. [Performance](#Performance)
6. [Model](#Model)

---

### Prerequisites
- Pytorch 1.1.0 + cuda 10.0
- You need to first install two repositories, [DCN_v2](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) and [MSSSIM](https://github.com/jorge-pessoa/pytorch-msssim), in the './model' directory, following their installation instructions respectively.
### Dataset
Download [GoPro]((https://seungjunnah.github.io/Datasets/gopro.html)) datasets and algin the blurry/sharp image pairs.
Organize the dataset in the following form:

```bash
|- Gopro_align_data 
|   |- train  % 2103 image pairs
|   |   |- GOPR0372_07_00_000047.png
|   |   |- ...
|   |- test   % 1111 image pairs
|   |   |- GOPR0384_11_00_000001.png
|   |   |- ...
```

### Training 
- To train motion offset estimation model, run the following command:
```bash
sh run_train.sh
```
Note that you can replace the argument ```offset_mode``` from ```lin/bilin/quad``` to decide the constraint of the estimated trajectory as ```linear/bi-linear/quadratic```

- To train the deblurring model, run the same command and change the argument ```blur_direction``` to ```deblur```


### Test
- To train motion offset estimation model, run the following command:
```bash
sh run_test.sh
```
- To train the deblurring model, run the same command and change the argument ```blur_direction``` to ```deblur```


### Performance
We provide some examples of our quadratic exposure trajectory and the cooresponding reblurred images.
<img src= "https://github.com/chosj95/MIMO-UNet/blob/main/img/Graph.jpg" width="50%">