# Motion-ETR
Exposure Trajectory Recovery from Motion Blur
Youjian Zhang, Chaoyue Wang, Dacheng Tao

## How to run

### Prerequisites
- Pytorch 1.1.0 + cuda 10.0
- You need to first install two repositories, [DCN_v2](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) and [MSSSIM](https://github.com/jorge-pessoa/pytorch-msssim), in the './model' directory, following their installation instructions.
### Dataset
Download GoPro datasets and algin the blurry/sharp image pairs.
Organize the dataset in the following form:

```bash
- Gopro_align_data 
    - train
        - GOPR0372_07_00_000047.png
        - ...
    - test
        - GOPR0384_11_00_000001.png
        - ...
```

### train
```bash
sh run_train.sh
```

### test
```bash
sh run_test.sh
```