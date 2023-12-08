<div align="center">

<h1>PrimDiffusion: Volumetric Primitives Diffusion for 3D Human Generation</h1>

<div>
    <a href='https://frozenburning.github.io/' target='_blank'>Zhaoxi Chen<sup>1</sup></a>&emsp;
    <a href='https://hongfz16.github.io/' target='_blank'>Fangzhou Hong<sup>1</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=TOZ9wR4AAAAJ&hl=en' target='_blank'>Haiyi Mei<sup>2</sup></a>&emsp;
    <a href='https://wanggcong.github.io/' target='_blank'>Guangcong Wang<sup>1</sup></a>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=jZH2IPYAAAAJ&hl=en' target='_blank'>Lei Yang<sup>2</sup></a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu<sup>1</sup></a>
</div>
<div>
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp; <sup>2</sup> Sensetime Research 
</div>
<div>
    <a href='https://nips.cc/Conferences/2023'>NeurIPS 2023</a>
</div>
<div>

<a target="_blank" href="https://arxiv.org/abs/2312.04559">
  <img src="https://img.shields.io/badge/arXiv-2312.04559-b31b1b.svg" alt="arXiv Paper"/>
</a>
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FFrozenBurning%2FPrimDiffusion&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</div>


<h4>TL;DR</h4>
<h5>PrimDiffusion generates 3D human by denoising a set of volumetric primitives. <br> Our method enables explicit pose, view and shape control with real-time rendering in high resolution.</h5>

### [Paper](https://arxiv.org/abs/2312.04559) | [Project Page](https://frozenburning.github.io/projects/primdiffusion) | [Video](https://youtu.be/zprHGZ7Gm7A)

<br>


<tr>
    <img src="./assets/teaser.gif" width="100%"/>
</tr>

</div>

## Updates
[12/2023] Source code released! :star_struck:

[09/2023] PrimDiffusion has been accepted to [NeurIPS 2023](https://nips.cc/Conferences/2023)! :partying_face:

## Citation
If you find our work useful for your research, please consider citing this paper:
```
@inproceedings{
chen2023primdiffusion,
title={PrimDiffusion: Volumetric Primitives Diffusion for 3D Human Generation},
author={Zhaoxi Chen and Fangzhou Hong and Haiyi Mei and Guangcong Wang and Lei Yang and Ziwei Liu},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023}
}
```

## Installation
We highly recommend using [Anaconda](https://www.anaconda.com/) to manage your python environment. You can setup the required environment by the following commands:
```bash
# clone this repo
git clone https://github.com/FrozenBurning/PrimDiffusion
cd PrimDiffusion

# install python dependencies
conda env create -f environment.yaml
conda activate primdiffusion
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
```

Build raymarching extensions:
```bash
cd dva
git clone https://github.com/facebookresearch/mvp
cd mvp/extensions/mvpraymarch
make -j4
```
Install Easymocap:
```bash
git clone https://github.com/zju3dv/EasyMocap
cd EasyMocap
pip install --user .
```
Install xformers for speedup (Optional): Please refer to the official repo for [installation](https://github.com/facebookresearch/xformers).


## Inference

### Download Pretrained Models
Download sample data, necessary assets, and pretrained model from [Google Drive](https://drive.google.com/drive/folders/1kiFsQyE1ycoaT0nlsFajrwLFk47_JSJZ?usp=sharing). [![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=yellow)](https://drive.google.com/drive/folders/1kiFsQyE1ycoaT0nlsFajrwLFk47_JSJZ?usp=sharing)

Register and download SMPL models [here](https://smpl.is.tue.mpg.de/). Please store the SMPL model together with downloaded files as follows:
```
├── ...
└── PrimDiffusion
    ├── visualize.py
    ├── README.md
    └── data
        └──checkpoints
            └── primdiffusion.pt
        └──smpl
            ├── basicModel_ft.npy
            ├── basicModel_vt.npy
            └── SMPL_NEUTRAL.pkl
        └──render_people
    ...
```

### Visualize Denoising Process and Novel Views
You can run the following script for generating 3D human with PrimDiffusion:
```bash
python visualize.py configs/primdiffusion_inference.yml ddim=False
```
Please specify the path to the pretrained model as `checkpoint_path` in the config file. Moreover, please specify `ddim=True` if you intend to use 100 steps DDIM sampler. The script will render and save videos under `output_dir` which is specified by the config file.

## Training

### Data Preparation
You could refer to the downloaded sample data at `./data/render_people` to prepare your own multiview dataset, and modify the corresponding path in the config file.

### Stage I Training 
```bash
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6666 train_stage1.py configs/renderpeople_stage1_fitting.yml
```
This will create a folder with checkpoints, config and a monitoring image at the `output_dir` specified in config file.

### Stage II Training
Please run the following command to launch the training of the diffusion model. Please set `pretrained_encoder` to the path of the latest checkpoint from Stage I. We also support training with mixed precision by default, please modify `train.amp` in the config file according to your usage.
```bash
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6666 train_stage2.py configs/renderpeople_stage2_primdiffusion.yml
```
Note that, we use 8 GPUs for training by default. Please adjust `--nproc_per_node` to the number you want. 



## License

Distributed under the S-Lab License. See [LICENSE](./LICENSE) for more information. Part of the code are also subject to the [LICENSE of DVA](https://github.com/facebookresearch/dva/blob/main/LICENSE).

## Acknowledgements
PrimDiffusion is implemented on top of the [DVA](https://github.com/facebookresearch/dva) and [Latent-Diffusion](https://github.com/CompVis/latent-diffusion). 
