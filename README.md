# ProcTHOR scene reconstruction

This repository implements a [LangSplat](https://github.com/minghanqin/LangSplat) reconstruction of houses from the [ProcTHOR-10k dataset](https://github.com/allenai/procthor-10k). Additionally it will be explained how to use LangSplat to reconstruct a scene from a video.
This is specifically explained for the system I ran it on.

## Hardware used

- Ubuntu 24.04 LTS
- Nvidia GeForce GTX 1070 with 8 GB of VRAM

## Software used

- Conda
- CUDA Toolkit 11.8
- text

# Setup

I created 4 conda environments
1. colmap
2. gaussian-splatting. Containing the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) project.
3. langsplat. Containing LangSplat itself in addition to [Segment-Anything-LangSplat](https://github.com/minghanqin/segment-anything-langsplat).
4. procthor. Containing the packages necessary to import, use, and modify ProcTHOR environments.

## Installing COLMAP

Something with installing an older gcc and g++ version.

``` shell
conda create -n colmap -c conda-forge colmap
```

## Installing 3D Gaussian Splatting

Following the instructions on the 3D Gaussian Splatting GitHub:
Clone the repository
```shell
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```
Then create the conda environment
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```

## Installing LangSplat

Following the instructions on the LangSplat GitHub:
Clone the repository
```shell
git clone https://github.com/minghanqin/LangSplat.git --recursive
```
Then create the conda environment
```shell
conda env create --file environment.yml
conda activate langsplat
```
Additionally install Segment-Anything-Langsplat like this
```
pip install git+https://github.com/minghanqin/segment-anything-langsplat.git
```
or clone the repository locally and install with (Probably did not do this..?)
```
git clone https://github.com/minghanqin/segment-anything-langsplat.git
cd segment-anything; pip install -e .
```

## Installing ProcTHOR

hmm

## Additional setup

Locate the package "ftfy" in the (?) environment and change line (?) from
```
from typing import Literal
```
to
```
try:
	from typing import Literal
except ImportError:
	from typing_extensions import Literal
```

Add a folder "ckpts" under the LangSplat folder, download a SAM checkpoint from [this link](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) and save it under the ckpts folder.
ViT-H is larger than ViT-L, which is larger than ViT-B. I used ViT-B due to my GPU.

# Images from videos

Using [FFmpeg](https://ffmpeg.org/), frames of a video can be extracted.
Simply run
```
ffmpeg -i <Video location> -r <Framerate> %05d.png
```
This will create \<Framerate> images per second from the video and save them in the current directory.

# Images from ProcTHOR

Getting images from a ProcTHOR house is done using the included "Images_from_ProcTHOR.ipynb" notebook. This is run in the "procthor" environment.

# Running LangSplat

Once the images have been acquired put them in a folder like this
```
<dataset name>
|	input
|	|	<image 0>
|	|	<image 1>
|	|	...
```
**Step 1:** Run COLMAP on that dataset in the "colmap" environment in the LangSplat folder
```
python convert.py -s <dataset location>
```
I put all datasets into a folder under the LangSplat folder called "datasets".

**Step 2:** Run 3D Gaussian Splatting on the dataset in the "gaussian-splatting" environment in the Gaussian Splatting folder
```
conda activate gaussian_splatting	#3DGS environment name
cd <gaussian splatting folder>
python train.py -s <dataset location> --checkpoint_iteration 30000 --eval
```
The output from this will be under ```gaussian_splatting/output```. The outputted folder can then be renamed into "\<dataset name>dat" (i.e. if the dataset is called proc2, then rename this folder into proc2dat) and moved into the dataset folder.

**Step 3**: Run LangSplat preprocessing on the dataset in the "langsplat" environment in the LangSplat folder.
```
conda activate langsplat	#LangSplat environment name
cd <langsplat folder>
```
If using ViT-H simply do
```
python preprocess.py --dataset_path <dataset location>
```
Otherwise open the "preprocess.py" file and change line 362 to vit_l or vit_b depending on which is used. Then do
```
python preprocess.py --dataset_path <dataset location> --sam_ckpt_path ckpts/<sam checkpoint file name>
```
Additionally, if this still causes problems with the GPU also do
```
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
```
before running the preprocess again.

**Step 4:** Run the autoencoder
If the dataset is called "proc2", then \<dataset name>dat means "proc2dat".
```
cd autoencoder
python train.py --dataset_path <dataset location> --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name <dataset name>dat
python test.py --dataset_path <dataset location> --dataset_name <dataset name>dat
```

**Step 5:** Train LangSplat
```
cd ..
python train.py -s <dataset location> -m output/<dataset name>dat --start_checkpoint <dataset location>/<dataset name>dat/chkpnt30000.pth --feature_level <level>
```
\<level> can be "1", "2", or "3", defining different levels of segmentation. For the full LangSplat output all three need to be run.

**Step 6:** Render that
```
python render.py -m output/<dataset name>dat_<level> --include feature
```
\<level> is as above.

**Step 7:** Evaluation
If there are ground truth masks and bounding boxes given, as is the case with ProcTHOR houses, the reconstruction can be evaluated.
First the file ```eval/eval.sh``` needs to be changed.
In line 2: ```CASE_NAME="<dataset name>dat"```
In line 5: ```gt_folder="<absolute path to where the bounding box and mask folders are stored>"```
Then
```
cd eval
sh eval.sh
```

## Example

With a dataset
```
proc2
|	input
|	|	<image 0>
|	|	<image 1>
|	|	...
```
With folder structure
```
Documents
|	LangSplat
|	|	datasets
|	|	|	proc2
|	|	|	|	input
|	|	|	|	|	<images>
|	|	|	proc2bboxes
|	|	|	proc2instances
|	|	ckpts
|	|	|	sam_vit_b_01ec64.pth
|	|	output
|	gaussian-splatting
|	|	output
```
Starting in "Documents/LangSplat" in "colmap" environment:
```
#Step 1
python convert.py -s datasets/proc2

#Step 2
conda activate gaussian_splatting
cd ..
cd gaussian-splatting
python train.py -s ../LangSplat/datasets/proc2 --checkpoint_iteration 30000 --eval
```
This then outputs a folder with some name under ```gaussian-splatting/output``` which we will move to here
```
Documents
|	LangSplat
|	|	datasets
|	|	|	proc2
|	|	|	|	input
|	|	|	|	|	<images>
|	|	|	|	proc2dat	#The moved folder
|	|	|	proc2bboxes
|	|	|	proc2instances
|	|	ckpts
|	|	|	sam_vit_b_01ec64.pth
|	|	output
```
Continuing
```
#Step 3
conda activate langsplat
cd ..
cd LangSplat
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
python preprocess.py --dataset_path datasets/proc2 --sam_ckpt_path ckpts/sam_vit_b_01ec64.pth

#Step 4
cd autoencoder
python train.py --dataset_path datasets/proc2 --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name proc2dat
python test.py --dataset_path datasets/proc2 --dataset_name proc2dat

#Step 5
cd ..
python train.py -s datasets/proc2 -m output/proc2dat --start_checkpoint datasets/proc2/proc2dat/chkpnt30000.pth --feature_level 3

#Step 6
python render.py -m output/proc2dat_3 --include_feature
```
Step 7
Change eval.sh:
Line 2: ```CASE_NAME="proc2dat"```
Line 5: ```gt_folder="<path to Documents>/LangSplat/datasets"```
Then run
```
cd eval
sh eval.sh
```
