# SSE-xMUDA
 Self-supervised Exclusive Learning for 3D Segmentation with Cross-Modal Unsupervised Domain Adaptation

## Preparation refer to [xMUDA](https://github.com/valeoai/xmuda)
### Prerequisites
Tested with
* PyTorch 1.4
* CUDA 10.0
* Python 3.8
* [SparseConvNet](https://github.com/facebookresearch/SparseConvNet)
* [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

### Installation
As 3D network we use SparseConvNet. It requires to use CUDA 10.0 (it did not work with 10.1 when we tried).
We advise to create a new conda environment for installation. PyTorch and CUDA can be installed, and SparseConvNet
installed/compiled as follows:
```
$ conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
$ pip install --upgrade git+https://github.com/facebookresearch/SparseConvNet.git
```

```
$ cd xmuda
$ pip install -ve .
```
The `-e` option means that you can edit the code on the fly.

### Datasets
#### NuScenes
Please download the Full dataset (v1.0) from the [NuScenes website](https://www.nuscenes.org) and extract it.

You need to perform preprocessing to generate the data for SSE-xMUDA.

Please edit the script `xmuda/data/nuscenes/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the NuScenes dataset
* `out_dir` should point to the desired output directory to store the pickle files

## Training on Day-to-Night
You can run the training with
```
$ python xmuda/train_sse.py --cfg=configs/nuscenes/day_night/xmuda.yaml OUTPUT_DIR ./output/day_night/1118SSE
```

### SSE-xMUDA<sub>PL</sub>
After having trained the xMUDA model, generate the pseudo-labels as follows:
```
$ python xmuda/test.py --cfg=configs/nuscenes/day_night/xmuda.yaml --pselab @/model_2d_100000.pth @/model_3d_100000.pth DATASET_TARGET.TEST "('train_...',)"
```
Note that we use the last model at 100,000 steps to exclude supervision from the validation set by picking the best
weights. The pseudo labels and maximum probabilities are saved as `.npy` file.
## Training on pseudo labels
```
$ python xmuda/train_sse.py --cfg=configs/nuscenes/day_night/xmuda_pl.yaml
```

## Testing
You can provide which checkpoints you want to use for testing. We used the ones
that performed best on the validation set during training (the best val iteration for 2D and 3D is
shown at the end of each training). Note that `@` will be replaced
by the output directory for that config file. For example:
```
$ python xmuda/test.py --cfg=configs/nuscenes/day_night/xmuda.yaml @/model_2d_065000.pth @/model_3d_095000.pth
```
You can also provide an absolute path without `@`. 





### Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{zhang2022self,
        title={Self-supervised Exclusive Learning for 3D Segmentation with Cross-Modal Unsupervised Domain Adaptation},
        author={Zhang, Yachao and Li, Miaoyu and Xie, Yuan and Li, Cuihua and Wang, Cong and Zhang, Zhizhong and Qu, Yanyun},
        booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
        pages={3338--3346},
        year={2022}
    }
### Acknowledgment
Note that this code is heavily borrowed from [xMUDA](https://github.com/valeoai/xmuda).
