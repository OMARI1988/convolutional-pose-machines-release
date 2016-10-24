# Convolutional Pose Machines
Shih-En Wei, Varun Ramakrishna, Takeo Kanade, Yaser Sheikh, "[Convolutional Pose Machines](http://arxiv.org/abs/1602.00134)", CVPR 2016.

This project is licensed under the terms of the GPL v2 license. By using the software, you are agreeing to the terms of the [license agreement](https://github.com/shihenw/convolutional-pose-machines-release/blob/master/LICENSE.md).

Contact: Muhannad Alomari, scmara@leeds.ac.uk.

## Before everything, the installation process will take about an hour.
- `mkdir ~/sk_cpm`
- `cd ~/sk_cpm`
- `git clone https://github.com/OMARI1988/convolutional-pose-machines-release.git`
- `git clone https://github.com/OMARI1988/caffe.git`

## Installing CUDA 8.0
- download cuda 8.0 from https://developer.nvidia.com/cuda-downloads, make sure to choose linux, x86_64, Ubuntu, 14.04, deb(local)
- `cd ~/wherever you installed cuda`
- `sudo dpkg -i cuda-repo-ubuntu1404_8-0-local_8.0.44-1_amd64.deb`
- `sudo apt-get update`
- `sudo apt-get install cuda`
- As part of the CUDA environment, you should add the following in the .bashrc file of your home folder.
```
export CUDA_HOME=/usr/local/cuda-8.0 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 
PATH=${CUDA_HOME}/bin:${PATH} 
export PATH
```

## installing cudNN
- you have to get cudNN (to-do, put the google drive link)
- `sudo pip install protobuf`
- `sudo pip install configobj`
- `sudo pip install --upgrade IPython`

## Building caffe
you have to get cudNN and sudo pip install protobuf and sudo pip install configobj and sudo pip install --upgrade IPython

- `cd ~/sk_cmp/caffe`
- `mkdir build`
- `cd build`
- `cmake ..`
- `make all`
- `make install`
- `make runtest`

## Pythonpath for cuda
- add python path of caffe to bashrc
- `export PYTHONPATH=$PYTHONPATH:"~/sk_cnn/caffe/python/"`

## Testing ConvNet and Download models
- `cd ~/sk_cpm/convolutional-pose-machines-release`
- `chmod u+x get_data.sh`
- `./get_data.sh`
- `cd testing`
- `./get_models.sh`
- `cd python`
- `python test.py`
