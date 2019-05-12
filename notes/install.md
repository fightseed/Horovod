# 安装 
## 一、主机安装
### 1.安装NCCL 2
在大多数情况下，使用NCCL 2将显着提高GPU版本的性能。NCCL 2提供针对NVIDIA GPU和各种网络设备（如RoCE或InfiniBand）优化的allreduce操作。  
#### ===>>> deb安装
(1)下载  
注册[NVIDIA Developer](https://developer.nvidia.com/developer-program),然后[下载](https://developer.nvidia.com/nccl)NCCL 2.x。   

（2）安装  
Installing NCCL on Ubuntu requires you to first add a repository to the APT system containing the NCCL packages, then installing the NCCL packages through APT. There are two repositories available; a local repository and a network repository. Choosing the later is recommended to easily retrieve upgrades when newer versions are posted.   
* Install the repository.
    * For the local NCCL repository:   
        `sudo dpkg -i nccl-repo-<version>.deb`
    * For the network repository:   
        `sudo dpkg -i nvidia-machine-learning-repo-<version>.deb`
* Update the APT database:   
`sudo apt update`
* Install the libnccl2 package with APT. Additionally, if you need to compile applications with NCCL, you can install the libnccl-dev package as well:  
**Note:** If you are using the network repository, the following command will upgrade CUDA to the latest version.   
`sudo apt install libnccl2 libnccl-dev`  
If you prefer to keep an older version of CUDA, specify a specific version, for example:   
`sudo apt install libnccl2=2.0.0-1+cuda8.0 libnccl-dev=2.0.0-1+cuda8.0`
#### ===>>> txz安装
(1)下载  
注册[NVIDIA Developer](https://developer.nvidia.com/developer-program),然后[下载](https://developer.nvidia.com/nccl)NCCL 2.x。     
(2） 安装   

Extract the NCCL package to your home directory or in /usr/local if installed as root for all users:
```
# cd /usr/local 
# tar xvf nccl-<version>.txz
```
When compiling applications, specify the directory path to where you installed NCCL, for example `/usr/local/nccl-<version>/`.
#### ===>>>[编译安装](https://github.com/NVIDIA/nccl)
(1)git clone & build & install
```
$ git clone https://github.com/NVIDIA/nccl

$ cd nccl
$ make -j src.build

$ # Install tools to create debian packages
$ sudo apt install build-essential devscripts debhelper
$ # Build NCCL deb package
$ make pkg.debian.build
$ ls build/pkg/deb/


```
#### ===>>> 验证
```
$ git clone https://github.com/NVIDIA/nccl-tests.git
$ cd nccl-tests
$ make
$ ./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```
### 2.安装[GPUDirect](https://developer.nvidia.com/gpudirect)(Optional)
If you're using an NVIDIA Tesla GPU and NIC with GPUDirect RDMA support, you can further speed up NCCL 2 by installing an [nv_peer_memory driver](https://github.com/Mellanox/nv_peer_memory).    
  
GPUDirect allows GPUs to transfer memory among each other without CPU involvement, which significantly reduces latency and load on CPU. NCCL 2 is able to use GPUDirect automatically for allreduce operation if it detects it.   
  
安装教程：https://github.com/Mellanox/nv_peer_memory   
### 3.安装[Open MPI](https://www.open-mpi.org/)
```
shell$ gunzip -c openmpi-4.0.1.tar.gz | tar xf -
shell$ cd openmpi-4.0.1
shell$ ./configure --prefix=/usr/local
<...lots of output...>
shell$ make all install
```
Note: Open MPI 3.1.3 has an issue that may cause hangs. It is recommended to downgrade to Open MPI 3.1.2 or upgrade to Open MPI 4.0.0.   
### 4. Horovod（with pip）
If you have installed NCCL 2 using the nccl-<version>.txz package, you should specify the path to NCCL 2 using the HOROVOD_NCCL_HOME environment variable.
```
$ HOROVOD_NCCL_HOME=/usr/local/nccl-<version> HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod
```
If you have installed NCCL 2 using the Ubuntu package, you can simply run:
```
$ HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod
```
Note: 即使GPU版本可用，一些具有高计算通信率的模型也可以在CPU上进行allreduce。要强制allreduce在CPU上发生，请传递device_dense='/cpu:0'给hvd.DistributedOptimizer：
```
opt = hvd.DistributedOptimizer(opt, device_dense='/cpu:0')
```

#### 高级：拥有专有的,针对GPU的网络优化的 MPI实施？(Have a proprietary MPI implementation with GPU support optimized for your network?)
仅当您拥有支持GPU的专有MPI实现（即不是Open MPI或MPICH）时，此部分才有意义。大多数用户应遵循以上部分之一。

如果您的MPI供应商在GPU 上执行allreduce操作的速度比NCCL 2快，您可以将Horovod配置为使用它：
```
$ HOROVOD_GPU_ALLREDUCE=MPI pip install --no-cache-dir horovod
```
此外，如果您的MPI供应商的实现支持GPU上的allgather和广播操作，您可以配置Horovod以使用它们：
```
$ HOROVOD_GPU_ALLREDUCE=MPI HOROVOD_GPU_ALLGATHER=MPI HOROVOD_GPU_BROADCAST=MPI pip install --no-cache-dir horovod
```
注意：Allgather分配输出张量，该输出张量与参与培训的进程数成比例。如果您发现自己的GPU内存不足，可以通过传递device_sparse='/cpu:0'到hvd.DistributedOptimizer以下命令强制allgather在CPU上发生 ：(Allgather allocates an output tensor which is proportionate to the number of processes participating in the training. If you find yourself running out of GPU memory, you can force allgather to happen on CPU by passing device_sparse='/cpu:0' to hvd.DistributedOptimizer:)
```
opt = hvd.DistributedOptimizer（opt，device_sparse = ' / cpu：0 '）
```
## 二、Docker安装
### 1.参考Dockerfile
```
FROM nvidia/cuda:9.0-devel-ubuntu16.04

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV TENSORFLOW_VERSION=1.12.0
ENV PYTORCH_VERSION=1.0.0
ENV CUDNN_VERSION=7.4.1.5-1+cuda9.0
ENV NCCL_VERSION=2.3.7-1+cuda9.0
ENV MXNET_URL=https://s3-us-west-2.amazonaws.com/mxnet-python-packages-gcc5/mxnet_cu90_gcc5-1.4.0-py2.py3-none-manylinux1_x86_64.whl

# Python 2.7 or 3.5 is supported by Ubuntu Xenial out of the box
ARG python=2.7
ENV PYTHON_VERSION=${python}

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install TensorFlow, Keras, PyTorch and MXNet
RUN pip install 'numpy<1.15.0' tensorflow-gpu==${TENSORFLOW_VERSION} keras h5py torch==${PYTORCH_VERSION} torchvision ${MXNET_URL}

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 pip install --no-cache-dir horovod && \
    ldconfig

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Download examples
RUN apt-get install -y --no-install-recommends subversion && \
    svn checkout https://github.com/horovod/horovod/trunk/examples && \
    rm -rf /examples/.svn

WORKDIR "/examples"
```
使用Dockerfile 构建：   
```
$ mkdir horovod-docker
$ wget -O horovod-docker/Dockerfile https://raw.githubusercontent.com/horovod/horovod/master/Dockerfile
$ docker build -t horovod:latest horovod-docker
```

### 2.DockerHub
* [Horovod DockerHub](https://hub.docker.com/r/horovod/horovod)   
* [自己的镜像](https://cloud.docker.com/repository/docker/fusimeng/ai.horovod)   
  
 