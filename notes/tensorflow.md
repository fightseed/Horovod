# Horovod在tensorflow上运行   
## 一、准备环境  
### 1.[镜像](https://github.com/fusimeng/Docker)  
使用官方镜像，修改后，  
`fusimeng/ai.horovod:v1`    
### 2.容器   
启动镜像（一机多卡）  
`$ nvidia-docker run -it horovod：latest`  
`root@c278c88dd552/examples#horovodrun -np 4 -H localhost:4 python keras_mnist_advanced.py  `   

启动镜像（多机多卡）  
Primary worker:  
`host1$ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh horovod:latest`  
`root@c278c88dd552:/examples# horovodrun -np 16 -H host1:4,host2:4,host3:4,host4:4 -p 12345 python keras_mnist_advanced.py`  
要设置要免密登录  
Secondary workers:  
`host2$ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh horovod:latest \
    bash -c "/usr/sbin/sshd -p 12345; sleep infinity"`




