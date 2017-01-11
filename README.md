## Deep Sliding Shapes for Amodal 3D Object Detection in RGB-D Images
S. Song, and J. Xiao. (CVPR2016)

### Compile code
Download CUDA 7.5 and cuDNN 3. You will need to register with NVIDIA.
```shell
cd code/marvin
./linux.sh
```
### Prepare data 
* download the processed RGBD data [here](http://dss.cs.princeton.edu/Release/sunrgbd_dss_data) by runing script:
     ```shell
     downloadData('../sunrgbd_dss_data','http://dss.cs.princeton.edu/Release/sunrgbd_dss_data/','.bin');
     ```
* or run dss_preparedata() to prepare your own data.

* download the image and hha images by runing script:
    downloadData('../image','http://dss.cs.princeton.edu/Release/image/','.tensor');
    downloadData('../hha','http://dss.cs.princeton.edu/Release/hha/','.tensor');
    
### 3D region proposal network:
* You can download the precomputed region proposal for [NYU](http://dss.cs.princeton.edu/Release/result/proposal/RPN_NYU/) and [SUNRGBD](http://dss.cs.princeton.edu/Release/result/proposal/RPN_SUNRGBD/) dataset from:
    by runing script:

    ```shell
    downloadData('../proposal','http://dss.cs.princeton.edu/Release/result/proposal/RPN_NYU/','.mat');
    downloadData('../proposal','http://dss.cs.princeton.edu/Release/result/proposal/RPN_SUNRGBD/','.mat');
    ```

* To train 3D region proposal network and extract 3D region proposal
       cd code/matlab_code/slidingAnchor
       run dss_prepareAnchorbox() to prepare training data.
       run RPN_extract() to extract 3D region proposal.
       You may need the segmentation result here:

    ```shell
    downloadData('../seg','http://dss.cs.princeton.edu/Release/seg/','.mat');
    ```
* Pretrained model and network defination can be found [here](http://dss.cs.princeton.edu/Release/pretrainedModels/DSS/RPN/multi_dpcv1/)
    

### 3D object detection network: 
1. change path in dss_initPath.m;
2. run dss_marvin_script(0,100,1,[]  ,1,'RPN_NYU',1,[],0,0);
3. Pretrained model and network defination can be found [here](http://dss.cs.princeton.edu/Release/pretrainedModels/DSS/ORN/)
    
### Notes :
* If matlab system call fails, you can try to run the command directly.
* The rotation matrixes for some of the images in the dataset are different from the original SUNRGB-D dataset,  so that the rotation only contains camera tilt angle (i.e. point cloud does not rotated on the x,y plane). We provide the data in this repo ```./external/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat```. All the results and ground truth boxes provided in this repo are using this rotation matrix. To convert the rotation matrix you can reference the code "changeRoomR.m"
   
