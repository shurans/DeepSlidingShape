* Compile code
    1. Download CUDA 7.5 and cuDNN 3. You will need to register with NVIDIA.
    2. cd code/marvin
    3. ./linux.sh

* Prepare data 
    download the processed RGBD data :
         http://dss.cs.princeton.edu/Release/sunrgbd_dss_data   
         by runing script:
         downloadData('../sunrgbd_dss_data','http://dss.cs.princeton.edu/Release/sunrgbd_dss_data/','.bin');
    or run dss_preparedata() to prepare your own data.

    download the image and hha images:
        http://dss.cs.princeton.edu/Release/hha
        http://dss.cs.princeton.edu/Release/image
        by runing script:
        downloadData('../image','http://dss.cs.princeton.edu/Release/image/','.tensor');
    
* 3D regoin proposal network:
    1. You can download the precomputed regoin proposal for NYU and SUNRGBD dataset from:
       http://dss.cs.princeton.edu/Release/result/proposal/RPN_SUNRGBD/
       http://dss.cs.princeton.edu/Release/result/proposal/RPN_NYU/
       by runing script:
       downloadData('../image','http://dss.cs.princeton.edu/Release/image/','.tensor');

    2. To train 3D regoin proposal network and extract 3D regoin proposal
    cd code/matlab_code/slidingAnchor
    run dss_prepareAnchorbox() to prepare training data.
    run RPN_extract() to extract 3D regoin proposal.

* 3D object detection network: 
    1. change path in dss_initPath.m;
    2. run dss_marvin_script(0,100,1,[]  ,1,'RPN_NYU',1,[],0,0);
    Pretrained model:     
    http://dss.cs.princeton.edu/Release/pretrainedModels/DSS/ORN/DSSnet_ORN_d.marvin

Note :
    If matlab system call fails, you can try to run the command directly.
   