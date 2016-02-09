rm marvin

#g++ -std=c++11 -Wfatal-errors -O3 -o marvin marvin.cpp -I. -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcublas -lcudnn

export PATH=$PATH:/usr/local/cuda/bin

nvcc -std=c++11 -O3 -o marvin marvin.cu -I. -I/usr/local/cudnn/v3/include/ -I/usr/local/cuda/include -L/usr/local/cudnn/v3/lib64 -L/usr/local/cuda/lib64 -lcudart -lcublas -lcudnn -lGLU -lOSMesa -I/opt/X11/include -L/opt/X11/lib


export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64


#wget https://www.dropbox.com/s/ed7azxjo36dp72c/marvin.hpp?dl=0 -O marvin.hpp

#wget https://www.dropbox.com/s/g5c64lbyd4259pw/marvin.hpp?dl=0 -O marvin.hpp

#g++ -std=c++11 -Wall -Wno-sign-compare -Wno-unused-result -Wno-reorder -g -O0 -o marvin marvin.cpp -I. -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcublas -lcudnn


# clang++ -std=c++11 -stdlib=libc++ -Wfatal-errors -O3 -o marvin marvin.cpp -I. -I/usr/local/cuda/include -I./cudnn-6.5-linux-x64-v2 -L/usr/local/cuda/lib -L./cudnn-6.5-linux-x64-v2 -lcudart -lcublas -lcudnn -stdlib=libc++
# export LD_LIBRARY_PATH=/usr/local/cuda/lib:./cudnn-6.5-osx-v2


# export LD_LIBRARY_PATH=/usr/local/cuda/lib:./cudnn-6.5-osx-v2

# export LD_LIBRARY_PATH=/usr/local/cuda/lib:./cudnn-7.0-osx-x64-v3.0-rc/lib


#g++ -Wfatal-errors -o marvin marvin.cpp ./cudnn-6.5-osx-v2/libcudnn_static.a -I. -I/usr/local/cuda/include -I./cudnn-6.5-osx-v2 -L/usr/local/cuda/lib -lcudart -lcublas -lm -lstdc++

#g++ -v -Wfatal-errors -O3 -o marvin marvin.cpp -I. -I/usr/local/cuda/include -I./cudnn-6.5-osx-v2 -L/usr/local/cuda/lib -L./cudnn-6.5-osx-v2 -lcudart -lcublas -lcudnn

#clang++ -std=c++11 -stdlib=libc++ -Weverything -O3 -o marvin marvin.cpp -I. -I/usr/local/cuda/include -I./cudnn-6.5-osx-v2 -L/usr/local/cuda/lib -L./cudnn-6.5-osx-v2 -lcudart -lcublas -lcudnn -stdlib=libc++


#g++ -v -Wfatal-errors -O3 -o marvin marvin.cpp -I. -I/usr/local/cuda/include -I./cudnn-7.0-osx-x64-v3.0-rc/include -L/usr/local/cuda/lib -L./cudnn-7.0-osx-x64-v3.0-rc/lib -lcudart -lcublas -lcudnn


#./marvin train mnist/lenet.json
#./marvin train DSS/debug.json
#./marvin train 3DShapeNet/3DShapeNet.json


# case 
# ./marvin train mnist/lenet.json


#./marvin train mnist/lenet.json

#./marvin train ImageNet/alexnet.json

#./marvin train ImageNet/alexnet.json ImageNet/caffe_pretrained.tensor

# ./marvin train ImageNet/alexnet.json ImageNet/caffe_pretrained.tensor 450000

#valgrind --leak-check=yes ./marvin train mnist/lenet.json

# g++ -c -Wfatal-errors -o marvin.o marvin.cpp -I. -I/usr/local/cuda/include -I./cudnn-6.5-osx-v2  -I/opt/local/include 

# gcc -v -o marvin -L/usr/local/cuda/lib -L./cudnn-6.5-osx-v2 -lcudart -lcublas -lcudnn -lm -lstdc++

# gcc -o mnistCUDNN mnistCUDNN.o -L/usr/local/cuda/lib -L/Users/jxiao/Dropbox/visiongpu/DeepND/cudnn-6.5-osx-v2 -L/Users/jxiao/Dropbox/visiongpu/DeepND/cuDNN/cudnn-sample-v2/FreeImage/lib/darwin -L/opt/local/lib -lnppi -lnppc -lfreeimage -lcudart -lcublas -lcudnn -lm -lstdc++

#./marvin test DSS/DSSnet_extractfea_gt.json fc5,fc6 DSS/fc5.tensor,DSS/fc6.tensor