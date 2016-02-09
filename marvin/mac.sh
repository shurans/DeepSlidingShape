rm marvin

#wget https://www.dropbox.com/s/ed7azxjo36dp72c/marvin.hpp?dl=0 -O marvin.hpp

#wget https://www.dropbox.com/s/g5c64lbyd4259pw/marvin.hpp?dl=0 -O marvin.hpp

#g++ -std=c++11 -Wall -Wno-sign-compare -Wno-unused-result -Wno-reorder -g -O0 -o marvin marvin.cpp -I. -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcublas -lcudnn

#g++ -std=c++11 -Wfatal-errors -O3 -o marvin marvin.cpp -I. -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcublas -lcudnn

#export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64

nvcc -std=c++11 -O3 -o marvin marvin.cu -I. -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcudart -lcublas -lcudnn
# export LD_LIBRARY_PATH=/usr/local/cuda/lib:./cudnn-6.5-osx-v2


# export LD_LIBRARY_PATH=/usr/local/cuda/lib:./cudnn-6.5-osx-v2

export LD_LIBRARY_PATH=/usr/local/cuda/lib

#g++ -Wfatal-errors -o marvin marvin.cpp ./cudnn-6.5-osx-v2/libcudnn_static.a -I. -I/usr/local/cuda/include -I./cudnn-6.5-osx-v2 -L/usr/local/cuda/lib -lcudart -lcublas -lm -lstdc++

#g++ -v -Wfatal-errors -O3 -o marvin marvin.cpp -I. -I/usr/local/cuda/include -I./cudnn-6.5-osx-v2 -L/usr/local/cuda/lib -L./cudnn-6.5-osx-v2 -lcudart -lcublas -lcudnn

#clang++ -std=c++11 -stdlib=libc++ -Weverything -O3 -o marvin marvin.cpp -I. -I/usr/local/cuda/include -I./cudnn-6.5-osx-v2 -L/usr/local/cuda/lib -L./cudnn-6.5-osx-v2 -lcudart -lcublas -lcudnn -stdlib=libc++


#g++ -v -Wfatal-errors -O3 -o marvin marvin.cpp -I. -I/usr/local/cuda/include -I./cudnn-7.0-osx-x64-v3.0-rc/include -L/usr/local/cuda/lib -L./cudnn-7.0-osx-x64-v3.0-rc/lib -lcudart -lcublas -lcudnn

# ./marvin train 3DShapeNet/3DShapeNet.json

# case 

./marvin train DSS/DSSnet.json
#./marvin train mnist/lenet.json

./marvin test mnist/lenet.json ip1,conv2 mnist/ip1.tensor,mnist/conv2.tensor


#./marvin train mnist/lenet_group_lrn_dropout_test_loss.json

#./marvin train mnist/lenet_group_lrn_dropout.json

#./marvin train ImageNet/alexnet.json

#valgrind --leak-check=yes ./marvin train mnist/lenet.json

# g++ -c -Wfatal-errors -o marvin.o marvin.cpp -I. -I/usr/local/cuda/include -I./cudnn-6.5-osx-v2  -I/opt/local/include 

# gcc -v -o marvin -L/usr/local/cuda/lib -L./cudnn-6.5-osx-v2 -lcudart -lcublas -lcudnn -lm -lstdc++

# gcc -o mnistCUDNN mnistCUDNN.o -L/usr/local/cuda/lib -L/Users/jxiao/Dropbox/visiongpu/DeepND/cudnn-6.5-osx-v2 -L/Users/jxiao/Dropbox/visiongpu/DeepND/cuDNN/cudnn-sample-v2/FreeImage/lib/darwin -L/opt/local/lib -lnppi -lnppc -lfreeimage -lcudart -lcublas -lcudnn -lm -lstdc++

