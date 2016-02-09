export PATH=$PATH:/usr/local/cuda/bin

nvcc -std=c++11 -O0 -g -o marvin marvin.cu -I. -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcublas -lcudnn -lGLU -lOSMesa -I/opt/X11/include -L/opt/X11/lib


export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64

