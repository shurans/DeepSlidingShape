#include "render_3d.hpp"

int sum(std::vector<int> dim){
  return std::accumulate(dim.begin(), dim.end(), 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////// 
// Scene3D Class
////////////////////////////////////////////////////////////////////////////////////////////////// 
unsigned long long get_timestamp_dss(){
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (unsigned long long)now.tv_sec * 1000000;
};


struct ImgObjInd{
  int ImageId;
  int ObjId; 
  int TarId;
  ImgObjInd(int i,int j,int k){
    ImageId = i;
    ObjId = j;
    TarId = k;
  }
};
struct RGBDpixel{
  uint8_t R;
  uint8_t G;
  uint8_t B;
  uint8_t D;
  uint8_t D_;
};
struct Box3D{
  unsigned int category;
  float base[9];
  float center[3];
  float coeff[3];
};

struct Box2D{
  unsigned int category;
  float tblr[4];
};

struct Box3Ddiff{
  float diff[6];
  float diff2d[4];
  int oreintation;
};


void __global__ compute_xyzkernel(float * XYZimage, float * Depthimage, float * K, float * R){
            int ix = blockIdx.x;
            int iy = threadIdx.x;
            int height = blockDim.x;
            //float depth = float(*((uint16_t*)(&(RGBDimage[iy + ix * height].D))))/1000.0;
            float depth = Depthimage[iy + ix * height];            
            // project the depth point to 3d
            float tdx = (float(ix + 1) - K[2]) * depth / K[0];
            float tdz =  - (float(iy + 1) - K[5]) * depth / K[4];
            float tdy = depth;
            
            XYZimage[3 * (iy + ix * height) + 0] = R[0] * tdx + R[1] * tdy + R[2] * tdz;
            XYZimage[3 * (iy + ix * height) + 1] = R[3] * tdx + R[4] * tdy + R[5] * tdz;
            XYZimage[3 * (iy + ix * height) + 2] = R[6] * tdx + R[7] * tdy + R[8] * tdz;
}


void __global__ compute_xyzkernel(float * XYZimage, RGBDpixel * RGBDimage, float * K, float * R){
            int ix = blockIdx.x;
            int iy = threadIdx.x;
            int height = blockDim.x;
            //
            //float depth = float(*((uint16_t*)(&(RGBDimage[iy + ix * height].D))))/1000.0;
            uint16_t D = (uint16_t)RGBDimage[iy + ix * height].D;
            uint16_t D_ = (uint16_t)RGBDimage[iy + ix * height].D_;
            D_ = D_<<8;
            float depth = float(D|D_)/1000.0;
            //printf("%d,%d,%f\n",RGBDimage[iy + ix * height].D,D_,depth);
            
            // project the depth point to 3d
            float tdx = (float(ix + 1) - K[2]) * depth / K[0];
            float tdz =  - (float(iy + 1) - K[5]) * depth / K[4];
            float tdy = depth;

            XYZimage[3 * (iy + ix * height) + 0] = R[0] * tdx + R[1] * tdy + R[2] * tdz;
            XYZimage[3 * (iy + ix * height) + 1] = R[3] * tdx + R[4] * tdy + R[5] * tdz;
            XYZimage[3 * (iy + ix * height) + 2] = R[6] * tdx + R[7] * tdy + R[8] * tdz;

}
void __global__ fillInBeIndexFull(unsigned int* beIndexFull, unsigned int* beIndex, unsigned int* beLinIdx, unsigned int len_beLinIdx){
     const int index = threadIdx.x + blockIdx.x * blockDim.x;
     if (index>=len_beLinIdx) {
        return;
     }
     else{
        beIndexFull[2*beLinIdx[index]+0] =  beIndex[2*index+0];
        beIndexFull[2*beLinIdx[index]+1] =  beIndex[2*index+1];
     }
}

class sceneMesh{
public:
  std::vector<Box3D> objects;
  std::string mesh_file;
};
struct mesh_meta{
  // x : left right 
  // y : height 
  // z : depth
  std::string mesh_file;
  float base[9];// oreintation 
  float center[3];
  float coeff[3];
};

enum Scene3DType { RGBD, Render, Mesh };

class Scene3D{
public:
  // defined in .list file
  std::vector<mesh_meta> mesh_List;

  std::string filename;
  std::string seqname;

  float K[9];
  float R[9];
  unsigned int width;
  unsigned int height;
  unsigned int len_pcIndex;
  unsigned int len_beIndex;
  unsigned int len_beLinIdx;
  std::vector<Box3D> objects;
  std::vector<Box2D> objects_2d_tight;
  std::vector<Box2D> objects_2d_full;

  bool GPUdata;
  Scene3DType DataType;
  // defined in .data file
  unsigned int* grid_range;
  float* begin_range;
  float grid_delta;
  RGBDpixel* RGBDimage;
  unsigned int* beIndex;
  unsigned int* beLinIdx;
  unsigned int* pcIndex;
  float* XYZimage;
  float* K_GPU;
  float* R_GPU;

  

  //Scene3D(): RGBDimage(NULL), beIndex(NULL), pcIndex(NULL), beLinIdx(NULL),XYZimage(NULL), grid_range(NULL), begin_range(NULL),K_GPU(NULL),R_GPU(NULL),GPUdata(false),isMesh(false){};
  Scene3D(){
      RGBDimage = NULL;
      beIndex = NULL;
      pcIndex = NULL;
      beLinIdx = NULL;
      XYZimage = NULL;
      grid_range = NULL;
      begin_range = NULL;
      K_GPU = NULL;
      R_GPU = NULL;

      GPUdata = false;
      DataType = RGBD;
  };

  void compute_xyz() {
    XYZimage = new float[width*height*3];
    //printf("scene->K:%f,%f,%f\n%f,%f,%f\n%f,%f,%f\n",K[0],K[1],K[2],K[3],K[4],K[5],K[6],K[7],K[8]);
    for (int ix = 0; ix < width; ix++){
      for (int iy = 0; iy < height; iy++){
          float depth = float(*((uint16_t*)(&(RGBDimage[iy + ix * height].D))))/1000.0;
          //printf("%d,%f\n",RGBDimage[iy + ix * height].D,RGBDimage[iy + ix * height].D_,depth);
          // project the depth point to 3d
          float tdx = (float(ix + 1) - K[2]) * depth / K[0];
          float tdz =  - (float(iy + 1) - K[5]) * depth / K[4];
          float tdy = depth;

          XYZimage[3 * (iy + ix * height) + 0] = R[0] * tdx + R[1] * tdy + R[2] * tdz;
          XYZimage[3 * (iy + ix * height) + 1] = R[3] * tdx + R[4] * tdy + R[5] * tdz;
          XYZimage[3 * (iy + ix * height) + 2] = R[6] * tdx + R[7] * tdy + R[8] * tdz;
      }

    }
  };
  void compute_xyzGPU() {
      if (!GPUdata){
         std::cout<< "Data is not at GPU cannot compute_xyz at GPU"<<std::endl;
         FatalError(__LINE__);
      }
      if (XYZimage!=NULL){
         std::cout<< "XYZimage!=NULL"<<std::endl;
         FatalError(__LINE__);
      }
      checkCUDA(__LINE__, cudaMalloc(&XYZimage, sizeof(float)*width*height*3));
      compute_xyzkernel<<<width,height>>>(XYZimage,RGBDimage,K_GPU,R_GPU);
  }

  void loadData2XYZimage(){
      //enum Scene3DType { RGBD, Render, Mesh };
      switch(DataType){
        case RGBD:
            {
              this ->load();
              this -> cpu2gpu();
              this -> compute_xyzGPU();
            }
            break;
        case Mesh:
            {
              this ->loadMesh2XYZimage();
            }
            break;
        case Render:
            {
              this ->loadrender2XYZimage();
            }
            break;
        }
  };

  void loadMesh2XYZimage(){
       std::vector<Mesh3D*> mesh_models(mesh_List.size());
       for (int i = 0 ;i < mesh_List.size();++i){
            mesh_models[i] = new Mesh3D(mesh_List[i].mesh_file);
            // scale and rotate and move to  its center 
            mesh_models[i]->zeroCenter();
            float scale_ratio = mesh_models[i]->scaleMesh(mesh_List[i].coeff);
            mesh_models[i]->roateMesh(R);
            mesh_models[i]->translate(mesh_List[i].center);
       }

       float  camRT[12] ={0};
       for (int i = 0; i<3; ++i){
          for (int j = 0; j<3; ++j){
            camRT[i*4+j] = R[j*3+i];
          }
       }

       float P[12];
       getProjectionMatrix(P,K, camRT);

       float* depth =  renderDepth(mesh_models, P, width, height);

       
       // copy to GPU
       checkCUDA(__LINE__, cudaMemcpy(K_GPU, (float*)K, sizeof(float)*9, cudaMemcpyHostToDevice));
       checkCUDA(__LINE__, cudaMemcpy(R_GPU, (float*)R, sizeof(float)*9, cudaMemcpyHostToDevice)); 
       float * depth_GPU;
       checkCUDA(__LINE__, cudaMalloc(&depth_GPU, sizeof(float)*width*height));
       checkCUDA(__LINE__, cudaMemcpy(depth_GPU, (float*)depth, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 

       // compute XYZimage
       checkCUDA(__LINE__, cudaMalloc(&XYZimage, sizeof(float)*width*height*3));
       compute_xyzkernel<<<width,height>>>(XYZimage,depth_GPU,K_GPU,R_GPU);

       // free memory
       checkCUDA(__LINE__, cudaFree(depth_GPU));
       delete[] depth;
       for (int i = 0 ;i <mesh_List.size();++i){
           delete mesh_models[i];
       }
       GPUdata = true;
  };

  void loadrender2XYZimage(){
       //Tensor<float>* depth = new Tensor<float>(filename);
       std::vector<Tensor<float>*> depthRender = readTensors<float>(filename);
       float* depth = new float[width*height];
       for(int ix=0; ix<width; ix++){
          for (int iy=0;iy<height;iy++){
            depth[iy + ix * height] = depthRender[0]->CPUmem[ix + iy * width] ;
          }
          //pbufferD[i] = float( m_near / (1.0 - double(pDepthBuffer[i])/double(4294967296)) ); 
       }
       float * depth_GPU;
       //checkCUDA(__LINE__,cudaDeviceSynchronize());
       checkCUDA(__LINE__, cudaMalloc(&K_GPU, sizeof(float)*9));
       checkCUDA(__LINE__, cudaMemcpy(K_GPU, (float*)K, sizeof(float)*9, cudaMemcpyHostToDevice));

      
       checkCUDA(__LINE__, cudaMalloc(&R_GPU, sizeof(float)*9));
       checkCUDA(__LINE__, cudaMemcpy(R_GPU, (float*)R, sizeof(float)*9, cudaMemcpyHostToDevice)); 

       checkCUDA(__LINE__, cudaMalloc(&depth_GPU, sizeof(float)*width*height));
       checkCUDA(__LINE__, cudaMemcpy(depth_GPU, (float*)depth, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 
       //checkCUDA(__LINE__,cudaDeviceSynchronize());
       checkCUDA(__LINE__, cudaMalloc(&XYZimage, sizeof(float)*width*height*3));
       //checkCUDA(__LINE__,cudaDeviceSynchronize());
       

       compute_xyzkernel<<<width,height>>>(XYZimage,depth_GPU,K_GPU,R_GPU);
       //checkCUDA(__LINE__,cudaDeviceSynchronize());
       checkCUDA(__LINE__, cudaFree(depth_GPU));
       for (int i = 0 ;i <depthRender.size();++i){
           delete depthRender[i];
       }
       delete depth;
       GPUdata = true;
  };

  int load(){
    int filesize =0;
    if (RGBDimage==NULL||beIndex==NULL||pcIndex==NULL||XYZimage==NULL){
      //std::cout<< "loading image "<< filename<<std::endl;
      free();
      FILE* fp = fopen(filename.c_str(),"rb");
      if (fp==NULL) { std::cout<<"in load() :fail to open file: "<<filename<<std::endl; exit(EXIT_FAILURE); }
      grid_range = new unsigned int[3];
      filesize += fread((void*)(grid_range), sizeof(unsigned int), 3, fp);
      
      begin_range = new float[3];
      filesize += fread((void*)(begin_range), sizeof(float), 3, fp);
      filesize += fread((void*)(&grid_delta), sizeof(float), 1, fp);

      RGBDimage = new RGBDpixel[width*height];
      filesize += fread((void*)(RGBDimage), sizeof(RGBDpixel), width*height, fp);

      filesize +=  fread((void*)(&len_beIndex), sizeof(unsigned int), 1, fp);
      beIndex   = new unsigned int [len_beIndex];
      filesize += fread((void*)(beIndex), sizeof(unsigned int), len_beIndex, fp);

      filesize +=  fread((void*)(&len_beLinIdx), sizeof(unsigned int), 1, fp);
      beLinIdx  = new unsigned int [len_beLinIdx];
      filesize += fread((void*)(beLinIdx), sizeof(unsigned int), len_beLinIdx, fp);

      filesize += fread((void*)(&len_pcIndex), sizeof(unsigned int), 1, fp);
      pcIndex   = new unsigned int [len_pcIndex];
      filesize += fread((void*)(pcIndex), sizeof(unsigned int), len_pcIndex, fp);
      fclose(fp);

      GPUdata = false;
      
      //std::cout<<"size of RGBDpixel: "<<sizeof(RGBDpixel)/sizeof(unsigned char)<<std::endl;
      //std::cout<<"RGB:"<<int(RGBDimage[0].R)<<","<<(int)RGBDimage[0].G<<","<<(int)RGBDimage[0].B<<","<<(int)RGBDimage[0].D<<","<<std::endl;

      //std::cout<<len<<std::endl;
      //std::cout<<pcIndex[0]<<"-"<<pcIndex[1]<<std::endl;
      //std::cout<<len<<std::endl;
      //std::cout<<beIndex[0]<<"-"<<beIndex[1]<<std::endl;
    }
    return filesize;
  };
  void cpu2gpu(){
    if (!GPUdata){
       if (beIndex!=NULL){
           unsigned int* beIndexCPU = beIndex;
           //checkCUDA(__LINE__,cudaDeviceSynchronize());
           checkCUDA(__LINE__, cudaMalloc(&beIndex, sizeof(unsigned int)*len_beIndex));
           //checkCUDA(__LINE__,cudaDeviceSynchronize());
           checkCUDA(__LINE__, cudaMemcpy(beIndex, beIndexCPU,sizeof(unsigned int)*len_beIndex, cudaMemcpyHostToDevice));
           delete [] beIndexCPU;
       }
       else{
           std::cout << "beIndex is NULL"<<std::endl;
       }

       if (beLinIdx!=NULL){
           unsigned int* beLinIdxCPU = beLinIdx;
           //checkCUDA(__LINE__,cudaDeviceSynchronize());
           checkCUDA(__LINE__, cudaMalloc(&beLinIdx, sizeof(unsigned int)*len_beLinIdx));
           //checkCUDA(__LINE__,cudaDeviceSynchronize());
           checkCUDA(__LINE__, cudaMemcpy(beLinIdx, beLinIdxCPU,sizeof(unsigned int)*len_beLinIdx, cudaMemcpyHostToDevice));
           delete [] beLinIdxCPU;
       }
       else{
           std::cout << "beLinIdx is NULL"<<std::endl;
       }

       // make it to full matrix to skip searching 
       unsigned int * beIndexFull;
       unsigned int sz = 2*sizeof(unsigned int)*(grid_range[0]+1)*(grid_range[1]+1)*(grid_range[2]+1);
       checkCUDA(__LINE__, cudaMalloc(&beIndexFull, sz));
       checkCUDA(__LINE__, cudaMemset(beIndexFull, 0, sz));
       int THREADS_NUM = 1024;
       int BLOCK_NUM = int((len_beLinIdx + size_t(THREADS_NUM) - 1) / THREADS_NUM);
       fillInBeIndexFull<<<BLOCK_NUM,THREADS_NUM>>>(beIndexFull,beIndex,beLinIdx,len_beLinIdx);
       checkCUDA(__LINE__,cudaGetLastError());
       checkCUDA(__LINE__, cudaFree(beIndex));      beIndex = NULL;
       checkCUDA(__LINE__, cudaFree(beLinIdx));     beLinIdx = NULL;
       beIndex = beIndexFull;

       if (pcIndex!=NULL){
          unsigned int* pcIndexCPU = pcIndex;
          checkCUDA(__LINE__, cudaMalloc(&pcIndex, sizeof(unsigned int)*len_pcIndex));
          checkCUDA(__LINE__, cudaMemcpy(pcIndex, pcIndexCPU,sizeof(unsigned int)*len_pcIndex, cudaMemcpyHostToDevice));
          delete [] pcIndexCPU;
       }
       else{
           std::cout << "pcIndexCPU is NULL"<<std::endl;
       }
       

       if (RGBDimage!=NULL){
         RGBDpixel* RGBDimageCPU = RGBDimage;
         checkCUDA(__LINE__, cudaMalloc(&RGBDimage, sizeof(RGBDpixel)*width*height));
         checkCUDA(__LINE__, cudaMemcpy( RGBDimage, RGBDimageCPU, sizeof(RGBDpixel)*width*height, cudaMemcpyHostToDevice));
         delete [] RGBDimageCPU;
       }
       else{
           std::cout << "RGBDimage is NULL"<<std::endl;
       }
       /* 
       if (XYZimage!=NULL){ 
          float * XYZimageCPU = XYZimage;
          checkCUDA(__LINE__, cudaMalloc(&XYZimage, sizeof(float)*width*height*3));
          checkCUDA(__LINE__, cudaMemcpy(XYZimage, XYZimageCPU, sizeof(float)*width*height*3, cudaMemcpyHostToDevice));
          delete [] XYZimageCPU;
       }
       else{
          std::cout << "XYZimage is NULL"<<std::endl;
       }
       */

       if (grid_range!=NULL){ 
          unsigned int * grid_rangeCPU = grid_range;
          checkCUDA(__LINE__, cudaMalloc(&grid_range, sizeof(unsigned int)*3));
          checkCUDA(__LINE__, cudaMemcpy(grid_range, grid_rangeCPU, 3*sizeof(unsigned int), cudaMemcpyHostToDevice));
          delete [] grid_rangeCPU;
       }
       else{
          std::cout << "grid_range is NULL"<<std::endl;
       }

       if (begin_range!=NULL){ 
          float * begin_rangeCPU = begin_range;
          checkCUDA(__LINE__, cudaMalloc(&begin_range, sizeof(float)*3));
          checkCUDA(__LINE__, cudaMemcpy(begin_range, begin_rangeCPU, sizeof(float)*3, cudaMemcpyHostToDevice));
          delete [] begin_rangeCPU;
       }
       else{
          std::cout << "grid_range is NULL"<<std::endl;
       }


       checkCUDA(__LINE__, cudaMalloc(&K_GPU, sizeof(float)*9));
       checkCUDA(__LINE__, cudaMemcpy(K_GPU, (float*)K, sizeof(float)*9, cudaMemcpyHostToDevice));

      
       checkCUDA(__LINE__, cudaMalloc(&R_GPU, sizeof(float)*9));
       checkCUDA(__LINE__, cudaMemcpy(R_GPU, (float*)R, sizeof(float)*9, cudaMemcpyHostToDevice)); 

       GPUdata = true;

    }
  };

  void free(){
    if (GPUdata){
      //std::cout<< "free GPUdata"<<std::endl;
      if (RGBDimage   !=NULL) {checkCUDA(__LINE__, cudaFree(RGBDimage));    RGBDimage = NULL;}
      if (beIndex     !=NULL) {checkCUDA(__LINE__, cudaFree(beIndex));      beIndex = NULL;}
      if (beLinIdx    !=NULL) {checkCUDA(__LINE__, cudaFree(beLinIdx));     beLinIdx = NULL;}
      if (pcIndex     !=NULL) {checkCUDA(__LINE__, cudaFree(pcIndex));      pcIndex = NULL;}
      if (XYZimage    !=NULL) {checkCUDA(__LINE__, cudaFree(XYZimage));     XYZimage = NULL;}
      if (R_GPU       !=NULL) {checkCUDA(__LINE__, cudaFree(R_GPU));        R_GPU = NULL;}
      if (K_GPU       !=NULL) {checkCUDA(__LINE__, cudaFree(K_GPU));        K_GPU = NULL;}
      if (grid_range  !=NULL) {checkCUDA(__LINE__, cudaFree(grid_range));   grid_range = NULL;}
      if (begin_range !=NULL) {checkCUDA(__LINE__, cudaFree(begin_range));  begin_range = NULL;}
      GPUdata = false;
    }
    else{
      //std::cout<< "free CPUdata"<<std::endl;
      if (RGBDimage   !=NULL) {delete [] RGBDimage;    RGBDimage   = NULL;}
      if (beIndex     !=NULL) {delete [] beIndex;      beIndex     = NULL;}
      if (beLinIdx    !=NULL) {delete [] beLinIdx;     beLinIdx    = NULL;}
      if (pcIndex     !=NULL) {delete [] pcIndex;      pcIndex     = NULL;}
      if (XYZimage    !=NULL) {delete [] XYZimage;     XYZimage    = NULL;}
      if (grid_range  !=NULL) {delete [] grid_range;   grid_range  = NULL;}
      if (begin_range !=NULL) {delete [] begin_range;  begin_range = NULL;}
    }
  };
  ~Scene3D(){
    free();
  };
};



__global__ void compute_TSDFGPUbox(StorageT* tsdf_data, float* R_data, float* K_data,  float* range, float grid_delta,  unsigned int *grid_range,
                                  RGBDpixel* RGBDimage,  unsigned int* star_end_indx_data ,unsigned int*  pc_lin_indx_data,float* XYZimage,
                                  const float* bb3d_data, int tsdf_size,int tsdf_size1,int tsdf_size2, int fdim, int im_w, int im_h, const int encode_type,const float scale)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;;
    int volume_size = tsdf_size * tsdf_size1 * tsdf_size2;
    if (index > volume_size) return;
    float delta_x = 2 * bb3d_data[12] / float(tsdf_size);  
    float delta_y = 2 * bb3d_data[13] / float(tsdf_size1);  
    float delta_z = 2 * bb3d_data[14] / float(tsdf_size2);  
    float surface_thick = 0.1;
    const float MaxDis = surface_thick + 20;
    //printf("delta_x:%f,%f,%f\n",R_data[0],R_data[1],R_data[2]); 
    // caculate tsdf for this box
    /*
    float x = float(index % tsdf_size);
    float y = float((index / tsdf_size) % tsdf_size);   
    float z = float((index / tsdf_size / tsdf_size) % tsdf_size);
    */
    float x = float((index / (tsdf_size1*tsdf_size2))%tsdf_size) ;
    float y = float((index / tsdf_size2) % tsdf_size1);
    float z = float(index % tsdf_size2);

    for (int i =0;i<fdim;i++){
        tsdf_data[index + i * volume_size] = GPUCompute2StorageT(ComputeT(0));
    }

    // get grid world coordinate
    float temp_x = - bb3d_data[12] + (x + 0.5) * delta_x;
    float temp_y = - bb3d_data[13] + (y + 0.5) * delta_y;
    float temp_z = - bb3d_data[14] + (z + 0.5) * delta_z;

    x = temp_x * bb3d_data[0] + temp_y * bb3d_data[3] + temp_z * bb3d_data[6]
        + bb3d_data[9];
    y = temp_x * bb3d_data[1] + temp_y * bb3d_data[4] + temp_z * bb3d_data[7]
        + bb3d_data[10];
    z = temp_x * bb3d_data[2] + temp_y * bb3d_data[5] + temp_z * bb3d_data[8]
        + bb3d_data[11]; 

    // project to image plane decides the sign
    // rotate back and swap y, z and -y
    float xx =   R_data[0] * x + R_data[3] * y + R_data[6] * z;
    float zz =   R_data[1] * x + R_data[4] * y + R_data[7] * z;
    float yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
    int ix = floor(xx * K_data[0] / zz + K_data[2]+0.5) - 1;
    int iy = floor(yy * K_data[4] / zz + K_data[5]+0.5) - 1;

    
    if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001){
        return;
    } 

    // find the most nearby point 
    float disTosurfaceMin = MaxDis;
    int idx_min = 0;
    int x_grid = floor((x-range[0])/grid_delta);
    int y_grid = floor((y-range[1])/grid_delta);
    int z_grid = floor((z-range[2])/grid_delta);
    //grid_range =  [w,d,h];  linearInd =x(i)*d*h+y(i)*h+z(i);
    //if (x_grid < 0 || x_grid >= grid_range[0] || y_grid < 0 || y_grid >= grid_range[1] || z_grid < 0 || z_grid >= grid_range[2]){
    if (x_grid < 0 || x_grid > grid_range[0] || y_grid < 0 || y_grid > grid_range[1] || z_grid < 0 || z_grid > grid_range[2]){
        return;
    }
    int linearInd =x_grid*grid_range[1]*grid_range[2]+y_grid*grid_range[2]+z_grid;      
    int search_region =1;
    if (star_end_indx_data[2*linearInd+0]>0){
        search_region =0;
    }  
    int find_close_point = -1;

    while(find_close_point<0&&search_region<3){
      for (int iix = max(0,x_grid-search_region); iix < min((int)grid_range[0],x_grid+search_region+1); iix++){
        for (int iiy = max(0,y_grid-search_region); iiy < min((int)grid_range[1],y_grid+search_region+1); iiy++){
          for (int iiz = max(0,z_grid-search_region); iiz < min((int)grid_range[2],z_grid+search_region+1); iiz++){
              unsigned int iilinearInd = iix*grid_range[1]*grid_range[2] + iiy*grid_range[2] + iiz;

              for (int pid = star_end_indx_data[2*iilinearInd+0]-1; pid < star_end_indx_data[2*iilinearInd+1]-1;pid++){
                 
                 //printf("%d-%d\n",star_end_indx_data[2*iilinearInd+0],star_end_indx_data[2*iilinearInd+1]);
                 unsigned int p_idx_lin = pc_lin_indx_data[pid];
                 float xp = XYZimage[3*p_idx_lin+0];
                 float yp = XYZimage[3*p_idx_lin+1];
                 float zp = XYZimage[3*p_idx_lin+2];
                 // distance
                 float xd = abs(x - xp);
                 float yd = abs(y - yp);
                 float zd = abs(z - zp);
                 if (xd < 2.0 * delta_x||yd < 2.0 * delta_x|| zd < 2.0 * delta_x){
                    float disTosurface = sqrt(xd * xd + yd * yd + zd * zd);
                    if (disTosurface < disTosurfaceMin){
                       disTosurfaceMin = disTosurface;
                       idx_min = p_idx_lin;
                       find_close_point = 1;
                       //printf("x:%f,%f,%f,xp,%f,%f,%f,xd%f,%f,%f,%f\n",x,y,z,xp,yp,zp,xd,yd,zd,disTosurfaceMin);
                       
                    }
                }
              } // for all points in this grid
            

          }
        }
      }
      search_region ++;
    }//while 
    
    float tsdf_x = MaxDis;
    float tsdf_y = MaxDis;
    float tsdf_z = MaxDis;


    float color_b =0;
    float color_g =0;
    float color_r =0;

    float xnear = 0;
    float ynear = 0;
    float znear = 0;
    if (find_close_point>0){
        
        xnear = XYZimage[3*idx_min+0];
        ynear = XYZimage[3*idx_min+1];
        znear = XYZimage[3*idx_min+2];
        tsdf_x = abs(x - xnear);
        tsdf_y = abs(y - ynear);
        tsdf_z = abs(z - znear);

        color_b = float(RGBDimage[idx_min].B)/255.0;
        color_g = float(RGBDimage[idx_min].G)/255.0;
        color_r = float(RGBDimage[idx_min].R)/255.0;

        //printf("x:%f,tsdf_x:%f,%f,%f\n",disTosurfaceMin,tsdf_x,tsdf_y,tsdf_z);          
    }
//printf("before : %f,%f,%f\n",tsdf_x,tsdf_y,tsdf_z);

    disTosurfaceMin = min(disTosurfaceMin/surface_thick,float(1.0));
    float ratio = 1.0 - disTosurfaceMin;
    float second_ratio =0;
    if (ratio > 0.5) {
       second_ratio = 1 - ratio;
    }
    else{
       second_ratio = ratio;
    }

    if (disTosurfaceMin > 0.999){
        tsdf_x = MaxDis;
        tsdf_y = MaxDis;
        tsdf_z = MaxDis;
    }

    
    if (encode_type == 101){ 
      tsdf_x = min(tsdf_x, surface_thick);
      tsdf_y = min(tsdf_y, surface_thick);
      tsdf_z = min(tsdf_z, surface_thick);
    }
    else{
      tsdf_x = min(tsdf_x, float(2.0 * delta_x));
      tsdf_y = min(tsdf_y, float(2.0 * delta_y));
      tsdf_z = min(tsdf_z, float(2.0 * delta_z));
    }

   

    float depth_project   = XYZimage[3*(ix * im_h + iy)+1];  
    if (zz > depth_project) {
      tsdf_x = - tsdf_x;
      tsdf_y = - tsdf_y;
      tsdf_z = - tsdf_z;
      disTosurfaceMin = - disTosurfaceMin;
      second_ratio = - second_ratio;
    }

    // encode_type 
    if (encode_type == 100||encode_type == 101){
      tsdf_data[index + 0 * volume_size] = GPUCompute2StorageT(tsdf_x);
      tsdf_data[index + 1 * volume_size] = GPUCompute2StorageT(tsdf_y);
      tsdf_data[index + 2 * volume_size] = GPUCompute2StorageT(tsdf_z);
    }
    else if(encode_type == 102){
      tsdf_data[index + 0 * volume_size] = GPUCompute2StorageT(tsdf_x);
      tsdf_data[index + 1 * volume_size] = GPUCompute2StorageT(tsdf_y);
      tsdf_data[index + 2 * volume_size] = GPUCompute2StorageT(tsdf_z);
      tsdf_data[index + 3 * volume_size] = GPUCompute2StorageT(color_b/scale);
      tsdf_data[index + 4 * volume_size] = GPUCompute2StorageT(color_g/scale);
      tsdf_data[index + 5 * volume_size] = GPUCompute2StorageT(color_r/scale);
    }
    else if(encode_type == 103){
      tsdf_data[index + 0 * volume_size] = GPUCompute2StorageT(ratio);
    }

    // scale feature 
    for (int i =0;i<fdim;i++){
        tsdf_data[index + i * volume_size] = GPUCompute2StorageT(scale* GPUStorage2ComputeT(tsdf_data[index + i * volume_size]));
    }

    //}// end for each index in each box
};

__global__ void compute_TSDFGPUbox_proj(StorageT* tsdf_data, float* R_data, float* K_data, RGBDpixel* RGBDimage, float* XYZimage,
                                      const float* bb3d_data, int tsdf_size,int tsdf_size1,int tsdf_size2, int fdim, int im_w, int im_h, const int encode_type,const float scale)
{
  const int index = threadIdx.x + blockIdx.x * blockDim.x;;
    int volume_size = tsdf_size * tsdf_size1 * tsdf_size2;
    if (index > volume_size) return;
    float delta_x = 2 * bb3d_data[12] / float(tsdf_size);  
    float delta_y = 2 * bb3d_data[13] / float(tsdf_size1);  
    float delta_z = 2 * bb3d_data[14] / float(tsdf_size2);  
    float surface_thick = 0.1;
    const float MaxDis = surface_thick + 20;

    float x = float((index / (tsdf_size1*tsdf_size2))%tsdf_size) ;
    float y = float((index / tsdf_size2) % tsdf_size1);
    float z = float(index % tsdf_size2);

    for (int i =0;i<fdim;i++){
        tsdf_data[index + i * volume_size] = GPUCompute2StorageT(ComputeT(0));
    }

    // get grid world coordinate
    float temp_x = - bb3d_data[12] + (x + 0.5) * delta_x;
    float temp_y = - bb3d_data[13] + (y + 0.5) * delta_y;
    float temp_z = - bb3d_data[14] + (z + 0.5) * delta_z;

    x = temp_x * bb3d_data[0] + temp_y * bb3d_data[3] + temp_z * bb3d_data[6]
        + bb3d_data[9];
    y = temp_x * bb3d_data[1] + temp_y * bb3d_data[4] + temp_z * bb3d_data[7]
        + bb3d_data[10];
    z = temp_x * bb3d_data[2] + temp_y * bb3d_data[5] + temp_z * bb3d_data[8]
        + bb3d_data[11]; 

    // project to image plane decides the sign
    // rotate back and swap y, z and -y
    float xx =   R_data[0] * x + R_data[3] * y + R_data[6] * z;
    float zz =   R_data[1] * x + R_data[4] * y + R_data[7] * z;
    float yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
    int ix = floor(xx * K_data[0] / zz + K_data[2]+0.5) - 1;
    int iy = floor(yy * K_data[4] / zz + K_data[5]+0.5) - 1;

    
    if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001){
        return;
    } 
    
    float x_project   = XYZimage[3*(ix * im_h + iy)+0];
    float y_project   = XYZimage[3*(ix * im_h + iy)+1];
    float z_project   = XYZimage[3*(ix * im_h + iy)+2]; 


    float tsdf_x = abs(x - x_project);
    float tsdf_y = abs(y - y_project);
    float tsdf_z = abs(z - z_project);

    float color_b = 0;
    float color_g = 0;
    float color_r = 0;
    if (RGBDimage!=NULL){
      color_b = float(RGBDimage[(ix * im_h + iy)].B)/255.0;
      color_g = float(RGBDimage[(ix * im_h + iy)].G)/255.0;
      color_r = float(RGBDimage[(ix * im_h + iy)].R)/255.0;
    }

    float disTosurfaceMin = sqrt(tsdf_x * tsdf_x + tsdf_y * tsdf_y + tsdf_z * tsdf_z);
    disTosurfaceMin = min(disTosurfaceMin/surface_thick,float(1.0));
    float ratio = 1.0 - disTosurfaceMin;
    float second_ratio =0;
    if (ratio > 0.5) {
       second_ratio = 1 - ratio;
    }
    else{
       second_ratio = ratio;
    }
    if (disTosurfaceMin > 0.999){
        tsdf_x = MaxDis;
        tsdf_y = MaxDis;
        tsdf_z = MaxDis;
    }

    tsdf_x = min(tsdf_x, float(2.0 * delta_x));
    tsdf_y = min(tsdf_y, float(2.0 * delta_y));
    tsdf_z = min(tsdf_z, float(2.0 * delta_z));

    if (zz > y_project) {
      tsdf_x = - tsdf_x;
      tsdf_y = - tsdf_y;
      tsdf_z = - tsdf_z;
      disTosurfaceMin = - disTosurfaceMin;
      second_ratio = - second_ratio;
    }

    // encode_type 
    if (encode_type == 0){
      tsdf_data[index + 0 * volume_size] = GPUCompute2StorageT(tsdf_x);
      tsdf_data[index + 1 * volume_size] = GPUCompute2StorageT(tsdf_y);
      tsdf_data[index + 2 * volume_size] = GPUCompute2StorageT(tsdf_z);
    }
    if (encode_type == 2){
      tsdf_data[index + 0 * volume_size] = GPUCompute2StorageT(tsdf_x);
      tsdf_data[index + 1 * volume_size] = GPUCompute2StorageT(tsdf_y);
      tsdf_data[index + 2 * volume_size] = GPUCompute2StorageT(tsdf_z);
      tsdf_data[index + 3 * volume_size] = GPUCompute2StorageT(color_b/scale);
      tsdf_data[index + 4 * volume_size] = GPUCompute2StorageT(color_g/scale);
      tsdf_data[index + 5 * volume_size] = GPUCompute2StorageT(color_r/scale);
    }
    // scale feature 
    for (int i =0;i<fdim;i++){
        tsdf_data[index + i * volume_size] = GPUCompute2StorageT(scale* GPUStorage2ComputeT(tsdf_data[index + i * volume_size]));
    }
}

void compute_TSDF_Space(Scene3D* scene , Box3D SpaceBox, StorageT* tsdf_data_GPU, std::vector<int> grid_size, int encode_type, float scale){
    
   scene->loadData2XYZimage(); 

   float* bb3d_data;
   checkCUDA(__LINE__, cudaMalloc(&bb3d_data,  sizeof(float)*15));
   checkCUDA(__LINE__, cudaMemcpy(bb3d_data    , SpaceBox.base, sizeof(float)*15, cudaMemcpyHostToDevice));
   unsigned int * grid_range = scene->grid_range;
   float* R_data = scene->R_GPU;
   float* K_data = scene->K_GPU;
   float* range  = scene->begin_range;
   RGBDpixel* RGBDimage = scene->RGBDimage;
   unsigned int* star_end_indx_data = scene->beIndex;
   unsigned int* pc_lin_indx_data = scene->pcIndex;
   float* XYZimage  = scene->XYZimage;

   int THREADS_NUM = 1024;
   int BLOCK_NUM = int((grid_size[1]*grid_size[2]*grid_size[3] + size_t(THREADS_NUM) - 1) / THREADS_NUM);

   compute_TSDFGPUbox<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_data_GPU, R_data, K_data, range, scene->grid_delta, grid_range, RGBDimage, 
                           star_end_indx_data, pc_lin_indx_data, XYZimage, bb3d_data, grid_size[1],grid_size[2],grid_size[3],grid_size[0], 
                           scene->width, scene->height, encode_type, scale);



   
   scene-> free();
   checkCUDA(__LINE__,cudaGetLastError());
   checkCUDA(__LINE__, cudaFree(bb3d_data));
      
}

void compute_TSDF (std::vector<Scene3D*> *chosen_scenes_ptr, std::vector<int> *chosen_box_id, StorageT* datamem, std::vector<int> grid_size, int encode_type, float scale) {
    // for each scene 
    int totalcounter = 0;
    float tsdf_size = grid_size[1];
    if (grid_size[1]!=grid_size[2]||grid_size[1]!=grid_size[3]){
        std::cerr << "grid_size[1]!=grid_size[2]||grid_size[1]!=grid_size[3]" <<std::endl;
        exit(EXIT_FAILURE);
    }

    int numeltsdf = grid_size[0]*tsdf_size*tsdf_size*tsdf_size;
    int THREADS_NUM = 1024;
    int BLOCK_NUM = int((tsdf_size*tsdf_size*tsdf_size + size_t(THREADS_NUM) - 1) / THREADS_NUM);
    float* bb3d_data;

    //int tmpD; cudaGetDevice(&tmpD); std::cout<<"GPU at LINE "<<__LINE__<<" = "<<tmpD<<std::endl;
    //checkCUDA(__LINE__,cudaDeviceSynchronize());
    checkCUDA(__LINE__, cudaMalloc(&bb3d_data,  sizeof(float)*15));
    
    //unsigned long long transformtime =0;
    //unsigned long long loadtime =0;
    //unsigned long long copygputime =0;
    //unsigned int sz = 0;
    Scene3D* scene_prev = NULL;
    for (int sceneId = 0;sceneId<(*chosen_scenes_ptr).size();sceneId++){
        // caculate in CPU mode
        //compute_TSDFCPUbox(tsdf_data,&((*chosen_scenes_ptr)[sceneId]),boxId,grid_size,encode_type,scale);
        // caculate in GPU mode
        
        //unsigned long long  time0,time1,time2,time3,time4;
        Scene3D* scene = (*chosen_scenes_ptr)[sceneId];
        //int tmpD; cudaGetDevice(&tmpD); std::cout<<"GPU at LINE "<<__LINE__<<" = "<<tmpD<<std::endl;
        // perpare scene
        if (scene!=scene_prev){
            if (scene_prev!=NULL){
               scene_prev -> free();
            }
            scene->loadData2XYZimage(); 
        }
        
        int boxId = (*chosen_box_id)[sceneId];
        checkCUDA(__LINE__, cudaMemcpy(bb3d_data, scene->objects[boxId].base, sizeof(float)*15, cudaMemcpyHostToDevice));

        unsigned int * grid_range = scene->grid_range;
        float* R_data = scene->R_GPU;
        float* K_data = scene->K_GPU;
        float* range  = scene->begin_range;
        
        RGBDpixel* RGBDimage = scene->RGBDimage;
        unsigned int* star_end_indx_data = scene->beIndex;
        unsigned int* pc_lin_indx_data = scene->pcIndex;
        float* XYZimage  = scene->XYZimage;
        
        // output
        StorageT * tsdf_data = &datamem[totalcounter*numeltsdf];

        //time3 = get_timestamp_dss();   
        //checkCUDA(__LINE__,cudaDeviceSynchronize());
         if (encode_type > 99){
            compute_TSDFGPUbox<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_data, R_data, K_data, range, scene->grid_delta, grid_range, RGBDimage, 
                           star_end_indx_data, pc_lin_indx_data, XYZimage, bb3d_data, grid_size[1],grid_size[2],grid_size[3], grid_size[0], 
                           scene->width, scene->height, encode_type, scale);

        }
        else{
          //std::cout<<"compute_TSDFGPUbox_proj"<<std::endl;
          compute_TSDFGPUbox_proj<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_data, R_data, K_data, RGBDimage, XYZimage,
                                                             bb3d_data, grid_size[1],grid_size[2],grid_size[3], grid_size[0], 
                                                             scene->width, scene->height, encode_type, scale);
        }
        
        checkCUDA(__LINE__,cudaDeviceSynchronize());
        checkCUDA(__LINE__,cudaGetLastError());
        //time4 = get_timestamp_dss();

        //

        ++totalcounter;

        scene_prev = scene;
        //loadtime += time1-time0;
        //copygputime += time2-time1;
        //transformtime += time4-time3;
    }
    checkCUDA(__LINE__, cudaFree(bb3d_data));
    
    // free the loaded images
    for (int sceneId = 0;sceneId<(*chosen_scenes_ptr).size();sceneId++){
        (*chosen_scenes_ptr)[sceneId]->free();
    }
    
    
    //std::cout << "compute_TSDF: read disk " << loadtime/1000 << " ms, " << "copygputime " 
    //<< copygputime/1000 << "transform " << transformtime/1000 << " ms" <<std::endl;  
}

Box3D processbox (Box3D box,float context_pad,int tsdf_size){
     if (context_pad > 0){
        float context_scale = float(tsdf_size) / (float(tsdf_size) - 2*context_pad);
        box.coeff[0] = box.coeff[0] * context_scale;
        box.coeff[1] = box.coeff[1] * context_scale;
        box.coeff[2] = box.coeff[2] * context_scale;
     }
     // change the oreintation 
     if (box.base[1]<0){
        box.base[0] = -1*box.base[0];
        box.base[1] = -1*box.base[1];
        box.base[2] = -1*box.base[2];
     }
     if (box.base[4]<0){
        box.base[3] = -1*box.base[3];
        box.base[4] = -1*box.base[4];
        box.base[5] = -1*box.base[5];
     }

     if(box.base[1]<box.base[4]){
        // swap first two row 
        float tmpbase[3];
        tmpbase[0] = box.base[0];
        tmpbase[1] = box.base[1];
        tmpbase[2] = box.base[2];

        box.base[0] = box.base[3];
        box.base[1] = box.base[4];
        box.base[2] = box.base[5];

        box.base[3] = tmpbase[0];
        box.base[4] = tmpbase[1];
        box.base[5] = tmpbase[2];
        float tmpcoeff =  box.coeff[0];
        box.coeff[0] = box.coeff[1];
        box.coeff[1] = tmpcoeff;
     }
     return box;
}
/*
int test(){
    std::string file_list = "DSS/boxfile/boxes_NYU_trainfea_debug.list";
    //std::string data_root = "DSS/sunrgbd_dss_data/";
    std::string data_root =  "/n/fs/modelnet/deepDetect/sunrgbd_dss_data/";
    std::vector<Scene3D*> scenes;

    //int count = 0;
    int object_count = 0;
    float scale =100;
    float context_pad =3;
    std::vector<int> grid_size {3,30,30,30};
    int encode_type =100;

    std::cout<<"loading file "<<file_list<<"\n";
    FILE* fp = fopen(file_list.c_str(),"rb");
    if (fp==NULL) { std::cout<<"fail to open file: "<<file_list<<std::endl; exit(EXIT_FAILURE); }
    while (feof(fp)==0) {
      Scene3D* scene = new Scene3D();
      unsigned int len = 0;
      fread((void*)(&len), sizeof(unsigned int), 1, fp);    
      if (len==0) break;
      scene->filename.resize(len);
      if (len>0) fread((void*)(scene->filename.data()), sizeof(char), len, fp);
      scene->filename = data_root+scene->filename+".bin"; 
      fread((void*)(scene->R), sizeof(float), 9, fp);
      fread((void*)(scene->K), sizeof(float), 9, fp);
      fread((void*)(&scene->height), sizeof(unsigned int), 1, fp);  
      fread((void*)(&scene->width), sizeof(unsigned int), 1, fp); 
      

      fread((void*)(&len),    sizeof(unsigned int),   1, fp);
      scene->objects.resize(len);
      if (len>0){
          for (int i=0;i<len;++i){
              Box3D box;
              fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp);
              fread((void*)(box.base),        sizeof(float), 9, fp);
              fread((void*)(box.center),      sizeof(float), 3, fp);
              fread((void*)(box.coeff),       sizeof(float), 3, fp);
              //process box pad contex oreintation 
              box = processbox (box, context_pad, grid_size[1]);
              scene->objects[i]=box;

              object_count++;
              //num_categories = max(num_categories, box.category);
            
              //printf("category:%d\n",box.category);
              //printf("box.base:%f,%f,%f,%f,%f,%f\n",box.base[0],box.base[1],box.base[2],box.base[3],box.base[4],box.base[5]);
              //printf("box.base:%f,%f,%f,%f,%f,%f\n",box.base[0],box.base[1],box.base[2],box.base[3],box.base[4],box.base[5]);
              //printf("box.center:%f,%f,%f\n",box.center[0],box.center[1],box.center[2]);
              //printf("box.coeff:%f,%f,%f\n",box.coeff[0],box.coeff[1],box.coeff[2]);
             
          }
      }
      scenes.push_back(scene);

    }
    fclose(fp);

    std::vector<Scene3D*> chosen_scenes;
    std::vector<int> chosen_box_id;
    for (int i = 0;i<scenes.size();++i){
       for (int j =0; j < scenes[i]->objects.size();++j){
            chosen_scenes.push_back(scenes[i]);
            chosen_box_id.push_back(j);
       } 
    }

    
    std::cout<<"object_count:" <<object_count <<std::endl;
    float* dataCPUmem = new float[(object_count)*3*30*30*30];
    StorageT* dataGPUmem;
    checkCUDA(__LINE__, cudaMalloc(&dataGPUmem, (object_count)*3*30*30*30*sizeof(float)));

    compute_TSDF(&chosen_scenes, &chosen_box_id, dataGPUmem,grid_size,encode_type,scale);
    checkCUDA(__LINE__, cudaMemcpy(dataCPUmem, dataGPUmem,(object_count)*3*30*30*30*sizeof(float), cudaMemcpyDeviceToHost) );
        

    std::string outputfile = "DSS/feature.bin";

    FILE * fid = fopen(outputfile.c_str(),"wb");
    fwrite(dataCPUmem,sizeof(float),(object_count)*3*30*30*30,fid);
    fclose(fid);
    return 1;
}
*/

class Scene3DDataLayer : public DataLayer {
public:
  int epoch_prefetch;
  //Tensor* dataCPU;
  Tensor<StorageT >* labelCPU;
  StorageT*   dataGPU;
  Tensor<StorageT >* bb_tar_diff;
  Tensor<StorageT >* bb_loss_weights;
  Tensor<StorageT >* hha_fea;
  Tensor<StorageT >* img_fea;
  Tensor<StorageT >* bb_2d_diff;
  Tensor<StorageT >* bb_2d_weights;
  Tensor<StorageT >* oreintation_label;
  Tensor<StorageT >* oreintation_label_w;


  std::future<void> lock;
  std::vector<int> batch_size;
  std::vector<int> bb_param_weight;
  
  std::vector<Scene3D*> scenes;
  std::vector<Box3Ddiff>  target_boxes;
  std::vector<Box2D> target_2dboxes;
  std::vector<ImgObjInd> imgobj_pos;
  std::vector<ImgObjInd> imgobj_neg;
  int counter_neg;
  int counter_pos;
  //std::vector<std::vector<ImgObjInd>> imgobj_pos_struct;
  //std::vector<std::vector<ImgObjInd>> imgobj_neg_struct;
  
  std::vector<std::vector<ImgObjInd>> imgobj_pos_cates;
  std::vector<int> counter_pos_cates;
  int num_percate;

  std::string file_list;
  std::string data_root;
  std::vector<int> grid_size;
  int encode_type;
  float scale;
  float context_pad;
  unsigned int num_categories;

  bool box_reg;
  bool is_render;
  bool is_combineimg;
  bool is_combinehha;
  int  imgfea_dim;
  std::string img_fea_folder;

  bool box_2dreg;
  bool orein_cls;

  int num_oreintation;
  

  int numofitems(){ 
    int total_number = 0;
    for (int i=0;i<scenes.size();++i){
      total_number += scenes[i]->objects.size();
    }
    return total_number;  
  };


  int numofitemsTruncated(){
    return sum(batch_size) * floor(double(numofitems())/double(sum(batch_size)));
  };

  void shuffle(){
    std::cout<< "!!!!!!!!!!! Should Call shuffle() with input  !!!!!!!!!!" <<std::endl;
    return;
  };

  void shuffle(int category_id){
       std::shuffle (imgobj_pos_cates[category_id].begin(),imgobj_pos_cates[category_id].end(), rng );
       std::cout<< "shuffle category "<< category_id <<std::endl;
  }

  void shuffle(bool neg, bool pos){
    if (pos){
      std::cout<< "shuffle positive data" <<std::endl;
      std::shuffle (imgobj_pos.begin(),imgobj_pos.end(), rng );
      /*
      ::shuffle (imgobj_pos_struct.begin(),imgobj_pos_struct.end(), rng );
      for (int i =0;i<imgobj_pos_struct.size();i++){
        ::shuffle (imgobj_pos_struct[i].begin(),imgobj_pos_struct[i].end(), rng );
      }
      int cnt = 0;
      for (int i =0;i < imgobj_pos_struct.size(); i++){
        for (int j =0; j < imgobj_pos_struct[i].size(); j++){
           imgobj_pos[cnt] = imgobj_pos_struct[i][j];
           cnt++;
        }
      }
      */
    }
    if (neg){
      std::cout<< "shuffle neg data" <<std::endl;
      std::shuffle (imgobj_neg.begin(),imgobj_neg.end(), rng );
      /*
      std::shuffle (imgobj_neg_struct.begin(),imgobj_neg_struct.end(), rng );
      for (int i =0;i < imgobj_neg_struct.size(); i++){
        std::shuffle (imgobj_neg_struct[i].begin(),imgobj_neg_struct[i].end(), rng );
      }
      int cnt = 0;
      for (int i =0;i < imgobj_neg_struct.size(); i++){
        for (int j =0; j < imgobj_neg_struct[i].size(); j++){
           imgobj_neg[cnt] = imgobj_neg_struct[i][j];
           cnt++;
        }
      }
      */

    }
  };

  void init(){
    epoch_prefetch = 0;
 
    train_me = false;
    counter_neg = 0;
    counter_pos = 0;
    imgobj_pos_cates.resize(num_categories);
    counter_pos_cates.resize(num_categories);
    for (int i = 0;i<num_categories;i++){
        counter_pos_cates[i] = 0;
    }

    if (phase==Testing){
      batch_size[0] = sum(batch_size);
      batch_size.resize(1);
    }


    // read files
    std::cout<<"loading file "<<file_list<<"\n";
    FILE* fp = fopen(file_list.c_str(),"rb");
    if (fp==NULL) { std::cout<<"fail to open file: "<<file_list<<std::endl; exit(EXIT_FAILURE); }
    size_t file_size;
    while (feof(fp)==0) {
      Scene3D* scene = new Scene3D();
      if (is_render) {scene->DataType = Render;}
      //std::vector<ImgObjInd> boxlist_pos;
      //std::vector<ImgObjInd> boxlist_neg;
      unsigned int len = 0;
      file_size += fread((void*)(&len), sizeof(unsigned int), 1, fp);    
      if (len==0) break;
      scene->filename.resize(len);
      if (len>0) 
        file_size += fread((void*)(scene->filename.data()), sizeof(char), len, fp);
      
      if (is_render) {
        scene->DataType = Render;
        scene->filename = data_root+scene->filename; 
      }
      else{
        scene->filename = data_root+scene->filename+".bin"; 
      }
      //std::cout<<scene->filename<<std::endl;
      file_size += fread((void*)(scene->R), sizeof(float), 9, fp);
      file_size += fread((void*)(scene->K), sizeof(float), 9, fp);
      file_size += fread((void*)(&scene->height), sizeof(unsigned int), 1, fp);
      file_size += fread((void*)(&scene->width), sizeof(unsigned int), 1, fp); 
      file_size += fread((void*)(&len),    sizeof(unsigned int),   1, fp);
      scene->objects.resize(len);
      if (len>0){
        for (int i=0;i<len;++i){
          Box3D box;
          file_size += fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp);
          file_size += fread((void*)(box.base),        sizeof(float), 9, fp);
          file_size += fread((void*)(box.center),      sizeof(float), 3, fp);
          file_size += fread((void*)(box.coeff),       sizeof(float), 3, fp);
          box = processbox (box, context_pad, grid_size[1]);
          scene->objects[i] = box;
          //num_categories = max(num_categories, box.category);
          // read target box if exist
          int tarId = -1;
          if (box_reg&&(phase==Training||is_render)){
            uint8_t hasTarget = 0;
            file_size += fread((void*)(&hasTarget), sizeof(uint8_t),   1, fp);
            if (hasTarget>0){
              Box3Ddiff box_tar_diff;
              file_size += fread((void*)(box_tar_diff.diff), sizeof(float), 6, fp);
              if (box_2dreg&&phase==Training){
                  file_size += fread((void*)(box_tar_diff.diff2d), sizeof(float), 4, fp);
                  float diff2d_full[4];
                  file_size += fread((void*)(diff2d_full), sizeof(float), 4, fp);
              }
              if (orein_cls&&phase==Training){
                  file_size += fread((void*)(&box_tar_diff.oreintation), sizeof(int), 1, fp);
                  //std::cout<<box_tar_diff.oreintation<<",";
              }
              
              tarId = target_boxes.size();
              target_boxes.push_back(box_tar_diff);
            }
          }
          // push to back ground forground list by category 
          // category 0 is negtive 
          if (box.category==0||phase==Testing){
            imgobj_neg.push_back(ImgObjInd(scenes.size(),i,tarId));
            //boxlist_neg.push_back(ImgObjInd(scenes.size(),i,tarId));
          }else{
            imgobj_pos.push_back(ImgObjInd(scenes.size(),i,tarId));
            imgobj_pos_cates[box.category].push_back(ImgObjInd(scenes.size(),i,tarId));
            //boxlist_pos.push_back(ImgObjInd(scenes.size(),i,tarId));
          }
          
        }
      }
      scenes.push_back(scene);
      //imgobj_neg_struct.push_back(boxlist_neg);
      //imgobj_pos_struct.push_back(boxlist_pos);
    }
    fclose(fp);
    std::cout<<"num_categories= "   << num_categories << std::endl;
    std::cout<<"all postive boxe= " << imgobj_pos.size() << std::endl;
    std::cout<<"all negative boxe= "<< imgobj_neg.size() << std::endl;

    if (phase!=Testing) {
      shuffle(true,true);
      for(int i = 1;i<imgobj_pos_cates.size();i++){
          shuffle(i);
          std::cout<<"class "<< i << "has " <<imgobj_pos_cates[i].size() <<" boxes."<< std::endl;
      }
    }

    for(int i = 1;i<imgobj_pos_cates.size();i++){
          std::cout<<"class "<< i << "has " <<imgobj_pos_cates[i].size() <<" boxes."<< std::endl;
    }
    std::cout<<"num of scenes = "   << scenes.size() << std::endl;

  };

  Scene3DDataLayer(std::string name_, Phase phase_, std::string file_list_, int batch_size_): DataLayer(name_), batch_size(batch_size_), file_list(file_list_){
    phase = phase_;
    init();
  };

  Scene3DDataLayer(JSON* json){
    SetValue(json, name,    "Scene3DData")
    SetValue(json, phase,   Training)
    SetOrDie(json, file_list  )
    SetOrDie(json, data_root  )
    SetOrDie(json, grid_size  )
    SetOrDie(json, batch_size   )
    SetOrDie(json, encode_type  )
    SetOrDie(json, scale    )
    SetOrDie(json, context_pad  )
    SetOrDie(json, GPU      )
    SetOrDie(json, box_reg    )
    SetOrDie(json, num_categories   )
    SetOrDie(json, bb_param_weight  )
    SetValue(json, num_percate, 0)
    SetValue(json, is_render  , 0)
    SetValue(json, is_combineimg  , false)
    SetValue(json, is_combinehha  , false)
    SetValue(json, imgfea_dim  , 0)
    SetValue(json, img_fea_folder, "")
    SetValue(json, box_2dreg, false)
    SetValue(json, num_oreintation, 20)
    SetValue(json, orein_cls, false)
    

    std::cout<<"grid_size:   ";  veciPrint(grid_size);  std::cout<<std::endl;
    std::cout<<"batch_size:  ";  veciPrint(batch_size); std::cout<<std::endl;
    std::cout<<"encode_type: " << encode_type << std::endl;
    std::cout<<"scale:     " << scale << std::endl;
    std::cout<<"context_pad: " << context_pad << std::endl;
    std::cout<<"Scene3DData GPU: " << GPU << std::endl;
    std::cout<<"data_root: " << data_root <<std::endl;
    std::cout<<"box_reg: "   << box_reg <<std::endl;
    std::cout<<"bb_param_weight: "; veciPrint(bb_param_weight);  std::cout<<std::endl;
    std::cout<<"num_percate: " << num_percate <<std::endl;
    std::cout<<"is_render: "   << is_render <<std::endl;
    std::cout<<"is_combineimg: "   << is_combineimg <<std::endl;
    std::cout<<"is_combinehha: "   << is_combinehha <<std::endl;

    std::cout<<"box_2dreg: "   << box_2dreg <<std::endl;
    std::cout<<"orein_cls: "   << orein_cls <<std::endl;
   
    
    
    if ((is_combineimg||is_combinehha)&&imgfea_dim==0){
      std::cout<<"imgfea_dim is "   << imgfea_dim <<std::endl;
      FatalError(__LINE__);
    }
    init();
  };

  ~Scene3DDataLayer(){
    if (lock.valid()) lock.wait();
    for(size_t i=0;i<scenes.size();i++){
      if (scenes[i]!=NULL) delete scenes[i];
    }
  };

  size_t Malloc(Phase phase_){
    if (phase == Training && phase_==Testing) return 0;
    size_t memoryBytes = 0;
    std::cout<< (train_me? "* " : "  ");
    std::cout<<name<<std::endl;

    std::vector<int> data_dim;
    std::vector<int> label_dim;  
    std::vector<int> bb_tar_diff_dim;
   
    data_dim.push_back(sum(batch_size));
    data_dim.insert( data_dim.end(), grid_size.begin(), grid_size.end() );
    label_dim.push_back(sum(batch_size));
    label_dim.push_back(1);   label_dim.push_back(1);   label_dim.push_back(1);   label_dim.push_back(1);

    bb_tar_diff_dim.push_back(sum(batch_size));
    bb_tar_diff_dim.push_back(bb_param_weight.size()*num_categories); 
    bb_tar_diff_dim.push_back(1); bb_tar_diff_dim.push_back(1); bb_tar_diff_dim.push_back(1);
    
    std::vector<int> img_fea_dim(5,1);
    img_fea_dim[0] = sum(batch_size);
    img_fea_dim[1] = imgfea_dim;
    
    labelCPU = new Tensor<StorageT>(label_dim);
    bb_tar_diff = new Tensor<StorageT>(bb_tar_diff_dim);
    bb_loss_weights = new Tensor<StorageT>(bb_tar_diff_dim);

    img_fea = new Tensor<StorageT>(img_fea_dim);
    hha_fea = new Tensor<StorageT>(img_fea_dim);

    out[0]->need_diff = false;
    out[0]->receptive_field.resize(data_dim.size()-2);  fill_n(out[0]->receptive_field.begin(),data_dim.size()-2,1);
    out[0]->receptive_gap.resize(data_dim.size()-2);  fill_n(out[0]->receptive_gap.begin(),  data_dim.size()-2,1);
    out[0]->receptive_offset.resize(data_dim.size()-2);  fill_n(out[0]->receptive_offset.begin(),  data_dim.size()-2,0);
    memoryBytes += out[0]->Malloc(data_dim);

    out[1]->need_diff = false;
    memoryBytes += out[1]->Malloc(label_dim);

    out[2]->need_diff = false;
    memoryBytes += out[2]->Malloc(bb_tar_diff_dim);

    out[3]->need_diff = false;
    memoryBytes += out[3]->Malloc(bb_tar_diff_dim);

    if (is_combineimg){
      out[4]->need_diff = false;
      out[4]->receptive_field.resize(img_fea_dim.size()-2);   fill_n(out[4]->receptive_field.begin(),   img_fea_dim.size()-2,1);
      out[4]->receptive_gap.resize(img_fea_dim.size()-2);     fill_n(out[4]->receptive_gap.begin(),     img_fea_dim.size()-2,1);
      out[4]->receptive_offset.resize(img_fea_dim.size()-2);  fill_n(out[4]->receptive_offset.begin(),  img_fea_dim.size()-2,0);
      memoryBytes += out[4]->Malloc(img_fea_dim);
    }
     if (is_combinehha){
      out[5]->need_diff = false;
      out[5]->receptive_field.resize(img_fea_dim.size()-2);   fill_n(out[5]->receptive_field.begin(),   img_fea_dim.size()-2,1);
      out[5]->receptive_gap.resize(img_fea_dim.size()-2);     fill_n(out[5]->receptive_gap.begin(),     img_fea_dim.size()-2,1);
      out[5]->receptive_offset.resize(img_fea_dim.size()-2);  fill_n(out[5]->receptive_offset.begin(),  img_fea_dim.size()-2,0);
      memoryBytes += out[5]->Malloc(img_fea_dim);
    }
    // 
    if (box_2dreg){
        std::vector<int> bb_2d_tar_diff;
        bb_2d_tar_diff.push_back(sum(batch_size));
        bb_2d_tar_diff.push_back(4*num_categories); 
        bb_2d_tar_diff.push_back(1); bb_2d_tar_diff.push_back(1); bb_2d_tar_diff.push_back(1);

        bb_2d_diff = new Tensor<StorageT>(bb_2d_tar_diff);
        bb_2d_weights = new Tensor<StorageT>(bb_2d_tar_diff);

        out[6]->need_diff = false;
        memoryBytes += out[6]->Malloc(bb_2d_tar_diff);

        out[7]->need_diff = false;
        memoryBytes += out[7]->Malloc(bb_2d_tar_diff);
    }

    if (orein_cls){
        std::vector<int> oreintation_label_dim(5,1);
        std::vector<int> oreintation_label_w_dim(5,1);
        oreintation_label_dim[0]   =  sum(batch_size);
        oreintation_label_dim[1]   =  1;
        oreintation_label_dim[2]   =  num_categories;

        oreintation_label_w_dim[0] =  sum(batch_size);
        oreintation_label_w_dim[1] =  num_oreintation;
        oreintation_label_w_dim[2] =  num_categories;

        oreintation_label = new Tensor<StorageT>(oreintation_label_dim);
        oreintation_label_w = new Tensor<StorageT>(oreintation_label_w_dim);

        out[8]->need_diff = false;
        memoryBytes += out[8]->Malloc(oreintation_label_dim);

        out[9]->need_diff = false;
        memoryBytes += out[9]->Malloc(oreintation_label_w_dim);

    }

    checkCUDA(__LINE__, cudaMalloc(&dataGPU, numel(data_dim) * sizeofStorageT) );
    lock = std::async(std::launch::async,&Scene3DDataLayer::prefetch,this);
    //prefetch();
    return memoryBytes;
  };

  void prefetch(){
     checkCUDA(__LINE__,cudaSetDevice(GPU));

    //return;
    //int tmpD; cudaGetDevice(&tmpD); std::cout<<"GPU at prefetch LINE: "<<__LINE__<<" = "<<tmpD<<std::endl;
        
    unsigned long long  time1 = get_timestamp();
    std::vector<Scene3D*> chosen_scenes;
    std::vector<int> chosen_box_id;
    std::vector<int> choose_img_id(sum(batch_size));
    std::vector<int> category_count(num_categories,0);
    

    int totalpos, totalneg, total_percate;
    int count_batch_box = 0;

    if (batch_size.size()==1||phase==Testing) {
       totalneg = sum(batch_size);
       totalpos = 0;
       total_percate =0;
    }
    else{
       totalneg = batch_size[0];
       total_percate = num_percate*num_categories;
       totalpos = batch_size[1]-total_percate;
    }
    // set bb_tar_diff, bb_loss_weights to zeros
    if (box_reg){
      memset(bb_tar_diff->CPUmem,     0, bb_tar_diff->numBytes());
      memset(bb_loss_weights->CPUmem, 0, bb_loss_weights->numBytes());
    }

    if (box_2dreg){
      memset(bb_2d_diff->CPUmem,      0, bb_2d_diff->numBytes());
      memset(bb_2d_weights->CPUmem,   0, bb_2d_weights->numBytes());
    }

    if(orein_cls){
      memset(oreintation_label->CPUmem,  0, oreintation_label->numBytes());
      memset(oreintation_label_w->CPUmem,0, oreintation_label_w->numBytes());
    }
    
    // get the negtive boxes

    for (int i =0; i< totalneg; ++i){
      int imgId = imgobj_neg[counter_neg].ImageId;
      int objId = imgobj_neg[counter_neg].ObjId;

      choose_img_id[count_batch_box] = imgId;
      chosen_scenes.push_back(scenes[imgId]);
      chosen_box_id.push_back(objId);

      labelCPU->CPUmem[count_batch_box] = CPUCompute2StorageT(ComputeT(scenes[imgId]->objects[objId].category));
      ++counter_neg;
      ++count_batch_box;
      category_count[0]++;
      if (counter_neg >= imgobj_neg.size()){
        counter_neg = 0;
        ++epoch_prefetch;
        if(phase!=Testing){
          shuffle(true,false);
        }
      }
    }
    /*
    std::cout<<"totalneg: "<<totalneg<<std::endl;
    std::cout<<"labelCPU totalneg"<<std::endl;
    for (int i =0;i<10;i++){
       std::cout<< labelCPU->CPUmem[i]<<" , ";
    }
    std::cout<<std::endl;
    */
    // get the per category boxes
    if(total_percate>0){
      for (int i = 1; i< num_categories; ++i){
        for (int j = 0; j< num_percate; ++j){
          
            int imgId = imgobj_pos_cates[i][counter_pos_cates[i]].ImageId;
            int objId = imgobj_pos_cates[i][counter_pos_cates[i]].ObjId;
            int tarId = imgobj_pos_cates[i][counter_pos_cates[i]].TarId;

            choose_img_id[count_batch_box] = imgId;
            chosen_scenes.push_back(scenes[imgId]);
            chosen_box_id.push_back(objId);
            labelCPU->CPUmem[count_batch_box] = CPUCompute2StorageT(ComputeT(scenes[imgId]->objects[objId].category));

            if (box_reg){
              int idx = count_batch_box*bb_param_weight.size()*num_categories + scenes[imgId]->objects[objId].category*bb_param_weight.size();
              for (int cid = 0;cid<6;cid++){
                  bb_tar_diff->CPUmem[idx+cid]  = CPUCompute2StorageT(target_boxes[tarId].diff[cid]);
              }
              for (int cid = 0;cid<bb_param_weight.size();cid++){
                  bb_loss_weights->CPUmem[idx+cid] = CPUCompute2StorageT(bb_param_weight[cid]);
              }

            }
            

           if (box_2dreg){
              int idx_2d = count_batch_box*4*num_categories + scenes[imgId]->objects[objId].category*4;
              for (int cid = 0;cid<4;cid++){
                  bb_2d_diff ->CPUmem[idx_2d+cid]  = CPUCompute2StorageT(target_boxes[tarId].diff2d[cid]);
              }
              for (int cid = 0;cid<4;cid++){
                  bb_2d_weights ->CPUmem[idx_2d+cid] = CPUCompute2StorageT(1);
              }
           }

           if (orein_cls){
               if(target_boxes[tarId].oreintation >= 0){
                  oreintation_label ->CPUmem[count_batch_box*num_categories+scenes[imgId]->objects[objId].category] = CPUCompute2StorageT(ComputeT(target_boxes[tarId].oreintation));

                  for (int cid = 0;cid<num_oreintation;cid++){
                      oreintation_label_w ->CPUmem[count_batch_box*num_oreintation*num_categories+cid*num_categories+scenes[imgId]->objects[objId].category] = CPUCompute2StorageT(1);
                  }
               }
           }
            

            ++count_batch_box;
            category_count[i]++;
            counter_pos_cates[i]++;

            if (counter_pos_cates[i] >= imgobj_pos_cates[i].size()){
                counter_pos_cates[i] = 0;
                if(phase!=Testing){
                  shuffle(i);
                }
            }
        }
      }
    }
    

    // get the positive boxes

    for (int i =0; i< totalpos; ++i){
        int imgId = imgobj_pos[counter_pos].ImageId;
        int objId = imgobj_pos[counter_pos].ObjId;
        int tarId = imgobj_pos[counter_pos].TarId;

        choose_img_id[count_batch_box] = imgId;
        chosen_scenes.push_back(scenes[imgId]);
        chosen_box_id.push_back(objId);

        labelCPU->CPUmem[count_batch_box] = CPUCompute2StorageT(ComputeT(scenes[imgId]->objects[objId].category));
        
        if (box_reg){
            int idx = count_batch_box*bb_param_weight.size()*num_categories + scenes[imgId]->objects[objId].category*bb_param_weight.size();
            for (int cid = 0;cid<6;cid++){
                bb_tar_diff->CPUmem[idx+cid]  = CPUCompute2StorageT(target_boxes[tarId].diff[cid]);
            }
            for (int cid = 0;cid<bb_param_weight.size();cid++){
                bb_loss_weights->CPUmem[idx+cid] = CPUCompute2StorageT(bb_param_weight[cid]);
            }
        }
        
        if (box_2dreg){
          int idx_2d = count_batch_box*4*num_categories + scenes[imgId]->objects[objId].category*4;
          for (int cid = 0;cid<4;cid++){
              bb_2d_diff -> CPUmem[idx_2d+cid]  = CPUCompute2StorageT(target_boxes[tarId].diff2d[cid]);
          }
          for (int cid = 0;cid<4;cid++){
              bb_2d_weights -> CPUmem[idx_2d+cid] = CPUCompute2StorageT(1);
          }
        }

        if (orein_cls){
           if(target_boxes[tarId].oreintation >= 0){
               oreintation_label ->CPUmem[count_batch_box*num_categories+scenes[imgId]->objects[objId].category] = CPUCompute2StorageT(ComputeT(target_boxes[tarId].oreintation));
              //std::cout << target_boxes[tarId].oreintation<<":"<<oreintation_label->CPUmem[count_batch_box]<<",";

              for (int cid = 0;cid<num_oreintation;cid++){
                  oreintation_label_w ->CPUmem[count_batch_box*num_oreintation*num_categories+cid*num_categories+scenes[imgId]->objects[objId].category] = CPUCompute2StorageT(1);
                  //std::cout << oreintation_label_w ->CPUmem[count_batch_box*num_oreintation+cid]<<",";
              }
           }
        }
            

        ++counter_pos;
        ++count_batch_box;
        category_count[scenes[imgId]->objects[objId].category]++;

        if (counter_pos >= imgobj_pos.size()){
          counter_pos = 0;
          std::cout<<imgobj_pos.size()<<","<<totalpos<<std::endl;
          shuffle(false,true);
        }
      
    }

    
    //
    if (is_combineimg){
        Tensor<StorageT> * feaTensor;
        for (int i =0; i < chosen_scenes.size(); i++){
            if (i == 0){
                feaTensor = new Tensor<StorageT>(img_fea_folder +"fc7_"+ std::to_string(choose_img_id[i]) + ".tensor");
            }
            else if(choose_img_id[i]!=choose_img_id[i-1]){
               delete feaTensor;
               feaTensor = new Tensor<StorageT>(img_fea_folder +"fc7_"+ std::to_string(choose_img_id[i]) + ".tensor");
            }
            //std::cout<<"i = "<<i<<"  imgfea_dim="<<imgfea_dim<<"  chosen_box_id[i]="<<chosen_box_id[i]<<std::endl;
            memcpy(img_fea->CPUmem + i*imgfea_dim, feaTensor->CPUmem + chosen_box_id[i]*imgfea_dim, sizeofStorageT*imgfea_dim);
        }
        delete feaTensor;
    }

    if (is_combinehha){
        Tensor<StorageT> * feaTensor;
        for (int i =0; i< chosen_scenes.size(); i++){
            if (i == 0){
              feaTensor = new Tensor<StorageT>(img_fea_folder +"da_fc7_"+  std::to_string(choose_img_id[i]) +".tensor");
            }
            else if(choose_img_id[i]!=choose_img_id[i-1]){
              delete feaTensor;
              feaTensor = new Tensor<StorageT>(img_fea_folder +"da_fc7_"+  std::to_string(choose_img_id[i]) +".tensor");
            }
            memcpy(hha_fea->CPUmem + i*imgfea_dim, feaTensor->CPUmem + chosen_box_id[i]*imgfea_dim, sizeofStorageT*imgfea_dim);
            //std::cout<<img_fea_folder +"da_fc7_"+  std::to_string(choose_img_id[i]) +".tensor"<<std::endl;
            //std::cout<<chosen_scenes[i]->filename<<std::endl;
            //std::cout<<"chosen_box_id: "<<chosen_box_id[i]<<std::endl;
        }
        delete feaTensor; 
        
    }
    //std::cin.ignore();
    // now, the list of scenes and wanted boxes are in chosen_scenes, compute the TSDF and store the data in dataCPU
    compute_TSDF(&chosen_scenes, &chosen_box_id, dataGPU, grid_size, encode_type, scale); 
    unsigned long long  time2 = get_timestamp();

    std::cout <<"category_count";veciPrint(category_count);std::cout<<std::endl;
    std::cout << "perfetch time " << (time2-time1)/1000 <<"ms"<<std::endl;
    


    /*
    std::string outputfile = "DSS/debug/feature.bin";
    FILE * fid = fopen(outputfile.c_str(),"wb");
    fwrite(dataCPU->CPUmem,sizeof(float),(sum(batch_size))*3*30*30*30,fid);
    fclose(fid);
    std::cout<< labelCPU->CPUmem[0]<<","<< labelCPU->CPUmem[1]<<","<<labelCPU->CPUmem[2]<<std::endl;
    */
    //dataCPU->CPUmem = float [sum(batch_size)* 6 * 30 * 30 * 30];
    //labelCPU->CPUmem = float [sum(batch_size)* 6 * 1 * 1 * 1];

    //std::cout<<"prefetch GPU="<<GPU<<std::endl;
    //checkCUDA(__LINE__,cudaSetDevice(GPU));
    /*
    if(phase==Training){
      oreintation_label->write("DSS/debug/oreintation_label.tensor");
      oreintation_label_w->write("DSS/debug/oreintation_label_w.tensor");
      std::cout<<"finish writing"<<std::endl;
      //std::cin.ignore();
      
    }
    */
    

  };

  void forward(Phase phase_){

    lock.wait();
    std::swap(out[0]->dataGPU,dataGPU);
    checkCUDA(__LINE__, cudaMemcpy(out[1]->dataGPU, labelCPU->CPUmem,        labelCPU->numBytes(),        cudaMemcpyHostToDevice) );

    if (box_reg){
       checkCUDA(__LINE__, cudaMemcpy(out[2]->dataGPU, bb_tar_diff->CPUmem,     bb_tar_diff->numBytes(),     cudaMemcpyHostToDevice) );
       checkCUDA(__LINE__, cudaMemcpy(out[3]->dataGPU, bb_loss_weights->CPUmem, bb_loss_weights->numBytes(), cudaMemcpyHostToDevice) );
    }

    if (is_combineimg){
        checkCUDA(__LINE__, cudaMemcpy(out[4]->dataGPU, img_fea->CPUmem,     img_fea->numBytes(),      cudaMemcpyHostToDevice) );
    }
    if (is_combinehha){
        checkCUDA(__LINE__, cudaMemcpy(out[5]->dataGPU, hha_fea->CPUmem,     hha_fea->numBytes(),      cudaMemcpyHostToDevice) );
    }

    if (box_2dreg){
      checkCUDA(__LINE__, cudaMemcpy(out[6]->dataGPU, bb_2d_diff->CPUmem,     bb_2d_diff->numBytes(),      cudaMemcpyHostToDevice) );
      checkCUDA(__LINE__, cudaMemcpy(out[7]->dataGPU, bb_2d_weights->CPUmem,  bb_2d_weights->numBytes(),   cudaMemcpyHostToDevice) );
    }

    if(orein_cls){
      //std::cout<<"copied"<<std::endl;
      checkCUDA(__LINE__, cudaMemcpy(out[8]->dataGPU, oreintation_label->CPUmem,    oreintation_label->numBytes(),     cudaMemcpyHostToDevice) );
      checkCUDA(__LINE__, cudaMemcpy(out[9]->dataGPU, oreintation_label_w->CPUmem,  oreintation_label_w->numBytes(),   cudaMemcpyHostToDevice) );
    }
    
    epoch = epoch_prefetch;
    lock = std::async(std::launch::async,&Scene3DDataLayer::prefetch,this);

    //prefetch();
    //checkCUDA(__LINE__, cudaMemcpy(out[0]->dataGPU, dataCPU->CPUmem,  dataCPU->numBytes() , cudaMemcpyHostToDevice) );
    //checkCUDA(__LINE__, cudaMemcpy(out[0]->dataGPU, dataGPU,  numel(data_dim) * sizeof(float) , cudaMemcpyDeviceToDevice) );
    /*
    {
      stringstream sstm;
      sstm << "/n/fs/modelnet/deepDetect/code/caffe_gpu2/debug/" << epoch << "_" << "bottom_0_conv1.tensor";
      std::string fname = sstm.str();
      Tensor* dataCPU = new Tensor(fname);
      std::cout<<"data ="; veciPrint(dataCPU->dim); std::cout<<std::endl;
      checkCUDA(__LINE__, cudaMemcpy(out[0]->dataGPU, dataCPU->CPUmem,  dataCPU->numBytes() , cudaMemcpyHostToDevice) );
      delete dataCPU;
    }
    {
      stringstream sstm;
      sstm << "/n/fs/modelnet/deepDetect/code/caffe_gpu2/debug/" << epoch << "_" << "bottom_1_loss.tensor";
      std::string fname = sstm.str();
      Tensor* labelCPU = new Tensor(fname);
      std::cout<<"label="; veciPrint(labelCPU->dim); std::cout<<std::endl;
      checkCUDA(__LINE__, cudaMemcpy(out[1]->dataGPU, labelCPU->CPUmem,  labelCPU->numBytes() , cudaMemcpyHostToDevice) );
      delete labelCPU;
    }

    epoch++;

    */

  };
};

class RPNDataLayer : public DataLayer {
public:
  int epoch_prefetch;
  //Tensor* dataCPU;
  StorageT*   dataGPU;
  Tensor<StorageT>* labelCPU;
  Tensor<StorageT>* label_weights;
  Tensor<StorageT>* bb_tar_diff;
  Tensor<StorageT>* bb_loss_weights;

  Tensor<StorageT>* labelCPU_1;
  Tensor<StorageT>* label_weights_1;
  Tensor<StorageT>* bb_tar_diff_1;
  Tensor<StorageT>* bb_loss_weights_1;

  Box3D SpaceBox;
  std::vector<int> grid_size;
  std::vector<int> label_size;
  std::vector<int> batch_size;


  std::vector<int> bb_param_weight;
  std::vector<int> size_group;
  std::vector<int> num_size_group;
  std::vector<int> size_group_map;

  std::vector<Scene3D*> scenes; // store the scenes without box 

  std::string file_list;
  std::string data_root;// to store the depth and image 
  std::string rpn_data_root;

  int encode_type;
  float scale;
  float context_pad;

  unsigned int num_categories;
  unsigned int num_anchors;
  std::vector<float> pos_overlap;
  float neg_overlap;
  float pos_weight;
  float neg_weight;

  std::future<void> lock;

  

  void shuffle(){
    std::shuffle(scenes.begin(),scenes.end(), rng );
  };
  int numofitems(){ 
    return scenes.size(); 
  };

  /*
  int numofitemsTruncated(){
    return sum(batch_size) * floor(double(numofitems())/double(sum(batch_size)));
  };
  */


  void init(){
    epoch_prefetch = 0;
    counter = 0;
    train_me = false;
    if (phase==Testing){
      batch_size[0] = sum(batch_size);
      batch_size.resize(1);
    }
    SpaceBox.category   = 0;
    float base[9] = {1,0,0,0,1,0,0,0,1};
    for (int i=0;i<9;i++){
      SpaceBox.base[i] = base[i];
    }
    SpaceBox.center[0] = 0; 
    SpaceBox.center[1] = 3.0; 
    SpaceBox.center[2] = -0.25; //{0.0 , 3.0 , -0.25};
    SpaceBox.coeff[0]  = 2.6; 
    SpaceBox.coeff[1]  = 2.6; 
    SpaceBox.coeff[2]  = 1.25; //{2.6 , 2.6 , 1.25};

    // get size_group
    if (size_group.size()!=num_anchors){
      std::cout<<"size_group.size()!=num_anchors"<<std::endl;
      FatalError(__LINE__);
    }

    size_group_map.resize(num_anchors);
    for(int i =0;i<size_group.size();i++){
      if (size_group[i]>=num_size_group.size()){
        num_size_group.resize(size_group[i]+1);
      }
      size_group_map[i] = num_size_group[size_group[i]];
      num_size_group[size_group[i]]++;
    }
    std::cout<<"num_size_group: "; veciPrint(num_size_group);  std::cout<<std::endl;
    std::cout<<"size_group_map: "; veciPrint(size_group_map);  std::cout<<std::endl;


     // read in image list 
    std::cout<<"loading file "<<file_list<<"\n";
    FILE* fp = fopen(file_list.c_str(),"rb");
    if (fp==NULL) { std::cout<<"fail to open file: "<<file_list<<std::endl; exit(EXIT_FAILURE); }
     // suffle 
    size_t file_size;
    while (feof(fp)==0) {
      Scene3D* scene = new Scene3D();
      unsigned int len = 0;
      file_size += fread((void*)(&len), sizeof(unsigned int), 1, fp);    
      if (len==0) break;
      scene->filename.resize(len);
      if (len>0) file_size += fread((void*)(scene->filename.data()), sizeof(char), len, fp);
      scene->seqname = scene->filename;
      scene->filename = data_root+scene->filename+".bin"; 
  
      file_size += fread((void*)(scene->R), sizeof(float), 9, fp);
      file_size += fread((void*)(scene->K), sizeof(float), 9, fp);
      file_size += fread((void*)(&scene->height), sizeof(unsigned int), 1, fp);
      file_size += fread((void*)(&scene->width), sizeof(unsigned int), 1, fp); 
      scenes.push_back(scene);

      //std::cout<<scene->seqname<<std::endl;
    }
    fclose(fp);
    //if (phase!=Testing) shuffle();
  };

  RPNDataLayer(std::string name_, Phase phase_, std::string file_list_, int batch_size_): DataLayer(name_), batch_size(batch_size_), file_list(file_list_){
    phase = phase_;
    init();
  };

  RPNDataLayer(JSON* json){
    SetValue(json, name,    "RPNDataLayer")
    SetValue(json, phase,   Training)
    SetOrDie(json, file_list    )
    SetOrDie(json, data_root    )
    SetOrDie(json, rpn_data_root  )
    SetOrDie(json, grid_size    )
    SetOrDie(json, batch_size     )
    SetOrDie(json, label_size     )
    SetOrDie(json, encode_type    )
    SetOrDie(json, scale      )
    SetOrDie(json, context_pad    )
    SetOrDie(json, GPU        )
    SetOrDie(json, num_categories   )
    SetOrDie(json, num_anchors    )
    SetOrDie(json, bb_param_weight  )
    SetOrDie(json, pos_overlap    )
    SetOrDie(json, neg_overlap    )
    SetOrDie(json, pos_weight     )
    SetOrDie(json, neg_weight     )
    SetOrDie(json, size_group     )

    

    std::cout<<"grid_size:   ";  veciPrint(grid_size);  std::cout<<std::endl;
    std::cout<<"batch_size:  ";  veciPrint(batch_size); std::cout<<std::endl;
    std::cout<<"encode_type: " << encode_type << std::endl;
    std::cout<<"scale:     " << scale << std::endl;
    std::cout<<"context_pad: " << context_pad << std::endl;
    std::cout<<"RPNDataLayer GPU: " << GPU << std::endl;
    std::cout<<"data_root: " << data_root <<std::endl;
    std::cout<<"bb_param_weight: ";  veciPrint(bb_param_weight);  std::cout<<std::endl;
    std::cout<<"label_size: ";     veciPrint(label_size);  std::cout<<std::endl;

    std::cout<<"pos_overlap: ";    vecfPrint(pos_overlap);  std::cout<<std::endl;
    std::cout<<"pos_weight:    " << pos_weight << std::endl;
    std::cout<<"neg_weight:    " << neg_weight << std::endl;
    init();
  };

  ~RPNDataLayer(){
    if (lock.valid()) lock.wait();
    for(size_t i=0;i<scenes.size();i++){
      if (scenes[i]!=NULL) delete scenes[i];
    }
  };

  size_t Malloc(Phase phase_){
    if (phase == Training && phase_==Testing) return 0;
    size_t memoryBytes = 0;
    std::cout<< (train_me? "* " : "  ");
    std::cout<<name<<std::endl;

    std::vector<int> data_dim;
    std::vector<int> label_dim;  
    std::vector<int> bb_tar_diff_dim;
    std::vector<int> label_weights_dim;

    // allocate data
    data_dim.push_back(sum(batch_size));
    data_dim.insert( data_dim.end(), grid_size.begin(), grid_size.end() );
    memoryBytes += numel(data_dim) * sizeof(float)/4;
    checkCUDA(__LINE__, cudaMalloc(&dataGPU, numel(data_dim) * sizeofStorageT) );
    out[0]->need_diff = false;
    out[0]->receptive_field.resize(data_dim.size()-2);  fill_n(out[0]->receptive_field.begin(),data_dim.size()-2,0.025);
    out[0]->receptive_gap.resize(data_dim.size()-2);  fill_n(out[0]->receptive_gap.begin(),  data_dim.size()-2,0.025);
    out[0]->receptive_offset.resize(data_dim.size()-2);  fill_n(out[0]->receptive_offset.begin(),  data_dim.size()-2,0);
    memoryBytes += out[0]->Malloc(data_dim);


    // allocate label 
    label_dim.push_back(sum(batch_size));
    label_dim.push_back(1);
    label_dim.push_back(num_size_group[0]);
    label_dim.insert(label_dim.end(), label_size.begin(), label_size.end() );   
    
    label_weights_dim.push_back(sum(batch_size));
    label_weights_dim.push_back(2);
    label_weights_dim.push_back(num_size_group[0]);
    label_weights_dim.insert(label_weights_dim.end(), label_size.begin(), label_size.end() );   

    bb_tar_diff_dim.push_back(sum(batch_size));
    bb_tar_diff_dim.push_back(bb_param_weight.size());  
    bb_tar_diff_dim.push_back(num_size_group[0]);
    bb_tar_diff_dim.insert(bb_tar_diff_dim.end(), label_size.begin(), label_size.end() ); 
    
    
    

    labelCPU = new Tensor<StorageT>(label_dim);
    label_weights = new Tensor<StorageT>(label_weights_dim);
    bb_tar_diff = new Tensor<StorageT>(bb_tar_diff_dim);
    bb_loss_weights = new Tensor<StorageT>(bb_tar_diff_dim);
    

    out[1]->need_diff = false;
    memoryBytes += out[1]->Malloc(label_dim);

    out[2]->need_diff = false;
    memoryBytes += out[2]->Malloc(label_weights_dim);

    out[3]->need_diff = false;
    memoryBytes += out[3]->Malloc(bb_tar_diff_dim);

    out[4]->need_diff = false;
    memoryBytes += out[4]->Malloc(bb_tar_diff_dim);


    if (num_size_group.size()>1){
      label_dim[2]     = num_size_group[1];
      label_weights_dim[2] = num_size_group[1];
      bb_tar_diff_dim[2]   = num_size_group[1];

      labelCPU_1 = new Tensor<StorageT>(label_dim);
      label_weights_1 = new Tensor<StorageT>(label_weights_dim);
      bb_tar_diff_1 = new Tensor<StorageT>(bb_tar_diff_dim);
      bb_loss_weights_1 = new Tensor<StorageT>(bb_tar_diff_dim);

      out[5]->need_diff = false;
      memoryBytes += out[5]->Malloc(label_dim);

      out[6]->need_diff = false;
      memoryBytes += out[6]->Malloc(label_weights_dim);

      out[7]->need_diff = false;
      memoryBytes += out[7]->Malloc(bb_tar_diff_dim);

      out[8]->need_diff = false;
      memoryBytes += out[8]->Malloc(bb_tar_diff_dim);

    }
    
    lock = std::async(std::launch::async,&RPNDataLayer::prefetch,this);
    //prefetch();
    return memoryBytes;
  };

  void prefetch(){
    checkCUDA(__LINE__,cudaSetDevice(GPU));
    // set mem to be zeros
    memset(labelCPU        ->CPUmem,    0, labelCPU->numBytes());
    memset(label_weights   ->CPUmem,    0, label_weights->numBytes());
    memset(bb_tar_diff     ->CPUmem,    0, bb_tar_diff->numBytes());
    memset(bb_loss_weights ->CPUmem,    0, bb_loss_weights->numBytes());
    if (num_size_group.size()>0){
      memset(labelCPU_1    ->CPUmem,   0, labelCPU_1->numBytes());
      memset(label_weights_1   ->CPUmem,   0, label_weights_1->numBytes());
      memset(bb_tar_diff_1   ->CPUmem,   0, bb_tar_diff_1->numBytes());
      memset(bb_loss_weights_1 ->CPUmem,   0, bb_loss_weights_1->numBytes());
    }
    // compute TSDF 
    for (int batch_id =0; batch_id < sum(batch_size); ++batch_id){
      
      StorageT * tsdf_data_GPU = &dataGPU[batch_id*grid_size[0]*grid_size[1]*grid_size[2]*grid_size[3]];
      compute_TSDF_Space(scenes[counter], SpaceBox, tsdf_data_GPU, grid_size, encode_type, scale); 

      std::string   spacefile = rpn_data_root + scenes[counter]->seqname + ".bin";
      FILE* fp = fopen(spacefile.c_str(),"rb");
      if (fp==NULL) { std::cout<<"fail to open file: "<<spacefile<<std::endl; exit(EXIT_FAILURE); }
      //std::cout<< spacefile << std::endl;

      int len = 0;
      size_t file_size = 0;
      file_size += fread(&len, sizeof(int), 1, fp);
      
      //std::cout<<len<<std::endl;
      //int* linearInd = new int[len];
      //fread(linearInd, sizeof(int), len, fp);
      uint8_t* anchor_Idx = new uint8_t[len];
      file_size += fread(anchor_Idx, sizeof(uint8_t), len, fp);

      uint8_t* x_Idx = new uint8_t[len];
      file_size += fread(x_Idx, sizeof(uint8_t), len, fp);

      uint8_t* y_Idx = new uint8_t[len];
      file_size += fread(y_Idx, sizeof(uint8_t), len, fp);

      uint8_t* z_Idx = new uint8_t[len];
      file_size += fread(z_Idx, sizeof(uint8_t), len, fp);
      
      float* oscf = new float[len];
      file_size += fread(oscf, sizeof(float), len, fp);
      
      float* diff = new float[6*len];
      file_size += fread(diff, sizeof(float), 6*len, fp);

      fclose(fp);

      int vol,vol1;
      vol = num_size_group[0] *label_size[0]*label_size[1]*label_size[2];
      if (num_size_group.size()>0){
        vol1 = num_size_group[1] *label_size[0]*label_size[1]*label_size[2];
      }
      //std::cout<<"vol"<<vol<<","<<vol1;

      std::vector<int> neg_list;
      int poscount = 0;
      int negcount = 0;
      std::vector<int> posnegcount={0,0,0,0};

      for(int idx = 0; idx<len; ++idx){
        float ov = oscf[idx];
        if (ov >= min(pos_overlap[0],pos_overlap[1])){
          // check in which btach and caculate idex
          int groupid = size_group[anchor_Idx[idx]];
          int linearInd = size_group_map[anchor_Idx[idx]]*label_size[0]*label_size[1]*label_size[2]
                  + x_Idx[idx]*label_size[1]*label_size[2]
                  + y_Idx[idx]*label_size[2] 
                  + z_Idx[idx];

          if (groupid == 0&& ov >= pos_overlap[0]){
            labelCPU->CPUmem[batch_id*vol+linearInd] = CPUCompute2StorageT(ComputeT(1));
            for (int pid = 0; pid<bb_param_weight.size(); ++pid){
              bb_tar_diff    ->CPUmem[batch_id*bb_param_weight.size()*vol+pid*vol+linearInd] = CPUCompute2StorageT(diff[6*idx+pid]);
              bb_loss_weights->CPUmem[batch_id*bb_param_weight.size()*vol+pid*vol+linearInd] = CPUCompute2StorageT(bb_param_weight[pid]);
            }
            label_weights->CPUmem[batch_id*2*vol + 0*vol + linearInd] = CPUCompute2StorageT(pos_weight);
            label_weights->CPUmem[batch_id*2*vol + 1*vol + linearInd] = CPUCompute2StorageT(pos_weight);
            posnegcount[0]++;
          }
          else if(groupid == 1&& ov >= pos_overlap[1]){
            labelCPU_1->CPUmem[batch_id*vol1+linearInd] = CPUCompute2StorageT(ComputeT(1));
            for (int pid = 0; pid<bb_param_weight.size(); ++pid){
              //std::cout<<diff[6*idx+pid];
              bb_tar_diff_1     ->CPUmem[batch_id*bb_param_weight.size()*vol1+pid*vol1+linearInd] = CPUCompute2StorageT(diff[6*idx+pid]);
              bb_loss_weights_1 ->CPUmem[batch_id*bb_param_weight.size()*vol1+pid*vol1+linearInd] = CPUCompute2StorageT(bb_param_weight[pid]);
            }
            //std::cout<<std::endl;
            label_weights_1 ->CPUmem[batch_id*2*vol1 + 0*vol1 + linearInd] = CPUCompute2StorageT(pos_weight);
            label_weights_1 ->CPUmem[batch_id*2*vol1 + 1*vol1 + linearInd] = CPUCompute2StorageT(pos_weight);
            posnegcount[1]++;
          }
          poscount++;
          
        }
        else if(ov < neg_overlap){
          neg_list.push_back(idx);
        }
      }

      int numneg = max(256-poscount,poscount);
      std::uniform_int_distribution<int> distribution(0,neg_list.size()-1);
      while (negcount<numneg) {
        int randnum = distribution(rng);
        int idx = neg_list[randnum];
        int groupid = size_group[anchor_Idx[idx]];
        int linearInd = size_group_map[anchor_Idx[idx]]*label_size[0]*label_size[1]*label_size[2]
              + x_Idx[idx]*label_size[1]*label_size[2]
              + y_Idx[idx]*label_size[2] 
              + z_Idx[idx];
        if (groupid == 0){
              if (CPUStorage2ComputeT(label_weights->CPUmem[batch_id*2*vol + 0*vol + linearInd]) < neg_weight){

                label_weights->CPUmem[batch_id*2*vol + 0*vol + linearInd] = CPUCompute2StorageT(neg_weight);
                label_weights->CPUmem[batch_id*2*vol + 1*vol + linearInd] = CPUCompute2StorageT(neg_weight);
                posnegcount[2]++;
                negcount++;
              }
        }
        else{
              if (CPUStorage2ComputeT(label_weights_1->CPUmem[batch_id*2*vol1 + 0*vol1 + linearInd]) < neg_weight){

                label_weights_1->CPUmem[batch_id*2*vol1 + 0*vol1 + linearInd] = CPUCompute2StorageT(neg_weight);
                label_weights_1->CPUmem[batch_id*2*vol1 + 1*vol1 + linearInd] = CPUCompute2StorageT(neg_weight);
                posnegcount[3]++;
                negcount++;
              }
        }
        
      }
            
      
      //std::cout << "poscount : "<< poscount <<" numneg"<<numneg <<std::endl;
      std::cout<<"posnegcount: "; veciPrint(posnegcount);  std::cout<<std::endl;

      delete[] anchor_Idx;
      delete[] x_Idx;
      delete[] y_Idx;
      delete[] z_Idx;
      delete[] oscf;
      delete[] diff;

      ++counter;
      if (counter >= scenes.size()){
          counter = 0;
          ++epoch_prefetch;
          shuffle();
      }

      /*
      
        if (counter==1&&phase!=Testing){
          std::cout << " poscount : "<< poscount << "." <<spacefile<<" numneg"<<numneg <<std::endl;
          float* dataCPUmem = new float[grid_size[0]*grid_size[1]*grid_size[2]*grid_size[3]];
        checkCUDA(__LINE__, cudaMemcpy(dataCPUmem, dataGPU,grid_size[0]*grid_size[1]*grid_size[2]*grid_size[3]*sizeof(float), cudaMemcpyDeviceToHost) );
          std::string outputfile = "DSS/debug/feature.bin";
        FILE * fid = fopen(outputfile.c_str(),"wb");
          fwrite(dataCPUmem,sizeof(float),grid_size[0]*grid_size[1]*grid_size[2]*grid_size[3],fid);
        fclose(fid);

        labelCPU->write("DSS/debug/labelCPU.tensor");
        label_weights->write("DSS/debug/label_weights.tensor");
        bb_tar_diff->write("DSS/debug/bb_tar_diff.tensor");
        bb_loss_weights->write("DSS/debug/bb_loss_weights.tensor");

        labelCPU_1->write("DSS/debug/labelCPU_1.tensor");
        label_weights_1->write("DSS/debug/label_weights_1.tensor");
        bb_tar_diff_1->write("DSS/debug/bb_tar_diff_1.tensor");
        bb_loss_weights_1->write("DSS/debug/bb_loss_weights_1.tensor");
        std::cout<<"finish writing"<<std::endl;
      }
      
      
      */

    }//for (int batch_id =0; batch_id< sum(batch_size); ++batch_id)
  };

  void forward(Phase phase_){
    lock.wait();
    std::swap(out[0]->dataGPU,dataGPU);
    checkCUDA(__LINE__, cudaMemcpy(out[1]->dataGPU, labelCPU->CPUmem,      labelCPU->numBytes(),      cudaMemcpyHostToDevice) );
    checkCUDA(__LINE__, cudaMemcpy(out[2]->dataGPU, label_weights->CPUmem,   label_weights->numBytes(),   cudaMemcpyHostToDevice) );
    checkCUDA(__LINE__, cudaMemcpy(out[3]->dataGPU, bb_tar_diff->CPUmem,   bb_tar_diff->numBytes(),     cudaMemcpyHostToDevice) );
    checkCUDA(__LINE__, cudaMemcpy(out[4]->dataGPU, bb_loss_weights->CPUmem, bb_loss_weights->numBytes(), cudaMemcpyHostToDevice) );
    if (num_size_group.size()>0){
      checkCUDA(__LINE__, cudaMemcpy(out[5]->dataGPU, labelCPU_1->CPUmem,      labelCPU_1->numBytes(),      cudaMemcpyHostToDevice) );
      checkCUDA(__LINE__, cudaMemcpy(out[6]->dataGPU, label_weights_1->CPUmem,   label_weights_1->numBytes(),     cudaMemcpyHostToDevice) );
      checkCUDA(__LINE__, cudaMemcpy(out[7]->dataGPU, bb_tar_diff_1->CPUmem,     bb_tar_diff_1->numBytes(),     cudaMemcpyHostToDevice) );
      checkCUDA(__LINE__, cudaMemcpy(out[8]->dataGPU, bb_loss_weights_1->CPUmem,   bb_loss_weights_1->numBytes(),   cudaMemcpyHostToDevice) );
    }
    epoch = epoch_prefetch;
    lock = std::async(std::launch::async,&RPNDataLayer::prefetch,this);
    //prefetch();
  };
};


class RenderMeshDataLayer : public DataLayer {
public:
    int epoch_prefetch;
    Tensor<StorageT >* labelCPU; 
    StorageT*   dataGPU;

    std::future<void> lock;
    std::vector<int> batch_size;

    std::string file_list;
    int context_pad;
    std::vector<int> grid_size;
    int encode_type;
    float scale;
    unsigned int num_categories;

    std::string off_root;
    std::vector<float> heigth_dis;
    std::vector<sceneMesh*> sceneMeshList;
    Tensor<StorageT >* oreintation_label;
    Tensor<StorageT >* oreintation_label_w;

    
    float* K_GPU;
    float* R_GPU;
    float* depth_GPU;
    float* XYZimage_GPU;
    float* bb3d_GPU;
    int num_oreintation;
    bool orein_cls;
    
    int im_w;
    int im_h;
    float camK[9];

    bool shuffle_data;

    int numofitems(){ 
      return sceneMeshList.size();  
    };

    int numofitemsTruncated(){
      return sum(batch_size) * floor(double(numofitems())/double(sum(batch_size)));
    };

    void shuffle(){
      std::shuffle(sceneMeshList.begin(),sceneMeshList.end(), rng );
      return;
    };

    void init(){
      epoch_prefetch = 0;
 
      train_me = false;
      counter =0;
      std::cout<<"loading file "<<file_list<<"\n";
      FILE* fp = fopen(file_list.c_str(),"rb");
      if (fp==NULL) { std::cout<<"fail to open file: "<<file_list<<std::endl; exit(EXIT_FAILURE); }

      
      unsigned int len = 0;
      size_t file_size = 0;
      file_size += fread((void*)(&len), sizeof(unsigned int), 1, fp);    
      std::string height_dis_file;
      height_dis_file.resize(len);
      if (len>0) file_size += fread((void*)(height_dis_file.data()), sizeof(char), len, fp);
      std::ifstream infile; 
        infile.open(height_dis_file); 
        int numcls;
        infile >> numcls; 
        heigth_dis.resize(2*numcls);
        std::vector<std::string> classname(numcls);
        for (int i =0;i<numcls;i++){ 
          infile >> classname[i];
          infile >> heigth_dis[2*i] ; 
          infile >> heigth_dis[2*i+1]; 
        }
        infile.close();
        
        std::cout<<"numcls "<<numcls<<"\n";
        std::cout<<"load file "<<height_dis_file<<"\n";

      while (feof(fp)==0) {
        sceneMesh* scene = new sceneMesh();
        unsigned int len = 0;
        file_size += fread((void*)(&len), sizeof(unsigned int), 1, fp);    
        if (len==0) break;
        
        scene->mesh_file.resize(len);
        if (len>0) file_size += fread((void*)(scene->mesh_file.data()), sizeof(char), len, fp);
        //mesh_file = off_root+mesh_file; 
        //scene->meshdata.readOFF(mesh_file);
        scene->objects.resize(1);
        Box3D box;
        file_size += fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp);
        file_size += fread((void*)(box.base),        sizeof(float), 9, fp);
        file_size += fread((void*)(box.center),      sizeof(float), 3, fp);
        file_size += fread((void*)(box.coeff),       sizeof(float), 3, fp);
        box = processbox (box, context_pad, grid_size[1]);
        scene->objects[0] = box;
        sceneMeshList.push_back(scene);
      
      }
      fclose(fp);
      std::cout<< "sceneMeshList length: "<<sceneMeshList.size()<<std::endl;
      if (shuffle_data) shuffle();

    };

    RenderMeshDataLayer(std::string name_, Phase phase_, std::string file_list_, int batch_size_): DataLayer(name_), batch_size(batch_size_), file_list(file_list_){
      phase = phase_;
      init();
    };

    RenderMeshDataLayer(JSON* json){
      SetValue(json, name,    "RenderMeshDataLayer")
      SetValue(json, phase,   Training)
      SetOrDie(json, file_list  )
      SetOrDie(json, off_root   )
      SetOrDie(json, grid_size  )
      SetOrDie(json, batch_size   )
      SetOrDie(json, encode_type  )
      SetOrDie(json, scale    )
      SetOrDie(json, context_pad  )
      SetOrDie(json, num_categories   )
      SetValue(json, num_oreintation, 20)
      SetValue(json, orein_cls, false)
      SetValue(json, shuffle_data, true)
      
      
      im_w = 640;
      im_h = 480;
      camK[0] = 5.1885790117450188e+02; camK[1] = 0; camK[2] = 320;
      camK[3] = 0; camK[4] = 5.1885790117450188e+02; camK[5] = 240;
      camK[6] = 0; camK[7] = 0; camK[8] = 1;
      //{fx, 0, cx, 0, fy, cy, 0, 0, 1};


      std::cout<<"grid_size:   ";  veciPrint(grid_size);  std::cout<<std::endl;
      std::cout<<"batch_size:  ";  veciPrint(batch_size); std::cout<<std::endl;
      std::cout<<"encode_type: " << encode_type << std::endl;
      std::cout<<"scale:     " << scale << std::endl;
      std::cout<<"context_pad: " << context_pad << std::endl;
      std::cout<<"orein_cls: " << orein_cls <<std::endl;
      std::cout<<"num_oreintation: " << num_oreintation <<std::endl;
      
      
      K_GPU      = NULL;
      R_GPU      = NULL;
      depth_GPU    = NULL;
      XYZimage_GPU = NULL;
      bb3d_GPU   = NULL;
      
      init();
    };

    ~RenderMeshDataLayer(){
      if (lock.valid()) lock.wait();
      for(size_t i=0;i<sceneMeshList.size();i++){
        if (sceneMeshList[i]!=NULL) delete sceneMeshList[i];
      }
      if (R_GPU != NULL){     checkCUDA(__LINE__, cudaFree(R_GPU));       R_GPU = NULL;}
      if (K_GPU != NULL){     checkCUDA(__LINE__, cudaFree(K_GPU));     K_GPU = NULL;}
      if (XYZimage_GPU !=NULL){ checkCUDA(__LINE__, cudaFree(XYZimage_GPU));  XYZimage_GPU = NULL;}
      if (bb3d_GPU     !=NULL){    checkCUDA(__LINE__, cudaFree(bb3d_GPU));    bb3d_GPU  = NULL;}
      if (depth_GPU    !=NULL){   checkCUDA(__LINE__, cudaFree(depth_GPU));   depth_GPU = NULL;}

    };

    size_t Malloc(Phase phase_){
      if (phase == Training && phase_==Testing) return 0;
      size_t memoryBytes = 0;
      std::cout<< (train_me? "* " : "  ");
      std::cout<<name<<std::endl;

      std::vector<int>data_dim,label_dim;

      data_dim.push_back(sum(batch_size));
      data_dim.insert( data_dim.end(), grid_size.begin(), grid_size.end() );
      label_dim.push_back(sum(batch_size));
      label_dim.push_back(1);   label_dim.push_back(1);   label_dim.push_back(1);   label_dim.push_back(1);

      labelCPU = new Tensor<StorageT>(label_dim);

      out[0]->need_diff = false;
      memoryBytes += out[0]->Malloc(data_dim);
      //std::cout<<"data_dim="; veciPrint(data_dim); std::cout<<std::endl;

      out[1]->need_diff = false;
      memoryBytes += out[1]->Malloc(label_dim);
      //std::cout<<"label_dim="; veciPrint(label_dim); std::cout<<std::endl;

      if (orein_cls){
        std::vector<int> oreintation_label_dim(5,1);
        std::vector<int> oreintation_label_w_dim(5,1);
        oreintation_label_dim[0]   =  sum(batch_size);
        oreintation_label_dim[1]   =  1;
        oreintation_label_dim[2]   =  num_categories;

        oreintation_label_w_dim[0] =  sum(batch_size);
        oreintation_label_w_dim[1] =  num_oreintation;
        oreintation_label_w_dim[2] =  num_categories;

        oreintation_label = new Tensor<StorageT>(oreintation_label_dim);
        oreintation_label_w = new Tensor<StorageT>(oreintation_label_w_dim);

        out[2]->need_diff = false;
        memoryBytes += out[2]->Malloc(oreintation_label_dim);

        out[3]->need_diff = false;
        memoryBytes += out[3]->Malloc(oreintation_label_w_dim);

      }

      checkCUDA(__LINE__, cudaMalloc(&dataGPU, numel(data_dim) * sizeofStorageT) );

      checkCUDA(__LINE__, cudaMalloc(&K_GPU, sizeof(float)*9));
      checkCUDA(__LINE__, cudaMalloc(&R_GPU, sizeof(float)*9));
      checkCUDA(__LINE__, cudaMalloc(&depth_GPU, sizeof(float)*im_w*im_h));
      checkCUDA(__LINE__, cudaMalloc(&bb3d_GPU,  sizeof(float)*15));
      checkCUDA(__LINE__, cudaMalloc(&XYZimage_GPU,  sizeof(float)*im_w*im_h*3));

      //lock = std::async(std::launch::async,&RenderMeshDataLayer::prefetch,this);
      prefetch();
      return memoryBytes;
    };

    void prefetch(){
      //checkCUDA(__LINE__,cudaSetDevice(GPU));
      if(orein_cls){
        memset(oreintation_label->CPUmem,  0, oreintation_label->numBytes());
        memset(oreintation_label_w->CPUmem,0, oreintation_label_w->numBytes());
      }

      //int tmpD; cudaGetDevice(&tmpD); std::cout<<"GPU at LINE "<<__LINE__<<" = "<<tmpD<<std::endl;

      std::uniform_int_distribution<int>   xzRot_dis(0,360);
      std::uniform_real_distribution<float> cam_tilt_dis(-5,0);
      std::uniform_real_distribution<float> objz_dis(1.500,5.000);//depth 
      std::uniform_real_distribution<float> objx_dis(-0.250,0.250);//left right

      int batch_id = 0;
      std::vector<int> category_count(num_categories,0);
      std::vector<int> oreintation_count(num_categories,0);
      unsigned long long  time0,time1,time2,time3;
      unsigned long long T1 = 0;
      unsigned long long T2 = 0;
      unsigned long long T3 = 0;
      while(batch_id < sum(batch_size)){
        time0 = get_timestamp_dss();
        
        Box3D objbox = sceneMeshList[counter]->objects[0];
        Mesh3D model(sceneMeshList[counter]->mesh_file);
        time1 = get_timestamp_dss();

        model.zeroCenter();
        float Ryzswi[9] = {1, 0, 0, 0, 0, 1, 0, -1, 0};
        model.roateMesh(Ryzswi); 
        std::normal_distribution<double > objh_dis (heigth_dis[2*objbox.category],heigth_dis[2*objbox.category+1]);

        float objh = objh_dis(rng);
        float sizeTarget[3] = {-1,objh,-1};// only scale height
        float scale_ratio = model.scaleMesh(sizeTarget);

        float angle_yaw = xzRot_dis(rng);
        float* R = genRotMat(angle_yaw);
        model.roateMesh(R);
        

        Point3D newcenter;
        newcenter.z = objz_dis(rng);
        newcenter.x = objx_dis(rng)*newcenter.z;
        newcenter.y = 0.5*objh-1; //newcenter.x = 0;  newcenter.y = -0.4;//1.24267; newcenter.z = 2.65842;
        model.translate(newcenter);
        time2 = get_timestamp_dss();   

        float  tilt_angle = cam_tilt_dis(rng);
        float* Rtilt = genTiltMat(tilt_angle);
        float  camRT[12] = {0};

        for (int i = 0; i<3; ++i){
          for (int j = 0; j<3; ++j){
            camRT[i*4+j] = Rtilt[i*3+j];
          }
        }
        
        
        float* depth =  renderCameraView(&model, camK, camRT,im_w, im_h);

        
        /************************ box in point could coordinate ************************/
        //transform the box 
        for (int i =0;i<3;++i){
          objbox.coeff[i] = scale_ratio*objbox.coeff[i];
        }
        
        Point3D bbcenter = model.getBoxCenter();
        objbox.center[0] = -1*bbcenter.x;
        objbox.center[1] = bbcenter.z;
        objbox.center[2] = bbcenter.y;
        float Rotate_boxbase[4];
        float angle_yaw_rad = -1*angle_yaw*3.14159265/180;
        Rotate_boxbase[0] = cos(angle_yaw_rad)*objbox.base[0]-sin(angle_yaw_rad)*objbox.base[1];
        Rotate_boxbase[1] = cos(angle_yaw_rad)*objbox.base[3]-sin(angle_yaw_rad)*objbox.base[4];

        Rotate_boxbase[2] = sin(angle_yaw_rad)*objbox.base[0]+cos(angle_yaw_rad)*objbox.base[1];
        Rotate_boxbase[3] = sin(angle_yaw_rad)*objbox.base[3]+cos(angle_yaw_rad)*objbox.base[4];

        objbox.base[0] = Rotate_boxbase[0];
        objbox.base[1] = Rotate_boxbase[2];
        objbox.base[3] = Rotate_boxbase[1];
        objbox.base[4] = Rotate_boxbase[3];

        float R_data[9];
        for (int i = 0; i<3; ++i){
          for (int j = 0; j<3; ++j){
            R_data[i*3+j] = Rtilt[j*3+i];
          }
        }

        
        int numValid = 0;
        for (int i=1;i<im_w*im_h;++i){
          if (depth[i]>0&depth[i]<8) {numValid++;}
          else {depth[i] = 10;}
        }
        

        if (numValid>500){
          // copy data to GPU 
          RGBDpixel * RGBDimage = NULL;

          checkCUDA(__LINE__, cudaMemcpy(K_GPU, (float*)camK, sizeof(float)*9, cudaMemcpyHostToDevice));
          checkCUDA(__LINE__, cudaMemcpy(R_GPU, (float*)R_data, sizeof(float)*9, cudaMemcpyHostToDevice)); 
          checkCUDA(__LINE__, cudaMemcpy(depth_GPU, (float*)depth, sizeof(float)*im_w*im_h, cudaMemcpyHostToDevice)); 
          checkCUDA(__LINE__, cudaMemcpy(bb3d_GPU, objbox.base, sizeof(float)*15, cudaMemcpyHostToDevice));

          compute_xyzkernel<<<im_w,im_h>>>(XYZimage_GPU,depth_GPU,K_GPU,R_GPU);

          StorageT * tsdf_data_GPU = &dataGPU[batch_id*grid_size[0]*grid_size[1]*grid_size[2]*grid_size[3]];
          int THREADS_NUM = 1024;
          int BLOCK_NUM = int((grid_size[1]*grid_size[2]*grid_size[3] + size_t(THREADS_NUM) - 1) / THREADS_NUM);
          //std::cout<<"BLOCK_NUM"<<BLOCK_NUM<<std::endl;
          compute_TSDFGPUbox_proj<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_data_GPU, R_GPU, K_GPU, RGBDimage, XYZimage_GPU,
                                                          bb3d_GPU, grid_size[1],grid_size[2],grid_size[3], grid_size[0], 
                                                          im_w, im_h, encode_type, scale);

          labelCPU->CPUmem[batch_id] = CPUCompute2StorageT(ComputeT(objbox.category));




          int oreintation = floor((360-angle_yaw)/18);
          if (orein_cls){
              oreintation_label -> CPUmem[batch_id*num_categories + objbox.category] = CPUCompute2StorageT(ComputeT(oreintation));
              for (int cid = 0;cid<num_oreintation;cid++){
                  oreintation_label_w -> CPUmem[batch_id*num_oreintation*num_categories + cid*num_categories + objbox.category] = CPUCompute2StorageT(1);
              }
          }
          oreintation_count[oreintation]++;
          category_count[objbox.category]++;
          batch_id++;

          /*
            for (int h=0; h<480; h=h+8){
               for (int w=0; w<640; w = w+4) if (depth[h+w*480]<8) std::cout<<".";  else std::cout<<" ";
               std::cout<<std::endl;
            }
            std::cout<<"oreintation"<<oreintation<<std::endl;
            std::cout<<"angle_yaw"<<angle_yaw<<std::endl;
            std::cout<<"objbox.category" <<objbox.category<<std::endl;
            std::cout<<sceneMeshList[counter]->mesh_file<<std::endl;
            std::cin.ignore();
            */
            

          /************debug*********************/
          /*
            checkCUDA(__LINE__,cudaDeviceSynchronize());
            for (int h=0; h<480; h=h+8){
              for (int w=0; w<640; w = w+4) if (depth[h+w*480]<8) std::cout<<".";  else std::cout<<" ";
              std::cout<<std::endl;
            }
            std::cout<<"R_data"<<std::endl;
            for (int i = 0; i<3; ++i){
              for (int j = 0; j<3; ++j){
                std::cout<<R_data[i*3+j]<<",";
              }
              std::cout<<std::endl;
            }
            std::cout<<"objbox.base"<<std::endl;
            for (int i = 0; i<3; ++i){
              for (int j = 0; j<3; ++j){
                std::cout<<objbox.base[i*3+j]<<",";
              }
              std::cout<<std::endl;
            }
            for (int j = 0; j<3; ++j){
              std::cout<<objbox.coeff[j]<<",";
            }
            std::cout<<std::endl;

            for (int j = 0; j<3; ++j){
                std::cout<<objbox.center[j]<<",";
            }
            std::cout<<std::endl;

            std::string outputfile = "DSS/debug/depth.bin";
            FILE * fid = fopen(outputfile.c_str(),"wb");
            checkCUDA(__LINE__, cudaMemcpy(depth, depth_GPU, im_w*im_h*sizeof(float), cudaMemcpyDeviceToHost) );
            fwrite(depth,sizeof(float),480*640,fid);
            fclose(fid);

            float* XYZimage_CPU = new float[3*im_w*im_h];
            checkCUDA(__LINE__, cudaMemcpy(XYZimage_CPU, XYZimage_GPU,3*im_w*im_h*sizeof(float), cudaMemcpyDeviceToHost) );

            float* dataCPUmem = new float[grid_size[0]*grid_size[1]*grid_size[2]*grid_size[3]];
            checkCUDA(__LINE__, cudaMemcpy(dataCPUmem, tsdf_data_GPU,grid_size[0]*grid_size[1]*grid_size[2]*grid_size[3]*sizeof(float), cudaMemcpyDeviceToHost) );
            
            outputfile = "DSS/debug/feature_renderSS.bin";
            fid = fopen(outputfile.c_str(),"wb");
            fwrite(dataCPUmem,sizeof(float),3*30*30*30,fid);
            fclose(fid);
            
            outputfile = "DSS/debug/XYZimage_GPU.bin";
            fid = fopen(outputfile.c_str(),"wb");
            fwrite(XYZimage_CPU,sizeof(float),3*im_w*im_h,fid);
            fclose(fid);


            */
            
            
          /************debug*********************/

        }
        //checkCUDA(__LINE__,cudaDeviceSynchronize());
        time3 = get_timestamp_dss();   
        T1 += time1-time0;
        T2 += time2-time1;
        T3 += time3-time2;

        delete[] depth;
        counter++;
        if (counter >= sceneMeshList.size()){
          counter = 0;
          ++epoch_prefetch;
          shuffle();
        }

        //std::cout<<counter<<std::endl;
        
      } 

      //std::cout << "load mesh" << T1/1000 << " ms, " << "trainsform " << T2/1000 <<" rendering + tsdf: "<<T3/1000<<std::endl;
      //<< copygputime/1000 << "transform " << transformtime/1000 << " ms" <<std::endl;  
      std::cout<<"category_count: "; veciPrint(category_count);  std::cout<<std::endl;
      std::cout<<"oreintation_count: "; veciPrint(oreintation_count);  std::cout<<std::endl;

      
      
    };

    void forward(Phase phase_){
      //lock.wait();
      std::swap(out[0]->dataGPU,dataGPU);
      checkCUDA(__LINE__, cudaMemcpy(out[1]->dataGPU, labelCPU->CPUmem,      labelCPU->numBytes(),      cudaMemcpyHostToDevice) );
      if(orein_cls){
        //std::cout<<"copied"<<std::endl;
        checkCUDA(__LINE__, cudaMemcpy(out[2]->dataGPU, oreintation_label->CPUmem,    oreintation_label->numBytes(),     cudaMemcpyHostToDevice) );
        checkCUDA(__LINE__, cudaMemcpy(out[3]->dataGPU, oreintation_label_w->CPUmem,  oreintation_label_w->numBytes(),   cudaMemcpyHostToDevice) );
      }
      epoch = epoch_prefetch;
      //lock = std::async(std::launch::async,&RenderMeshDataLayer::prefetch,this);
      prefetch();
    };
};//end of RenderMeshDataLayer

class Scene2DDataLayer : public DataLayer {
public:
  int epoch_prefetch;
  Tensor<StorageT>*   rois;
  StorageT*  HHAdata_GPU;
  StorageT*  IMAdata_GPU;

  std::vector<int> imagesize;
  int batch_size;
  int numBoxperImage;
  std::vector<Scene3D*> scenes;
  std::string file_list;
  std::string hha_root;
  std::string image_root;
  std::future<void> lock;
  //float up_scale;
  int box_type;
  //int up_size;
  //
  int numofitems(){ 
    int total_number = scenes.size()*numBoxperImage;
    return total_number;  
  };
   void shuffle(){
    return;
  };
  void init(){
    epoch_prefetch = 0;
 
    train_me = false;
    counter = 0;

    std::cout<<"loading file "<<file_list<<"\n";
    FILE* fp = fopen(file_list.c_str(),"rb");
    if (fp==NULL) { std::cout<<"fail to open file: "<<file_list<<std::endl; exit(EXIT_FAILURE); }

    while (feof(fp)==0) {
      Scene3D* scene = new Scene3D();
      unsigned int len = 0;
      size_t file_size = 0;
      file_size += fread((void*)(&len), sizeof(unsigned int), 1, fp);    
      if (len==0) break;
      scene->filename.resize(len);
      if (len>0) file_size += fread((void*)(scene->filename.data()), sizeof(char), len, fp);
     

      file_size += fread((void*)(scene->R), sizeof(float), 9, fp);
      file_size += fread((void*)(scene->K), sizeof(float), 9, fp);
      file_size += fread((void*)(&scene->height), sizeof(unsigned int), 1, fp);
      file_size += fread((void*)(&scene->width), sizeof(unsigned int), 1, fp); 
      file_size += fread((void*)(&len),    sizeof(unsigned int),   1, fp);
      scene->objects.resize(len);
      
      //std::cout<<scene->filename <<std::endl; 
      //std::cout<<len <<std::endl; 
      //std::cin.ignore();
      for (int bid = 0;bid<len;bid++){
         Box2D box;
         file_size += fread((void*)(&(box.category)), sizeof(unsigned int),   1, fp);
         file_size += fread((void*)(box.tblr),        sizeof(float), 4, fp);
         scene->objects_2d_tight.push_back(box);
         
         uint8_t hasTarget = 0;
         file_size += fread((void*)(&hasTarget), sizeof(uint8_t),   1, fp);
         if (hasTarget>0){
          std::cout<<" sth wrong in line "   << __LINE__ << std::endl;
         }

         file_size += fread((void*)(box.tblr),   sizeof(float), 4, fp);
         scene->objects_2d_full.push_back(box);
         file_size += fread((void*)(&hasTarget), sizeof(uint8_t),   1, fp);
         if (hasTarget>0){
          std::cout<<" sth wrong in line "   << __LINE__ << std::endl;
         }
      }
      scenes.push_back(scene);
    }
    fclose(fp);
    //std::cout<<scenes.size()
    //std::cin.ignore();


  }
  Scene2DDataLayer(std::string name_, Phase phase_, std::string file_list_, int batch_size_): DataLayer(name_), batch_size(batch_size_), file_list(file_list_){
    phase = phase_;
    init();
  };

  Scene2DDataLayer(JSON* json){
    SetValue(json, name,    "Scene2DData")
    SetValue(json, phase,   Training)
    SetOrDie(json, file_list        )
    SetOrDie(json, image_root       )
    SetOrDie(json, hha_root         )
    SetOrDie(json, imagesize        )
    SetOrDie(json, batch_size       )
    SetOrDie(json, numBoxperImage   )
    //SetOrDie(json, up_scale         )
    SetOrDie(json, box_type         )
    //std::cout<<"up_scale :  "<<up_scale<<std::endl;
    init();
  };

  ~Scene2DDataLayer(){
    if (lock.valid()) lock.wait();
    for(size_t i=0;i<scenes.size();i++){
      if (scenes[i]!=NULL) delete scenes[i];
    }
  };

  size_t Malloc(Phase phase_){
    if (phase == Training && phase_==Testing) return 0;
    size_t memoryBytes = 0;
    std::cout<< (train_me? "* " : "  ");
    std::cout<<name<<std::endl;
    std::vector<int> roi_data_dim(4,1);
    std::vector<int> image_data_dim(4,1);


    roi_data_dim[0] = batch_size*numBoxperImage;
    roi_data_dim[1] = 5;

    image_data_dim[0] = batch_size;
    image_data_dim[1] = 3;
    image_data_dim[2] = imagesize[0];
    image_data_dim[3] = imagesize[1];

    rois = new Tensor<StorageT>(roi_data_dim);
    checkCUDA(__LINE__, cudaMalloc(&IMAdata_GPU, sizeof(float)*batch_size*3*imagesize[0]*imagesize[1]));
    checkCUDA(__LINE__, cudaMalloc(&HHAdata_GPU, sizeof(float)*batch_size*3*imagesize[0]*imagesize[1]));
    
    out[0]->need_diff = false;
    memoryBytes += out[0]->Malloc(roi_data_dim);
    veciPrint(out[0]->dim); std::cout<<std::endl;

    out[1]->need_diff = false;
    memoryBytes += out[1]->Malloc(image_data_dim);

    out[2]->need_diff = false;
    memoryBytes += out[2]->Malloc(image_data_dim);
    //lock = std::async(std::launch::async,&Scene2DDataLayer::prefetch,this);
    prefetch();
    return memoryBytes;
  };

  void prefetch(){
     checkCUDA(__LINE__,cudaSetDevice(GPU));
     memset(rois->CPUmem, 0, rois->numBytes());
     float up_scale = std::min(float(imagesize[0])/float(scenes[counter]->height),float(imagesize[1])/float(scenes[counter]->width));
     for (int batch_id = 0;batch_id < batch_size;batch_id++){
       for(int box_id = 0; box_id < scenes[counter]->objects_2d_tight.size();box_id++){
        rois->CPUmem[batch_id*numBoxperImage+5*box_id+0] = CPUCompute2StorageT((float)batch_id);
        for (int i =0;i<4;i++){
          if (box_type==0){
              rois->CPUmem[batch_id*numBoxperImage+5*box_id+i+1] = CPUCompute2StorageT((float)(scenes[counter]->objects_2d_tight[box_id].tblr[i]*up_scale-1));
          }
          else if(box_type==1){
             rois->CPUmem[batch_id*numBoxperImage+5*box_id+i+1] = CPUCompute2StorageT((float)(scenes[counter]->objects_2d_full[box_id].tblr[i]*up_scale-1));
          }
          else{
            FatalError(__LINE__);
          }
        }
       }


       //size_t numel_batch_all = 3*imagesize[0]*imagesize[1];
       Tensor<StorageT>*  IMA = new Tensor<StorageT>(image_root+ scenes[counter]->filename + ".tensor");
       IMA->writeGPU(&IMAdata_GPU[batch_id*3*imagesize[0]*imagesize[1]]);       

       // read HH image 
       Tensor<StorageT>*  HHA = new Tensor<StorageT>(hha_root  + scenes[counter]->filename + ".tensor");
       HHA->writeGPU(&HHAdata_GPU[batch_id*3*imagesize[0]*imagesize[1]]);

       counter++;
     }
     if (counter >= scenes.size()){
         counter = 0;
         epoch_prefetch++;
     }
    
  }

  void forward(Phase phase_){
    //lock.wait();
    checkCUDA(__LINE__,cudaDeviceSynchronize());
    checkCUDA(__LINE__, cudaMemcpy(out[0]->dataGPU, rois->CPUmem, rois->numBytes(),     cudaMemcpyHostToDevice) );
    std::swap(out[1]->dataGPU,IMAdata_GPU);
    std::swap(out[2]->dataGPU,HHAdata_GPU);
    epoch = epoch_prefetch;
    prefetch();
    //lock = std::async(std::launch::async,&Scene2DDataLayer::prefetch,this);
  };

};//Scene2DDataLayer

