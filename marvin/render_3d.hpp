//g++ test_render.cpp -o test_render -lGLU -lOSMesa -I/opt/X11/include -L/opt/X11/lib
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <GL/osmesa.h>
#include <GL/glu.h>
#include <cmath> // sin cos


struct Point3D{
	float x;
	float y;
	float z;
};

float * genRotMat(float angle_yaw){
		//angle_yaw in degree
		angle_yaw = angle_yaw*3.14159265/180;
		float* R = new float[9];
		R[0] = cos(angle_yaw); R[1] = 0; R[2] = -sin(angle_yaw);
		R[3] = 0; 			   R[4] = 1; R[5] = 0;
		R[6] = sin(angle_yaw); R[7] = 0; R[8] = cos(angle_yaw);
		//R = {cos(angle_yaw), 0, -sin(angle_yaw), 0, 1, 0, sin(angle_yaw), 0, cos(angle_yaw)};
		return R;
}

float * genTiltMat(float theta){
		theta = theta*3.14159265/180;
		float* R = new float[9];
		R[0] = 1; R[1] = 0; R[2] = 0;
		R[3] = 0; R[4] = cos(theta); R[5] = -sin(theta);
		R[6] = 0; R[7] = sin(theta); R[8] =  cos(theta);

		//R = {1, 0, 0, 0, cos(theta), -sin(theta),0, sin(theta), cos(theta)};

		return R;
}

class Mesh3D{
public:
	std::vector < Point3D > vertex;
	std::vector < std::vector< int > > face;
	Mesh3D(){};
	Mesh3D(std::string filename){ 
		if(filename.substr(filename.find_last_of(".") + 1) == "off") {
			readOFF(filename); 
		}
		else{
			readPLYbin(filename); 
		}
	};

	Point3D getCenter(){
		Point3D center= getBoxCenter();
		/*
		center.x = 0;
		center.y = 0;
		center.z = 0;
		for (int i=0;i<vertex.size();++i){
			center.x += vertex[i].x;
			center.y += vertex[i].y;
			center.z += vertex[i].z;
		}
		if (vertex.size()>0){		
			center.x /= float(vertex.size());
			center.y /= float(vertex.size());
			center.z /= float(vertex.size());
		}
		*/
		return center;
	};
	Point3D getBoxCenter(){
		Point3D center;
		float minv[3] = {1000000,1000000,1000000};
		float maxv[3] = {-1000000,-1000000,-1000000};;
		for (int i=0;i<vertex.size();++i){
			minv[0] = min(minv[0],vertex[i].x);
	      	minv[1] = min(minv[1],vertex[i].y);
	      	minv[2] = min(minv[2],vertex[i].z);

	      	maxv[0] = max(maxv[0],vertex[i].x);
	      	maxv[1] = max(maxv[1],vertex[i].y);
	      	maxv[2] = max(maxv[2],vertex[i].z);
		}
		center.x = 0.5*(minv[0]+maxv[0]);
		center.y = 0.5*(minv[1]+maxv[1]);
		center.z = 0.5*(minv[2]+maxv[2]);
		
		return center;
	};

	void translate(Point3D T){
		for (int i=0;i<vertex.size();++i){
			vertex[i].x += T.x;
			vertex[i].y += T.y;
			vertex[i].z += T.z;
		}
	};

	void translate(float* T){
		for (int i=0;i<vertex.size();++i){
			vertex[i].x += T[0];
			vertex[i].y += T[1];
			vertex[i].z += T[2];
		}
	};

	void roateMesh(float* R){
		//need to call zeroCenter() first
		// rotate
		for (int i=0;i<vertex.size();++i){
			float ix = vertex[i].x;
	      	float iy = vertex[i].y;
	      	float iz = vertex[i].z;
			vertex[i].x = R[0] * ix + R[1] * iy + R[2] * iz;
			vertex[i].y = R[3] * ix + R[4] * iy + R[5] * iz;
			vertex[i].z = R[6] * ix + R[7] * iy + R[8] * iz;
		}
	}
	float scaleMesh(float size[3]){
		// need to call zeroCenter() first
		float minv[3] = {1000000,1000000,1000000};
		float maxv[3] = {-1000000,-1000000,-1000000};;
		for (int i=0;i<vertex.size();++i){
			minv[0] = min(minv[0],vertex[i].x);
	      	minv[1] = min(minv[1],vertex[i].y);
	      	minv[2] = min(minv[2],vertex[i].z);

	      	maxv[0] = max(maxv[0],vertex[i].x);
	      	maxv[1] = max(maxv[1],vertex[i].y);
	      	maxv[2] = max(maxv[2],vertex[i].z);
		}
		
		float scale_ratio = size[1] / (maxv[1]-minv[1]);
		//std::cout<<"size"<<size[1]<<","<<maxv[1]<<","<<minv[1]<<","<<scale_ratio<<std::endl;
		for (int i=0;i<vertex.size();++i){
			vertex[i].x = vertex[i].x*scale_ratio;
			vertex[i].y = vertex[i].y*scale_ratio;
			vertex[i].z = vertex[i].z*scale_ratio;
		}
		return scale_ratio;
	}


	void zeroCenter(){
		Point3D center = getCenter();
		center.x = - center.x;
		center.y = - center.y;
		center.z = - center.z;
		translate(center);
	};
	void readOFF(const std::string filename){
		std::string readLine;
		std::ifstream fin(filename.c_str());
		getline(fin,readLine);
		if (readLine != "OFF") std::cerr << "The file to read should be OFF." << std::endl;
		int delimiterPos_1, delimiterPos_2, delimiterPos_3;

		//getline(fin,readLine);
		//std::cout<<"readLine[0]="<<readLine[0]<<std::endl;
		//std::cout<<"readLine[0]="<<(!(readLine[0]=='#' || readLine[0]==' ' || readLine[0]=='\n' || readLine[0]=='\r'))<<std::endl;

		do { getline(fin,readLine); } while((readLine[0]=='#' || readLine[0]==' ' || readLine[0]=='\n' || readLine[0]=='\r'));
		
		//std::cout<<"std::endl"<<std::endl;

		delimiterPos_1 = readLine.find(" ", 0);
		int nv = atoi(readLine.substr(0,delimiterPos_1+1).c_str());
		delimiterPos_2 = readLine.find(" ", delimiterPos_1);
		int nf = atoi(readLine.substr(delimiterPos_1,delimiterPos_2 +1).c_str());

		//std::cout<<"nv="<<nv<<std::endl;
		//std::cout<<"nf="<<nf<<std::endl;

		vertex.resize(nv);
		face.resize(nf);
		for (int n=0; n<nv; n++){
			do { getline(fin,readLine); } while((readLine[0]=='#' || readLine[0]==' ' || readLine[0]=='\n' || readLine[0]=='\r'));
			delimiterPos_1 = readLine.find(" ", 0);
			vertex[n].x = atof(readLine.substr(0,delimiterPos_1).c_str());
			delimiterPos_2 = readLine.find(" ", delimiterPos_1+1);
			vertex[n].y = atof(readLine.substr(delimiterPos_1,delimiterPos_2 ).c_str());
			delimiterPos_3 = readLine.find(" ", delimiterPos_2+1);
			vertex[n].z = atof(readLine.substr(delimiterPos_2,delimiterPos_3 ).c_str());
		}
		for (int n=0; n<nf; n++){
			do { getline(fin,readLine); } while((readLine[0]=='#' || readLine[0]==' ' || readLine[0]=='\n' || readLine[0]=='\r'));
			delimiterPos_1 = readLine.find(" ", 0);
			face[n].resize(atoi(readLine.substr(0,delimiterPos_1).c_str()));
			for (int i=0;i<face[n].size();++i){
				delimiterPos_2 = readLine.find(" ", delimiterPos_1+1);
				face[n][i] = atoi(readLine.substr(delimiterPos_1,delimiterPos_2).c_str());
				delimiterPos_1 = delimiterPos_2;
			}
		}
		fin.close();
	};
	void writeOFF(const std::string filename){
		std::ofstream fout(filename.c_str());
		fout<<"OFF"<<std::endl;
		fout<<vertex.size()<<" "<<face.size()<<" 0"<<std::endl;
		for(int n=0;n<vertex.size();++n)
			fout<<vertex[n].x<<" "<<vertex[n].y<<" "<<vertex[n].z<<std::endl;
		for(int n=0;n<face.size();++n){
			fout<<face[n].size();
			for (int i=0;i<face[n].size();++i) fout<<" "<<face[n][i];
			fout<<std::endl;
		}
		fout.close();
	};

	void readPLY(const std::string filename){
		std::ifstream fin(filename.c_str());
		int num_vertex, num_face, num_edge;
		std::string str;
		getline(fin,str); getline(fin,str); getline(fin,str);
		fin>>str>>str>>num_vertex;  getline(fin,str);
		getline(fin,str); getline(fin,str); getline(fin,str);
		fin>>str>>str>>num_face;  getline(fin,str);
		getline(fin,str);
		fin>>str>>str>>num_edge;  getline(fin,str);
		getline(fin,str); getline(fin,str); getline(fin,str);
		//std::cout<<"num_vertex="<<num_vertex<<std::endl;
		//std::cout<<"num_face  ="<<num_face<<std::endl;
		//std::cout<<"num_edge  ="<<num_edge<<std::endl;
		vertex.resize(num_vertex);
		for (int i=0;i<num_vertex;++i){
			fin>>vertex[i].x>>vertex[i].y>>vertex[i].z;
		}
		face.resize(num_face);
		for (int i=0;i<num_face;++i){
			int num_vertex_index;
			fin>>num_vertex_index;
			face[i].resize(num_vertex_index);
			for (int j=0;j<num_vertex_index; j++)
				fin>>face[i][j];
		}
		fin.close();
	};
	void readPLYbin(const std::string filename){
		std::ifstream fin(filename.c_str());
		int num_vertex, num_face;// num_edge;
		std::string str;
		getline(fin,str); getline(fin,str); //getline(fin,str);
		fin>>str>>str>>num_vertex;  getline(fin,str);
		getline(fin,str); getline(fin,str); getline(fin,str);
		fin>>str>>str>>num_face;  getline(fin,str);
		getline(fin,str);
		//fin>>str>>str>>num_edge;  getline(fin,str);
		//getline(fin,str); 
		//getline(fin,str); 
		getline(fin,str);
		//std::cout<<"num_vertex="<<num_vertex<<std::endl;
		//std::cout<<"num_face  ="<<num_face<<std::endl;
		//std::cout<<"num_edge  ="<<num_edge<<std::endl;
		vertex.resize(num_vertex);

		fin.read((char*)(&(vertex[0].x)), num_vertex*3*sizeof(float));

		face.resize(num_face);
		for (int i=0;i<num_face;++i){
			uint8_t num_vertex_index;
			fin.read((char*)(&num_vertex_index),sizeof(uint8_t));
			face[i].resize(num_vertex_index);
			fin.read((char*)(&(face[i][0])),num_vertex_index*sizeof(int));
		}
		fin.close();
	};
	void writePLY(const std::string filename){
		std::ofstream fout(filename.c_str());
		fout<<"ply"<<std::endl;
		fout<<"format ascii 1.0"<<std::endl;
		//fout<<"comment format for RenderMe"<<std::endl;
		fout<<"element vertex "<<vertex.size()<<std::endl;
		fout<<"property float x"<<std::endl;
		fout<<"property float y"<<std::endl;
		fout<<"property float z"<<std::endl;
		fout<<"element face "<<face.size()<<std::endl;
		fout<<"property list uchar int vertex_index"<<std::endl;
		//fout<<"element edge "<<0<<std::endl;
		//fout<<"property int vertex1"<<std::endl;
		//fout<<"property int vertex2"<<std::endl;
		fout<<"end_header"<<std::endl;
		for (int i=0;i<vertex.size();++i){
			fout<<vertex[i].x<<" "<<vertex[i].y<<" "<<vertex[i].z<<std::endl;
		}
		for (int i=0;i<face.size();++i){
			fout<<face[i].size();
			for (int j=0;j<face[i].size();++j){
				fout<<" "<<face[i][j];
			}
			fout<<std::endl;
		}
		fout.close();
	};
	/*
	void writePLY(const std::string filename){
		std::ofstream fout(filename.c_str());
		fout<<"ply"<<std::endl;
		fout<<"format ascii 1.0"<<std::endl;
		fout<<"comment format for RenderMe"<<std::endl;
		fout<<"element vertex "<<vertex.size()<<std::endl;
		fout<<"property float x"<<std::endl;
		fout<<"property float y"<<std::endl;
		fout<<"property float z"<<std::endl;
		fout<<"element face "<<face.size()<<std::endl;
		fout<<"property list uchar int vertex_index"<<std::endl;
		fout<<"element edge "<<0<<std::endl;
		fout<<"property int vertex1"<<std::endl;
		fout<<"property int vertex2"<<std::endl;
		fout<<"end_header"<<std::endl;
		for (int i=0;i<vertex.size();++i){
			fout<<vertex[i].x<<" "<<vertex[i].y<<" "<<vertex[i].z<<std::endl;
		}
		for (int i=0;i<face.size();++i){
			fout<<face[i].size();
			for (int j=0;j<face[i].size();++j){
				fout<<" "<<face[i][j];
			}
			fout<<std::endl;
		}
		fout.close();
	};
	*/
};

float* renderDepth(Mesh3D* model, float* projection, int render_width, int render_height){

	unsigned char * pbufferRGB = new unsigned char [3 * render_width * render_height];


	float m_near = 0.3;
	float m_far = 1e8;
	int m_level = 0;

	// Step 1: setup off-screen mesa's binding 
	OSMesaContext ctx = OSMesaCreateContextExt(OSMESA_RGB, 32, 0, 0, NULL );
	// Bind the buffer to the context and make it current
	if (!OSMesaMakeCurrent(ctx, (void*)pbufferRGB, GL_UNSIGNED_BYTE, render_width, render_height)) {
		std::cerr << "OSMesaMakeCurrent failed!: " << render_width << ' ' << render_height << std::endl;
		return NULL;
	}
	OSMesaPixelStore(OSMESA_Y_UP, 0);

	// Step 2: Setup basic OpenGL setting
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glDisable(GL_CULL_FACE);
	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK);
	glPolygonMode(GL_FRONT, GL_FILL);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glClearColor(m_clearColor[0], m_clearColor[1], m_clearColor[2], 1.0f); // this line seems useless
	glViewport(0, 0, render_width, render_height);

	// Step 3: Set projection matrices
	double scale = (0x0001) << m_level;
	double final_matrix[16];

	// new way: faster way by reuse computation and symbolic derive. See sym_derive.m to check the math.
	double inv_width_scale  = 1.0/(render_width*scale);
	double inv_height_scale = 1.0/(render_height*scale);
	double inv_width_scale_1 =inv_width_scale - 1.0;
	double inv_height_scale_1_s = -(inv_height_scale - 1.0);
	double inv_width_scale_2 = inv_width_scale*2.0;
	double inv_height_scale_2_s = -inv_height_scale*2.0;
	double m_far_a_m_near = m_far + m_near;
	double m_far_s_m_near = m_far - m_near;
	double m_far_d_m_near = m_far_a_m_near/m_far_s_m_near;
	final_matrix[ 0]= projection[2+0*3]*inv_width_scale_1 + projection[0+0*3]*inv_width_scale_2;
	final_matrix[ 1]= projection[2+0*3]*inv_height_scale_1_s + projection[1+0*3]*inv_height_scale_2_s;
	final_matrix[ 2]= projection[2+0*3]*m_far_d_m_near;
	final_matrix[ 3]= projection[2+0*3];
	final_matrix[ 4]= projection[2+1*3]*inv_width_scale_1 + projection[0+1*3]*inv_width_scale_2;
	final_matrix[ 5]= projection[2+1*3]*inv_height_scale_1_s + projection[1+1*3]*inv_height_scale_2_s; 
	final_matrix[ 6]= projection[2+1*3]*m_far_d_m_near;    
	final_matrix[ 7]= projection[2+1*3];
	final_matrix[ 8]= projection[2+2*3]*inv_width_scale_1 + projection[0+2*3]*inv_width_scale_2; 
	final_matrix[ 9]= projection[2+2*3]*inv_height_scale_1_s + projection[1+2*3]*inv_height_scale_2_s;
	final_matrix[10]= projection[2+2*3]*m_far_d_m_near;
	final_matrix[11]= projection[2+2*3];
	final_matrix[12]= projection[2+3*3]*inv_width_scale_1 + projection[0+3*3]*inv_width_scale_2;
	final_matrix[13]= projection[2+3*3]*inv_height_scale_1_s + projection[1+3*3]*inv_height_scale_2_s;  
	final_matrix[14]= projection[2+3*3]*m_far_d_m_near - (2*m_far*m_near)/m_far_s_m_near;
	final_matrix[15]= projection[2+3*3];

	// matrix is ready. use it
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixd(final_matrix);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Step 3: render the mesh 

	for (unsigned int i = 0; i < model->face.size() ; ++i) {
		glBegin(GL_POLYGON);
		for (unsigned int j=0; j < model->face[i].size(); ++j){
			int vi = model->face[i][j];
			glVertex3f(model->vertex[vi].x, model->vertex[vi].y, model->vertex[vi].z);
		}
		glEnd();
	}

	glFinish(); // done rendering

	////////////////////////////////////////////////////////////////////////////////
	unsigned int* pDepthBuffer;
	GLint outWidth, outHeight, bitPerDepth;
	OSMesaGetDepthBuffer(ctx, &outWidth, &outHeight, &bitPerDepth, (void**)&pDepthBuffer);

	float* pbufferD = new float[outWidth*outHeight];
	for(int ix=0; ix<outWidth; ix++){
		for (int iy=0;iy<outHeight;iy++){
			float depthvalue = float( m_near / (1.0 - double(pDepthBuffer[iy * outWidth + ix])/double(4294967296)) );
			pbufferD[iy + ix * outHeight] = depthvalue;
		}
		//pbufferD[i] = float( m_near / (1.0 - double(pDepthBuffer[i])/double(4294967296)) );
		
	}
	OSMesaDestroyContext(ctx);

	delete [] pbufferRGB;
	return pbufferD;
}


float* renderDepth(std::vector<Mesh3D*> modelList, float* projection, int render_width, int render_height){

	unsigned char * pbufferRGB = new unsigned char [3 * render_width * render_height];


	float m_near = 0.3;
	float m_far = 1e8;
	int m_level = 0;

	// Step 1: setup off-screen mesa's binding 
	OSMesaContext ctx = OSMesaCreateContextExt(OSMESA_RGB, 32, 0, 0, NULL );
	// Bind the buffer to the context and make it current
	if (!OSMesaMakeCurrent(ctx, (void*)pbufferRGB, GL_UNSIGNED_BYTE, render_width, render_height)) {
		std::cerr << "OSMesaMakeCurrent failed!: " << render_width << ' ' << render_height << std::endl;
		return NULL;
	}
	OSMesaPixelStore(OSMESA_Y_UP, 0);

	// Step 2: Setup basic OpenGL setting
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glDisable(GL_CULL_FACE);
	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK);
	glPolygonMode(GL_FRONT, GL_FILL);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glClearColor(m_clearColor[0], m_clearColor[1], m_clearColor[2], 1.0f); // this line seems useless
	glViewport(0, 0, render_width, render_height);

	// Step 3: Set projection matrices
	double scale = (0x0001) << m_level;
	double final_matrix[16];

	// new way: faster way by reuse computation and symbolic derive. See sym_derive.m to check the math.
	double inv_width_scale  = 1.0/(render_width*scale);
	double inv_height_scale = 1.0/(render_height*scale);
	double inv_width_scale_1 =inv_width_scale - 1.0;
	double inv_height_scale_1_s = -(inv_height_scale - 1.0);
	double inv_width_scale_2 = inv_width_scale*2.0;
	double inv_height_scale_2_s = -inv_height_scale*2.0;
	double m_far_a_m_near = m_far + m_near;
	double m_far_s_m_near = m_far - m_near;
	double m_far_d_m_near = m_far_a_m_near/m_far_s_m_near;
	final_matrix[ 0]= projection[2+0*3]*inv_width_scale_1 + projection[0+0*3]*inv_width_scale_2;
	final_matrix[ 1]= projection[2+0*3]*inv_height_scale_1_s + projection[1+0*3]*inv_height_scale_2_s;
	final_matrix[ 2]= projection[2+0*3]*m_far_d_m_near;
	final_matrix[ 3]= projection[2+0*3];
	final_matrix[ 4]= projection[2+1*3]*inv_width_scale_1 + projection[0+1*3]*inv_width_scale_2;
	final_matrix[ 5]= projection[2+1*3]*inv_height_scale_1_s + projection[1+1*3]*inv_height_scale_2_s; 
	final_matrix[ 6]= projection[2+1*3]*m_far_d_m_near;    
	final_matrix[ 7]= projection[2+1*3];
	final_matrix[ 8]= projection[2+2*3]*inv_width_scale_1 + projection[0+2*3]*inv_width_scale_2; 
	final_matrix[ 9]= projection[2+2*3]*inv_height_scale_1_s + projection[1+2*3]*inv_height_scale_2_s;
	final_matrix[10]= projection[2+2*3]*m_far_d_m_near;
	final_matrix[11]= projection[2+2*3];
	final_matrix[12]= projection[2+3*3]*inv_width_scale_1 + projection[0+3*3]*inv_width_scale_2;
	final_matrix[13]= projection[2+3*3]*inv_height_scale_1_s + projection[1+3*3]*inv_height_scale_2_s;  
	final_matrix[14]= projection[2+3*3]*m_far_d_m_near - (2*m_far*m_near)/m_far_s_m_near;
	final_matrix[15]= projection[2+3*3];

	// matrix is ready. use it
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixd(final_matrix);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Step 3: render the mesh 
	for (unsigned int modelid = 0; modelid < modelList.size(); modelid++){
		Mesh3D* model = modelList[modelid];

		for (unsigned int i = 0; i < model->face.size() ; ++i) {
			glBegin(GL_POLYGON);
			for (unsigned int j=0; j < model->face[i].size(); ++j){
				int vi = model->face[i][j];
				glVertex3f(model->vertex[vi].x, model->vertex[vi].y, model->vertex[vi].z);
			}
			glEnd();
		}
	}
	glFinish(); // done rendering

	////////////////////////////////////////////////////////////////////////////////
	unsigned int* pDepthBuffer;
	GLint outWidth, outHeight, bitPerDepth;
	OSMesaGetDepthBuffer(ctx, &outWidth, &outHeight, &bitPerDepth, (void**)&pDepthBuffer);

	float* pbufferD = new float[outWidth*outHeight];
	for(int ix=0; ix<outWidth; ix++){
		for (int iy=0;iy<outHeight;iy++){
			float depthvalue = float( m_near / (1.0 - double(pDepthBuffer[iy * outWidth + ix])/double(4294967296)) );
			pbufferD[iy + ix * outHeight] = depthvalue;
		}
		//pbufferD[i] = float( m_near / (1.0 - double(pDepthBuffer[i])/double(4294967296)) );
		
	}
	OSMesaDestroyContext(ctx);

	delete [] pbufferRGB;
	return pbufferD;
}

void getProjectionMatrix(float* P,float* K, float* RT){
	 float RTinv[12];
  	 RTinv[0] = RT[0]; RTinv[1] = RT[4]; RTinv[2] = RT[8];  
  	 RTinv[4] = RT[1]; RTinv[5] = RT[5]; RTinv[6] = RT[9];   
  	 RTinv[8] = RT[2]; RTinv[9] = RT[6]; RTinv[10] = RT[10]; 
  	 for (int i=0;i<3;i++){
        RTinv[3+i*4] = -1*(RT[0+i]*RT[3]+RT[4+i]*RT[7]+RT[8+i]*RT[11]);
  	 }

     //inverse first row
     for (int i=0;i<4;i++){
      	RTinv[i] = -1*RTinv[i];
     } 

  	 //float ratio =1;
	 // Camera Paramte
  	 for (int i =0;i<4;i++){
	    P[0+i*3] = (K[0]*RTinv[0+i]+K[2]*RTinv[8+i]);
	    P[1+i*3] = (K[4]*RTinv[4+i]+K[5]*RTinv[8+i]);
	    P[2+i*3] = RTinv[8+i];
  	 }
};

float* renderCameraView(Mesh3D* model,float* K, float* RT, int m_width, int m_height ){
	float RTinv[12];
  	RTinv[0] = RT[0]; RTinv[1] = RT[4]; RTinv[2] = RT[8];  
  	RTinv[4] = RT[1]; RTinv[5] = RT[5]; RTinv[6] = RT[9];   
  	RTinv[8] = RT[2]; RTinv[9] = RT[6]; RTinv[10] = RT[10]; 
  	for (int i=0;i<3;i++){
      RTinv[3+i*4] = -1*(RT[0+i]*RT[3]+RT[4+i]*RT[7]+RT[8+i]*RT[11]);
  	}

  	//inverse first row
   for (int i=0;i<4;i++){
      RTinv[i] = -1*RTinv[i];
   } 

  	//float ratio =1;
	// Camera Paramte
	float P[12];
  	for (int i =0;i<4;i++){
	    P[0+i*3] = (K[0]*RTinv[0+i]+K[2]*RTinv[8+i]);
	    P[1+i*3] = (K[4]*RTinv[4+i]+K[5]*RTinv[8+i]);
	    P[2+i*3] = RTinv[8+i];
  	}
  	float* depth =  renderDepth(model, P, m_width, m_height);
  	return depth;


}



