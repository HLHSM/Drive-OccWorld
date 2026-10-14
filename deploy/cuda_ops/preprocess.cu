#include <cuda_runtime.h>
#include <cmath>
#include <cuda_fp16.h>

__global__ void normalize_resize(const unsigned char* input,float* output,int cameras,
    int in_width,int in_height,int out_width,int out_height) {
  int index=blockIdx.x*blockDim.x+threadIdx.x;
  int area=out_width*out_height;
  if(index>=cameras*3*area) return;
  int camera=index/(3*area),channel=(index/area)%3;
  int x=index%out_width,y=(index/out_width)%out_height;
  float sx=(x+.5f)*in_width/out_width-.5f;
  float sy=(y+.5f)*in_height/out_height-.5f;
  sx=fminf(fmaxf(sx,0.f),float(in_width-1));
  sy=fminf(fmaxf(sy,0.f),float(in_height-1));
  int x0=int(floorf(sx)),y0=int(floorf(sy));
  int x1=min(x0+1,in_width-1),y1=min(y0+1,in_height-1);
  float dx=sx-x0,dy=sy-y0;
  const unsigned char* image=input+camera*in_width*in_height*3;
  float top=image[(y0*in_width+x0)*3+channel]*(1.f-dx)+image[(y0*in_width+x1)*3+channel]*dx;
  float bottom=image[(y1*in_width+x0)*3+channel]*(1.f-dx)+image[(y1*in_width+x1)*3+channel]*dx;
  float mean=channel==0?103.530f:(channel==1?116.280f:123.675f);
  output[index]=top*(1.f-dy)+bottom*dy-mean;
}

extern "C" int agrioccPreprocess(const unsigned char* input,float* output,int cameras,
    int in_width,int in_height,int out_width,int out_height,cudaStream_t stream) {
  int count=cameras*3*out_width*out_height;
  normalize_resize<<<(count+255)/256,256,0,stream>>>(input,output,cameras,in_width,in_height,out_width,out_height);
  return int(cudaGetLastError());
}

template <typename T> __device__ float logit_value(T x) { return float(x); }
template <> __device__ float logit_value(__half x) { return __half2float(x); }

template <typename T>
__global__ void semantic_argmax(const T* logits,unsigned char* labels,int* invalid,
    int height,int width,int depth,int classes) {
  int index=blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=height*width*depth) return;
  int z=index%depth,y=(index/depth)%width,x=index/depth/width;
  // Match original evaluate_occ's token reshape followed by X/Y transpose.
  int source=((y*height+x)*depth+z)*classes;
  float maximum=-INFINITY; int category=0;
  for(int c=0;c<classes;++c) {
    float value=logit_value(logits[source+c]);
    if(!isfinite(value)) atomicExch(invalid,1);
    if(value>maximum) { maximum=value; category=c; }
  }
  labels[index]=static_cast<unsigned char>(category);
}

extern "C" int agrioccArgmax(const void* logits,unsigned char* labels,int* invalid,
    int height,int width,int depth,int classes,int half,cudaStream_t stream) {
  int count=height*width*depth;
  auto error=cudaMemsetAsync(invalid,0,sizeof(int),stream);
  if(error!=cudaSuccess) return int(error);
  if(half)
    semantic_argmax<<<(count+255)/256,256,0,stream>>>(static_cast<const __half*>(logits),labels,invalid,height,width,depth,classes);
  else
    semantic_argmax<<<(count+255)/256,256,0,stream>>>(static_cast<const float*>(logits),labels,invalid,height,width,depth,classes);
  return int(cudaGetLastError());
}
