#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cstring>
#include <string>
#include "../cuda_ops/msda.h"
using namespace nvinfer1;

class AttentionPlugin final : public IPluginV2DynamicExt {
 public:
  const char* getPluginType() const noexcept override { return "AgriOccMSDeformAttn"; }
  const char* getPluginVersion() const noexcept override { return "1"; }
  int getNbOutputs() const noexcept override { return 1; }
  int initialize() noexcept override { return 0; }
  void terminate() noexcept override {}
  size_t getSerializationSize() const noexcept override { return sizeof(int); }
  void serialize(void* buffer) const noexcept override { int version=1; std::memcpy(buffer,&version,sizeof(version)); }
  void destroy() noexcept override { delete this; }
  IPluginV2DynamicExt* clone() const noexcept override { return new AttentionPlugin(*this); }
  void setPluginNamespace(const char* ns) noexcept override { ns_=ns; }
  const char* getPluginNamespace() const noexcept override { return ns_.c_str(); }
  DataType getOutputDataType(int,const DataType* types,int) const noexcept override { return types[0]; }
  DimsExprs getOutputDimensions(int,const DimsExprs* input,int,IExprBuilder& builder) noexcept override {
    DimsExprs output; output.nbDims=3; output.d[0]=input[0].d[0]; output.d[1]=input[3].d[1];
    output.d[2]=builder.operation(DimensionOperation::kPROD,*input[0].d[2],*input[0].d[3]); return output;
  }
  bool supportsFormatCombination(int pos,const PluginTensorDesc* desc,int,int) noexcept override {
    if(desc[pos].format!=TensorFormat::kLINEAR) return false;
    if(pos==1 || pos==2) return desc[pos].type==DataType::kINT32;
    if(pos==0) return desc[0].type==DataType::kFLOAT || desc[0].type==DataType::kHALF;
    return desc[pos].type==desc[0].type;
  }
  void configurePlugin(const DynamicPluginTensorDesc*,int,const DynamicPluginTensorDesc*,int) noexcept override {}
  size_t getWorkspaceSize(const PluginTensorDesc*,int,const PluginTensorDesc*,int) const noexcept override { return 0; }
  int enqueue(const PluginTensorDesc* input,const PluginTensorDesc*,const void* const* inputs,void* const* outputs,void*,cudaStream_t stream) noexcept override {
    const auto& v=input[0].dims; const auto& l=input[3].dims;
    return agriocc_msda(inputs[0],static_cast<const int*>(inputs[1]),static_cast<const int*>(inputs[2]),inputs[3],inputs[4],outputs[0],v.d[0],v.d[1],v.d[2],v.d[3],l.d[1],l.d[3],l.d[4],input[0].type==DataType::kHALF,stream)==cudaSuccess?0:1;
  }
  void attachToContext(cudnnContext*,cublasContext*,IGpuAllocator*) noexcept override {}
  void detachFromContext() noexcept override {}
 private: std::string ns_="agriocc";
};

class Creator final : public IPluginCreator {
 public:
  const char* getPluginName() const noexcept override { return "AgriOccMSDeformAttn"; }
  const char* getPluginVersion() const noexcept override { return "1"; }
  const PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }
  IPluginV2* createPlugin(const char*,const PluginFieldCollection*) noexcept override { auto* p=new AttentionPlugin; p->setPluginNamespace(ns_.c_str()); return p; }
  IPluginV2* deserializePlugin(const char*,const void* data,size_t length) noexcept override {
    if(length!=sizeof(int)) return nullptr; int version=0; std::memcpy(&version,data,sizeof(version));
    return version==1?createPlugin(nullptr,nullptr):nullptr;
  }
  void setPluginNamespace(const char* ns) noexcept override { ns_=ns; }
  const char* getPluginNamespace() const noexcept override { return ns_.c_str(); }
 private: std::string ns_="agriocc"; PluginFieldCollection fields_{0,nullptr};
};

extern "C" bool initLibAgriOccPlugins() {
  static Creator creator;
  static bool registered=getPluginRegistry()->registerCreator(creator,"agriocc");
  return registered;
}
