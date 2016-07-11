#ifndef DECONVOLUTION3D_LAYER_HPP_
#define DECONVOLUTION3D_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
//#include "caffe/layers/neuron_layers.hpp"
//#include "caffe/loss_layers.hpp"
//#include "caffe/data_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {



template <typename Dtype>
class DeConvolution3DLayer : public Layer<Dtype> {
 public:
  explicit DeConvolution3DLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual ~DeConvolution3DLayer();
  
  virtual inline const char* type() const { return "DeConvolution3D"; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  /*virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);*/
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  /*virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);*/
  virtual inline bool reverse_dimensions() { return true; }
  
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape();
  
  /*int kernel_size_;
  int kernel_depth_;*/
  vector<int> kernel_shape_;
  vector<int> stride_shape_;
  /*int stride_;
  int temporal_stride_;*/
  int num_;
  int channels_;
  vector<int> pad_shape_;
  vector<int> input_shape_;
  /*int pad_;
  int temporal_pad_;*/
  int length_;
  int height_;
  int width_;
  int num_output_;
  vector<int> output_shape_;
  int filter_group_;
  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
  //shared_ptr<SyncedMemory> bias_multiplier_;
  bool bias_term_;
  int M_;
  int K_;
  int N_;

  int conv_out_spatial_dim_;
  int kernel_dim_;
  int output_offset_;
  int bottom_offset_, top_offset_, weight_offset_, bias_offset_;
};

}

#endif /* DECONVOLUTION3D_LAYER_HPP_ */
