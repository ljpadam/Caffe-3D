#include <vector>
#include "caffe/layer.hpp"
#include "caffe/layers/deconvolution3d_layer.hpp"
#include "caffe/util/vol2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DeConvolution3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  ConvolutionParameter conv_param =
    this->layer_param_.convolution_param();
  // Configure the kernel size, padding, stride, and inputs.
  CHECK(conv_param.has_kernel_shape())
      << "Kernel shape is required.";
  if (conv_param.has_pad_shape()) {
    CHECK_EQ(conv_param.kernel_shape().dim_size(),
             conv_param.pad_shape().dim_size())
        << "Kernel and Pad shape don't match !";
  }
  if (conv_param.has_stride_shape()) {
    CHECK_EQ(conv_param.kernel_shape().dim_size(),
             conv_param.stride_shape().dim_size())
        << "Kernel and Stride shape don't match !";
  }
  for (int i = 0; i < conv_param.kernel_shape().dim_size(); ++i) {
    kernel_shape_.push_back(conv_param.kernel_shape().dim(i));
    CHECK_GT(kernel_shape_[i], 0) << "Filter dimensions cannot be zero.";
  }
  if (conv_param.has_pad_shape()) {
    for (int i = 0; i < conv_param.kernel_shape().dim_size(); ++i) {
      pad_shape_.push_back(conv_param.pad_shape().dim(i));
    }
  } else {
    pad_shape_ = std::vector<int>(kernel_shape_.size(), 0);
  }
  if (conv_param.has_stride_shape()) {
    for (int i = 0; i < conv_param.kernel_shape().dim_size(); ++i) {
      stride_shape_.push_back(conv_param.stride_shape().dim(i));
    }
  } else {
    stride_shape_ = std::vector<int>(kernel_shape_.size(), 1);
  }
  
  
  
  /*kernel_size_ = this->layer_param_.convolution_param().kernel_size();
  kernel_depth_ = this->layer_param_.convolution_param().kernel_depth();
  stride_ = this->layer_param_.convolution_param().stride();
  temporal_stride_ = this->layer_param_.convolution_param().temporal_stride();
  pad_ = this->layer_param_.convolution_param().pad();
  temporal_pad_ = this->layer_param_.convolution_param().temporal_pad();*/
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  length_ = bottom[0]->length();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_output_ = this->layer_param_.convolution_param().num_output();
  filter_group_ = this->layer_param_.convolution_param().group();
  CHECK_GT(num_output_, 0);

  // number of output filters must be divided by filter_group
  CHECK_EQ(num_output_ % filter_group_, 0);

  // The vol2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.

  int height_out = (height_- 1) * stride_shape_[0] + kernel_shape_[0] - 2 * pad_shape_[0];
  int width_out = (width_ -1)*stride_shape_[1] + kernel_shape_[1] - 2 * pad_shape_[1];
  int length_out = (length_ -1)* stride_shape_[2] + kernel_shape_[2] - 2 * pad_shape_[2];

  col_buffer_.Reshape(
      1, channels_ * kernel_shape_[0] * kernel_shape_[1] * kernel_shape_[2], length_out, height_out, width_out);


  bias_term_ = this->layer_param_.convolution_param().bias_term();

  // Figure out the dimensions for individual gemms.
  M_ = num_output_ / filter_group_; // doing convolution filter_group_ times per volume
  K_ = channels_ * kernel_shape_[0] * kernel_shape_[1] * kernel_shape_[2];
  N_ = length_out * height_out * width_out;

  // output size
  top[0]->Reshape(bottom[0]->num(), num_output_, length_out, height_out, width_out);

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }

    

    //Todo: need revise the weights initialization
    // Initialize the weights
    this->blobs_[0].reset(new Blob<Dtype>(
        num_output_, channels_, kernel_shape_[0], kernel_shape_[1], kernel_shape_[2]));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    //std::cout<<"\n\n\nweight\n\n"<< string(this->layer_param_.convolution_param().weight_filler())<<std::endl;
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }

  // Set up the bias filler
  if (bias_term_) {
	vector<int> bias_multiplier_shape(1, N_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
        caffe_set(bias_multiplier_.count(), Dtype(1),
                  bias_multiplier_.mutable_cpu_data());
	vector<int> bias_shape(1, num_output_);
	this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
	shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
		  this->layer_param_.convolution_param().bias_filler()));
	bias_filler->Fill(this->blobs_[1].get());
  }
}

template <typename Dtype>
void DeConvolution3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
	  num_ = bottom[0]->shape(0);
	  CHECK_EQ(bottom[0]->shape(1),
	           channels_) << "Input size incompatible with convolution kernel.";
	  input_shape_ = bottom[0]->shape();
	  // TODO: generalize to handle inputs of different shapes.
	  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
	    CHECK_EQ(num_, bottom[bottom_id]->shape(0))
	        << "Inputs must have same num.";
	    CHECK_EQ(channels_, bottom[bottom_id]->shape(1))
	        << "Inputs must have same channels.";
	    for (int i = 0; i < bottom[0]->num_axes(); ++i) {
	      CHECK_EQ(input_shape_[i],
	               bottom[bottom_id]->shape(i)) << "Inputs must have same shape.";
	    }
	  }
	  // Shape the tops.
	  compute_output_shape();
	  for (int top_id = 0; top_id < top.size(); ++top_id) {
	    top[top_id]->Reshape(output_shape_);
	  }

	  conv_out_spatial_dim_ = 1;
	  for (int i = 2; i < output_shape_.size(); ++i) {
	    conv_out_spatial_dim_ *= output_shape_[i];
	  }

	  kernel_dim_ = channels_;
	  for (int i = 0; i < kernel_shape_.size(); ++i) {
	    kernel_dim_ *= kernel_shape_[i];
	  }
	  weight_offset_ = num_output_ * kernel_dim_ /filter_group_ /filter_group_;
	  output_offset_ = num_output_ * conv_out_spatial_dim_ /filter_group_;
	  // Set up the all ones "bias multiplier" for adding biases by BLAS
	  if (bias_term_) {
	    vector<int> bias_multiplier_shape(1, conv_out_spatial_dim_);
	    bias_multiplier_.Reshape(bias_multiplier_shape);
	    caffe_set(bias_multiplier_.count(), Dtype(1),
	              bias_multiplier_.mutable_cpu_data());
	  }

	  bottom_offset_ = 1;
	  for (int i = 1; i < input_shape_.size(); ++i) {
	    bottom_offset_ *= input_shape_[i];
	  }
	  bottom_offset_ /= filter_group_;
	  top_offset_ = 1;
	  for (int i = 1; i < output_shape_.size(); ++i) {
	    top_offset_ *= output_shape_[i];
	  }
	  top_offset_ /= filter_group_;
	return;
}
template <typename Dtype>
void DeConvolution3DLayer<Dtype>::compute_output_shape()
{
	output_shape_.clear();
	output_shape_.push_back(num_);
	output_shape_.push_back(num_output_);

	  for (int i = 2; i < input_shape_.size(); ++i) {
		int dim = (input_shape_[i]-1) * stride_shape_[i-2] + kernel_shape_[i-2] - 2*pad_shape_[i-2];
		if (dim > 1) {
		  output_shape_.push_back(dim);

		}
	  }
	return;
}

template <typename Dtype>
void DeConvolution3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (top)[0]->mutable_cpu_data();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  int weight_offset = M_ * K_;
  int top_offset = M_ * N_;

  for (int n = 0; n < num_; ++n) {
	// First, im2col
    vol2col_cpu(bottom_data + bottom[0]->offset(n), channels_, length_, height_,
	    		  width_, kernel_shape_[1], kernel_shape_[0], pad_shape_[1], pad_shape_[0], stride_shape_[1], stride_shape_[0], col_data);

    // Second, inner-product without filter groups
	for (int g=0 ; g < filter_group_; ++g) {
      /*caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
	    (Dtype)1., weight + g * weight_offset, col_data,
	    (Dtype)0., top_data + (*top)[0]->offset(n) + g * top_offset);*/
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, N_, K_,
	      (Dtype)1., weight + g * weight_offset, col_data,
		  (Dtype)0., top_data + (top)[0]->offset(n) + g * top_offset);
	}

      // third, add bias
	if (bias_term_) {
	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
	   N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
	   reinterpret_cast<const Dtype*>(bias_multiplier_.cpu_data()),
	   (Dtype)1., top_data + (top)[0]->offset(n));
    }

  }
  return;
}

template <typename Dtype>
void DeConvolution3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* bottom_data = (bottom)[0]->cpu_data();
  Dtype* bottom_diff = (bottom)[0]->mutable_cpu_diff();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  Dtype* col_diff = col_buffer_.mutable_cpu_diff();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;

  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    memset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
    for (int n = 0; n < num_; ++n) {
	  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
		  1., top_diff + top[0]->offset(n),
		  reinterpret_cast<const Dtype*>(bias_multiplier_.cpu_data()), 1.,
		  bias_diff);
    }
  }

  int weight_offset = M_ * K_;
  int top_offset = M_ * N_;

  memset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
  for (int n = 0; n < num_; ++n) {
	// since we saved memory in the forward pass by not storing all col data,
	// we will need to recompute them.
	vol2col_cpu(bottom_data + (bottom)[0]->offset(n), channels_, length_, height_,
					  width_, kernel_shape_[1], kernel_shape_[0], pad_shape_[1], pad_shape_[0], stride_shape_[1],
					  stride_shape_[0], col_data);
	// gradient w.r.t. weight. Note that we will accumulate diffs.
	for (int g=0; g<filter_group_; ++g){
		/*caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
				(Dtype)1., top_diff + top[0]->offset(n) + g * top_offset,
				col_data, (Dtype)1.,
				weight_diff + g * weight_offset);*/
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_, M_, N_,
				(Dtype)1., 
				col_data,top_diff + top[0]->offset(n) + g * top_offset, (Dtype)1.,
				weight_diff + g * weight_offset);
	}

	// gradient w.r.t. bottom data, if necessary
	if (propagate_down[0]) {
	  // compute first filter group -> col_diff
	  /*caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
	    (Dtype)1., weight,
		top_diff + top[0]->offset(n),
		(Dtype)0., col_diff);*/
	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, N_, M_,
	    (Dtype)1., weight, top_diff + top[0]->offset(n), (Dtype)1., col_diff);

	  // accumulate the other filter groups -> col_diff
	  for (int g=1; g<filter_group_; ++g){
	    /*caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
	      (Dtype)1., weight + g * weight_offset,
		  top_diff + top[0]->offset(n) + g * top_offset,
		  (Dtype)1., col_diff);	*/
	    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, N_, M_,
	    (Dtype)1., weight + g * weight_offset, top_diff + top[0]->offset(n) + g * top_offset, (Dtype)1., col_diff);
	  }

	  // vol2im back to the data
	  col2vol_cpu(col_diff, channels_, length_, height_, width_, kernel_shape_[1], kernel_shape_[0], pad_shape_[1],
		  pad_shape_[0], stride_shape_[1], stride_shape_[0], bottom_diff + (bottom)[0]->offset(n));
	}

  }
}

INSTANTIATE_CLASS(DeConvolution3DLayer);
REGISTER_LAYER_CLASS(DeConvolution3D);

}  // namespace caffe
