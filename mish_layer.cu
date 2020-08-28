#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/mish_layer.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
__global__ void MishForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
      Dtype x=in[index];
      Dtype tmp=exp(x)+1;
      out[index]=x-2*x/(tmp*tmp+1);
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void MishBackward(const int n, const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype x=in_data[index];
    Dtype tmp=exp(x)+1;
    Dtype tmp1=tmp*tmp+1;
    Dtype diff_tmp=1-2/tmp1+4*x*(tmp*tmp-tmp)/(tmp1*tmp1);
    out_diff[index]=in_diff[index]*diff_tmp;
  }
}

template <typename Dtype>
void MishLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();  
  //Dtype* backward_buff_data=backward_buff_.mutable_gpu_data();
  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, backward_buff_.mutable_gpu_data());
  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  MishForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void MishLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  if (top[0] == bottom[0]) {
    bottom_data = backward_buff_.gpu_data();
  }
  Dtype* bottom_diff= bottom[0]->mutable_gpu_diff();

  // Propagate to bottom
  if (propagate_down[0]) {
    MishBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(MishLayer);
}  // namespace caffe
