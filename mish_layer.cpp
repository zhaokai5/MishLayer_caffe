#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/mish_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype mish(Dtype x) {
  Dtype tmp=exp(x)+1;
  return x-2*x/(tmp*tmp+1);
}
template <typename Dtype>
inline Dtype mish_back(Dtype x) {
  Dtype tmp=exp(x)+1;
  Dtype tmp1=tmp*tmp+1;

  return 1-2/tmp1+4*x*(tmp*tmp-tmp)/(tmp1*tmp1);
}

template <typename Dtype>
void MishLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    //is_test_= this->phase_ == TEST;
    //if(!is_test_){
    //    backward_buff_.ReshapeLike(*bottom[0]);
    //}    
}

template <typename Dtype>
void MishLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // CHECK_GE(bottom[0]->num_axes(), 2)
  //     << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    backward_buff_.ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void MishLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();  
  Dtype* backward_buff_data=backward_buff_.mutable_cpu_data();
  if(bottom[0] == top[0]){
    // caffe_set(count,static_cast<Dtype>(0),backward_buff_data);
    caffe_copy(count,bottom_data,backward_buff_data);
  }
  for (int i = 0; i < count; ++i) {
      Dtype x_bottom=bottom_data[i];
      top_data[i]=mish(x_bottom);
      //if(!is_test_){
      //  backward_buff_data[i]=mish_back(x_bottom);
      //}
    }

}

template <typename Dtype>
void MishLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  if (top[0] == bottom[0]) {
    bottom_data = backward_buff_.cpu_data();
  }

  Dtype* bottom_diff= bottom[0]->mutable_cpu_diff();
  
  if (this->param_propagate_down_[0]) {
    //caffe_mul(count,top_diff,back_data,bottom_diff);
    for(int i=0;i<count;i++){
      bottom_diff[i]=mish_back(bottom_data[i])*top_diff[i];
      // bottom_diff
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(MishLayer);
#endif

INSTANTIATE_CLASS(MishLayer);
REGISTER_LAYER_CLASS(Mish);

}  // namespace caffe
