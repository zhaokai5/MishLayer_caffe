# MishLayer_caffe
MishLayer for caffe

1. compile
copy mish_layer.hpp to $CAFFE_ROOTDIR/include/caffe/layer/ 
copy mish_layer.cpp mish_layer.cu to $CAFFE_ROOTDIR/src/caffe/layer/ 

make

2. add Mish layer to net.prototxt
layer {
  name: "Mish1"
  type: "Mish"
  bottom: "conv1"
  top: "conv1"
}
