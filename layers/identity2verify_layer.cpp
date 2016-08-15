
#include "caffe/layers/identity2verify_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <algorithm>
#include <vector>

namespace caffe {

template <typename Dtype>
void Identity2VerifyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void Identity2VerifyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    vector<int> top_shape = bottom[0]->shape();
    top_shape[0] /= 2;
    top[0]->Reshape(top_shape);
    top[1]->Reshape(top_shape);
	
	vector<int> label_shape = bottom[1]->shape();
	label_shape[0] /= 2;
	top[2]->Reshape(label_shape);
}

template <typename Dtype>
void Identity2VerifyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const int feature_size = bottom[0]->count(1);
    for (int n = 0; n < bottom[0]->num(); ++ n) {
        caffe_copy(
                feature_size, 
                bottom[0]->cpu_data() + n * feature_size, 
                top[n & 1]->mutable_cpu_data() + (n / 2) * feature_size
                );
    }
	
	const int label_size = bottom[1]->count(1);
	for (int n = 0; n < bottom[1]->num(); n += 2) {
		Dtype label;
		
		if( abs(*(bottom[1]->cpu_data() + n * label_size) - 
				*(bottom[1]->cpu_data() + (n+1) * label_size)) < 1)
		{
			label = Dtype(1.0);
		}
		else
		{
			label= Dtype(0.0);		
		}
		
		caffe_copy(
                label_size, 
                &label,
                top[2]->mutable_cpu_data() + (n / 2) * label_size
                );
    }
}

template <typename Dtype>
void Identity2VerifyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const int feature_size = bottom[0]->count(1);
    for (int n = 0; n < bottom[0]->num(); ++ n) {
        caffe_copy(
                feature_size,
                top[n & 1]->cpu_diff() + (n / 2) * feature_size,
                bottom[0]->mutable_cpu_diff() + n * feature_size
                );
    }
}

#ifdef CPU_ONLY
STUB_GPU(Identity2VerifyLayer);
#endif

INSTANTIATE_CLASS(Identity2VerifyLayer);
REGISTER_LAYER_CLASS(Identity2Verify);

}  // namespace caffe

