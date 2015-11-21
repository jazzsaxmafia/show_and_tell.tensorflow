# Neural Caption Generator
* Implementation of "Show and Tell" http://arxiv.org/abs/1411.4555
 * Borrowed some code and ideas from Andrej Karpathy's NeuralTalk.
* You need flickr30k data (images and annotations)
 
### 코드
* make_flickr_dataset.py : Extracting feats of flickr30k images, and save them in './data/feats.npy' 
* model_tensorflow.py : TensorFlow Version
* model_theano.py : Theano Version
 
#### 사용방법
* Flickr30k Dataset Download
* Extract VGG Featues of Flicker30k images (make_flickr_dataset.py)
* Train: run train() in  model_tensorflow.py or model_theano.py
* Test: run test_v2() in model_tensorflow.py or test() in model_theano.py. 
 * parameters: VGG FC7 feature of test image, trained model path
