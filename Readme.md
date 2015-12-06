# Neural Caption Generator
* Tensorflow implementation of "Show and Tell" http://arxiv.org/abs/1411.4555
 * Borrowed some code and ideas from Andrej Karpathy's NeuralTalk.
* You need flickr30k data (images and annotations)
 
### Code
* make_flickr_dataset.py : Extracting feats of flickr30k images, and save them in './data/feats.npy' 
* model_tensorflow.py : TensorFlow Version
* model_theano.py : Theano Version
 
#### Usage
* Flickr30k Dataset Download
* Extract VGG Featues of Flicker30k images (make_flickr_dataset.py)
* Train: run train() in  model_tensorflow.py or model_theano.py
* Test: run test() in model_tensorflow.py or model_theano.py. 
 * parameters: VGG FC7 feature of test image, trained model path
