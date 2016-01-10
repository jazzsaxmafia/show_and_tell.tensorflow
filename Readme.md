# Neural Caption Generator
* Tensorflow implementation of "Show and Tell" http://arxiv.org/abs/1411.4555
 * Borrowed some code and ideas from Andrej Karpathy's NeuralTalk.
* You need flickr30k data (images and annotations)
 
### Code
* make_flickr_dataset.py : Extracting feats of flickr30k images, and save them in './data/feats.npy' 
* model.py : TensorFlow Version
 
#### Usage
* Flickr30k Dataset Download
* Extract VGG Featues of Flicker30k images (make_flickr_dataset.py)
* Train: run train() in  model.py
* Test: run test() or test_tf() in model.py
 * parameters: VGG FC7 feature of test image, trained model path
 * Once you download Tensorflow VGG Net (one of the links below), you don't need Caffe when testing.

#### Downloading data/trained model
* Extraced FC7 data: [download](https://drive.google.com/file/d/0B5o40yxdA9PqTnJuWGVkcFlqcG8/view?usp=sharing)
 * This is used in train() function in model.py. You can skip feature extraction part by using this.
* Pretrained model [download](https://drive.google.com/file/d/0B5o40yxdA9PqeW4wY0wwZXhrZkE/view?usp=sharing)
 * This is used in test() and test_tf() in model.py. If you do not have time for training, or if you just want to check out captioning, download and test the model.
* Tensorflow VGG net [download](https://drive.google.com/file/d/0B5o40yxdA9PqSGtVODN0UUlaWTg/view?usp=sharing)
 * This file is used in test_tf() in model.py
* Along with the files above, you might want to download flickr30k annotation data from [link](http://shannon.cs.illinois.edu/DenotationGraph/) 

![alt tag](https://github.com/jazzsaxmafia/show_and_tell.tensorflow/blob/master/result.jpg)

### License
* BSD license
