# Neural Caption Generator
* Implementation of "Show and Tell" http://arxiv.org/abs/1411.4555
 * Andrej Karpathy의 NeuralTalk 참고함.
* flickr30k 데이터 필요.
 
### 코드
* make_flickr_dataset.py : 단어 dictionary 생성
* model_tensorflow.py : TensorFlow 버전
* model_theano.py : Theano 버전
 
#### 사용방법
* Flickr30k Dataset Download
* Flicker30k 이미지에 대한 VGG feature 추출 (별도로 해야 함)
* make_flickr_dataset.py 이용해 dictionary 생성
* Train: model_tensorflow.py 에서 train()
* Test: model_tensorflow.py에서 test(). 
 * parameters: 테스트 이미지의 VGG FC7 feature, 학습된 모델 path
