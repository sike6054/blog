---
title: "Inception-v3 논문 정리"
date: 2019-07-01 08:31:11 -0400
tags: AI ComputerVision Paper Inception-v3
categories:
  - Paper
toc: true
---

## Paper Information

SZEGEDY, Christian, et al. **"Rethinking the inception architecture for computer vision"**. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. p. 2818-2826.
> a.k.a. [Inception-v3 paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)


---
## Abstract
Convolutional network는 다양한 분야에서 state-of-the-art 성능을 가진 computer vision solution들의 핵심이다. 2014년부터 very deep convolutional network가 주류를 이뤘으며, 다양한 벤치마크에서 실질적인 성능 이득을 얻었다. 이 경우 모델의 크기나 계산 비용이 증가하긴 하지만, 충분한 양의 학습 데이터만 제공된다면 대부분의 작업에서 즉각적인 성능 향상이 이루어진다. 또한 convolution의 계산적인 효율성이나 적은 수의 parameter를 사용한다는 특징으로 인해, mobile vision이나 big-data scenario와 같은 다양한 케이스에 적용 가능하게 한다.

이 논문에서는 다음의 두 방법을 통해, 네트워크의 크기를 효율적으로 키우는 방법을 탐색한다.

1. **Suitably factorized convolutions**

2. **Aggressive regularization**

<br/>
제안하는 방법을 ILSVRC 2012 classification challenge의 validation set에 테스트하여, state-of-the-art 기법에 비해 상당한 성능 향상을 보였다. Inference 한 번에 소요되는 계산 비용이 [ less than 25 million paramters / 5 billion multiply-adds ]인 네트워크를 가지고 테스트 한 결과는 다음과 같다.
    
- Single frame evaluation에서 top-1 error가 21.2%이고, top-5 error가 5.6%
  
- 4가지 모델을 ensemble한 multi-crop evaluation에서 top-1 error가 17.3%이고, top-5 error가 3.5%

---
## 1. Introduction

2012년도 ImageNet 대회에서 우승했던 [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)은, 이후 다양한 종류의 컴퓨터 비전 분야에 성공적으로 적용됐다.
- [Object detection (R-CNN)](https://arxiv.org/pdf/1311.2524.pdf)

- [Segmentation (FCN)](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

- [Human pose estimation (DeepPose)](https://arxiv.org/pdf/1312.4659.pdf)

- [Video classification](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/42455.pdf)

- [Object tracking](https://papers.nips.cc/paper/5192-learning-a-deep-compact-image-representation-for-visual-tracking.pdf)

- [Super-resolution](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.642.1999&rep=rep1&type=pdf)

<br/>
이에 따라, 더 좋은 성능의 CNN을 발견하는 것에 초점을 둔 연구들이 성행했으며, 2014년 이후로는 더 깊고 넓어진 네트워크를 활용하면서 성능이 크게 향상됐다. [VGG](https://arxiv.org/pdf/1409.1556.pdf)와 [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)은 ILSVRC 2014에서 비슷한 성과를 나타냈으며, 이로부터 classification 성능의 향상이 다양한 응용 분야에서의 상당한 성능 향상으로 이어지는 경향이 있음을 알 수 있었다.

<br/>
즉, CNN 구조의 개선으로부터 visual feature에 의존하는 대부분의 컴퓨터 비전 분야에서의 성능 향상이 이뤄질 수 있음을 의미한다. 또한, 네트워크의 질이 높아지면서 CNN을 위한 새로운 응용 도메인이 생겼다.
>Detection의 region proposal의 경우와 같이, [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)의 성능이 hand-engineered solution만큼 성능이 나오지 않았던 분야에서도 CNN을 활용할 만한 상황이란 말이다.

<br/>
[VGG](https://arxiv.org/pdf/1409.1556.pdf)는 구조가 단순하다는 장점이 있지만, 계산 비용이 비싸다. 반면, [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)의 Inception 구조는 memory나 computational budget에 대해 엄격한 제약 조건이 주어지는 경우에서도 잘 수행되도록 설계되어있다.
>[AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)은 약 60 million, [VGG](https://arxiv.org/pdf/1409.1556.pdf)는 [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)의 약 3배 정도 되는 parameter를 사용하는 반면, [GoogleNet](https://arxiv.org/pdf/1409.4842.pdf)은 상대적으로 훨씬 적은 500만 개의 parameter만을 사용한다.

<br/>
Inception 구조의 계산 비용은 VGG나 혹은 보다 좋은 성능의 후속 네트워크들보다 훨씬 적다. 따라서 모바일과 같이 memory나 computational capacity가 제한된 환경에서 합리적인 비용으로 big-data를 다루는 경우에 inception network를 활용할 수 있다.

<br/>
물론, memory usage에 특화 된 solution을 사용하거나, 계산 트릭으로 특정 동작의 수행을 최적화하는 등의 방법으로도 문제를 어느 정도 완화시킬 수는 있다. 하지만, 이는 계산 복잡성이 가중될 뿐만 아니라, 이와 같은 기법들이 inception 구조의 최적화에도 적용된다면 효율성의 격차가 다시 벌어질 것이다.

<br/>
Inception 구조의 복잡성은 네트워크의 변경을 더욱 어렵게 만든다. 구조를 단순하게 확장한다면, 계산적인 장점의 많은 부분이 손실될 수 있다. 또한, [GoogleNet](https://arxiv.org/pdf/1409.4842.pdf)에서는 구조적인 결정들에 대한 디자인 철학을 명확하게 설명하지 않아, 효율성을 유지하면서 새로운 use case에 적용하는 등의 구조 변경이 훨씬 어려워진다.
>[이전 포스트](https://sike6054.github.io/blog/paper/second-post/)에서도 알아봤지만, [GoogleNet](https://arxiv.org/pdf/1409.4842.pdf)에서는 inception 구조를 설계한 목적 위주로 설명하고 있으며, 구조적인 결정에 대한 디자인 철학을 직접적으로 언급하진 않았다.

<br/>
만약 어떤 Inception 모델의 capacity를 올릴 필요가 있다고 판단되는 경우, 모든 filter bank의 크기를 2배 늘리는 간단한 방법을 취할 수 있다. 하지만, 이 경우 계산 비용과 parameter의 수는 4배가 될 것이며, 만약 이에 따른 이점이 적다면 real world에서는 사용할 수 없거나 불합리한 방법일 것이다.
>2배 늘려서 4배가 된다는 것을 보면, filter bank는 convolution filter를 뜻하는 것으로 보인다.

<br/>
이 논문에서는 CNN의 효율적인 확장에 유용한 몇 가지 일반적인 원칙과 최적화 아이디어를 설명하는 것으로 시작한다. 이는 근처 구성 요소들의 구조적인 변화에 따른 영향을 완화해주는 dimensional reduction과 inception의 병렬 구조를 충분히 사용함으로써 가능하다.
>설명하는 원칙들은 inception-type의 네트워크에만 국한된 원칙이 아니며, 일반적으로 inception style building block 구조가 이러한 제약 사항들을 자연스럽게 통합한다는 점 때문에 이들의 효과를 관찰하기 더 쉽다고 한다.

<br/>
또한, 모델의 높은 성능을 유지하기 위해서는 몇 가지 지침 원칙을 준수할 필요가 있으니 주의할 필요가 있다.

---
## 2. General Design Principles
여기에서는 large-sacle 데이터에 대한 실험을 근거하여, CNN의 다양한 구조적인 결정들에 대한 몇 가지 디자인 원칙을 설명한다.

<br/>
이 시점에서 아래 원칙들의 유용성은 추측에 기반하며, 이들의 정확성이나 타당성 평가를 위해선 추가적인 실험적 증거가 필요할 것이다.
>이러한 원칙들에서 크게 벗어나면 네트워크의 성능이 저하되는 경향이 있으며, 해당 부분을 수정하면 일반적으로 구조가 개선된다.

<br/>
### (1) Avoid representational bottlenecks
Feed-forward 네트워크는 input layer로부터 classifier나 regressor에 이르는 비순환 그래프로 나타낼 수 있으며, 정보가 흐르는 방향을 명확하게 알 수 있다.

<br/>
Input에서 output까지의 모든 layer의 경우, 각 layer를 통과하는 정보량에 접근할 수 있다. 이 때, **극단적인 압축으로 인한 정보의 bottleneck 현상이 발생하지 않도록 해야한다**.
>특히, 네트워크의 초반부에서 일어나는 bottleneck을 주의할 필요가 있다. 결국에는 모든 정보의 출처가 입력 데이터인데, 초반에서부터 bottleneck이 일어나서 정보 손실이 발생한다면 아무리 네트워크가 깊어진다 한들, 정보의 원천이 부실해지므로 성능의 한계가 발생하기 때문으로 생각된다.

<br/>
일반적으로 input에서 final representation까지 도달하면서 representation size가 서서히 감소해야 한다.
>Representation은 각 layer의 출력으로 생각하면 된다. 일반적으로 pooling 과정을 통해 feature map size가 작아지는데, 이 과정의 필요성을 말하는 것으로 보인다.

<br/>
이론적으로, correlation structure와 같은 중요한 요소를 버리기 때문에, 정보를 dimensionality of representation으로만 평가할 수 없다.

<br/>
### (2) Higher dimensional representations are easier to process locally within a network
CNN에서 activations per tile을 늘리면 disentangled feature를 많이 얻을 수 있으며, 네트워크가 더 빨리 학습하게 될 것이다.
>Conv layer의 filter map 개수를 늘리면, 다양한 경우의 activated feature map을 탐지할 수 있고, 이를 통해 네트워크의 학습이 빨라질 수 있다는 뜻으로 보인다. (modify)

<br/>
### (3) Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power
이를테면, 더 많은 convolution을 수행하기 전에, 심각한 부작용 없이 input representation의 dimension reduction이 가능하므로, 그 후에 spatial aggregation할 수 있다. 또한, 이러한 signal은 쉽게 압축 할 수 있어야한다는 점을 감안하면, dimension reduction으로 인해 학습 속도가 빨라질 것이다.
>Convolution 연산을 spatial aggregation이라 표현하는 것으로 보인다. Signal의 압축은, 학습 과정에서 네트워크의 각 layer를 거쳐가며, 원하는 동작을 위한 판단에 필요한 feature를 입력 데이터로부터 추출하는 작업을 signal의 압축 과정으로 생각한다면 쉽게 이해할 수 있다. 즉, **convolution을 다수 수행하는 경우에는 적절한 dimension reduction을 해주는 것이 빠른 학습에 도움 된다**는 것으로 보면 된다.
>
>이는 **출력이 spatial aggregation에 사용되는 경우, 인접한 unit 간의 강력한 상관 관계로 인해 dimension reduction 중의 정보 손실이 훨씬 줄어들 것이라는 가설**에 근거한 원칙이다.

<br/>
### (4) Balance the width and depth of the network
네트워크의 optimal performance는 [ 각 stage의 filter의 개수 / 네트워크의 depth ]의 밸런싱으로 달성 할 수 있다.
>Width는 각 stage의 filter의 개수에 해당한다.

<br/>
네트워크의 width와 depth를 모두 늘리는 것은 성능 향상에 도움될 수 있다. 이 때, 둘을 병렬적으로 증가 시킨다면 일정한 계산량에 대한 optimal improvement에 도달할 수 있다.
>즉, **늘릴 때 늘리더라도, computational budget이 depth와 width 간에 균형 잡힌 방식으로 할당되도록 네트워크를 구성**해야 최적의 성능을 보일 것이다.

<br/>
<br/>
이러한 원칙들이 타당하긴 하지만, 이를 활용해서 네트워크의 성능을 향상시키는 것은 간단하지 않다. 따라서, 모호한 상황에서만 이 아이디어들을 고려하도록 하자.

---
## 3. Factorizing Convolutions with Large Filter Size
GoogLeNet 네트워크의 이득 중 상당 부분은 dimension reduction를 충분히 사용함으로써 발생한 것이다. 이는 계산 효율이 좋은 방식으로 convolution을 factorizing하는 것의 특별한 케이스로 볼 수 있다.

<br/>
1x1 conv layer 다음에 3x3 conv layer가 오는 경우를 생각해보자. 비전 네트워크에서는 인접한 activation들의 출력 간에 높은 상관 관계가 예상된다. 따라서, aggregation 전에 이들의 activation이 줄어들 수 있으며, 유사한 표현력의 local representation을 가지는 것으로 볼 수 있다.
>상관 관계가 높은 activation 간에는 유사한 표현력을 지니며, 이들의 수가 줄어들더라도 상관없는 것으로 생각된다. ???진ㅉ?

<br/>
이 장에서는 특히 모델의 계산 효율을 높이는 목적을 고려하여, 다양한 환경에서의 convolution factorizing 방법들을 알아본다.

<br/>
Inception network는 fully convolutional하기 때문에, 각 weight는 activation 당 하나의 곱셈에 해당한다. 따라서, 계산 비용을 줄이면 paramter의 수가 줄어들게 된다.

<br/>
이는 적절한 factorizing이 이뤄지면 더 많은 disentangled parameter를 얻을 수 있으며, 이에 따라 빠른 학습이 가능하다는 것을 의미한다.(check)

<br/>
또한 메모리를 포함한 계산 비용의 절감을 통해, single computer에서 모델의 각 복제본들을 학습할 수 있는 능력을 유지하면서 네트워크의 filter-bank size를 늘릴 수 있다.
>[이전 포스트](https://sike6054.github.io/blog/paper/second-post/)에서도 알아봤었지만, [GoogleNet](https://arxiv.org/pdf/1409.4842.pdf)에서는 [DistBelief](https://www.cs.toronto.edu/~ranzato/publications/DistBeliefNIPS2012_withAppendix.pdf)라는 이름의 프레임워크로 분산 학습을 수행했었다. 반면, 이 논문에서는 [TensorFlow에서 자체 개발한 분산 학습 시스템](https://arxiv.org/pdf/1603.04467.pdf)을 이용하고 있으며, 실험에서는 50개의 복제본(replica)에 대한 분산 학습을 진행했다고 한다.

<br/>
### 3.1 Factorization into smaller convolutions
보다 큰 spatial filter를 갖는 convolution은 계산적인 측면에서 불균형하게 비싼 경향이 있다.
>n개의 filter로 이루어진 5x5 convolution 연산의 경우, 같은 수의 filter를 사용하는 3x3의 convolution보다 계산 비용이 25/9로, 약 2.78배 더 비싸다.

<br/>
물론, 보다 큰 filter는 이전 layer의 출력에서 더 멀리 떨어진 unit activation 간의 신호 종속성을 포착할 수 있기 때문에, filter의 크기를 줄이면 그만큼 표현력을 위한 비용이 커지게 된다. 그래서 논문의 저자들은 5x5 convolution을 동일한 input size와 output depth를 가지면서, 더 적은 parameter를 가진 multi-layer 네트워크로 대체할 방법에 대해 고민한다. (check)

<br/>
Fig.1은 5x5 convolution의 computational graph를 확대한 것이다. 각 출력은 입력에 대해 5x5 filter가 sliding하는 형태의 소규모 fully-connected 네트워크처럼 보인다.
![Fig.1](/blog/images/Inception-v3, Fig.1(removed).png )
>**Fig.1** <br/>Mini-network replacing the 5x5 convolutions.

<br/>
여기선 vision network를 구축하고 있기 때문에, fully-connected component를 2-layer convolution로 대체하여 translation invariance을 다시 이용하는 것이 자연스러워 보인다.
>여기서 말하는 2-layer convolution은 Fig.1에서 보였다. 즉, 첫 번째 layer는 3x3 conv이고, 두 번째 layer는 첫 번째 layer의 3x3 output grid 위에 연결된 fully-connected layer이다. 이와 같이 input activation grid에 sliding하는 filter를 5x5 conv에서 2-layer 3x3 conv로 대체하는 것이 이 절에서 제안하는 factorizing 방법이다. (Fig.2와 Fig.3 비교)
>
>Translation invariance는 입력에 shift가 일어난 경우에도 변함 없이 학습한 패턴을 캡처하는 convolution 방식의 특성을 말하는 것이다. 각종 invariance-type은 아래 그림을 참조하자.
>
><br/>
>![Extra.1](/blog/images/Inception-v3, Extra.1(removed).png )
>
><br/>[그림 출처](https://stats.stackexchange.com/questions/208936/what-is-translation-invariance-in-computer-vision-and-convolutional-neural-netwo/208949#208949)

<br/>
![Fig.2](/blog/images/Inception-v3, Fig.4(removed).png )
>**Fig.2** <br/>[GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)에서 제안 된 기존의 inception module

<br/>
![Fig.3](/blog/images/Inception-v3, Fig.5(removed).png )
>**Fig.3** <br/>2장의 원칙 3을 위해, 이 절에서 제안한 inception module

<br/>
이 구조는 인접한 unit 간의 weight를 공유함으로써 parameter 수를 확실히 줄여준다. 절감되는 계산 비용을 예측 분석하기 위해, 일반적인 상황에 적용 할 수 있는 몇 가지 단순한 가정을 해보자. 우선 $n = \alpha m$로 가정한다. 즉, activation이나 unit의 개수를 상수 $\alpha$에 따라 결정한다.
>5x5 convolution을 수행하는 경우엔 $\alpha$가 일반적으로 1보다 약간 크며, GoogLeNet의 경우엔 약 1.5를 사용했었다.

<br/>
5x5 conv layer를 2-layer로 바꾸는 경우, 두 단계로 확장하는 것이 합리적이다. 여기선 문제를 단순화 하기 위해, 확장을 하지 않는 $\alpha =1$을 고려한다.
>2-layer의 경우, 각 단계에서 filter 수를 ${\sqrt \alpha}$만큼 증가시키는 방법을 취할 수 있다.

<br/>
만약 인접한 grid tile 간에 계산 결과를 재사용하지 않으면서, 단순히 5x5 convolution sliding만 하게 된다면 계산 비용이 증가하게 될 것이다. 이 때, 5x5 convolution sliding을 인접한 tile 간의 activation을 재사용하는 형태의 2-layer 3x3 convolution으로 나타낼 수 있으며, 이 경우에는 $\frac{9+9}{25}=0.72$배로 계산량이 감소된다.
>이 경우는 factorizing을 통해 28%의 상대적 이득을 얻는 것에 해당한다.

이 경우에도 parameter들은 각 unit의 activation 계산에서 정확히 한 번씩만 사용되므로, parameter 개수에 대해서도 정확히 동일한 절약이 일어난다.

<br/>
물론, 이 방법에 대해서도 두 가지 의문점이 생길 수 있다.

1. 위와 같은 replacement로부터 표현력 손실이 발생하는가?

2. 계산의 linear part에 대한 factorizing이 목적인 경우엔, 2-layer 중 first layer에서는 linear activation을 유지해야 하는가?

<br/>
이에 대한 몇 가지 실험을 통해, factorization에 linear activation을 사용하는 것이, 모든 단계에서 ReLU를 사용하는 것보다 성능이 좋지 않은 것을 확인했다고 한다. 실험 결과는 아래 Fig.4를 참조하자.

<br/>
![Fig.4](/blog/images/Inception-v3, Fig.2(removed).png )
>**Fig.4** <br/>빨간 dash line에 해당하는 **Linear**는 activation으로 linear와 ReLU를 사용한 것이며, 파란 solid line에 해당하는 **ReLU**는 두 activation 모두 ReLU를 사용한 결과이다.
>
>결과는 3.86 million iteration 후에 top-1 validation accuracy가 각각 76.2%와 77.2%로 측정됐다.

<br/>
저자들은 이러한 이득들이 네트워크가 학습할 수 있는 space of variation을 확대해준다고 보며, 특히 BN을 사용하는 경우에 그런 경향이 강하다고 한다. Dimension reduction에서 linear activation을 사용하는 경우에도 비슷한 효과를 볼 수 있다. (check)

<br/>
### 3.2 Spatial Factorization into Asymmetric Convolutions
3.1절에 따르면, filter의 크기가 3x3보다 큰 convolution은 항상 3x3 convolution의 sequence로 축소될 수 있으므로, 이를 이용하는 것은 보통 효율적이지 않다고 볼 수 있다.

<br/>
물론 2x2 convolution과 같이 더 작은 단위로 factorizing을 할 수도 있지만, $n\times 1$과과 같은 asymmetric convolution을 사용하는 것이 훨씬 좋은 것으로 밝혀졌다.

<br/>
3x1 convolution 뒤에 1x3 convolution을 사용한 2-layer를 sliding 하는 것과, 3x3 convolution의 receptive field는 동일하다. Fig.5 참조.

<br/>
![Fig.5](/blog/images/Inception-v3, Fig.3(removed).png )
>**Fig.5** <br/>Mini-network replacing the 3x3 convolutions.

<br/>
여전히 입출력의 filter 수가 같은 경우에는, 같은 수의 output filter에 대해 2-layer solution이  $\frac{3+3}{9}=0.66$배로 계산량이 감소된다.
>3x3 convolution을 두 개의 2x2 convolution으로 나누는 경우에는 계산량이 $\frac{4+4}{9}=0.89$배로 절약되어, asymmetric fatorizing보다 효과가 적은 것을 알 수 있다.

<br/>
이론적으로 더 나가보자면, Fig.6과 같이 nxn convolution은 1xn 뒤에 nx1 convolution이 오는 형태로 대체할 수 있으며, 여기서 n이 커짐에 따라 계산 비용 절감이 극적으로 증가한다고 주장할 수 있다.

<br/>
![Fig.6](/blog/images/Inception-v3, Fig.6(removed).png )
>**Fig.6** <br/>$n \times n$ convolution을 factorizing한 inception module이다. 제안 된 구조에서는 $17 \times 17$ grid에서 n=7로 적용했다.

<br/>
실험을 통해 이와 같은 factorization이 grid-size가 큰 초반부의 layer에서는 잘 동작하지 않지만, medium grid-size인 중후반 layer에서는 7x1 과 1x7 convolution을 사용하여 매우 좋은 결과를 얻을 수 있었다.
>여기서 medium grid-size는 $m \times m$ feature map의 $m$이 12~20정도인 경우를 말한다.


---
## 4. Utility of Auxiliary Classifiers
[GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)은 very deep network의 수렴을 개선시키기 위해 보조 분류기(Auxiliary Classifier)를 도입했다.

보조 분류기는 원래 동기는 다음과 같다.

1. useful한 gradient를 하위 layer로 밀어 넣어, 즉시 useful하게 만들기 위함

2. Very deep network의 vanishing gradient 문제를 해결하여, 학습 중의 수렴을 개선시키기 위함

<br/>
이 외에도, [Lee 등의 연구](https://arxiv.org/pdf/1409.5185.pdf) 등에서 보조 분류기가 보다 안정적인 학습과 더 나은 수렴을 촉진한다고 주장했다.

<br/>
하지만, 학습 초기에는 보조 classifier들이 수렴을 개선시키지 않는다는 흥미로운 결과를 발견했다고 한다. 높은 정확도에 도달하기 전까지의 학습 과정에서는 보조 분류기의 유무랑 관계없이 유사한 성능을 보였지만, 학습이 끝날 무렵에는 보조 분류기가 있는 네트워크에서 정확도를 앞지르기 시작하다가 결과적으론 조금 더 높은 성능에 도달하며 학습이 종료됐다고 한다.

<br/>
[GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)에서는 두 개의 보조 분류기가 각각 다른 stage에 사용됐었지만, 하위 stage의 보조 분류기 하나를 제거하더라도 최종 성능에 악영향을 미치지 않았다고 한다. 사용하는 보조 분류기는 Fig.7과 같다.

<br/>
![Fig.7](/blog/images/Inception-v3, Fig.8(removed).png )
>**Fig.7** <br/>제안 된 구조에서 사용하는 보조 분류기로, grid size가 17x17인 layer 중 가장 상위에 위치한다. [BN](https://arxiv.org/pdf/1502.03167.pdf)을 사용할 경우에는 top-1 accuracy가 0.4%정도 증가했다.

<br/>
이 두 관찰 결과가 의미하는 바는, 원래의 [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)에서 세운 가설인 "**보조 분류기가 low-level feature의 발전에 도움이 된다**"는 것이 잘못된 것일 수도 있음을 뜻한다. 그 대신, 저자들은 **이러한 보조 분류기가 regularizer로 동작한다**고 주장한다.
>이는 보조 분류기에서 [BN](https://arxiv.org/pdf/1502.03167.pdf)이나 [drop-out](https://arxiv.org/pdf/1207.0580.pdf)이 사용되는 경우에, 주 분류기의 결과가 더 좋아진다는 사실이 근거가 되는 주장이라 한다. 이는 또한, [BN](https://arxiv.org/pdf/1502.03167.pdf)이 regularizer 역할을 한다는 추측에 대한 미약한 증거가 된다고 한다.

---
## 5. Efficient Grid Size Reduction
전통적으로 CNN은 pooling 연산을 통해서 feature map의 grid size를 줄인다. 이 때, representational bottleneck을 피하기 위해, pooling을 적용하기 전에 activated filter의 dimension이 확장된다.
>예를 들어, $d \times d$ grid에 k개의 filter로부터 시작해서, $\frac{d}{2} \times \frac{d}{2}$ grid와 2k개의 filter에 도달하려면, 먼저 2k개의 filter로 stride가 1인 convolution을 계산한 후에 pooling을 수행한다.
>
>Grid size가 줄어들기만 하는건 grid에 들어있던 정보를 보다 저차원의 데이터로 압축하는 것이기 때문에, 이를 병목 현상으로 볼 수 있다. 이 때문에, filter의 개수를 먼저 늘려준다면 정보의 병목 현상을 완화시키는 효과가 있는 것이다.

<br/>
하지만, 이는 네트워크의 전체 계산 비용이 pooling 이전의 확장 단계에서 일어나는 $2{d^2}k$ 에 크게 좌우하게 된다는 것을 의미한다.
>논문에서는 $2{d^2}k^2$라고 되어있다. 하지만, stride가 1이라면 각 convolution filter마다 $d^2$번씩 계산하며, filter의 개수인 $2k$개만큼 곱해지면 총 $2{d^2}k$개가 맞는 것으로 보인다.

<br/>
만약 convloution과 pooling의 순서를 바꾼다면, 계산 비용이 4분의 1로 감소 된 $2{{\frac{d}{2}}^2}k$가 된다. 하지만, 이는 representation의 전반적인 차원이 ${{\frac{d}{2}}^2}k$로 낮아져서 표현력이 떨어지게 되고, 이는 곧 representational bottleneck을 야기한다. Fig.8 참조.
>여기도 마찬가지로, 논문에서는 $2{{\frac{d}{2}}^2}k$대신 $2{{\frac{d}{2}}^2}k^2$로 나타나 있다. 하지만, ${{\frac{d}{2}}^2}k$는 제대로 계산됐다.

<br/>
![Fig.8](/blog/images/Inception-v3, Fig.9(removed).png )
>**Fig.8** <br/>Grid size를 줄이는 두 가지 방법. 좌측의 솔루션은 representational bottleneck을 피하라는 1번 원칙을 반하며, 우측의 경우엔 계산 비용이 3배나 비싸다.

<br/>
이 장에서는 representational bottleneck을 피하면서, 계산 비용도 줄일 수 있는 구조를 제안한다. 제안하는 방법은 stride가 2인 block 2개를 병렬로 사용한다. 각 블록은 pooling layer와 conv layer로 이루어져 있으며, pooling은 maximum 혹은 average를 사용한다. 두 block의 filter bank는 Fig.9에 나타난 것처럼 concatenate로 연결 된다.

<br/>
![Fig.9](/blog/images/Inception-v3, Fig.10(removed).png )
>**Fig.9** <br/>좌측은 filter bank를 확장하면서 grid size를 줄이는 inception module이다. 이는 계산 비용도 저렴하면서도 representational bottleneck을 피하므로, 원칙 1을 준수한다. 우측 다이어그램은 좌측과 같은 솔루션을 나타내지만, grid size의 관점에서 나타냈기 때문에 표현 방식이 다른 것 뿐이다.

<br/>
그냥 넘어가기 전에, 제안한 방법이 상대적으로 얼마나 저렴해지는지 알아보자. 우선 Fig.9의 우측 다이어그램을 보면, convolution part와 pooling part에 각각 $2k$의 절반인 $k$만큼 할당한다는 것을 알 수 있다.
>이 논문에서는 parametric operation만을 비용으로 계산하고 있기 때문에, convolution part만 계산하면 된다.

<br/>
좌측 다이어그램에 따르면, convolution part는 두 개의 branch로 이뤄져있음을 알 수 있다. 여기서 두 branch 간의 filter 수의 비율은 언급되지 않았지만, 절반으로 가정하고 계산해보자. 우선 2-layer인 branch에서는 stride가 1인 것과 2인 conv layer에서 각각 ${d^2}k$와 ${\frac{d}{2}}^2k$만큼 비용이 발생하며, 1-layer인 branch에서는 ${\frac{d}{2}}^2k$만큼 비용이 발생한다. 이를 다 더하면, **총 $\frac{3}{2} {d^2}k$만큼 비용**이 발생한다. 기존의 $2{d^2}k$에 비하면, **제안하는 방법의 계산 비용이 25% 저렴**하다고 볼 수 있다.

---
## 6. Inception-v2
위에서 언급한 것들을 결합하여, 새로운 아키텍처를 제안한다. 이 구조는 ILSVRC 2012 classification benchmark에서 향상된 성능을 보였다. 네트워크의 개요는 Table.1에서 보인다.

<br/>
![Table.1](/blog/images/Inception-v3, Table.1(removed).png )
>**Table.1** <br/>제안 된 네트워크 구조의 개요이다.
>
>각 module의 출력 크기는 다음 module의 입력 크기에 해당하며, 각 inception block 사이에는 Fig.9에 나타난 reduction 기법을 사용한다.
>
>Padded라고 표기 된 convolution과, grid size 유지가 필요한 inception module을 제외하고는 padding을 사용하지 않는다.
>>출력 크기를 보면, 최초의 inception block이 시작되기 전의 'conv'는 'conv padded'가 되어야 하는 것으로 보인다.
>
>원칙 4를 위해, 다양한 filter bank size를 택하고 있다.
>
>>각 inception 모듈에 해당하는 Figure 번호는 논문 기준이며, 본 포스팅 기준으로는 파란색 번호인 Fig.3, Fig.6, Fig.10에 해당한다.

<br/>
3.1절의 아이디어를 기반으로, 기존의 7x7 convolution을 3개의 3x3 convolution으로 factorizing 했다.

<br/>
네트워크의 inception part는 기존의 inception 모듈에 3.1절의 factorizing 기법만 사용한 기존의 inception module이 3개 뒤따른다. 이 때, 각 입력 grid는 35x35x288에 해당하며, 마지막에는 5장의 reduction 기법으로 grid가 17x17x768로 축소된다.
>두 방법은 각각 Fig.3과 Fig.9의 inception module을 말한다.

<br/>
다음은 3.2절의 asymmetric fatorizing 기법을 이용한 inception module이 5개 뒤따르며, 마찬가지로 5장의 reduction 기법에 의해 grid가 8x8x1280으로 축소된다.
>두 방법은 각각 Fig.6과 Fig.9의 inception module을 말한다.

<br/>
Grid size가 8x8로 가장 축소 된 단계에서는 Fig.10의 inception module이 2개 뒤따른다. 각 tile에 대한 출력의 filter bank size는 2048이 된다.

<br/>
![Fig.10](/blog/images/Inception-v3, Fig.7(removed).png )
>**Fig.10** <br/>Filter bank size를 확장한 inception module이며, grid size가 가장 축소됐을 때 사용한다. Grid size가 작아졌다는 것은 그 만큼 high dimensional representation이란 것이기 때문에, 2번 원칙에 따라 locally하게 처리하는 것이다.

<br/>
Inception module 내부의 filter bank size를 포함한 네트워크 구조의 자세한 정보는, 본 논문의 tar 파일에 포함된 model.txt에 나와 있다.
>나한텐 안 줬다. 그래서 하단의 Keras 구현 코드는 GoogLeNet을 참고한 것이다.

<br/>
아무튼 저자들은 2장의 원칙들을 준수하는 한, 구조의 다양한 변화에도 비교적 안정적인 성능을 보인다는 것을 알 수 있었다고 한다. 제안 된 네트워크는 42-layer나 됨에도 불구하고, 계산 비용이 GoogLeNet(22-layer)보다 약 2.5배만 비싸며, VGGNet보다 훨씬 효율적이다.

---
## 7. Model Regularization via Label Smoothing

---
## 8. Training Methodology

---
## 9. Performance on Lower Resolution Input

---
## 10. Experimental Results and Comparisons

---
## 11. Conclusions


---
작성 중

<br/>
{% include disqus.html %}
