---
title: "(Inception-v3) Rethinking the inception architecture for computer vision 번역 및 추가 설명과 Keras 구현"
date: 2019-07-01 08:31:11 -0400
tags: AI ComputerVision Paper Inception-v3
categories:
  - Paper
toc: true
sitemap :
  priority : 1.0
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

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
>Conv layer의 filter map 개수를 늘리면, 다양한 경우의 activated feature map을 탐지할 수 있고, 이를 통해 네트워크의 학습이 빨라질 수 있다는 뜻으로 보인다.

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
>상관 관계가 높은 activation 간에는 유사한 표현력을 지니며, 이들의 수가 줄어들더라도 상관없는 것으로 생각된다.

<br/>
이 장에서는 특히 모델의 계산 효율을 높이는 목적을 고려하여, 다양한 환경에서의 convolution factorizing 방법들을 알아본다.

<br/>
Inception network는 fully convolutional하기 때문에, 각 weight는 activation 당 하나의 곱셈에 해당한다. 따라서, 계산 비용을 줄이면 paramter의 수가 줄어들게 된다.

<br/>
이는 적절한 factorizing이 이뤄지면 더 많은 disentangled parameter를 얻을 수 있으며, 이에 따라 빠른 학습이 가능하다는 것을 의미한다.

<br/>
또한 메모리를 포함한 계산 비용의 절감을 통해, single computer에서 모델의 각 복제본들을 학습할 수 있는 능력을 유지하면서 네트워크의 filter-bank size를 늘릴 수 있다.
>[이전 포스트](https://sike6054.github.io/blog/paper/second-post/)에서도 알아봤었지만, [GoogleNet](https://arxiv.org/pdf/1409.4842.pdf)에서는 [DistBelief](https://www.cs.toronto.edu/~ranzato/publications/DistBeliefNIPS2012_withAppendix.pdf)라는 이름의 프레임워크로 분산 학습을 수행했었다. 반면, 이 논문에서는 [TensorFlow에서 자체 개발한 분산 학습 시스템](https://arxiv.org/pdf/1603.04467.pdf)을 이용하고 있으며, 실험에서는 50개의 복제본(replica)에 대한 분산 학습을 진행했다고 한다.

<br/>
### 3.1 Factorization into smaller convolutions
보다 큰 spatial filter를 갖는 convolution은 계산적인 측면에서 불균형하게 비싼 경향이 있다.
>n개의 filter로 이루어진 5x5 convolution 연산의 경우, 같은 수의 filter를 사용하는 3x3의 convolution보다 계산 비용이 $$\frac{25}{9}$$로, 약 2.78배 더 비싸다.

<br/>
물론, 보다 큰 filter는 이전 layer의 출력에서 더 멀리 떨어진 unit activation 간의 신호 종속성을 포착할 수 있기 때문에, filter의 크기를 줄이면 그만큼 표현력을 위한 비용이 커지게 된다. 그래서 논문의 저자들은 5x5 convolution을 동일한 input size와 output depth를 가지면서, 더 적은 parameter를 가진 multi-layer 네트워크로 대체할 방법에 대해 고민한다.

<br/>
Fig.1은 5x5 convolution의 computational graph를 확대한 것이다. 각 출력은 입력에 대해 5x5 filter가 sliding하는 형태의 소규모 fully-connected 네트워크처럼 보인다.

<br/>
![Fig.1](/blog/images/Inception-v3, Fig.1(removed).png )
>**Fig.1** <br/>Mini-network replacing the $$5\times 5$$ convolutions.

<br/>
여기선 vision network를 구축하고 있기 때문에, fully-connected component를 2-layer convolution로 대체하여 translation invariance을 다시 이용하는 것이 자연스러워 보인다.
>여기서 말하는 2-layer convolution은 Fig.1에서 보였다. 즉, 첫 번째 layer는 $$3\times 3$$ conv이고, 두 번째 layer는 첫 번째 layer의 $$3\times 3$$ output grid 위에 연결된 fully-connected layer이다. 이와 같이 input activation grid에 sliding하는 filter를 $$5\times 5$$ conv에서 2-layer $$3\times 3$$ conv로 대체하는 것이 이 절에서 제안하는 factorizing 방법이다. (Fig.2와 Fig.3 비교)
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
이 구조는 인접한 unit 간의 weight를 공유함으로써 parameter 수를 확실히 줄여준다. 절감되는 계산 비용을 예측 분석하기 위해, 일반적인 상황에 적용 할 수 있는 몇 가지 단순한 가정을 해보자. 우선 $$n = \alpha m$$로 가정한다. 즉, activation이나 unit의 개수를 상수 $$\alpha$$에 따라 결정한다.
>5x5 convolution을 수행하는 경우엔 $$\alpha$$가 일반적으로 1보다 약간 크며, GoogLeNet의 경우엔 약 1.5를 사용했었다.

<br/>
5x5 conv layer를 2-layer로 바꾸는 경우, 두 단계로 확장하는 것이 합리적이다. 여기선 문제를 단순화 하기 위해, 확장을 하지 않는 $$\alpha =1$$을 고려한다.
>2-layer의 경우, 각 단계에서 filter 수를 $${\sqrt \alpha}$$만큼 증가시키는 방법을 취할 수 있다.

<br/>
만약 인접한 grid tile 간에 계산 결과를 재사용하지 않으면서, 단순히 $$5\times 5$$ convolution sliding만 하게 된다면 계산 비용이 증가하게 될 것이다. 이 때, $$5\times 5$$ convolution sliding을 인접한 tile 간의 activation을 재사용하는 형태의 2-layer $$3\times 3$$ convolution으로 나타낼 수 있으며, 이 경우에는 $$\frac{9+9}{25}=0.72$$배로 계산량이 감소된다.
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
저자들은 이러한 이득들이 네트워크가 학습할 수 있는 space of variation을 확대해준다고 보며, 특히 output activation을 [batch-normalize](https://arxiv.org/pdf/1502.03167.pdf)하는 경우에 그런 경향이 강하다고 한다. Dimension reduction에 linear activation을 사용하는 경우에도 비슷한 효과를 볼 수 있다고 한다.
>네트워크가 학습할 수 있는 space of variation은, 모델의 capacity를 말한다.

<br/>
### 3.2 Spatial Factorization into Asymmetric Convolutions
3.1절에 따르면, filter의 크기가 3x3보다 큰 convolution은 항상 $$3\times 3$$ convolution의 sequence로 축소될 수 있으므로, 이를 이용하는 것은 보통 효율적이지 않다고 볼 수 있다.

<br/>
물론 $$2\times 2$$ convolution과 같이 더 작은 단위로 factorizing을 할 수도 있지만, $$n\times1$$과과 같은 asymmetric convolution을 사용하는 것이 훨씬 좋은 것으로 밝혀졌다.

<br/>
3x1 convolution 뒤에 1x3 convolution을 사용한 2-layer를 sliding 하는 것과, $$3\times 3$$ convolution의 receptive field는 동일하다. Fig.5 참조.

<br/>
![Fig.5](/blog/images/Inception-v3, Fig.3(removed).png )
>**Fig.5** <br/>Mini-network replacing the $$3\times 3$$ convolutions.

<br/>
여전히 입출력의 filter 수가 같은 경우에는, 같은 수의 output filter에 대해 2-layer solution이  $$\frac{3+3}{9}=0.66$$배로 계산량이 감소된다.
>3x3 convolution을 두 개의 $$2\times 2$$ convolution으로 나누는 경우에는 계산량이 $$\frac{4+4}{9}=0.89$$배로 절약되어, asymmetric fatorizing보다 효과가 적은 것을 알 수 있다.

<br/>
이론적으로 더 나가보자면, Fig.6과 같이 nxn convolution은 $$1\times n$$ 뒤에 $$n\times 1$$ convolution이 오는 형태로 대체할 수 있으며, 여기서 $$n$$이 커짐에 따라 계산 비용 절감이 극적으로 증가한다고 주장할 수 있다.

<br/>
![Fig.6](/blog/images/Inception-v3, Fig.6(removed).png )
>**Fig.6** <br/>$$n \times n$$ convolution을 factorizing한 inception module이다. 제안 된 구조에서는 $$17 \times 17$$ grid에서 $$n=7$$로 적용했다.

<br/>
실험을 통해 이와 같은 factorization이 grid-size가 큰 초반부의 layer에서는 잘 동작하지 않지만, medium grid-size인 중후반 layer에서는 $$7\times 1$$ 과 $$1\times 7$$ convolution을 사용하여 매우 좋은 결과를 얻을 수 있었다.
>여기서 medium grid-size는 $$m \times m$$ feature map의 $$m$$이 12~20정도인 경우를 말한다.


---
## 4. Utility of Auxiliary Classifiers
[GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)은 very deep network의 수렴을 개선시키기 위해 보조 분류기(Auxiliary Classifier)를 도입했다.

보조 분류기는 원래 동기는 다음과 같다.

1. Useful한 gradient를 하위 layer로 밀어 넣어, 즉시 useful하게 만들기 위함

2. Very deep network의 vanishing gradient 문제를 해결하여, 학습 중의 수렴을 개선시키기 위함

<br/>
이 외에도, [Lee 등의 연구](https://arxiv.org/pdf/1409.5185.pdf) 등에서 보조 분류기가 보다 안정적인 학습과 더 나은 수렴을 촉진한다고 주장했다.

<br/>
하지만, 학습 초기에는 보조 classifier들이 수렴을 개선시키지 않는다는 흥미로운 결과를 발견했다고 한다. 높은 정확도에 도달하기 전까지의 학습 과정에서는 보조 분류기의 유무랑 관계없이 유사한 성능을 보였지만, 학습이 끝날 무렵에는 보조 분류기가 있는 네트워크에서 정확도를 앞지르기 시작하다가 결과적으론 조금 더 높은 성능에 도달하며 학습이 종료됐다고 한다.

<br/>
[GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)에서는 두 개의 보조 분류기가 각각 다른 stage에 사용됐었지만, 하위 stage의 보조 분류기 하나를 제거하더라도 최종 성능에 악영향을 미치지 않았다고 한다. 사용하는 보조 분류기는 Fig.7과 같다.

<br/>
![Fig.7](/blog/images/Inception-v3, Fig.8(removed).png )
>**Fig.7** <br/>제안 된 구조에서 사용하는 보조 분류기로, grid size가 $$17\times 17$$인 layer 중 가장 상위에 위치한다. [BN](https://arxiv.org/pdf/1502.03167.pdf)을 사용할 경우에는 top-1 accuracy가 0.4%정도 증가했다.

<br/>
이 두 관찰 결과가 의미하는 바는, 원래의 [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)에서 세운 가설인 "**보조 분류기가 low-level feature의 발전에 도움이 된다**"는 것이 잘못된 것일 수도 있음을 뜻한다. 그 대신, 저자들은 **이러한 보조 분류기가 regularizer로 동작한다**고 주장한다.
>이는 보조 분류기에서 [BN](https://arxiv.org/pdf/1502.03167.pdf)이나 [drop-out](https://arxiv.org/pdf/1207.0580.pdf)이 사용되는 경우에, 주 분류기의 결과가 더 좋아진다는 사실이 근거가 되는 주장이라 한다. 이는 또한, [BN](https://arxiv.org/pdf/1502.03167.pdf)이 regularizer 역할을 한다는 추측에 대한 미약한 증거가 된다고 한다.

---
## 5. Efficient Grid Size Reduction
전통적으로 CNN은 pooling 연산을 통해서 feature map의 grid size를 줄인다. 이 때, representational bottleneck을 피하기 위해, pooling을 적용하기 전에 activated filter의 dimension이 확장된다.
>예를 들어, $$d\times d$$ grid에 $$k$$개의 filter로부터 시작해서, $$\frac{d}{2}\times \frac{d}{2}$$ grid와 $$2k$$개의 filter에 도달하려면, 먼저 $$2k$$개의 filter로 stride가 1인 convolution을 계산한 후에 pooling을 수행한다.
>
>Grid size가 줄어들기만 하는건 grid에 들어있던 정보를 보다 저차원의 데이터로 압축하는 것이기 때문에, 이를 병목 현상으로 볼 수 있다. 이 때문에, filter의 개수를 먼저 늘려준다면 정보의 병목 현상을 완화시키는 효과가 있는 것이다.

<br/>
하지만, 이는 네트워크의 전체 계산 비용이 pooling 이전의 확장 단계에서 일어나는 $$2{d^2}k$$ 에 크게 좌우하게 된다는 것을 의미한다.
>논문에서는 $$2{d^2}k^2$$라고 되어있다. 하지만, stride가 1이라면 각 convolution filter마다 $$d^2$$번씩 계산하며, filter의 개수인 $$2k$$개만큼 곱해지면 총 $$2{d^2}k$$개가 맞는 것으로 보인다.

<br/>
만약 convloution과 pooling의 순서를 바꾼다면, 계산 비용이 4분의 1로 감소 된 $$2(\frac{d}{2})^2k$$가 된다. 하지만, 이는 representation의 전반적인 차원이 $$(\frac{d}{2})^2k$$로 낮아져서 표현력이 떨어지게 되고, 이는 곧 representational bottleneck을 야기한다. Fig.8 참조.
>여기도 마찬가지로, 논문에서는 $$2(\frac{d}{2})^2k$$대신 $$2(\frac{d}{2})^2k^2$$로 나타나 있다. 하지만, $$(\frac{d}{2})^2k$$는 제대로 계산됐다.

<br/>
![Fig.8](/blog/images/Inception-v3, Fig.9(removed).png )
>**Fig.8** <br/>Grid size를 줄이는 두 가지 방법. 좌측의 솔루션은 representational bottleneck을 피하라는 1번 원칙을 반하며, 우측의 경우엔 계산 비용이 3배나 비싸다.

<br/>
이 장에서는 representational bottleneck을 피하면서, 계산 비용도 줄일 수 있는 구조를 제안한다. 제안하는 방법은 stride가 2인 block 2개를 병렬로 사용한다. 각 블록은 pooling layer와 conv layer로 이루어져 있으며, pooling은 maximum 혹은 average를 사용한다. 두 block의 filter bank는 Fig.9에 나타난 것처럼 concatenate로 연결 된다.

<br/>
![Fig.9](/blog/images/Inception-v3, Fig.10(removed).png )
>**Fig.9** <br/>좌측은 filter bank를 확장하면서 grid size를 줄이는 inception module이다. 이는 계산 비용도 저렴하면서도 representational bottleneck을 피하므로, 원칙 1을 준수한다. 우측 다이어그램은 좌측과 같은 솔루션을 나타내지만, grid size의 관점에서 나타냈기 때문에 표현 방식이 다른 것 뿐이다.

<br/>
그냥 넘어가기 전에, 제안한 방법이 상대적으로 얼마나 저렴해지는지 알아보자. 우선 Fig.9의 우측 다이어그램을 보면, convolution part와 pooling part에 각각 $$2k$$의 절반인 $$k$$만큼 할당한다는 것을 알 수 있다.
>이 논문에서는 parametric operation만을 비용으로 계산하고 있기 때문에, convolution part만 계산하면 된다.

<br/>
좌측 다이어그램에 따르면, convolution part는 두 개의 branch로 이뤄져있음을 알 수 있다. 여기서 두 branch 간의 filter 수의 비율은 언급되지 않았지만, 절반으로 가정하고 계산해보자. 우선 2-layer인 branch에서는 stride가 1인 것과 2인 conv layer에서 각각 $${d^2}k$$와 $$(\frac{d}{2})^2k$$만큼 비용이 발생하며, 1-layer인 branch에서는 $$(\frac{d}{2})^2k$$만큼 비용이 발생한다. 이를 다 더하면, **총 $$\frac{3}{2} {d^2}k$$만큼 비용**이 발생한다. 기존의 $$2{d^2}k$$에 비하면, **제안하는 방법의 계산 비용이 25% 저렴**하다고 볼 수 있다.

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
3.1절의 아이디어를 기반으로, 기존의 $$7\times 7$$ convolution을 3개의 $$3\times 3$$ convolution으로 factorizing 했다.

<br/>
네트워크의 inception part는 기존의 inception 모듈에 3.1절의 factorizing 기법만 사용한 기존의 inception module이 3개 뒤따른다. 이 때, 각 입력 grid는 $$35\times 35\times 288$$에 해당하며, 마지막에는 5장의 reduction 기법으로 grid가 $$17\times 17\times 768$$로 축소된다.
>두 방법은 각각 Fig.3과 Fig.9의 inception module을 말한다.

<br/>
다음은 3.2절의 asymmetric fatorizing 기법을 이용한 inception module이 5개 뒤따르며, 마찬가지로 5장의 reduction 기법에 의해 grid가 $$8\times 8\times 1280$$으로 축소된다.
>두 방법은 각각 Fig.6과 Fig.9의 inception module을 말한다.

<br/>
Grid size가 $$8\times 8$$로 가장 축소 된 단계에서는 Fig.10의 inception module이 2개 뒤따른다. 각 tile에 대한 출력의 filter bank size는 $$2048$$이 된다.

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
이 장에서는 학습 중 label-dropout의 marginalized effect를 추정하여, classifier layer를 regularize하는 메커니즘을 제안한다.

<br/>
우선 각 학습 데이터 $$x$$에 대해, 각 label $$k$$에 대한 확률을 계산한다.
- $$k \in {1 ... K}$$.

- $$p(k\mid x) = \frac{e^{z_k}}{\sum_{i=1}^K e^{z_i}}$$.

- $$z_i$$는 *logit* 혹은 *unnormalized log-probability*다.
>*logit*은 *weighted sum* 정도로 생각하면 된다.

<br/>
이 학습 데이터의 label $$q(k|x)$$에 대한 ground-truth distribution을 고려하여 정규화하면, $$\sum_{k} q(k|x)=1$$이 된다.

<br/>
편의상, $$p$$와 $$q$$를 데이터 $$x$$와 독립인 것으로 생각하자. 학습 데이터에 대한 loss는 cross entropy $$\ell$$로 정의된다.
- $$\ell = -\sum_{k=1}^K {\log (p(k))}q(k)$$.

<br/>
위의 $$\ell$$을 minimize하는 것은, label의 log-likelihood를 maximize하는 것과 동일하다.
>Label은 ground-truth distribution인 $$q(k)$$에 따라 선택된다.

<br/>
Cross entropy loss는 logit $$z_k$$에 대해 미분 가능하므로, deep network의 gradient 학습에 사용될 수 있다. Gradient는 다음의 단순한 form을 따른다.
- $$\frac{\partial \ell}{\partial z_k} = p(k) - q(k)$$.
- Bounded in $$[-1, 1]$$

<br/>
예제 x와 레이블 y가 주어지면, log-likelihood는 $$q(k) = \delta_{k,y}$$에 대해 maximize 된다.
>여기서 $$\delta_{k,y}$$는, $$k=y$$일 때 1이고, 그렇지 않은 경우에는 0인 [Kronecker delta](https://en.wikipedia.org/wiki/Kronecker_delta)이다. 즉, **one-hot encoded label**.
>
>원문에서는 [Dirac delta](https://en.wikipedia.org/wiki/Dirac_delta_function)라고 되어있으나, 정의에 따르면 [Kronecker delta](https://en.wikipedia.org/wiki/Kronecker_delta)에 해당한다.

<br/>
유한한 $$z_k$$에 대해 maximum을 달성 할 수는 없지만, $$z_y \gg z_k, \forall k \neq y$$인 경우에는 maximum에 근접할 수 있다.
>즉, ground-truth label에 해당하는 logit이, 나머지 모든 logit들보다 훨씬 큰 경우에는 maximum log-likelihood에 근접할 수 있다.

<br/>
그러나, 이 경우에는 두 가지 문제점이 생길 수 있다.

1. 학습 데이터에 over-fitting 될 수도 있다.
>모델이 각 학습 데이터 $$x$$를 ground-truth label에 모든 확률을 할당하도록 학습한다면, 일반화 성능을 보장할 수 없다.

2. Largest logit과 나머지 logit 간의 차이가 매우 커지도록 유도된다.
>이 특성이 $$[-1, 1]$$의 값인 bounded gradient $$\frac{\partial \ell}{\partial z_k}$$와 함께 쓰이게 되면 모델의 적응력을 감소시킨다.

<br/>
위 문제들의 원인을 직관적으로 유추해보면, 모델이 prediction에 대한 confidence를 너무 높게 갖기 때문에 발생하는 것으로 볼 수 있다.

<br/>
이 장에서는 모델의 confidence가 낮아지도록 유도하는 간단한 메커니즘을 제안한다. 만약, 학습 label의 log-likelihood를 maximize하는 것이 목표라면 바람직하지 않은 방법일 수도 있지만, 이는 **모델의 일반화 성능 및 적응력 향상**에 도움되는 기법이다.

<br/>
방법은 매우 간단하다. 학습 데이터 $$x$$에 독립인 **label distribution $$u(k)$$**와 **smoothing parameter인 $$\epsilon$$**을 고려하자.

<br/>
Ground-truth label이 $$y$$일 때, label distribution $$q(k\mid x) = \delta_{k,y}$$ 를 다음과 같이 바꿀 수 있다.

- $$q'(k\mid x) = (1-\epsilon)\delta_{k,y} + \epsilon u(k)$$
>Original ground-truth distribution인 $$q(k\mid x)$$와 fixed distribution인 $$u(k)$$에 $$1 - \epsilon$$과 $$\epsilon$$이 각각 가중치로 곱해진 혼합식이다.
>
>여기서 **$$q(k\mid x) = \delta_{k,y}$$**는 흔히들 알고 있는 **one-hot encoded label**이며, **$$q'(k\mid x)$$**는 **label smoothing 기법이 적용 된 새로운 label**이다.
>
>결국, 기존의 label인 $$q(k\mid x)$$가 $$y=k$$일 경우에 1, 나머지는 0으로 채워지는 one-hot encoded 형태면, **새로운 label인 $$q'(k\mid x)$$는 $$y=k$$일 경우에 $$1-\epsilon$$, 나머지는 $$\epsilon u(k)$$ 값으로 채워지는 형태**이다.

<br/>
이는 다음의 방법으로 얻어지는 label $$k$$의 distribution으로 볼 수 있다.

1. Ground-truth label $$k = y$$를 설정한다.

2. Probability $$\epsilon$$에 대해, $$k$$를 $$u(k)$$에서 추출 된 샘플로 대체한다.

<br/>
저자들은 $$u(k)$$에 prior distribution을 사용하라고 언급했다. 실험에서는 uniform distribution $$u(k) = \frac{1}{K}$$를 사용하고 있으며, 이는 다음의 식과 동일하다.

- $$q'(k) = (1-\epsilon)\delta_{k, y} + \frac{\epsilon}{K}$$.
>여기서 $$K$$는 class 개수다. ImageNet classification의 경우에는 $$K=1000$$.

<br/>
이와 같은 ground-truth label distribution의 변형을, **label-smoothing regularization** 또는 **LSR**이라고 칭한다. 이 **LSR**은 **largest logit이 나머지 logit들과의 차이가 매우 커지지 않게**하려는 목적을 달성할 수 있게 해준다.
>즉, probability를 극단적인 형태로 나눠갖는 현상을 완화시킨다.

<br/>
실제로 LSR이 적용되면, $$q'(k)$$는 큰 cross-entropy 값을 가지게 될 것이다.
>$$q'(k)$$가 $$\delta_{k,y}$$ 와는 달리, positive lower bound를 가지기 때문이다.

<br/>
또한, LSR은 cross-entropy에 대해서도 수식화 될 수 있다.

- $$H(q', p) = -\sum_{k=1}^K \log {p(k)q'(k)} = (1-\epsilon)H(q,p) + \epsilon H(u,p)$$
>즉, LSR은 기존의 single cross-entropy loss인 $$H(q,p)$$를, 두 loss $$H(q,p)$$와 $$H(u,p)$$로 대체하는 것과 동일하다.

<br/>
두 번째 loss인 $$H(u,p)$$는, prior인 **$$u$$로부터 얻어지는** predicted label distribution **$$p$$의 deviation**에 대해 계산되며, 첫 번째 loss에 비해 상대적으로 $$\frac{\epsilon}{1-\epsilon}$$만큼 가중된다.
>이 때, $$H(u, p) = D_{KL}(u\parallel p) + H(u)$$와 $$H(u)$$가 고정되어 있기 때문에, deviation이 KL divergence에 따라 동일하게 측정될 수 있음에 유의하자.

<br/>
$$u$$가 uniform distribution일 때의 $$H(u,p)$$는, predicted distribution인 $$p$$가 얼마나 uniform하지 않은지에 대한 척도이다.
>이는 negative entropy인 $$-H(p)$$로도 측정될 수 있지만, 동일한 값은 아니다.

<br/>
실험은 $$K = 1000$$ class인 ImageNet에 대해 진행했으며, 이 때는 $$u(k) = \frac{1}{1000}$$과 $$\epsilon = 0.1$$을 사용했다. ILSVRC 2012 dataset에 대한 실험 결과, top-1 error와 top-5 error에 대한 성능이 약 0.2% 향상하는 일관적인 결과를 얻었다. Table.3 참조.

---
## 8. Training Methodology
[TensorFlow 분산 학습 시스템](https://arxiv.org/pdf/1603.04467.pdf)에서 50개의 모델 복제본(replica)이 각각 NVidia Kepler GPU 상에서 stochastic gradient로 학습됐다. 학습은 batch size를 32로 하여, 100 epoch만큼 학습했다.

<br/>
초반 실험에서는 decay가 0.9인 [momentum](https://www.cs.toronto.edu/~fritz/absps/momentum.pdf)을 사용했었으며, best model은 decay가 0.9인 [RMSProp](https://arxiv.org/pdf/1609.04747.pdf)과, $$\epsilon = 1.0$$을 사용할 때 얻었다.
>여기서 $$\epsilon$$은 optimizer인 [RMSProp](https://arxiv.org/pdf/1609.04747.pdf)의 hyperparameter를 말한다.

<br/>
Learning rate는 0.045에서 시작하여, 두 번의 epoch마다 0.94를 곱했다. 또한, threshold가 2.0인 [gradient clipping](http://proceedings.mlr.press/v28/pascanu13.pdf)이 안정적인 학습에 도움된다는 것을 발견했다.

<br/>
모델의 평가는, 시간이 지남에 따라 계산 된 parameter의 running average로 이뤄졌다.

---
## 9. Performance on Lower Resolution Input
비전 네트워크의 대표적인 use case는, detection의 post-classification을 위한 것이다.
>Detected object에 대한 classification을 말한다.

<br/>
[Multibox](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Erhan_Scalable_Object_Detection_2014_CVPR_paper.pdf)에도 post-classification 작업이 포함됐다. 여기서는 single object를 포함하면서 상대적으로 작은 image patch에 대한 분석이 포함된다.
>Image patch에 대한 분석은 patch의 중앙 부분이 어떤 object에 해당하는지 여부를 결정하고, object가 존재하는 경우엔 class를 결정하는 작업이다.

<br/>
이러한 use case에서의 문제는 object가 상대적으로 작으면서, low-resolution인 점이다. 따라서, lower resolution input에 대한 적절한 처리 방법이 필요하게 된다.

<br/>
일반적으로, higher resolution의 receptive field를 사용하는 모델이, recognition 성능이 크게 향상되는 경향이 있다고 알려져 있다.

<br/>
일단, **첫 번째 layer의 receptive field의 resolution이 증가했을 때의 효과**와, **model이 커짐에 따른 capacitance 및 computation에 대한 효과**를 구별하는 것이 중요하다.
>Receptive field의 resolution이 커진다는 것은, convolution filter와 input 간의 weighted sum의 계산에 사용되는 pixel의 수가 많아진다는 것이다. Resolution이 커질수록 더 넓은 범위의 인근 pixel들을 고려하여 패턴을 학습할 수 있게 된다.
>
>모델의 capacitance가 크다는건 많은 parameter를 가지는 것이며, 그만큼 더 복잡한 관계에 대한 패턴을 학습할 수 있는 여지가 생긴다. 예를 들어, 3-layer를 가지는 CNN으로 MNIST dataset에 대한 학습을 진행하면 우수한 성능을 얻을 수 있지만, ImageNet dataset에 대한 학습을 진행하면 민망한 성능을 얻게 되는 것과 유사한 이치다.

<br/>
만약, 모델을 수정하지 않고 input resolution만 변경한다면, 계산 비용이 훨씬 저렴한 모델로, 보다 어려운 작업에 대한 학습을 하게 된다. 이 경우, 계산량이 줄어드는만큼 솔루션의 견고함도 떨어지게 된다.

<br/>
정확한 평가를 위해서는, 모델의 세부 사항을 "hallucinate" 할 수 있도록 모호한 징후들을 분석해야한다.
>이 또한 계산 비용이 많이 드는 작업이다.

<br/>
아직도 의문인건, 계산량이 일정하게 유지되면서 input resolution이 높아지는 것이 얼마나 도움되는가 하는 점이다.
>Receptive field의 resolution를 말하는 것으로 보인다.

<br/>
일정한 계산량을 유지하는 간단한 방법은, 입력이 lower resolution인 경우엔 처음 두 layer에서 stride를 줄이거나, 네트워크의 첫 번째 pooling layer를 제거하면 된다. 이를 위해 다음 세 가지 실험을 수행했다.

1. stride가 2인 299x299 receptive field를 사용하고, 첫 번째 layer 다음에 max pooling을 사용.

2. stride가 1인 151x151 receptive field를 사용하고, 첫 번째 layer 다음에 max pooling을 사용.

3. stride가 1인 79x79 receptive field를 사용하고, 첫 번째 layer 다음에 pooling layer가 없음.

>입력의 크기가 각각 299x299, 151x151, 79x79고, 첫 번째 conv layer의 stride가 각각 2, 1, 1인 것을 말한다.

<br/>
위 실험에 해당하는 3개의 네트워크는 거의 동일한 계산 비용을 갖는다.
>실제로는 3번 네트워크가 약간 저렴하지만, pooling layer의 계산 비용은 총 비용의 1% 이내 수준이기 때문에 무시한다.

<br/>
각각의 네트워크는 수렴 될 때까지 학습했으며, 성능은 ImageNet ILSVRC 2012 classification benchmark의 validation set에 대해 측정됐다.

<br/>
결과는 Table.2에서 보인다. lower-resolution 네트워크가 학습하는 데 오래 걸리긴 하지만, 성능은 higher resolution 네트워크에 근접한다.

<br/>
![Table.2](/blog/images/Inception-v3, Table.2(removed).png )
>**Table.2** <br/>Receptive field size의 변화에 따른 성능 비교이다. 모든 케이스의 computational cost는 동일하다.

<br/>
만약, input resolution에 따라 단순하게 네트워크 크기를 줄이게 된다면 성능이 저하된다. 하지만, 이는 어려운 작업에 대해서 16배나 저렴한 모델이기 때문에, 이를 비교하는 것은 불공평하다.

<br/>
또한 Table.2의 결과들은, [R-CNN](https://arxiv.org/pdf/1311.2524.pdf)의 smaller object에 대해, high-cost low resoluton 네트워크의 사용 가능성을 암시한다.

---
## 10. Experimental Results and Comparisons
Table.3은 6장에서 제안한 Inception-v2에 대한 실험 결과를 보여준다.

<br/>
![Table.3](/blog/images/Inception-v3, Table.3(removed).png )
>**Table.3** <br/>다양한 기법들에 대한 누적 효과를 비교하는 single-crop 성능이다.
>
>BN-Inception은 [BN](https://arxiv.org/pdf/1502.03167.pdf)에서 측정된 성능이다.
>
>Inception-v2의 적용 기법들은 누적되며, **Inception-v3**는 **Inception-v2에 모든 기법들을 적용한 경우**를 말한다.

<br/>
각 Inception-v2 행은, 각 기법에 이전 기법들을 누적 적용한 학습 결과를 표시한다.
>예를 들어, **Inception-v2 Label Smoothing** 행은 **Inception-v2**에 **RMSProp**과 **Label Smoothing**을 모두 적용한 결과이다.

<br/>
- Label Smoothing은 7장에서 설명한 방법이다.

- Factorized 7x7은 첫 번째 7x7 conv layer를 3x3 conv layer의 시퀀스로 factorizing하는 것을 포함한다.

- BN-auxiliary는 보조 분류기에서 conv layer 뿐만 아니라, FC layer에도 [BN](https://arxiv.org/pdf/1502.03167.pdf)이 적용 된 버전이다.

<br/>
성능은 multi-crop과 ensemble에 대해 평가했다. Table.4 및 Table.5 참조.

<br/>
![Table.4](/blog/images/Inception-v3, Table.4(removed).png )
>**Table.4** <br/>Sinle-mode, multi-crop 실험 결과이다. ILSVRC 2012 classification 성능이 가장 잘 나온 Inception-v3의 성능만 비교한다.

<br/>
![Table.5](/blog/images/Inception-v3, Table.5(removed).png )
>**Table.5** <br/>Ensemble evaluation results comparing multi-model, multi-crop reported results.

<br/>
모든 성능은 ILSVRC-2012 validation set의 non-blacklisted example 48238개에 대해 평가됐다.
>non-blacklisted example은 [이 논문](https://arxiv.org/pdf/1409.0575.pdf)에서 제안했다고 한다.

<br/>
50000개의 example에 대해 평가했으며, 결과는 top-5 error에서 약 0.1%, top-1 error에서 약 0.2% 떨어졌다.
>Test set 50000개에 대한 평가이고, validation set에 대한 성능과 비교한 것으로 보인다.

<br/>
대충 다 나왔으니, inception-v3를 keras로 구현해보자. 논문에서는 모델에 대한 자세한 설명이나, 도식이 따로 제공되지 않았다. [Tensorflow github](https://github.com/tensorflow/models/tree/master/research/inception)에 제공되는 모델의 도식은 아래와 같다.

<br/>
![Extra.2](/blog/images/Inception-v3, Extra.2(removed).png )
>**Inception-v3 구조**

<br/>
위 구조는 논문의 설명과 다른 부분이 있다.

- 네트워크 초반의 6번 째 layer
>Table.3에 따르면, conv layer가 있을 자리다.

- Inception module 안에서의 pooling method
>논문에서는 따로 언급이 없었으므로, GoogLeNet에서와 같은 MaxPooling을 기본 pooling method로 고려한다.

- 두 번째 inception block의 개수
>Table.1에 따르면 5개가 와야하는데 4개만 있다.

- Dimension reduction module의 형태
>논문의 Fig.9와 다르며, 둘 끼리도 형태가 상이하다.

<br/>
이 외에도 [Tensorflow github](https://github.com/tensorflow/models/tree/master/research/inception)이나 [Keras github](https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py)에서 제공되는 코드는, 위 구조에 따라 작성됐다.

<br/>
아래의 keras 구현 코드는 논문의 구조에 맞춰서 구현했으며, 논문에 제공되지 않은 각종 hyperparameter는 위의 두 페이지에서 적당히 참조했다.
>Inception module 내부의 각 conv filter 개수, dropout rate, loss weight 등을 포함한다.

<br/>
우선 Fig.3, Fig.6, Fig.10의 각 inception module과 Fig.9의 reduction module을 구현하면 다음과 같다.
``` python
def conv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu'):
    x = Conv2D(filters, (kernel_size[0], kernel_size[1]), padding=padding, strides=strides)(x)    
    x = BatchNormalization()(x)
    
    if activation:
        x = Activation(activation)(x)
    
    return x

def inception_f3(input_tensor, filter_channels, name=None):
    filter_b1, filter_b2, filter_b3, filter_b4 = filter_channels
    
    branch_1 = conv2d_bn(input_tensor, filter_b1[0], (1, 1))
    branch_1 = conv2d_bn(branch_1, filter_b1[1], (3, 3))
    branch_1 = conv2d_bn(branch_1, filter_b1[2], (3, 3))
    
    branch_2 = conv2d_bn(input_tensor, filter_b2[0], (1, 1))
    branch_2 = conv2d_bn(branch_2, filter_b2[1], (3, 3))
    
    branch_3 = MaxPooling2D((3, 3), strides=1, padding='same')(input_tensor)
    branch_3 = conv2d_bn(branch_3, filter_b3, (1, 1))
    
    branch_4 = conv2d_bn(input_tensor, filter_b4, (1, 1))
    
    filter_concat = Concatenate(name=name)([branch_1, branch_2, branch_3, branch_4]) if not name==None else Concatenate()([branch_1, branch_2, branch_3, branch_4])
    
    return filter_concat

def inception_f6(input_tensor, filter_channels, n=7, name=None):
    filter_b1, filter_b2, filter_b3, filter_b4 = filter_channels
    
    branch_1 = conv2d_bn(input_tensor, filter_b1[0], (1, 1))
    branch_1 = conv2d_bn(branch_1, filter_b1[1], (1, n))
    branch_1 = conv2d_bn(branch_1, filter_b1[2], (n, 1))
    branch_1 = conv2d_bn(branch_1, filter_b1[3], (1, n))
    branch_1 = conv2d_bn(branch_1, filter_b1[4], (n, 1))
    
    branch_2 = conv2d_bn(input_tensor, filter_b2[0], (1, 1))
    branch_2 = conv2d_bn(branch_2, filter_b2[1], (1, n))
    branch_2 = conv2d_bn(branch_2, filter_b2[2], (n, 1))
    
    branch_3 = MaxPooling2D((3, 3), strides=1, padding='same')(input_tensor)
    branch_3 = conv2d_bn(branch_3, filter_b3, (1, 1))
    
    branch_4 = conv2d_bn(input_tensor, filter_b4, (1, 1))
    
    filter_concat = Concatenate(name=name)([branch_1, branch_2, branch_3, branch_4]) if not name==None else Concatenate()([branch_1, branch_2, branch_3, branch_4])
    
    return filter_concat

def inception_f10(input_tensor, filter_channels, name=None):
    filter_b1, filter_b2, filter_b3, filter_b4 = filter_channels
    
    branch_1 = conv2d_bn(input_tensor, filter_b1[0], (1, 1))
    branch_1 = conv2d_bn(branch_1, filter_b1[1], (3, 3))
    branch_1a = conv2d_bn(branch_1, filter_b1[2][0], (1, 3))
    branch_1b = conv2d_bn(branch_1, filter_b1[2][1], (3, 1))
    branch_1 = Concatenate()([branch_1a, branch_1b])
    
    branch_2 = conv2d_bn(input_tensor, filter_b2[0], (1, 1))
    branch_2a = conv2d_bn(branch_2, filter_b2[1][0], (1, 3))
    branch_2b = conv2d_bn(branch_2, filter_b2[1][1], (3, 1))
    branch_2 = Concatenate()([branch_2a, branch_2b])
    
    branch_3 = MaxPooling2D((3, 3), strides=1, padding='same')(input_tensor)
    branch_3 = conv2d_bn(branch_3, filter_b3, (1, 1))
    
    branch_4 = conv2d_bn(input_tensor, filter_b4, (1, 1))
    
    filter_concat = Concatenate(name=name)([branch_1, branch_2, branch_3, branch_4]) if not name==None else Concatenate()([branch_1, branch_2, branch_3, branch_4])
    
    return filter_concat
    
def inception_dim_reduction(input_tensor, filter_channels, name=None):
    filter_b1, filter_b2 = filter_channels
    
    branch_1 = conv2d_bn(input_tensor, filter_b1[0], (1, 1))
    branch_1 = conv2d_bn(branch_1, filter_b1[1], (3, 3))
    branch_1 = conv2d_bn(branch_1, filter_b1[2], (3, 3), strides=2)
    
    branch_2 = conv2d_bn(input_tensor, filter_b2[0], (1, 1))
    branch_2 = conv2d_bn(branch_2, filter_b2[1], (3, 3), strides=2)
    
    branch_3 = MaxPooling2D((3, 3), strides=2, padding='same')(input_tensor)
    
    filter_concat = Concatenate(name=name)([branch_1, branch_2, branch_3]) if not name==None else Concatenate()([branch_1, branch_2, branch_3])
    
    return filter_concat
```

<br/>
다음은 위 module들을 이용해서 inception-v3 구조를 만든다.
``` python
def Inception_v3(model_input):
    x = conv2d_bn(model_input, 32, (3, 3), padding='valid', strides=2) # (299, 299, 3) -> (149, 149, 32)
    x = conv2d_bn(x, 32, (3, 3), padding='valid') # (147, 147, 32) -> (147, 147, 32)
    x = conv2d_bn(x, 64, (3, 3), padding='same') # (147, 147, 32) -> (147, 147, 64)
    
    x = MaxPooling2D((3, 3), strides=2, padding='valid')(x) # (147, 147, 64) -> (73, 73, 64)
    
    x = conv2d_bn(x, 80, (3, 3), padding='valid') # (73, 73, 64) -> (71, 71, 80)
    x = conv2d_bn(x, 192, (3, 3), padding='valid', strides=2) # (71, 71, 80) -> (35, 35, 192)
    x = conv2d_bn(x, 288, (3, 3), padding='same') # (35, 35, 192) -> (35, 35, 288)
    
    x = inception_f3(x, [[64, 96, 96], [48, 64], 64 , 64]) # (35, 35, 288)
    x = inception_f3(x, [[64, 96, 96], [48, 64], 64 , 64]) # (35, 35, 288)
    x = inception_f3(x, [[64, 96, 96], [48, 64], 64 , 64], name='block_inception_f3') # (35, 35, 288)
    
    x = inception_dim_reduction(x, [[64, 96, 96], [256, 384]], name='block_reduction_1') # (35, 35, 288) -> (17, 17, 768)
    
    x = inception_f6(x, [[128, 128, 128, 128, 192], [128, 128, 192], 192, 192]) # (17, 17, 768)
    x = inception_f6(x, [[160, 160, 160, 160, 192], [160, 160, 192], 192, 192]) # (17, 17, 768)
    x = inception_f6(x, [[160, 160, 160, 160, 192], [160, 160, 192], 192, 192]) # (17, 17, 768)
    x = inception_f6(x, [[192, 192, 192, 192, 192], [192, 192, 192], 192, 192]) # (17, 17, 768)
    x_a = inception_f6(x, [[192, 192, 192, 192, 192], [192, 192, 192], 192, 192], name='block_inception_f6') # (17, 17, 768)
    
    x = inception_dim_reduction(x_a, [[128, 192, 192], [192, 320]], name='block_reduction_2') # (17, 17, 768) -> (8, 8, 1280)
    
    x = inception_f10(x, [[448, 384, [384, 384]], [384, [384, 384]], 192, 320]) # (8, 8, 1280) -> (8, 8, 2048)
    x = inception_f10(x, [[448, 384, [384, 384]], [384, [384, 384]], 192, 320], name='block_inception_f10') # (8, 8, 2048)
    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.8)(x)
    
    x = Dense(classes, activation=None)(x)
    
    model_output = Dense(classes, activation='softmax', name='main_classifier')(x) # 'softmax'
    
    # Auxiliary Classifier
    auxiliary = AveragePooling2D((5, 5), strides=3, padding='valid')(x_a) # (17, 17, 768) -> (5, 5, 768)
    auxiliary = conv2d_bn(auxiliary, 128, (1, 1)) # (5, 5, 768) -> (5, 5, 128)
    
    auxiliary = conv2d_bn(auxiliary, 1024, K.int_shape(auxiliary)[1:3], padding='valid') # (5, 5, 768) -> (1, 1, 1024)
    auxiliary = Flatten()(auxiliary) # (1, 1, 1024)
    auxiliary_output = Dense(classes, activation='softmax', name='auxiliary_classifier')(auxiliary)
    
    model = Model(model_input, [model_output, auxiliary_output])
    
    return model
```

<br/>
다음은 7장의 label smoothing과 8장의 learning rate 정책을 적용한다.
``` python
classes = 10
smoothing_param = 0.1

def smoothed_categorical_crossentropy(y_true, y_pred): 
    if smoothing_param > 0:
        smooth_positives = 1.0 - smoothing_param 
        smooth_negatives = smoothing_param / classes 
        y_true = y_true * smooth_positives + smooth_negatives 

    return K.categorical_crossentropy(y_true, y_pred)

class LearningRateSchedule(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%2 == 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr*0.94)
```

<br/>
위 코드들을 통합하여 학습하는 코드는 다음과 같다.
``` python
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Dense, Flatten, BatchNormalization, AveragePooling2D
from keras.layers import Concatenate
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from keras.optimizers import RMSprop
from keras.datasets import cifar10

import keras.backend as K
import numpy as np

classes = 10
smoothing_param = 0.1

def Upscaling_Data(data_list, reshape_dim):
    ...

def conv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu'):
    ...

def inception_f3(input_tensor, filter_channels, name=None):
    ...

def inception_f6(input_tensor, filter_channels, n=7, name=None):
    ...

def inception_f10(input_tensor, filter_channels, name=None):
    ...

def inception_dim_reduction(input_tensor, filter_channels, name=None):
    ...

def Inception_v3(model_input):
    ...

def smoothed_categorical_crossentropy(y_true, y_pred): 
    ...

class LearningRateSchedule(Callback):
    ...

input_shape = (299, 299, 3)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = Upscaling_Data(x_train, input_shape)
x_test = Upscaling_Data(x_test, input_shape)

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

y_train = to_categorical(y_train, num_classes=classes)
y_test = to_categorical(y_test, num_classes=classes)

model_input = Input( shape=input_shape )

model = Inception_v3(model_input)

optimizer = RMSprop(lr=0.045, epsilon=1.0, decay=0.9)
filepath = 'weights/' + model.name + '.h5'
callbacks_list = [ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_weights_only=True, save_best_only=True, mode='auto', period=1),
                  CSVLogger(model.name + '.log'),
                  LearningRateSchedule()]

model.compile(optimizer, 
        	loss={'main_classifier' : smoothed_categorical_crossentropy,
               'auxiliary_classifier' : smoothed_categorical_crossentropy},
                loss_weights={'main_classifier' : 1.0, 
                              'auxiliary_classifier' : 0.4},
                metrics=['acc'])

history = model.fit(x_train, [y_train, y_train], batch_size=32, epochs=100, validation_split=0.2, callbacks=callbacks_list)
```
>물론, 이번에도 CIFAR-10 데이터를 $$299\times 299\times 3$$로 upscaling하여 학습하는 코드다.


---
## 11. Conclusions
이 논문에서는 CNN을 확장하기 위한 몇 가지 디자인 원칙을 제공하고, inception 구조에서 이에 대한 연구를 진행했다.

<br/>
제공한 원칙들을 따르면, 단순하고 일체화 된 구조에 비해, 적은 계산 비용을 갖는 고성능 비전 네트워크를 구성할 수 있게 해준다.

<br/>
가장 성능이 좋았던 Inception-v3의 경우, ILSVR 2012 classification에 대한 single-crop 성능이 top-1 error와 top-5 error에서 각각 21.2%, 5.6%에 도달했다.
>BN-Inception보다 계산 비용이 2.5배 정도로 적게 증가했다.

<br/>
제안 된 방법은 denser network를 기반의 모델 중 best인 것보다 훨씬 적은 계산을 사용한다.
>Table.4와 Table.5의 [PReLU](https://arxiv.org/pdf/1502.01852.pdf) 결과를 말한다. 
>
>계산 비용이 6배 저렴하고, 최소 5배는 적은 parameter를 사용하면서도 top-5 error와 top-1 error가 각각 상대적으로 25%와 14% 만큼 낮아졌다.

<br/>
Inception-v3 모델 4개를 ensemble한 multi-crop 성능은 top-5 error가 3.5%이다.
>이는 당시의 최고 성능을 25% 이상 줄인 것으로, ILSVRC 2014 GoogLeNet ensemble error에 비해 거의 절반이다.

<br/>
또한, 79x79의 낮은 receptive field resolution에서도 높은 성능을 얻을 수 있음을 입증했다.
>이는 상대적으로 작은 object를 탐지하는 시스템에 도움될 수 있다.

<br/>
이 논문에서는 네트워크 내부에서의 **factorizing convolution 기법**과 **적극적인 dimension reduction**으로, 어떻게 **높은 성능을 유지**하면서도, 비교적 **낮은 계산 비용**이 드는 네트워크를 만들 수 있는지에 대해 알아봤다.

<br/>
적은 수의 parameter와 BN이 사용 된 보조 분류기, label-smoothing 기법이 함께 사용되면, 크지 않은 규모의 학습 데이터 상에서도, 고성능의 네트워크를 학습 할 수 있다.

---

<br/>
<br/>
{% include disqus.html %}
