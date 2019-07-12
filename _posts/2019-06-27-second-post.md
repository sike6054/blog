---
title: "GoogLeNet 논문 정리"
date: 2019-06-18 17:22:11 -0400
tags: AI ComputerVision Paper GoogLeNet
categories: 
  - Paper
toc: true
comments: true
---

## Paper Information

SZEGEDY, Christian, et al. **"Going deeper with convolutions"**. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2015. p. 1-9.
> a.k.a. [GoogLeNet paper](https://arxiv.org/pdf/1409.4842.pdf)


---
## Abstract

이 논문에서는 **"Inception"**이란 이름의 deep convolutional neural network architecture를 제안한다. 저자들은 제안한 방법을 통해 ILSVRC 2014의 classification/detection 분야에서 state-of-the-art 성능을 달성했다.
>ILSVRC 2014에 제출했던 **GoogLeNet**은 이 Inception 구조를 모듈 단위로 활용해서 구현한 네트워크다.

<br/>
Inception architecture의 대표적인 특징은, 네트워크 내부의 computing resource을 효율적으로 활용한다는 점이다. 이는 신중하게 설계된 디자인을 통해 이뤄졌으며, 연산 비용을 일정하게 유지하면서 네트워크의 depth/width를 증가시킨다.
<br/>
이 architecture의 성능 최적화와 관련된 사항은 Hebbian principle과 multi-scale processing에 대한 직관에 기반하여 결정했다.
>여기서 말하는 Hebbian principle은 **"neurons that fire together, wire together"**을 말한다. 즉, 동시에 활성화 된 노드들 간에는 연관성이 있다는 의미 정도로 생각할 수 있다.

<br/>
ILSVRC 2014에 제출 된 모델은 22-layer로 이루어진 **GoogLeNet**으로, classification/detection 분야에서 그 성능을 평가했다.

---
## 1. Introduction
최근, deep learning 및 convolutional network의 발전으로 인해, classification/detection 성능이 극적으로 향상됐다. 한 가지 고무적인 소식은, 이 결과들이 하드웨어의 성능 향상이나 dataset의 규모 증가, 모델의 크기 증가로부터 얻어진 것이 아니라, 새로운 아이디어나 새로운 알고리즘, 개선된 네트워크 구조로부터 얻어진 결과들이 대부분이라는 것이다.
>ILSVRC 2014의 top entry들은 detection에서 classification dataset 이외에 새로운 데이터를 사용하지 않았다.

<br/>
ILSVRC 2014에 제출한 GoogLeNet은, 2년 전에 우승했던 [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)보다 12배나 적은 parameter를 사용함에도, 훨씬 좋은 성능을 보였다. object detection의 경우, 단순히 크고 깊게 만든 네트워크 보다는, [R-CNN](https://arxiv.org/pdf/1311.2524.pdf)과 같이 classical computer vision 알고리즘과 deep architecture의 시너지 효과로부터 큰 성능 향상이 이루어졌다.
>R-CNN은 Regions with Convolutional Neural Networks를 줄인 것이다.

<br/>
또 다른 중요한 요소는, 모바일이나 임베디드 상에서 지속적으로 동작하기 위해선 알고리즘의 효율성이 중요하다는 점이다. 논문에서는 정확도의 수치보다 효율성을 더 고려하여 deep architecture를 설계했으며, 대부분의 실험에서 사용된 모델은 inference time의 연산량이 1.5 billion multiply-add 정도를 유지하도록 설계됐다.
>모바일이나 임베디드의 경우, 특히 전력 및 메모리 사용 면의 효율성이 중요하다고 언급했다. 또한, 제안하는 구조가 성능 향상에 목적을 두지 않은만큼 학문적인 흥미는 떨어질 수 있으나, large dataset을 갖는 real world에서도 합리적인 비용으로 이 구조를 사용할 수 있을 것이라는 점에 중점을 뒀다.

<br/>
Inception은 [NIN](https://arxiv.org/pdf/1312.4400.pdf) 논문과 함께, "we need to go deeper"라는 유명한 인터넷 밈에서 유래한 이름이다. 이 논문에서는 "deep"이라는 단어를 두 가지 의미로 사용한다. 
1. **Inception module**이라는 새로운 구조의 도입
2. Network의 depth가 늘어남
    
일반적으로 Inception 모델은 [NIN](https://arxiv.org/pdf/1312.4400.pdf)의 논리로부터 영감을 얻었으며, [Arora](https://arxiv.org/pdf/1310.6343.pdf)의 이론적 연구가 지침이 된 것으로 볼 수 있다. Inception 구조의 이점은 ILSVRC 2014 classification 및 detection 분야에서 실험적으로 검증됐으며, 당시의 state-of-the-art보다 훨씬 뛰어난 성능을 보였다.
>NIN은 Network-in-Network를 줄인 것이다.

---
## 2. Related Work
[LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)을 필두로, CNN는 일반적인 standard structure를 가지게 됐다. 이 기본적인 디자인의 변형은 image classification에서 널리 사용됐으며, MNIST와 CIFAR, ImageNet classification challenge에서 state-of-the-art 성능을 얻었다.
>stacked conv layer 뒤에 contrast normalization이나 max-pooling layer가 선택적으로 뒤따르며, 그 뒤에는 하나 이상의 FC layer가 이어지는 형태를 말한다.

<br/>
ImageNet과 같은 큰 dataset의 경우, layer의 수와 크기를 늘리면서 [dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)을 사용하여 overfitting을 피하는 것이 최근의 추세였다. 또한, max-pooling layer가 정확한 공간 정보를 잃어 버릴 수 있다는 우려가 있었음에도, [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)은 localization과 object detection 및 human pose estimation 분야에서 성공적인 성과를 거뒀다.

<br/>
영장류의 시각 피질에 대한 신경 과학 모델에서 영감을 얻은 [Serre의 연구](http://cbcl.mit.edu/publications/ps/serre-wolf-poggio-PAMI-07.pdf)에서는 multiple scale을 다루기 위해 크기가 다른 fixed Gabor filter들을 사용했다. 이 논문에서는 비슷한 전략을 사용하지만, Inception architecture의 모든 filter가 학습한다는 점에서 차이가 있다. 또한, GoogLeNet의 경우에는 Inception layer가 여러 번 반복되어 22-layer deep model로 구현된다.

<br/>
[Network-in-Network](https://arxiv.org/pdf/1312.4400.pdf)는 네트워크의 표현력(representational power of neural networks)을 증가시키기 위한 접근법이다. 이 모델은 1x1 conv layer가 네트워크에 추가되어 depth를 증가시키는데, 이 접근 방식을 논문에서 제안하는 구조에 많이 사용한다. 하지만, 제안하는 방법에서 1x1 convolution이 가지는 목적은 두 가지다.

1. computational bottleneck을 제거하기 위한 dimension reduction module로써 사용

2. depth를 늘리는 것과 더불어, 큰 성능의 저하없이 width의 증가를 위함

>이를 이용하지 않는다면 네트워크의 크기가 제한될 수 있기 때문에, 저자들은 1번 목적을 가장 중요하게 여긴다.

<br/>
당시 object detection 분야의 state-of-the-art 성능을 가진 모델는 [R-CNN](https://arxiv.org/pdf/1311.2524.pdf)이었다. 이 모델는 전체적인 detection problem을 두 개의 subproblem으로 분해한다.

1. color와 texture 같은 low-level feature를 활용하여, category에 구애받지 않는 방식으로 object location proposal을 생성
    
2. CNN classifier를 사용하여, 해당 location들의 object category를 식별

<br/>
이러한 two stage approach는 low-level feature를 활용한 bounding box segmentation의 정확성 뿐만 아니라, state-of-the-art CNN의 강력한 classification power를 활용할 수 있다. 제안하는 방법을 이용한 detection submission에도 이와 유사한 pipeline을 적용했지만, 두 단계 모두에 대한 개선 사항이 있었다. 
    
1. object bounding box의 높은 recall을 위한 [multi-box prediction](https://arxiv.org/pdf/1312.2249.pdf)
    
2. bounding box proposal을 보다 잘 분류하기 위한 ensemble approach

>잠깐 검색해봐도 segmentation은 semantic/instance segmentation이나 나오지, bounding box segmentation은 보이지 않았다. R-CNN에서는 bounding box의 후보군을 추출하기 위한 1st-stage에서 [selective search](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf) 알고리즘을 사용하고 있는데, 이 알고리즘이 exhaustive search와 segmentation의 장점을 결합한 것이라 한다. 이걸 뜻하는 segmentation인지, 단순히 '분할'이라는 의미로 사용된건지는 잘 모르겠으나, [selective search](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)가 가진 성능도 취한다는 의미 정도로만 이해해도 문제 없어 보인다.
>
>Recall은 실제로 true인 데이터를 true라고 예측한 결과의 비율을 말한다. [multi-box prediction](https://arxiv.org/pdf/1312.2249.pdf) 논문을 참조하면, 당시 기준으로 localization 분야에서 우승한 모델에서 사용한 기존의 방법으로는 동일한 종류의 object가 여러 개의 instance를 가질 때, output을 복제하는 등의 단순한 방법을 통해야만 이를 처리할 수 있다고 한다. 이를 해결하기 위해 제안한 방법이 [multi-box prediction](https://arxiv.org/pdf/1312.2249.pdf)이며, 동일한 object에 대한 multiple instance를 detecting하는 것은 당연히 recall이 높아지는 결과가 될 것이다.

---
## 3. Motivation and High Level Considerations
Deep network의 성능을 향상시키는 가장 직접적인 방법은 depth나 width를 늘려서 네트워크의 크기를 증가시키는 것이며, 이는 특히 labeled training data가 많은 경우에 쉽고 안전하게 고성능의 모델을 학습하는 방법이다.
>depth는 network level 수, width는 각 level의 node 개수를 말한다.

<br/>
그러나, 이 간단한 solution에는 두 가지의 주요한 문제점이 있다.

1. 일반적으로 bigger size는 더 많은 수의 parameter를 가진다는 것을 의미하므로, 네트워크가 overfitting 되기 쉬운 경향이 있으며, 특히 labeled training set이 제한적일 경우에는 더욱 심해진다.

2. 네트워크의 크기가 증가함에 따라, computaional resource이 극도로 증가한다. 

>1번 문제는 ImageNet과 같이 다양한 클래스로 세분화 된 datasets를 다루는 경우에 생기는 주요 bottleneck이다. Fig.1과 같이 생김새가 흡사한 경우, 사진만으로는 사람이 직접 visual category를 분류하는 것은 전문가일지라도 상당히 어려워 보인다.
>
>2번 문제는 conv layer의 filter의 수가 증가하면 계산량이 제곱으로 증가하는 것을 예로 들 수 있다. 이 때, 추가된 filter에 해당하는 대부분의 weight가 0에 가까워지는 등, 비효율적으로 사용되는 경우에는 computaional resource의 낭비로 이어지게 된다. 따라서, 성능을 높이는 것이 주 목적인 경우에도 크기를 무작정 증가시키기 보다는 computing resource를 효율적으로 분배하는 것이 바람직하다.

<br/>
![Fig.1](/blog/images/GoogLeNet, Fig.1(removed).png )
>**Fig.1** <br/>ImageNet dataset에서 label이 각각 Siberian husky와 Eskimo dog인 데이터를 하나씩 보여준다.

<br/>
위 두 가지 문제를 해결하는 근본적인 방법은 FC layer나 convolution 내부를 sparse한 것으로 교체하여, sparsity을 도입하는 것이다. 이는 biological system의 모방 외에도, [Arora의 연구](https://arxiv.org/pdf/1310.6343.pdf)로부터 견고한 이론적 토대를 얻을 수 있다는 장점이 있다.
>[Arora의 연구](https://arxiv.org/pdf/1310.6343.pdf)는 dataset의 확률 분포를 sparse하고 큰 네트워크로 나타낼 수 있다면, 선행 layer activation들의 correlation statistic 분석 및 highly correlated output을 가진 뉴런들을 클러스터링 함으로써, 최적화 된 네트워크를 구성할 수 있다는 내용이라 한다. 이에는 엄격한 condition이 따라야 하지만, Hebbian principle을 참조하면 실제로는 덜 엄격한 condition 하에서도 적용할 수 있다고 한다. 
>
>즉, sparse한 경우에는 대부분의 노드가 비활성화 되고 일부만 활성화 될텐데, 이 때 엄격한 condition에 따라 상관관계를 분석함으로써 활성화 될 노드를 정한다면 최적화 된 네트워크가 구성될 것이라는 말이다.

<br/>
>Biological system의 모방이라는 부분을 잠깐 살펴보자. 노벨 생리의학상을 수여한 두 신경생리학자 Hubel과 Wiesel의 [연구1](https://physoc.onlinelibrary.wiley.com/doi/pdf/10.1113/jphysiol.1962.sp006837), [연구2](https://physoc.onlinelibrary.wiley.com/doi/epdf/10.1113/jphysiol.1968.sp008455) 등에서는 고양이와 원숭이를 실험 대상으로 진행한 연구 결과가 나와있다. 지금도 CNN 관련 기초 자료 중 상당수는 아래와 유사한 그림을 보여주면서 이들의 연구를 아이디어 배경으로 언급하고 있다. <br/>
>
>![Extra.1](/blog/images/GoogLeNet, Extra.1(removed).png )
>
><br/>
>CNN의 배경에 잠깐 언급할 때는 보통, 실험체의 시신경들이 특정 패턴에 뉴런이 반응하는 것을 발견했고 이에 착안한 방법이 CNN이다 정도로만 설명한다. 하지만, 실제로 각 패턴에 반응하는 뉴런의 집합은 전체 뉴런에 비해 극히 일부이므로 sparse하게 동작하는 것으로 볼 수 있으며, 논문에서는 이러한 맥락에서 biological system의 모방이라 표현한 것으로 보인다. 또한, 시신경 뉴런들의 이러한 동작에 대한 체계적인 연구가 [Sparse Coding](http://www.chaos.gwdg.de/~michael/CNS_course_2004/papers_max/OlshausenField1997.pdf)에서 진행됐다고 한다.
>
><br/>
>Deep neural network와 관련해 sparsity 키워드를 검색해보면 실제로 [dropout](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)으로 예를 드는 사이트가 굉장히 많이 나온다. [Dropout](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)은 정해진 비율만큼 랜덤하게 해당 layer의 노드를 deactivation 시키는 기법이다. 이 과정을 거치면 실제로 activated node의 수가 줄어들며, 자연스럽게 sparse한 구조가 이뤄진다. 아래의 그림을 보자.
>
><br/>
>![Extra.2](/blog/images/GoogLeNet, Extra.2(removed).png )
>
><br/>
>[Dropout](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)을 사용한 오른쪽의 경우에는 각 feature map의 채널이 특정 패턴에 집중하는 것으로 보인다. 반면, 이를 사용하지 않은 왼쪽의 경우에는 각 채널들이 모든 종류의 패턴을 학습하려 드는 오지라퍼 정도로 느껴진다. 
>
>[Dropout](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)에서는 실제로 sparsity를 도입하기 위한 것보단, node 간의 co-adaption을 줄이기 위한 방법으로 node를 랜덤하게 deactivation한 것이라 한다. 아래 그래프를 참조하면 실제로 [dropout](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)을 사용한 경우에는 deactivated node의 수가 압도적으로 많은 것으로 나타난다.
><br/>
>
>![Extra.3](/blog/images/GoogLeNet, Extra.3(removed).png )
>
><br/>
>하지만, [dropout](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)의 경우에는 deactivation일 뿐이지, deactivated node를 계산에서 제외하도록 sparse data structure를 이용하는 건 아닌 것으로 알고 있다. 아래는 [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)에서 사용하는 표다.
>
><br/>
>![Extra.4](/blog/images/GoogLeNet, Extra.4(removed).png )
>
><br/>
>위의 표는, pre-layer의 feature map 채널이 6개이고, post-layer의 feature map 채널이 16개일 때 사용되는 표이다. 이 표는 pre-layer의 output이 column번 째의 채널을 형성할 때, pre-layer의 row번 째의 채널에 해당하는 데이터를 사용할 지에 대한 여부이다. 즉, output을 만들기 위한 데이터를 선택적으로 취하겠다는 것이다. 이는 네트워크 안에서의 symmetry를 깨고, 합리적인 연결만을 유지하는 것이 목적이다. Symmetry를 깨는 이유는, 각 feature map들이 서로 다르면서 상호 보완적인 feature를 추출하도록 유도하는 효과가 있기 때문이라 한다.
>
>Sparse한 것으로 대체 한다는 것은 [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)과 같이 직접적 선택 혹은 통계적 분석을 이용한 간접 선택, 또는 [dropout](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)과 같은 방법들로부터 발생하거나, 학습 중 자연스럽게 발생한 deactivated node를 다음 layer와의 연산 과정에 포함되지 않도록 sparse data structure로 변환하는 방법으로 생각할 수 있다.


<br/>
하지만, 불행하게도 오늘날의 computing infra는 non-uniform sparse data structure에서 수행되는 계산에 관련해선 매우 비효율적이다.
>Sparsity를 도입하여 arithmetic operation의 수가 100배 감소하더라도, 참조하면서 발생되는 lookup이나 cache miss로 인한 오버 헤드가 이를 상회하는 역효과를 낳는다. 또한, CPU나 GPU에 최적화 된 numerical library가 개발되면서 dense matrix multiplication의 고속 연산이 가능해졌고, 이에 따라 operation의 수가 줄어듦으로 인한 이득이 점점 감소했다.

<br/>
Vision 목적의 최신 학습 모델에서는 convolution을 사용하는 것만으로 spatial domain에서의 sparsity를 활용한다. 그러나 convolution은 이전 layer의 patch에 대한 dense connection의 집합으로 구현된다.
>convolution 과정 중, 한 위치에서 filter가 입력과 곱해져서 해당하는 pixel의 출력이 만들어지는 부분을 하나의 dense connection으로 표현하는 것으로 보인다.

<br/>
CNN 계열에서, [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)이후로는 symmetry를 깨고 학습을 향상시키기 위해서 feature 차원에서의 random 혹은 sparse connection table을 사용 했었지만, 이후에는 병렬 계산을 더 최적화하기 위해 [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)과 같은 full connection으로 추세가 다시 바뀌었다. Computer vision 분야의 state-of-the-art 기법들은 많은 수의 filter와 큰 batch size로부터 효율적인 dense computation이 가능한 구조들을 따르고 있다.

<br/>
Dense matrix 연산에 적합한 하드웨어를 활용한다는 조건 하에서, 위의 이론처럼 filter-level과 같은 중간 단계에서 sparsity을 이용할 방법이 있는가 하는 의문이 든다.

[연구1](https://graal.ens-lyon.fr/~bucar/papers/ucca2D.pdf)과 같이 sparse matrix의 계산에 대한 수많은 연구들은, sparse matrix를 상대적으로 densely한 submatrix로 clustering 하는 방법이 sparse matrix multiplication에서 쓸만한 성능을 보였다고 한다. GoogLeNet 저자는 이 연구들을 두고, 가까운 미래에 non-uniform deeplearning architecture의 자동화 기법에 이와 유사한 방법이 활용 될 가능성이 있을거라 생각했다고 한다.

<br/>
Inception architecture는 [Arora의 연구](https://arxiv.org/pdf/1310.6343.pdf)에서 말한 sparse structure에 대한 근사화를 포함해, dense하면서도 쉽게 사용할 수 있도록 정교하게 설계된 network topology construction 알고리즘을 평가하기 위한 사례 연구로 시작됐다.

<br/>
상당한 추측에 근거한 연구였지만, [NIN](https://arxiv.org/pdf/1312.4400.pdf)에 기반하여 구현된 기준 네트워크와 비교했을 때, 이른 시점에서 약간의 성능 향상이 관찰됐다. 이는 약간의 튜닝으로도 gap이 넓어졌으며, Inception이 [R-CNN](https://arxiv.org/pdf/1311.2524.pdf)과 [Scalable object detection](https://arxiv.org/pdf/1312.2249.pdf)에서 base network로 사용되는 경우, localization과 object detection 분야에서 성능 향상 효과가 있다는 것이 증명되었다.

<br/>
그럼에도 신중하게 접근할 필요가 있다. Inception architecture가 computer vision에서 성공적으로 자리 잡았음에도 불구하고, 이 architecture가 원래 의도했던 원칙에 의해 얻어진 구조인지에 대해서는 미심쩍은 부분이 있기 때문이다. 이를 확인하기 위해서는 훨씬 더 철저한 분석과 검증이 필요하다.

---
## 4. Architectural Details
Inception architecture의 주요 아이디어는 computer vision을 위한 convolutional network의
1.  optimal local sparse structure 근사화
2. 쉽게 이용 가능한 dense component 구성

을 위한 방법을 고려한다.
<br/>

Translation invariant를 가정하고, convolutional building block으로 네트워크를 만들 것이다. 이를 위해 필요한 것은 optimal local construction을 찾고, 이를 공간적으로 반복하는 것이다.
>Translation invariant은 이미지 내의 어떤 특징이 평행 이동 되더라도 활성화 여부가 달라지지 않는 성질이다.

<br/>
[Arora의 연구](https://arxiv.org/pdf/1310.6343.pdf)는 마지막 layer의 correlation statistics를 분석하여, high correlation unit의 그룹으로 clustering하는 방식의 layer-by-layer construction을 제안한다.
>Clustered unit은 다음 layer의 unit으로 구성되고, 이전 layer의 unit에 연결 된다.

<br/>
이 논문에서는 이전 layer의 각 unit이 입력 이미지의 일부 영역에 해당하며, 이러한 unit들이 filter bank로 그룹화 된다고 가정한다. 이에 따라 lower layer에서는 correlated unit들이 local region에 집중 될 것이다. 즉, 한 영역에 많은 cluster가 집중 될 것이며, 이는 [NIN](https://arxiv.org/pdf/1312.4400.pdf)에서 제안 된 것처럼 다음 layer에서의 1x1 conv layer로 수행될 수 있다.
>큰 patch size의 convolution을 사용함으로써, 적은 수의 cluster로도 더 넓게 펼쳐질 수 있으며, 이를 통해 patch의 수가 감소될 수도 있다. Patch는 filter로 생각하면 된다.

<br/>
또한, patch-alignment issus를 피하기 위해, 현재의 Inception architecture에서 filter size를 1x1, 3x3 및 5x5로 제한한다. 이는 필요성보다는 편의성에 더 중점을 둔 결정이다.
>Filter size가 짝수일 경우에는 patch의 중심을 어디로 해야 할지도 정해야하는데, 이를 두고 patch-alignment issue라고 한다.

<br/>
즉, 제안 된 구조는 이러한 layer들의 output filter bank 조합이라는 것을 의미한다.
>이는 다음 layer의 입력으로 연결 될 single output vector가 된다.

<br/>
또한, 현재의 성공적인 convolutional network들에서 pooling 연산이 필수적이었단 사실은, Fig.2와 같이 각 단계에서 alternative parallel pooling path를 추가하는 것이 이점을 가질 것임을 암시한다.

<br/>
![Fig.2](/blog/images/GoogLeNet, Fig.2(removed).png )
>**Fig.2** <br/>제안하는 inception module의 각 버전별 구조이다.

<br/>
이러한 "Inception modules"이 서로의 위에 쌓이게 되면, 출력의 correlation statistics가 달라질 수 있다. Higher layers에서는 높은 수준으로 추상화 된 feature에서 추출되므로 공간 집중도가 감소 할 것으로 예상되며, 이는 higher layer로 갈수록 3x3 혹은 5x5 convolution의 비율이 증가해야 함을 암시한다.

<br/>
하지만, Fig.2의 (a)와 같은 naive한 inception 모듈의 경우, 5x5 convolution를 적게 사용하더라도 filter가 많아지면 비용이 엄청나게 비싸진다는 문제가 있다. 여기에 pooling unit이 추가되면, 이 문제가 더욱 부각된다.
>Fig.2를 보면 Inception module은 convolution 결과들을 concatenation으로 연결하기 때문에, output 채널의 수가 많아진다는 것을 알 수 있다. 여기에 output의 채널 수가 input과 같은 pooling unit이 추가되면, 단계가 거듭 될수록 채널이 이전 layer의 2배 이상 누적되므로 치명적일 수 있다.

<br/>
따라서, 이 구조는 optimal sparse structure를 처리 할 수 있음에도 매우 비효율적으로 수행되기 때문에 몇 단계 내에서 계산량이 급증한다는 문제에 봉착하게 된다.

<br/>
다음은 inception architecture에서 이를 타개하기 위한 두 번째 아이디어를 설명한다. 이는 계산 요구량이 너무 많이 증가 할 경우에 현명하게 차원을 줄이는게 목적이며, 성공적인 embedding에 기반한 방법이다. 저차원의 embedding은 상대적으로 큰 image patch에 대한 많은 정보를 포함 할 수 있다. 하지만, 이러한 embedding은 정보가 고밀도로 압축 된 형식이며, 압축 된 정보는 처리하기가 더 어렵워진다.

[Arora의 연구](https://arxiv.org/pdf/1310.6343.pdf)의 조건에 따르면, representation은 대부분의 위치에서 sparse하게 유지되어야하며, 필요한 경우에는 신호를 압축해야 한다. 따라서, 계산량이 많은 3x3이나 5x5 convolution 이전에는 reduction을 위한 1x1 convolution이 사용된다. 이 때, activation으로 ReLU를 사용함으로써 이중 목적으로 활용다. 최종 결과는 Fig.2의 (b)와 같다.
>단순한 계산량 감소 목적 외에도, non-linearity를 함께 취하겠다는 의미로 보인다.

<br/>
Inception network는 상기 타입의 module들이 서로 쌓여서 구성된 네트워크이며, 가끔씩 feature map size를 줄이기 위해 strides가 2인 max-pooling를 사용한다. 또한, 학습 중의 메모리 효율성을 고려하여, 하위 layer에서는 전통적인 convolution을 유지하고, 상위 layer에서만 Inception module을 사용하는 것이 좋다고 판단한다.
>이는 비효율적인 인프라를 고려하기 위한 것이며, 꼭 필요한 절차는 아니다.

<br/>
Inception architecture는 후반부의 stage에서도 계산 복잡도의 폭발적인 증가 없이 각 단계의 unit 수를 크게 늘릴 수 있다는 것다. 이는, 더 큰 patch size를 갖는 expensive convolution에 앞서, dimensionality reduction을 보편적으로 사용함에 따라 가능해진다.

<br/>
또한, 이 디자인은 시각적 정보를 다양한 scale 상에서 처리된 후에 종합함으로써, 다음 stage에서 서로 다른 scale로부터 feature를 동시에 추상화 한다는 타당한 직관을 따른다.


<br/>
즉, **Inception 구조는 계산 자원을 효율적으로 사용하므로, 성능은 약간 떨어지더라도 저렴한 계산 비용으로 깊고 넓은 네트워크를 구축할 수 있다** 는 말이다. 저자들은 사용 가능한 모든 knobs와 levers로 computational resource의 밸런스를 조절하여, 유사한 성능의 non-Inception architecture보다 3~10배 빠른 네트워크를 만들 수 있음을 발견했다고 한다. 물론 이 경우에는 신중한 수동적 설계가 필요하다.
>'사용 가능한 모든 knobs와 levers'에 해당하는 정확한 워딩은 'all the available knobs and levers'인데, 자주 사용하는 상투적 표현도 아닌 것 같다. 그냥 inception module에 관련된 각종 parameter를 변화하면서 적절한 값을 찾아봤다는 정도로 생각된다.


---
## 5. GoogLeNet
**GoogLeNet**은 ILSVRC 2014 competition에 제출한 Inception architecture의 특정 형태를 말한다. 우리는 약간 더 좋은 성능을 가지는 더 깊고 넓은 Inception network를 사용했다. 하지만, 이 네트워크를 ensemble 했을 땐 약간의 성능 향상만 이뤄졌다.

<br/>
경험에 따르면, 정확한 architectural parameter의 영향은 상대적으로 미미하기 때문에, 세부적인 네트워크 정보는 생략한다. Table.1은 ILSVRC에서 사용 된 가장 일반적인 Inception 구조를 보여준다. 이 네트워크는 ensemble에 사용된 7개의 모델 중 6개에 사용됐다.
>Ensemble에 사용 된 6개의 네트워크는 서로 다른 image patch sampling 방법에 따라 학습됐다.

<br/>
![Table.1](/blog/images/GoogLeNet, Table.1(removed).png )
>**Table.1** <br/>ILSVRC 2014 competition에 제출됐던 GoogLeNet의 구조이다. "#3x3 reduce"와 "#5x5 reduce"는 각각 3x3, 5x5 convolution 전에 사용 된 reduction layer의 1x1 filter 개수를 나타낸다. Inception 모듈에서의 "pool proj" 열 값은 max-pooling 뒤에 따라오는 projection layer의 1x1 filter 개수를 나타낸다.

<br/>
GoogLeNet은 Inception module 내부의 reduction/projection layer를 포함한 모든 convolution에서 ReLU를 사용한다. Receptive field는 zero mean RGB data에서 224x224 크기를 가진다. 또한, parameter가 있는 경우만 계산하면 22-layer에 해당한다. 네트워크에 사용 된 layer를 개별적으로 카운팅하면 약 100개 정도 사용하고 있다.
>정확한 개수는 layer를 카운팅하는 방법에 따라 다르다.

<br/>
이 네트워크는 계산적인 효율성과 실용성을 염두하여 설계됐으므로, 특히 메모리가 적은 경우를 포함하여, 제한적인 computational resource를 갖는 device에서도 inference를 수행할 수 있다.

<br/>
네트워크에 linear layer가 추가됐음에도 classifier 이전에 average pooling을 사용하는 것은 [NIN](https://arxiv.org/pdf/1312.4400.pdf)에 근거한 방법이다. Linear layer를 사용하면 네트워크를 다른 label set에 쉽게 적용시킬 수 있지만, 대부분의 경우에는 어떤 주요한 효과를 기대하기 보다는 편의를 위해 사용한다. FC layer 대신 average pooling를 사용하여 top-1 accuracy가 약 0.6% 향상됐다. 하지만, 이 경우에도 여전히 [dropout](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)의 사용이 필수적이었다.
>여기서 Linear layer는 FC layer를 말한다.
>
>이 논문이 작성 될 시점에는 [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf)이 없었다. 이후에 연구 된 [Inception-V3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)에서는 [BN](https://arxiv.org/pdf/1502.03167.pdf)을 활용하면서 [dropout](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf) 얘기를 그닥 하질 않았으니, 필수까진 아닌 것으로 보인다.

<br/>
네트워크가 깊어지는 경우, 모든 layer를 통해 gradient를 효과적인 방식으로 전달할 수 있는 능력이 중요해진다. Shallower network가 이 부분에서 강력한 성능을 보인다는 것은, 네트워크의 중간 layer에 의해 생성 된 feature의 식별성이 높아야 한다는 것을 암시한다.

<br/>
GoogLeNet에서는 보조 분류기(auxiliary classifier)를 중간 layer에 연결시켜 추가함으로써, 하위 layer에서의 높은 식별성을 예상하고 있다. 또한, 저자들은 보조 분류기의 추가가 regularization 효과와 함께 vanishing gradient problem를 해결해주는 것으로 생각한다. 

<br/>
보조 분류기는 Inception (4a)와 (4d)의 출력에서 뻗어 나온 작은 convolution network의 형태를 취한다. 학습 중에는 이들의 loss에 weight를 적용하여 네트워크의 총 loss에 더하며, inference 시에는 이를 제거한다.
>보조 분류기의 loss는 30%만 고려하고 있다.

<br/>
최근의 실험에 따르면 보조 분류기의 효과는 상대적으로 작으며, 같은 효과를 위해선 이 중에 하나만 있으면 된다.
>약 0.5%의 성능 향상이 이뤄졌다.


<br/>
보조 분류기를 포함한 측면의 추가 네트워크에 대한 정확한 구조는 다음과 같다. 최종 형태는 Fig.3 참조

- Filter size가 5x5이고 strides가 3인 average pooling layer. 출력의 shape은 (4a)와 (4d)에서 각각 4x4x512와 4x4x528이다

- Dimension reduction을 위한 1x1 conv layer(128 filters) 및 ReLU

- FC layer(1024 nodes) 및 ReLU

- Dropout layer (0.7)

- Linear layer에 softmax를 사용한 1000-class classifier.

<br/>
![Fig.3](/blog/images/GoogLeNet, Fig.3(removed).png )
>**Fig.3** <br/>ILSVRC 2014 competition에 제출됐던 GoogLeNet의 도식이다.

<br/>
글만 쓰면 정말 재미가 없다. Fig.3을 keras로 구현해보자.
``` python
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, AveragePooling2D, Flatten, Concatenate
from keras.optimizers import SGD
from keras.callbacks import Callback

from keras.utils import to_categorical
from keras.datasets import cifar10
import numpy as np

class LearningRateSchedule(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%8 == 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr*0.96)

class LocalResponseNormalization(Layer):
    def __init__(self, n=5, alpha=1e-4, beta=0.75, k=2, **kwargs):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LocalResponseNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(LocalResponseNormalization, self).build(input_shape)

    def call(self, x):
        _, r, c, f = self.shape 
        squared = K.square(x)
        pooled = K.pool2d(squared, (self.n, self.n), strides=(1,1), padding="same", pool_mode='avg')
        summed = K.sum(pooled, axis=3, keepdims=True)
        averaged = self.alpha * K.repeat_elements(summed, f, axis=3)
            
        denom = K.pow(self.k + averaged, self.beta)
        
        return x / denom 
    
    def get_output_shape_for(self, input_shape):
        return input_shape

def Upscaling_Data(data_list, reshape_dim):
    ...

def inception(input_tensor, filter_channels):
    filter_1x1, filter_3x3_R, filter_3x3, filter_5x5_R, filter_5x5, pool_proj = filter_channels
    
    branch_1 = Conv2D(filter_1x1, (1, 1), strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(input_tensor)
    
    branch_2 = Conv2D(filter_3x3_R, (1, 1), strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(input_tensor)
    branch_2 = Conv2D(filter_3x3, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(branch_2)

    branch_3 = Conv2D(filter_5x5_R, (1, 1), strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(input_tensor)
    branch_3 = Conv2D(filter_5x5, (5, 5), strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(branch_3)
    
    branch_4 = MaxPooling2D((3, 3), strides=1, padding='same')(input_tensor)
    branch_4 = Conv2D(pool_proj, (1, 1), strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(branch_4)
    
    DepthConcat = Concatenate()([branch_1, branch_2, branch_3, branch_4])
    
    return DepthConcat
    
def GoogLeNet(model_input, classes=10):
    conv_1 = Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(model_input) # (112, 112, 64)
    pool_1 = MaxPooling2D((3, 3), strides=2, padding='same')(conv_1) # (56, 56, 64)
    LRN_1 = LocalResponseNormalization()(pool_1) # (56, 56, 64)
    
    conv_2 = Conv2D(64, (1, 1), strides=1, padding='valid', activation='relu')(LRN_1) # (56, 56, 64)
    conv_3 = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu')(conv_2) # (56, 56, 192)
    LRN_2 = LocalResponseNormalization()(conv_3) # (56, 56, 192)
    pool_2 = MaxPooling2D((3, 3), strides=2, padding='same')(LRN_2) # (28, 28, 192)
    
    inception_3a = inception(pool_2, [64, 96, 128, 16, 32, 32]) # (28, 28, 256)
    inception_3b = inception(inception_3a, [128, 128, 192, 32, 96, 64]) # (28, 28, 480)
    
    pool_3 = MaxPooling2D((3, 3), strides=2, padding='same')(inception_3b) # (14, 14, 480)
    
    inception_4a = inception(pool_3, [192, 96, 208, 16, 48, 64]) # (14, 14, 512)
    inception_4b = inception(inception_4a, [160, 112, 224, 24, 64, 64]) # (14, 14, 512)
    inception_4c = inception(inception_4b, [128, 128, 256, 24, 64, 64]) # (14, 14, 512)
    inception_4d = inception(inception_4c, [112, 144, 288, 32, 64, 64]) # (14, 14, 528)
    inception_4e = inception(inception_4d, [256, 160, 320, 32, 128, 128]) # (14, 14, 832)
    
    pool_4 = MaxPooling2D((3, 3), strides=2, padding='same')(inception_4e) # (7, 7, 832)
    
    inception_5a = inception(pool_4, [256, 160, 320, 32, 128, 128]) # (7, 7, 832)
    inception_5b = inception(inception_5a, [384, 192, 384, 48, 128, 128]) # (7, 7, 1024)
    
    avg_pool = GlobalAveragePooling2D()(inception_5b)
    dropout = Dropout(0.4)(avg_pool)
    
    linear = Dense(1000, activation='relu')(dropout)
    
    model_output = Dense(classes, activation='softmax', name='main_classifier')(linear) # 'softmax'
    
    # Auxiliary Classifier
    auxiliary_4a = AveragePooling2D((5, 5), strides=3, padding='valid')(inception_4a)
    auxiliary_4a = Conv2D(128, (1, 1), strides=1, padding='same', activation='relu')(auxiliary_4a)
    auxiliary_4a = Flatten()(auxiliary_4a)
    auxiliary_4a = Dense(1024, activation='relu')(auxiliary_4a)
    auxiliary_4a = Dropout(0.7)(auxiliary_4a)
    auxiliary_4a = Dense(classes, activation='softmax', name='auxiliary_4a')(auxiliary_4a)
    
    auxiliary_4d = AveragePooling2D((5, 5), strides=3, padding='valid')(inception_4d)
    auxiliary_4d = Conv2D(128, (1, 1), strides=1, padding='same', activation='relu')(auxiliary_4d)
    auxiliary_4d = Flatten()(auxiliary_4d)
    auxiliary_4d = Dense(1024, activation='relu')(auxiliary_4d)
    auxiliary_4d = Dropout(0.7)(auxiliary_4d)
    auxiliary_4d = Dense(classes, activation='softmax', name='auxiliary_4d')(auxiliary_4d)
    
    
    model = Model(model_input, [model_output, auxiliary_4a, auxiliary_4d])
    
    return model


input_shape = (224, 224, 3)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = Upscaling_Data(x_train, input_shape)
x_test = Upscaling_Data(x_test, input_shape)

x_train = np.float32(x_train / 255.)
x_test = np.float32(x_test / 255.)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model_input = Input( shape=input_shape )

model = GoogLeNet(model_input, 10)

optimizer = SGD(momentum=0.9)

model.compile(optimizer, 
        	loss={'main_classifier' : 'categorical_crossentropy',
                    'auxiliary_4a' : 'categorical_crossentropy',
                        'auxiliary_4d' : 'categorical_crossentropy'},
                loss_weights={'main_classifier' : 1.0, 
                                'auxiliary_4a' : 0.3, 
                                'auxiliary_4d' : 0.3}, 
                metrics=['acc'])

model.fit(x_train, 
        {'main_classifier' : y_train, 
                'auxiliary_4a' : y_train, 
                'auxiliary_4d' : y_train},  
        	epochs=100, batch_size=32, callbacks=LearningRateSchedule())

```

>귀찮으니 CIFAR-10 dataset을 upscaling해서 테스트 한다. ImageNet에 적용하려면, GoogLeNet 함수를 호출하면서 class 수를 1000으로 넣으면 된다. 보조 분류기 설명에서 FC layer를 하나만 언급했는데 Fig.3에는 두 개가 있다. 가볍게 무시하고 하나만 사용했다.
>
>6장에서는 optimizer로 asynchronous SGD를 사용한다고 되어있는데, 지금은 distributed learning framework에서 학습하는게 아니므로 가볍게 무시하고 SGD를 사용한다.
>
>mementum과 learning rate는 6장의 내용에 따라 정했으며, 그 외에 언급하지 않은 내용은 default로 두거나, 임의로 아무 값이나 넣어뒀다.
>
>n번 째의 epoch마다 learning rate scheduling을 적용하려면 직접 구현하는 수 밖에 없다더라.
>
>LocalResponseNormalization 역시 keras에서 제공하지 않아서 [여기](https://datascienceschool.net/view-notebook/d19e803640094f76b93f11b850b920a4/)를 참조했다.


---
## 6. Training Methodology
GoogLeNet은 [DistBelief](https://www.cs.toronto.edu/~ranzato/publications/DistBeliefNIPS2012_withAppendix.pdf)라는 distributed machine learning system을 사용하여 학습됐다.
>구글에서 개발 된 대규모 분산 학습 프레임워크로, 적당한 양의 모델과 데이터 병렬성을 이용하여 학습한다고 한다.

<br/>
학습은 CPU 기반으로만 진행했었으나, high-end GPU를 몇 개 사용하여 학습하는 경우에는 1주일 내에 수렴 가능할 것으로 추정된다.
>모바일이나 임베디드 시스템 상에서도 inference가 가능하도록 설계하는게 목적이었기 때문으로 생각된다.

<br/>
학습에선 momentum을 0.9로 한 [asynchronous SGD](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45187.pdf)를 사용했고, learning rate schedule은 8 epoch마다 4% 감소하도록 적용했다.
>분산 학습에 사용되는 optimizer라고 한다. ([참고](https://www.facebook.com/groups/smartbean2/permalink/1879507065431662/))

<br/>
inference time에 사용 될 최종 모델을 생성하기 위해 [Polyak averaging](https://pdfs.semanticscholar.org/6dc6/1f37ecc552413606d8c89ffbc46ec98ed887.pdf)이 사용됐다.
>[cs231n 강의 슬라이드](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf)에서는 Polyak averaging의 동작을 다음의 한 줄로 표현했음. 부가 설명은 [강의 동영상](https://www.youtube.com/watch?v=_JB0AO7QxSA) 참조
> 
>**INSTEAD of using actual parameter vector, keep a moving average of the parameter vector and use that at test time.**

<br/>
이미지 샘플링 방법은 ILSVRC 2014 competition까지의 몇 달 동안 크게 변화됐었다. 이미 수렴한 모델들은 다른 옵션을 사용하여 학습됐으며, 때로는 dropout이나 learning rate 등의 hyperparameter를 변경하기도 했다.
>그래서 이 네트워크를 학습하는 가장 효과적인 방법에 대한 가이드를 제공하긴 어렵다고 함.

<br/>
문제를 더 복잡하게하기 위해, 모델 중 일부는 상대적으로 작은 크기의 crop 상에서 주로 학습했고, 다른 모델들은 더 큰 크기의 crop 상에서 학습했다.
>이는 [Andrew Howard의 연구](https://arxiv.org/ftp/arxiv/papers/1312/1312.5402.pdf)에서 영감을 얻은 방법이라 한다.

<br/>
Competition 이후에는, 종횡비를 [3/4, 4/3]로 제한하여 8% ~ 100%의 크기에서 균등 분포로 patch sampling 하는 것이 매우 잘 작동한다는 것을 발견했다. 또한, [Andrew Howard의 연구](https://arxiv.org/ftp/arxiv/papers/1312/1312.5402.pdf)의 'photometric distortion'이 overfitting 방지에 유용하다는 것을 발견했다.


---
## 7. ILSVRC 2014 Classification Challenge Setup and Results
ILSVRC 2014 classification challenge는 이미지를 Imagenet 계층에서 1000개의 카테고리 중 하나로 분류하는 작업을 포함한다. [ training / validation / test ] 데이터는 각각 [ 약 1,200,000 / 50,000 / 100,000 ]개의 이미지로 구성되어 있다.

<br/>
각 이미지는 ground truth로 하나의 카테고리만 연관되어 있으며, 성능 평가는 classifier의 prediction 중 highest scoring을 기반으로 측정되며, 대개 두 종류의 수치를 본다.

1. **top-1 accuracy rate**는 ground truth를 highest score class와 비교하여 측정한다.
        
2. **top-5 error rate**는 ground truth을 predicted score 상에서의 최상위 5개 class와 비교하여 측정한다. 최상위 5개의 class 안에 ground truth가 속하기만 하면 순위에 상관없이 정답으로 간주한다.

>이 challenge에서는 top-5 error rate로 ranking을 결정한다.

<br/>
GoogLeNet은 external data를 학습에 사용하지 않는 challenge에 참여했다. 논문에 언급 된 학습 기법 외에도, testing 시에는 더 높은 성능을 위해 일련의 기법들을 사용했다.

<br/>
- 동일한 GoogLeNet 모델의 7가지 버전(wider version도 하나 포함)을 독립적으로 학습했으며, 이들을 이용한 ensemble prediction을 수행했다. 각 모델들은 동일한 weight initialization과 learning rate policy로 학습했으며, sampling 방법과 shuffle로 인한 학습 데이터의 순서에서만 차이가 있다.

<br/>
- 테스트 과정에서는 [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)보다 더 적극적인 cropping 방식을 적용했다. shorter side가 각각 [256 / 288 / 320 / 352]인 4가지 scale로 이미지의 크기를 조정하여 [ left / center / right ]의 square를 취한다. 각 square에 대해 [ 모서리 4개 / 중앙 ]에서 224x224 crop 및 square 자체를 224x224 크기로 resize한 것과, 이들의 미러링 된 버전을 취한다. 물론 충분한 개수의 crop이 있는 경우에는 benefit이 적어지므로, 실제로는 이러한 적극적인 cropping이 필요하지 않을 수도 있다.
>세로 방향의 이미지의 경우, [ left / center / right ] 대신, [ top / center / bottom ]에서 square를 취하며, 이미지 당 총 4x3x6x2 = 144개의 crop이 생성 됨.
>
>비슷한 접근법이 전년도에 [Andrew Howard](https://arxiv.org/ftp/arxiv/papers/1312/1312.5402.pdf)의 엔트리에서 사용됐었지만, 이들이 제안한 방법이 약간 더 좋다는 것을 경험적으로 입증됐다.
>
>아래의 그림에서 한 눈에 알아보자.
>
><br/>
>![Extra.5](/blog/images/GoogLeNet, Extra.5(removed).gif )

<br/>
- Final prediction은 [ multiple crop / all the individual classifier ]에 대한 softmax probability를 평균하여 얻는다. 실험 중에 [ max pooling over crops / averaging over classifiers ] 같은 validation data에 대한 대안적 접근법들을 분석했었으나, 단순한 averaging보다 열등한 성능을 보였다.

<br/>
논문의 나머지 부분에서는 final submission의 전반적인 성능에 기여하는 여러 요소들을 분석한다. Challenge에 제출한 final submission은 validation과 test set에서 모두 6.67%의 top-5 error를 기록하고, 1위를 차지했다. 이는 2012년의 SuperVision approach에 비해, 56.5%의 상대적 오류 감소율을 보인 것이며, 학습에 external data를 사용하는 2013년도 best approach인 Clarifai에 비해선 약 40%의 상대적 감소를 보였다. 최근 3년 간의 best approach에 대한 통계를 Table.2에서 보인다.

<br/>
![Table.2](/blog/images/GoogLeNet, Table.2(removed).png )
>**Table.2** <br/>Classification performance.


<br/>
Table.3에서는 예측할 때 사용되는 [ model / crop ]의 수를 변경하면서 얻은 성능들을 보여준다.
![Table.3](/blog/images/GoogLeNet, Table.3(removed).png )
>**Table.3** <br/>Classification 성능에 대한 비교다. 하나의 모델만 사용할 때는 validation data에 대해 lowest top-1 error rate을 가진 모델을 선택했으며, 모든 수치는 testing data를 cheating하지 않기 위해, validation data에 대한 결과만 본다.


---
## 8. ILSVRC 2014 Detection Challenge Setup and Results
ILSVRC 2014의 detection 분야는 이미지 내에서 200개의 possible class object 주위에 bounding box를 생성하는 것이다. Detected object가 [ ground truth와 class 일치 / bounding box가 50% 이상 overlap ] 되는 경우에 정답인 것으로 계산된다.
>Overlap의 정도는 [Jaccard Index](https://ko.wikipedia.org/wiki/%EC%9E%90%EC%B9%B4%EB%93%9C_%EC%A7%80%EC%88%98)를 이용하여 측정한다.

<br/>
Extraneous detection은 false positive으로 간주하여 페널티를 준다. Classification과는 달리, 각 이미지는 많은 object를 포함하거나 전혀 포함하지 않을 수 있으며, 크기도 다를 수 있으며, mAP로 성능을 평가한다.

<br/>
GoogLeNet의 detection approach는 [R-CNN](https://arxiv.org/pdf/1311.2524.pdf)과 비슷하며, region classifier를 Inception model로 보강했다. 또한 region proposal step은 object bounding box의 recall을 더 높이기 위해 [multi-box prediction](https://arxiv.org/pdf/1312.2249.pdf)을 사용한 [selective search](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)와 결합하여 보강했다.

<br/>
또한, false positive의 수를 줄이기 위해 super pixel size를 2배 증가시켰다. 이를 통해 [selective search](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf) 알고리즘에서 얻어지는 proposal의 수가 반으로 줄어든다.
>Superpixel 참고 [[1]](https://extremenormal.tistory.com/entry/Superpixel%EC%9D%B4%EB%9E%80) [[2]](https://hwiyong.tistory.com/84)

<br/>
여기에 [multi-box](https://arxiv.org/pdf/1312.2249.pdf)에서 나온 200개의 region proposal을 더하여, 총 개수는 [R-CNN](https://arxiv.org/pdf/1311.2524.pdf)에서 사용 된 proposal의 약 60%정도지만, coverage는 92%에서 93%로 늘어났다.
>Proposal의 개수를 줄고 coverage가 늘어남으로 인해, single model의 경우 mAP가 1% 정도 향상되는 효과가 있었다.

<br/>
각 region에 대한 classification에는 6개의 GoogLeNet을 ensemble하여 사용하여, 정확도가 40%에서 43.9%로 향상됐다. ILSVRC 2014에 제출 시에는 시간적 한계로 인해, R-CNN에서 사용했던 bounding box regression를 사용하지 않았다.

<br/>
다음은 top detection 결과들과, 이후의 진행 상황들을 보여준다. 2013년의 결과에 비해 정확도가 거의 두 배로 향상됐으며, 최고 성능을 보인 팀들은 모두 CNN를 사용했다. Table.4의 공식 score와 각 팀의 전략들을 보여준다.

<br/>
![Table.4](/blog/images/GoogLeNet, Table.4(removed).png )
>**Table.4** <br/>Comparison of detection performances.


External data는 일반적으로 detection model의 pre-training 용도로 ILSVRC12 classification data를 사용하지만, 일부 팀에서는 localization data의 사용에 대해서도 언급했다. Localization bounding box는 detection data에 포함되어 있지 않기 때문에, general bounding box regressor를 pre-training 할 수 있다는 장점이 있다. 
>동일한 방식으로 classification에 대해서도 pre-training 할 수 있다.
>
>GoogLeNet은 pre-training에 localization data를 사용하지 않은 결과이다.

<br/>
Table.5에서는 single model만을 사용한 결과를 비교한다. 최고 성능을 낸 모델은 Deep Insight이지만, 이 모델 3개를 ensemble 한 경우엔 0.3%만 향상 된 반면, GoogLeNet의 ensemble에선 훨씬 더 큰 향상 효과를 얻었다.

![Table.5](/blog/images/GoogLeNet, Table.5(removed).png )
>**Table.5** <br/>Single model performance for detection.

---
## 9. Conclusions
GoogLeNet의 실험 결과는, 쉽게 이용 가능한 dense building blocks에 의해 optimal sparse structure를 근사화하는 것이, computer vision을 위한 네트워크의 성능 개선에 효과 있는 방법이라는 확실한 증거를 제시한다.
>쉽게 이용 가능한 기존의 conv layer로 optimal sparse structure를 나름 근사화 해봤고, 실험해보니 실제로 성능 개선에 효과가 있었다는 말이다.

<br/>
이 방법의 가장 큰 장점은 [ shallower / narrower ] 한 architecture에 비해, computational cost가 약간 증가 함에도 상당한 성능 향상을 얻을 수 있다는 것이다.

<br/>
Object detection에서는 [ context의 활용 / bounding box regression ]을 수행하지 않아도 경쟁력 있는 성능을 보였으며, 이는 inception architecture의 강점에 대한 또 다른 증거가 된다. 또한, classification과 detection의 결과들은, 유사한 크기의 훨씬 비싼 non-Inception-type 네트워크와 성능이 비슷할 것으로 예상된다.

<br/>
그럼에도, GoogLeNet approach는 sparser architecture로 바꾸는 것이 실현 가능하며 유용한 아이디어라는 확실한 증거를 제시한다. 이는 inception architecture의 아이디어를 다른 도메인에 적용하는 것 외에도, [Arora의 연구](https://arxiv.org/pdf/1310.6343.pdf)에 기반한 자동화 된 방식으로, 더 sparser하게 개선 된 구조에 대한 향후 연구를 제안한다.


---

<div id="disqus_thread"></div>
<script>
/**
 *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
 *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables
 */
/*
var disqus_config = function () {
	this.page.url = "{{ site.url }}{{ page.url }}";  // Replace PAGE_URL with your page's canonical URL variable
	this.page.identifier = "{{ page.id }}"; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
*/
  
//var disqus_shortname = '{{ site.disqus }}';

(function() {  // DON'T EDIT BELOW THIS LINE
	var d = document, s = d.createElement('script');
	s.src = 'https://aroddarys-personal-lab.disqus.com/embed.js'; // https://
	s.setAttribute('data-timestamp', +new Date());
	(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript" rel="nofollow">comments powered by Disqus.</a></noscript>
