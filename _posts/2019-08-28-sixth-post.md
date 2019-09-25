---
title: "(DenseNet) Densely Connected Convolutional Networks 번역 및 추가 설명과 Keras 구현"
date: 2019-08-28 07:24:11 -0400
tags: AI ComputerVision Paper DenseNet DenseBlock
categories:
  - Paper
toc: true
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Paper Information

HUANG, Gao, et al. **"Densely Connected Convolutional Networks"**. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2017. p.4700-4708.
> a.k.a. [DenseNet paper](https://arxiv.org/pdf/1608.06993.pdf)

<br/>
Keras 구현 코드 삽입 예정.

---
## Abstract
입력에 가까운 layer와 출력에 가까운 layer 사이에 shorter connection이 포함되면, 더 깊고 정확하면서 효율적으로 학습할 수 있다.

<br/>
이 논문에서는 이러한 관찰을 받아들여, feed-forward 방식으로 각 layer를 다른 모든 layer에 연결한 **DenseNet**을 소개한다.

<br/>
$$L$$개의 layer가 있는 기존의 CNN은 각 layer가 후속 layer와의 connection만 존재하는 $$L$$-connection 형태인 반면, **DenseNet**은 $$\frac{L(L + 1)}{2}$$개의 connection으로 이루어진다.

<br/>
각 layer에는 모든 선행 layer의 feature-map이 input으로 사용되며, 각 feature-map은 모든 후속 layer의 input으로 사용된다.

<br/>
**DenseNet**에는 몇 가지 장점이 있다.
- Vanishing-gradient 문제 완화

- 견고한 feature propagation

- Feature reuse 장려

- Parameter 수의 큰 감소

<br/>
경쟁이 치열한 4가지의 object recognition benchmark task에서 제안 된 아키텍처를 평가한다.
>CIFAR-10 / CIFAR-100 / SVHN / ImageNet

<br/>
**DenseNet**은 대부분의 state-of-the-art에 비해 크게 개선됐음에도, high performance 달성에 적은 양의 계산이 요구된다.

<br/>
코드 및 pre-trained model은 [저자의 github](https://github.com/liuzhuang13/DenseNet)에서 확인할 수 있다.
>Torch로 구현된 코드가 제공된다.
>
>본 포스트의 후반부에는 Keras로 구현한 코드가 추가되어 있다.

---
## 1. Introduction
CNN은 visual object recognition을 위한 machine learning approach 중 압도적으로 많이 사용된다.

<br/>
CNN은 [20년 전에 소개](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)됐었지만, 하드웨어와 네트워크 구조의 개선을 통해 최근에서야 deep CNN의 학습이 가능해졌다.

<br/>
[LeNet5](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)는 5-layer, [VGG](https://arxiv.org/pdf/1409.1556.pdf)는 19-layer로 구성됐으며, [Highway Network](https://arxiv.org/pdf/1507.06228.pdf)와 [ResNet](https://arxiv.org/pdf/1512.03385.pdf)만이 100-layer의 장벽을 넘어섰다.
>논문 작성 당시인 2016년 기준임.

<br/>
CNN이 점점 깊어짐에 따라, 새롭게 연구할 문제가 발생했다.
- **Input**이나 **gradient**에 대한 정보가 여러 layer를 통과하는 경우에는, 네트워크의 양단에 도달하는 시점에 이 정보가 **vanish** 혹은 **wash out** 될 수 있다.

<br/>
최근에는 이러한 문제나 관련된 문제를 해결하려는 연구가 많았다.
- [ResNet](https://arxiv.org/pdf/1512.03385.pdf)과 [Highway Network](https://arxiv.org/pdf/1507.06228.pdf)는 **identity connection**을 통해, signal이 한 layer에서 그 다음 layer로 건너뛴다.

- [Stochastic depth](https://arxiv.org/pdf/1603.09382.pdf)는 **학습 중에 layer를 무작위로 drop** 함으로써 ResNet을 단축시켜, information/gradient flow를 개선한다.

- [FractalNet](https://arxiv.org/pdf/1605.07648.pdf)은 네트워크에 많은 short path를 유지하면서 large depth로 만들기 위해, **여러 개의 parallel layer sequence와 서로 다른 수의 convolutional block을 반복적으로 결합**한다.

<br/>
이러한 접근 방식들은 네트워크 토폴로지나 학습 절차에 따라 다르지만, 모두 다음과 같은 특징을 가진다.
- 선행 layer에서 후속 layer로 향하는 **short path**를 만든다.

<br/>
이러한 통찰을 통해, 이 논문에서는 simple connectivity pattern의 구조를 제안한다.
- 네트워크의 layer 간 information flow를 극대화하기 위해, feature-map size가 동일한 모든 layer가 직접 연결된다.

<br/>
Feed-forward 특성을 유지하기 위해, 각 layer는 모든 선행 layer로부터 additional input을 취하며, 각 feature-map은 모든 후속 layer로 전달된다. **Fig.1**은 이 레이아웃을 개략적으로 보여준다.

<br/>
![Fig.1](/blog/images/DenseNet, Fig.1(removed).png )
>**Fig.1** <br/>A 5-layer dense block with a growth rate of $$k = 4$$.
>
>각 layer들은 모든 선행 layer의 feature-map을 input으로 사용한다.

<br/>
결정적으로 [ResNet](https://arxiv.org/pdf/1512.03385.pdf)은 feature들이 layer로 전달되기 전에 summation을 통해 결합되지만, 제안하는 구조에서는 **feature들을 concatenation**하여 결합된다.

<br/>
따라서, 제 $$\ell^{th}$$ layer는 모든 선행 conv block의 feature-map들로 구성된 $$\ell$$ 개의 input을 가지며, 각 feature-map은 모든 $$L-\ell$$개의 후속 layer로 전달된다.

<br/>
이는 기존의 $$L$$-layer 아키텍처에서, $$L$$개의 connection 대신 $$\frac{L(L+1)}{2}$$ 개의 connection을 도입한다.

<br/>
이러한 dense connectivity pattern으로 인해, 이 구조를 **DenseNet**이라 부른다.
>Dense Convolutional Networks

<br/>
Dense connectivity pattern에서 중복되는 feature-map은 다시 학습할 필요가 없기 때문에, DenseNet은 기존의 CNN보다 적은 수의 parameter만 필요하다.
>이는 densely connect로 인해 parameter가 많아질 것이라는 직관에 반대되는 효과다.

<br/>
이전에 연구된 네트워크 구조들과 DenseNet의 차이점을 다음과 같이 볼 수 있다.
- 전통적인 feed-forward 구조는 state를 가진 알고리즘으로 볼 수 있다. 여기서 state는 layer-to-layer로 전달되며, 각 layer는 이전 layer의 state를 읽고 후속 layer에 쓴다. 이 구조에서는 단순히 state를 변경할 뿐만 아니라, 보존해야하는 정보도 함께 전달한다.

- [ResNet](https://arxiv.org/pdf/1512.03385.pdf)에서는 추가적인 identity transformation을 통해, 명시적으로 정보를 보존한다.

- [ResNet의 변형 연구](https://arxiv.org/pdf/1603.09382.pdf)에서는 많은 layer가 거의 기여하지 않으며, 실제로 학습 중에 무작위로 drop 될 수 있음을 보여준다.

- [또 다른 ResNet 연구](https://arxiv.org/pdf/1604.03640.pdf)에서는 ResNet의 state를 unrolled RNN과 비슷하게 만들었지만, 각 layer가 자체 weight를 가지기 때문에 parameter의 수가 상당히 많아진다.

- **DenseNet**은 네트워크에 추가 된 정보와 보존 된 정보를 명시적으로 구분한다.

<br/>
DenseNet의 layer는 very narrow(e.g. 12 filters per layer)하며 네트워크의 "collective knowledge"에 적은 수의 feature-map set만 추가하고, 나머지 feature-map은 변경하지 않는다. 또한, 최종 classifier는 네트워크의 모든 feature-map에 기반하여 결정한다.
>Feature-map의 concatenation과 feature reuse에 관련한 내용이다.

<br/>
더 나은 parameter efficiency 외에도, DenseNet의 큰 장점 중 하나는 네트워크 전체에서 개선 된 information flow와 gradient를 통해 쉽게 학습할 수 있다는 것이다.

<br/>
각 layer는 loss function과 original input signal로부터의 gradient에 직접 액세스 할 수 있으므로, 유사 deep supervision이 이루어진다. 이는 deeper network architecture의 학습에 도움된다.
>[Deeply Supervised Network(DSN)](https://arxiv.org/pdf/1409.5185.pdf)에 대한 내용, 포스트 후반부 참조.

<br/>
또한, dense connection은 regularizing effect를 포함하므로, 더 작은 training set에 대한 overfitting을 줄인다.
>[stochastic depth](https://arxiv.org/pdf/1603.09382.pdf)에 대한 내용, 포스트 후반부 참조.

<br/>
경쟁이 치열한 4가지 benchmark dataset에서 DenseNet의 성능을 평가한다.
>**CIFAR-10, CIFAR-100, SVHN, ImageNet**에 해당

<br/>
DenseNet은 기존의 알고리즘보다 훨씬 적은 수의 parameter를 요구하는 경향이 있다.

<br/>
또한, DenseNet은 대부분의 benchmark에서 state-of-the-art 성능을 능가한다.

---
## 2. Related Work
네트워크 아키텍처에 대한 연구는 오래전부터 neural network 연구의 일부였으며, 최근 neural network에 대한 인기가 늘어나면서 이 분야의 연구도 다시 활발해졌다.

<br/>
최신 network들의 layer 수는 갈수록 증가됐으며, 이는 아키텍처 간의 차이를 증폭시킴과 동시에, 다양한 connectivity pattern의 탐구 및 오래된 연구의 아이디어를 다시 찾게되는 계기가 됐다.

<br/>
DenseNet과 유사한 [cascade structure](https://papers.nips.cc/paper/207-the-cascade-correlation-learning-architecture.pdf)는 1980년대 neural network 분야에서 이미 연구됐었다.
>이 연구는 fully connected MLP를 layer-by-layer 방식으로 학습하는 것에 중점을 둔다.

<br/>
보다 최근에는 batch gradient descent로 학습 된 [fully connected cascade network](http://www.eng.auburn.edu/~wilambm/pap/2010/Neural%20Network%20Learning%20Without%20Backpropagation.pdf)가 제안됐다.
>이 연구에서 제안하는 방법은 small dataset에는 효과적이지만, parameter가 수백 개인 네트워크로만 확장이 가능하다.

<br/>
다른 연구에서는 skip-connection을 통해 CNN의 multi-level feature를 활용하는 것이 다양한 vision task에서 효과적인 것으로 밝혀졌다.
>다른 연구 [[1](https://arxiv.org/pdf/1411.5752.pdf)] [[2](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)] [[3](https://arxiv.org/pdf/1212.0142.pdf)] [[4](https://arxiv.org/pdf/1505.05232.pdf)]

<br/>
[AdaNet](https://arxiv.org/pdf/1607.01097.pdf)은 cross-layer connection을 가진 네트워크를 위한 theoretical framework를 연구했다.
>여기서 말하는 cross-layer connection을 가진 네트워크가 DenseNet과 유사한 구조다.

<br/>
[Highway Networks](https://arxiv.org/pdf/1507.06228.pdf)는 100개 이상의 layer로 이루어진 end-to-end 네트워크를 효과적으로 학습시키는 방법을 제공한 최초의 아키텍처 중 하나이다.
>Gating unit과 bypassing path를 사용하면 Highway Networks를 어려움없이 최적화 할 수 있다고 한다.

<br/>
Bypassing path(우회 경로)는 very deep network의 학습에서 핵심적인 요소인 것으로 여겨졌으며, [ResNet](https://arxiv.org/pdf/1512.03385.pdf)에서도 이를 지지하는 연구 결과가 나타났다.
>[ResNet](https://arxiv.org/pdf/1512.03385.pdf)에서는 pure identity mapping이 bypassing path로 사용됐다.
>
>ImageNet이나 COCO object detection과 같은 까다로운 image recognition, localization, detection 분야에서 수많은 기록을 갈아치웠다.

<br/>
또한, 최근에는 1202-layer ResNet을 성공적으로 학습시킨 [stochastic depth](https://arxiv.org/pdf/1603.09382.pdf)가 제안됐다.
>학습 중에 무작위로 layer를 drop하는 방법을 통해 목표를 달성했다. 또한, 이 연구에서는 모든 layer가 필요한 것이 아니며, deep network에는 중복되거나 불필요한 layer가 많이 존재함을 강조한다. **DenseNet**은 여기서 부분적으로 영감을 받았다고 한다.

[pre-activation으로 구성한 ResNet](https://arxiv.org/pdf/1603.05027.pdf)은, 1000개 이상의 layer를 가진 최신 네트워크를 쉽게 학습할 수 있다.
>ResNet-v2라고도 불린다. ResNet의 skip-connection을 활용하는 최신 논문들은 대개 이 구조를 따른다.

<br/>
네트워크를 더 깊게 만드는 또 다른 approach는 네트워크 width를 늘리는 것이다.

<br/>
[GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)은 "Inception module"을 사용하여, 다양한 크기의 filter로 생성 된 feature-map들을 연결한다.

<br/>
[다른 연구](https://openreview.net/pdf?id=lx9l4r36gU2OVPy8Cv9g)에서는 wide generalized residual block을 가진 ResNet의 변형이 연구됐다.

<br/>
실제로 [Wide ResNet](https://arxiv.org/pdf/1605.07146.pdf)에서는 ResNet에서 각 layer의 filter 개수를 늘리는 것만으로도 depth가 충분하다면, 성능을 향상시킬 수 있음을 보였다.

<br/>
[FractalNet](https://arxiv.org/pdf/1605.07648.pdf)도 wide network structure를 사용하여, 여러 dataset에서 경쟁력있는 결과를 달성했다.

<br/>
**DenseNet**은 extremly deep 하거나 wide한 구조로부터 representational power를 끌어내는 대신, **feature의 재사용을 통해 네트워크의 잠재력을 활용**함으로써, 학습하기 쉬우면서도 효율적인 parameter를 가진 압축 모델을 만든다.
>다른 layer에서 학습한 feature-map들을 연결하면, 후속 layer의 input에서 variation이 증가하고 efficiency가 향상된다.
>
>**이는 DenseNet와 ResNet의 주요 차이점이다.**

<br/>
DenseNets는 다른 layer의 feature-map을 연결하는 Inception network에 비해, 더 간단하고 효율적이다.
>덜 간단하고 비효율적이라는 Inception network [[1](https://arxiv.org/pdf/1409.4842.pdf)] [[2](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)]

<br/>
주목할만한 또 다른 혁신적인 네트워크 구조가 있다.

<br/>
[Network in Network(NIN)](https://arxiv.org/pdf/1312.4400.pdf)은 보다 복잡한 feature을 추출하기 위해, convolution layer의 filter에 micro MLP를 포함한다.

<br/>
[Deeply Supervised Network(DSN)](https://arxiv.org/pdf/1409.5185.pdf)에서 internal layer들은 auxiliary classifier(보조 분류기)에 의해 directly supervised 되며, 이에 따라 초반부의 layer에 전달되는 gradient가 강화된다.
>네트워크의 출력부에 위치한 classifier 외에도, 앞쪽에 배치된 auxiliary classifier에서도 gradient가 전달된다.

Ladder Network는 autoencoder에 lateral connection을 도입하여, semi-supervised learning에 대한 인상적인 성능을 보였다.
>Ladder Network [[1](https://arxiv.org/pdf/1507.02672.pdf)] [[2](https://arxiv.org/pdf/1511.06430.pdf)]

<br/>
[Deeply-Fused Net(DFN)](https://arxiv.org/pdf/1605.07716.pdf)에서는 different base network의 intermediate layer를 결합하여, information flow를 향상시키는 방법을 제안했다.

<br/>
또한, reconstruction loss를 minimize하는 방식으로 네트워크를 확장하는 것이 image classification model의 성능을 향상시킨다는 연구도 있다.
>[그 연구](https://arxiv.org/pdf/1606.06582.pdf)

---
## 3. DenseNets
Convolutional network를 통과하는 single image $$x_0$$를 고려하자.

<br/>
네트워크는 $$L$$개의 layer로 구성되며, 각 layer는 non-linear transformation $$H_{\ell}(\cdot)$$를 포함한다.
>$$\ell$$은 layer의 index다.

<br/>
여기서 $$H_{\ell}(\cdot)$$는 [BN](https://arxiv.org/pdf/1502.03167.pdf), ReLU, Pooling, Convolution과 같은 연산의 복합 함수가 될 수 있다.

<br/>
또한, $$\ell^{th}$$ layer의 출력을 $$x_{\ell}$$로 표시한다.

<br/>
### ResNets
Traditional convolutional feed-forward network는 $$\ell^{th}$$ layer의 출력을 $$(\ell+1)^{th}$$ layer의 입력으로 연결하여, 다음과 같은 layer transition을 발생시킨다.
>**$$x_{\ell} = H_{\ell}(x_{\ell-1})$$**

<br/>
[ResNet](https://arxiv.org/pdf/1512.03385.pdf)은 identity function으로 non-linear transformation을 우회하는 skip-connection을 추가한다. (**Eqn.1** 참조)

<br/>
>**Eqn.1**
>
>**$$x_{\ell} = H_{\ell}(x_{\ell-1})+x_{\ell-1}$$**

<br/>
ResNets의 장점은, identity function을 통해 gradient가 후반 layer에서 전반 layer로 직접 흐를 수 있다는 것이다.

<br/>
그러나 identity function과 $$H_{\ell}$$의 출력이 summation으로 결합되어, 네트워크의 information flow를 방해할 수 있다.

<br/>
### Dense connectivity
Layer 간의 information flow를 개선시키기 위한 새로운 connectivity pattern을 제안한다.
>모든 layer가 모든 후속 layer로의 direct connection을 갖는다. (**Fig.1** 참조)

<br/>
결과적으로, $$\ell^{th}$$ layer는 모든 선행 layer의 feature-map인 $$x_0,...,x_{\ell-1}$$을 입력으로 받아들인다. (**Eqn.2** 참조)

<br/>
>**Eqn.2**
>
>**$$x_{\ell} = H_{\ell}([x_0,x_1,...,x_{\ell-1}])$$**

<br/>
여기서 $$[x_0,x_1,...,x_{\ell-1}]$$은 layer $$0,...,\ell-1$$에서 생성 된 feature-map의 연결을 나타낸다.

<br/>
이러한 dense connectivity로 인해, 이 네트워크 아키텍처를 Dense Convolutional Network(DenseNet)라고 부른다.

<br/>
구현의 편의를 위해, **Eqn.2**의 $$H_{\ell}(\cdot)$$에 들어가는 multiple input을 single tensor로 연결한다.

<br/>
### Composite function
[ResNet-v2](https://arxiv.org/pdf/1603.05027.pdf)에 따라, $$H_{\ell}(\cdot)$$는 세 개의 연속 연산으로 이루어진 복합 함수로 정의된다.
- [BN](https://arxiv.org/pdf/1502.03167.pdf), ReLU, 3x3 conv layer가 뒤따르는 복합 함수다.

<br/>
### Pooling layers
**Eqn.2**에서 사용된 concatenation operation은 feature-map의 크기가 변경되면 수행할 수 없지만, convolutional network에는 feature-map의 크기를 변경하는 down-sampling layer가 필수로 사용된다.

<br/>
따라서 down-sampling의 용이성을 위해, 네트워크를 multiple dense block으로 나눈다. (**Fig.2** 참조)

<br/>
![Fig.2](/blog/images/DenseNet, Fig.2(removed).png )
>**Fig.2** <br/>A deep DenseNet with three dense blocks.
>
>인접한 두 block 사이의 layer를 **transition layer**라고 하며, convolution과 pooling을 통해 feature-map의 크기를 변경한다.

<br/>
실험에 사용 된 transition layer는 [BN](https://arxiv.org/pdf/1502.03167.pdf)과 1x1 conv layer, 2x2 avg pooling layer가 뒤따르는 형태로 구성된다.

<br/>
### Growth rate
각 함수 $$H_{\ell}$$이 $$k$$개의 feature-map을 생성한다면, $$\ell^{th}$$ layer는 $$k_0 + k\times (\ell-1)$$개의 feature-map을 입력으로 가진다.
>$$k_0$$은 해당 dense block의 input feature-map 개수다.

<br/>
DenseNet과 기존의 네트워크 구조의 중요한 차이점은, very narrow layer(e.g. $$k = 12$$)을 가질 수 있다는 것이다.

<br/>
여기서 hyperparameter $$k$$를 네트워크의 **growth rate**라고 한다.

<br/>
4장에서는 상대적으로 작은 growth rate로도 state-of-the-art 성능을 얻기에 충분하다는 것을 보여준다.

<br/>
이러한 효과에 대한 한 가지 이유로는, 각 layer들이 block 내의 모든 이전 feature-map에 접근함에 따라, 네트워크의 "collective knowledge"에 액세스 된다는 것이다.

<br/>
Feature-map을 네트워크의 global state로 볼 수 있으며, 각 layer는 각자의 $$k$$ feature-map에 이 state를 추가한다.

<br/>
Growth rate는 각 layer가 global state에 기여하는 new information의 양을 조절한다.

<br/>
한 번 쓰여진 global state는 네트워크의 어디에서나 액세스 할 수 있으며, 기존의 네트워크 아키텍처와 달리 layer-to-layer로 복제할 필요가 없다.
>Concatenate로 연결되어 있기 때문이다.

<br/>
### Bottleneck layers
각 layer는 출력으로 $$k$$개의 feature-map만 생성하지만, 입력은 일반적으로 더 많은 feature-map으로 이루어져 있다.

<br/>
[ResNet](https://arxiv.org/pdf/1512.03385.pdf)과 [Inception](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)에서는 각 3x3 convolution 전에 1x1 convolution을 bottleneck layer로 도입하여, 입력 feature-map의 개수를 줄이고 계산 효율을 향상시킬 수 있음을 알 수 있다.

<br/>
이 디자인은 DenseNet에 특히 효과적이며, 이러한 bottleneck layer를 이용한다. 즉, BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)으로 이루어진 $$H_{\ell}$$을 이용하며, 이를 DenseNet-B 라고 칭한다.

<br/>
실험에서는 각각의 1x1 convolution이 $$4k$$개의 feature-map을 생성하도록 한다.

<br/>
### Compression
모델을 보다 소형으로 만들기 위해, transition layer에서 feature-map의 개수를 줄일 수 있다.

<br/>
Dense block이 $$m$$개의 feature-map을 포함하는 경우, 뒤따르는 transition layer에서 출력 feature-map을 $$\lfloor{\theta m}$$개 생성한다.

<br/>
여기서 $$0\lt \theta \lt 1$$은 **compression factor**라고 한다.
>$$\theta = 1$$ 인 경우, transition layer의 feature-map 개수는 변경되지 않는다.

<br/>
$$\theta \lt 1$$ 인 DenseNet을 DenseNet-C라고 칭하며, 실험에서는 $$\theta = 0.5$$로 설정한다.

<br/>
Bottleneck layer와 $$\theta \lt 1$$ 인 transition layer를 모두 사용하는 모델을 DenseNet-BC라고 칭한다.

<br/>
### Implementation Details
ImageNet을 제외한 모든 dataset에 대한 실험에는, 각각 동일한 수의 layer를 가진 3개의 dense block으로 구성 된 DenseNet을 사용한다.

- 첫 번째 dense block에 들어가기 전에, input image를 입력으로 하며, 16개(DenseNet-BC는 growth rate의 2배)의 feature-map을 출력으로 하는 convolution이 수행된다.

- Kernel size가 3x3인 conv layer의 경우, feature-map의 크기를 고정하기 위해 zero-padding을 사용한다.

- 연속되는 dense block 사이에는 1x1 convolution과 2x2 average pooling이 뒤따라오는 transition layer를 사용한다.

- 마지막 dense block의 끝에는 global average pooling 후에 softmax classifier가 뒤따른다.

- 3개의 dense block에서 feature-map 크기는 각각 32x32, 16x16, 8x8이다.

<br/>
실험에서 사용한 configuration은 다음과 같다. Hyperparameter인 $$L$$과 $$k$$는 각각 layer 개수와 growth rate에 해당한다.

<br/>
**Basic DenseNet**
- $${L = 40, k = 12}$$.

- $${L = 100, k = 12}$$.

- $${L = 100, k = 24}$$.

<br/>
**DenseNet-BC**
- $${L = 100, k = 12}$$.

- $${L = 250, k = 24}$$.

- $${L = 190, k = 40}$$.

<br/>
ImageNet에 대한 실험에서는 224x224 크기의 input image에 4개의 dense block이 있는 DenseNet-BC 구조를 사용한다.

<br/>
초기 conv layer는 stride가 2인 7x7 크기의 $$2k$$ convolution으로 구성되며, 다른 모든 layer의 feature-map 개수는 $$k$$를 따른다.

<br/>
ImageNet에 사용된 정확한 네트워크 구성은 **Table.1**을 참조하자.

<br/>
![Table.1](/blog/images/DenseNet, Table.1(removed).png )
>**Table.1** <br/>DenseNet architectures for ImageNet.
>
>모든 네트워크에서 growth rate는 $$k = 32$$를 사용하며, 표시된 conv layer는 모두 BN-ReLU-Conv로 이루어진 동작에 해당한다.

---
## 4. Experiments
여러 benchmark dataset에서 DenseNet의 효과를 실험적으로 입증하고, state-of-the-art 특히 ResNet 및 ResNet의 변형 모델들과 성능을 비교한다.

<br/>
### 4.1. Datasets

### CIFAR
두 개의 [CIFAR dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- 32x32 pixel의 color natural image로 구성된다.

- CIFAR-10(C10)과 CIFAR-100(C100)은 각각 10개, 100개 class로 구성된다.

- 제공되는 training, test set은 각각 50,000개와 10,000개의 이미지가 포함되어 있으며, 실험에서는 5,000개의 training data를 holdout validation set으로 사용한다.

- 두 dataset에는 널리 사용되는 standard data augmentation 기법을 사용하며, 이를 적용한 실험 결과는 dataset 이름 끝에 "+"라고 표시 된다.
>[ResNet](https://arxiv.org/pdf/1512.03385.pdf), [NIN](https://arxiv.org/pdf/1312.4400.pdf), [stochastic depth](https://arxiv.org/pdf/1603.09382.pdf) 등에서 사용한 mirroring/shifting 기법 등을 참조하고 있다.

- 전처리를 위해 channel mean과 standard deviation으로 normalize한다.

- 수행에는 50,000개의 training set을 모두 사용하고, 학습 종료 시에 test error를 측정한다.

<br/>
### SVHN
[Street View House Numbers(SVHN) dataset](http://ufldl.stanford.edu/housenumbers/)
- 32x32 color number image로 구성된다.

- Training, test set은 각각 73,257개와 26,032개의 이미지가 포함되어 있으며, additional traning을 위한 531,131개의 이미지도 제공된다.

- 여러 연구에서 채택한 관행에 따라 data augmentation을 수행하지 않은 training set만을 사용하며, 이 중 6,000개의 이미지는 validation set으로 분리 된다.
>[Maxout](https://arxiv.org/pdf/1302.4389.pdf), [NIN](https://arxiv.org/pdf/1312.4400.pdf) 등을 참조한다.

- 학습 중 validation error가 가장 낮은 모델을 택하여 test error를 측정한다.

- [Wide ResNet](https://arxiv.org/pdf/1605.07146.pdf)에 따라 pixel 값을 255로 나누어 [0, 1] 범위로 전처리한다.

<br/>
### ImageNet
[ILSVRC 2012 classification dataset](http://image-net.org/challenges/LSVRC/2012/index)
- 1,000개 class로 구성된다.

- Training, validation set은 각각 약 120만개와 50,000개로 구성된다.

- [ResNet](https://arxiv.org/pdf/1512.03385.pdf)에서와 동일한 data augmentation 기법을 적용하며, test 시에는 224x224 크기로 single-crop 또는 10-crop을 적용한다.

- [ResNet](https://arxiv.org/pdf/1512.03385.pdf), [ResNet-v2](https://arxiv.org/pdf/1603.05027.pdf), [stochastic depth](https://arxiv.org/pdf/1603.09382.pdf)에 따라 validation set에 대한 classification error를 측정한다.

<br/>
### 4.2. Training
모든 네트워크는 stochastic gradient descent(SGD)로 학습된다.

<br/>
**CIFAR / SVHN**
- Batch size를 64로 하고, 각각 300 / 40회의 epoch 동안 학습을 진행한다.

- 초기 learning rate는 0.1이며, 총 epoch의 50%와 75%에 도달하는 시점에 10으로 나눈다.

<br/>
**ImageNet**
- Batch size를 256으로 하고, 90회의 epoch 동안 학습을 진행한다.

- 초기 learning rate는 0.1이며, 30, 60번 째 epoch에서 10으로 나눈다.

<br/>
**Common**
- [Training and investigating Residual Nets](http://torch.ch/blog/2016/02/04/resnets.html)에 따라, 10e-4의 weight decay와 dampening을 제외한 [Nesterov momentum](https://www.cs.toronto.edu/~fritz/absps/momentum.pdf)을 0.9로 사용한다.

- [He initialization](https://arxiv.org/pdf/1502.01852.pdf)으로 weight 초기화

<br/>
C10, C100, SVHN과 같이 data augmentation을 제외한 실험에서는 첫 번째를 제외한 각 conv layer의 뒤에 rate가 0.2인 [dropout layer](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)를 추가한다.

<br/>
Test error는 각 task 및 model setting에 대해 한 번씩 측정됐다.

<br/>
DenseNet의 naive implementation에는 memory inefficiency가 포함될 수 있다.
>GPU의 memory consumption을 줄이려면, DenseNet의 memory-efficient implementation에 대한 [technical report](https://arxiv.org/pdf/1707.06990.pdf)를 참조하자.

<br/>
### 4.3. Classification Results on CIFAR and SVHN
이 장에서는 CIFAR 및 SVHN에 대한 DenseNet 실험 결과를 알아본다.

<br/>
다양한 depth $$L$$과 growth $$k$$에 대해 실험했으며, 주요 결과는 **Table.2**를 참조하자.

<br/>
![Table.2](/blog/images/DenseNet, Table.2(removed).png )
>**Table.2** <br/>Error rates(%) on CIFAR and SVHN datasets.
>
>모든 method의 성능을 뛰어넘은 결과는 bold로 표시됐으며, 전체에서 가장 좋은 결과는 파란색으로 표시된다.
>
>**"+"**로 표시된 열은 standard data augmentation(translation and/or mirroring)을 적용한 경우다.
>
>**"*"**로 표시된 결과는 직접 수행해서 얻은 결과를 나타낸다.
>
>Data augmentation을 적용하지 않은 C10, C100, SVHN의 경우에는 dropout을 포함한다.
>
>DenseNet은 ResNet보다 적은 수의 parameter를 사용하면서도 성능이 좋다.
>
>Data augmentation을 적용하지 않은 경우에는 DenseNet의 성능이 큰 차이로 앞선다.

<br/>
### Accuracy
가장 눈에 띄는 결과는 **Table.2**의 맨 아래에 있는 $$L = 190$$, $$k = 40$$ 인 DenseNet-BC의 성능이다. 모든 CIFAR dataset에서 기존의 state-of-the-art 성능을 능가하고 있다.

<br/>
C10+과 C100+에서는 각각 3.46%, 7.18%의 오류율을 얻었으며, 이는 [Wide ResNet](https://arxiv.org/pdf/1605.07146.pdf)의 오류율보다 상당히 낮다.

<br/>
C10과 C100에 대한 최고 성능은 훨씬 고무적인 결과다. 둘 다 drop-path regularization을 사용한 [FractalNet](https://arxiv.org/pdf/1605.07648.pdf)보다 30%정도 낮은 오류율을 보였다.

<br/>
SVHN에서 Dropout을 사용한 $$L = 100$$, $$k = 24$$ 인 DenseNet도 [Wide ResNet](https://arxiv.org/pdf/1605.07146.pdf)의 최고 성능을 능가했지만, 250-layer DenseNet-BC는 더 낮은 depth 버전에 비해 성능을 더 향상시키지 않았다.

<br/>
이는 SVHN이 비교적 쉬운 작업이기 때문에 extremely deep model이 traning set에 overfitting 된 것으로 볼 수 있다.

<br/>
### Capacity
Compression 또는 bottle layer가 없으면, $$L$$과 $$k$$가 증가함에 따라 DenseNet의 성능이 향상되는 경향이 있다.

<br/>
이는 주로 model capacity의 증가로 인한 것이며, C10+와 C100+에 대한 결과에서 잘 설명된다.

<br/>
C10+의 경우, parameter의 개수가 1.0M에서 7.0M, 27.2M로 증가함에 따라, 오류가 5.24%에서 4.10%, 3.74 %로 떨어지며, C100+에서도 비슷한 경향이 관찰된다.

<br/>
이는 DenseNet이 bigger and deeper model의 향상된 representational power를 활용할 수 있음을 나타낸다.

<br/>
또한, 이는 residual network의 overfitting이나 optimization difficulty로부터 고통받지 않음을 나타낸다.

<br/>
### Parameter Efficiency
**Table.2**의 결과는 DenseNet이 다른 아키텍처(특히 ResNet)보다 더 효율적으로 parameter를 활용함을 나타낸다.

<br/>
Bottleneck과 dimension reduction 기능을 갖춘 transition layer를 포함한 DenseNet-BC는 특히 parameter-efficient 하다.

<br/>
예를 들어, 250-layer model에는 15.3M개의 paramter만 있지만, 30M개 이상의 parameter가 있는 [FractalNet](https://arxiv.org/pdf/1605.07648.pdf)이나 [Wide ResNet](https://arxiv.org/pdf/1605.07146.pdf)과 같은 다른 모델보다 일관되게 성능이 뛰어나다.

<br/>
또한 $$L = 100$$, $$k = 12$$ 인 DenseNet-BC는 1001-layer의 pre-activation ResNet보다 90%나 적은 parameter를 사용하면서도 비슷한 성능을 달성한다.
>**DenseNet-BC** vs **1001-layer ResNet-v2**
>
>C10+ : **4.51%** vs **4.62%**
>
>C100+ : **22.27%** vs **22.71%**

<br/>
**Fig.4**의 오른쪽 그래프는 C10+에 대한 두 네트워크의 training loss와 test error를 보여준다.

<br/>
1001-layer ResNet은 더 낮은 training loss로 수렴하지만, 유사한 test error로 수렴된다. 아래에서는 이를 더 자세히 분석한다.

<br/>
### Overfitting
Parameter를 효율적으로 사용하는 것에 따른 부작용 중 하나는, DenseNet이 overfitting되는 경향이 적다는 것이다.
>편향(Bias)에 관련된 내용이다.

<br/>
Data augmentation을 적용하지 않은 경우, 이전 모델들에 비해 DenseNet의 성능 개선이 특히 두드러진다.

<br/>
C10에서는 7.33%에서 5.19%로 약 29%의 상대적 오류 감소를 보였으며, C100에서는 28.20%에서 19.64%로 약 30%의 상대적 오류 감소를 보였다.

<br/>
실험에서는 단일 환경에서 잠재적인 overfitting을 관찰했다.
- C10에서는 $$k = 12$$를 $$k = 24$$로 높여 parameter를 4배 증가시켰고, 그 결과 오류율이 5.77%에서 5.83%로 약간 증가했다.

<br/>
반면, DenseNet-BC의 bottleneck 및 compression layer는 이러한 경향에 효과적으로 대처하는 것으로 나타났다.

<br/>
### 4.4. Classification Results on ImageNet
이 장에서는 ImageNet classification task에 대한 DenseNet-BC의 성능을 state-of-the-art ResNet 아키텍처와 비교 평가한다.
>마찬가지로 다양한 depth $$L$$과 growth $$k$$에 대해 실험한 결과를 보여준다.

<br/>
두 아키텍처를 공정하게 비교하기 위해, 공개적으로 사용 가능한 [Torch implementation for ResNet](https://github.com/facebook/fb.resnet.torch)을 채택하여 preprocessing 및 optimization setting의 차이와 같은 모든 요소를 제거했다.

<br/>
실험에는 단순히 ResNet을 DenseNet-BC로 교체했으며, 모든 실험 세팅은 ResNet에 사용 된 설정과 동일하게 유지한다.

<br/>
ImageNet에 대한 DenseNet의 single-crop 및 10-crop validation error는 **Table.3**을 참조하자.

<br/>
![Table.3](/blog/images/DenseNet, Table.3(removed).png )
>**Table.3** <br/>The top-1 and top-5 error rates on the ImageNet validation set, with single-crop / 10-crop testing.

<br/>
**Fig.3**은 DenseNet과 ResNet의 parameter 및 FLOP의 수에 따른 single-crop top-1 validation error를 보여준다.

<br/>
![Fig.3](/blog/images/DenseNet, Fig.3(removed).png )
>**Fig.3** <br/>Comparison of the DenseNets and ResNets top-1 error rates(single-crop testing) on the ImageNet validation dataset.
>
>A function of learned parameters (left) and FLOPs during test-time (right).

<br/>
**Fig.3**의 결과는 DenseNet이 SOTA ResNet과 동등한 성능을 보이며, 비슷한 성능의 달성에 필요한 parameter의 수와 계산량이 훨씬 적음을 보여준다.

<br/>
예를 들어, 20M개의 parameter가 있는 DenseNet-201은 40M개 이상의 parameter가 있는 101-layer ResNet과 유사한 validation error를 보인다.

<br/>
**Fig.3**의 오른쪽 그래프에서도 유사한 경향을 관찰할 수 있으며, FLOP의 수에 대한 validation error를 보인다.
>ResNet-50과 유사한 양의 parameter와 계산량을 갖는 DenseNet은, 비용이 두 배 수준인 ResNet-101과 동등한 성능을 보인다.

<br/>
실험에 사용된 hyperparameter setting은 ResNet에 최적화 된 것이다. 즉, DenseNet에는 최적화 되지 않은 상태로 실험한 결과이다. 따라서, 보다 광범위한 hyperparameter 탐색을 통해 ImageNet에 대한 DenseNet의 성능을 추가로 향상시킬 수 있다.

---
## 5. Discussion
표면적으로 DenseNet은 ResNet과 매우 유사하다.
- **Eqn.2**는 $$H_{\ell}(\cdot)$$에 대한 입력의 summation 대신 concatenation한다는 점에서만 **Eqn.1**과 다르다.

<br/>
하지만, 이러한 작은 차이로부터 두 네트워크 아키텍처의 동작이 크게 달라진다.

<br/>
### Model compactness
입력을 concatenation 함으로써, 모든 DenseNet layer에서 학습된 feature-map을 모든 후속 layer에서 액세스 할 수 있게 된다.

<br/>
이를 통해 네트워크 전체에서 feature reuse가 촉진되며, 보다 compact한 모델로 이어진다.

<br/>
**Fig.4**에서는 DenseNet의 모든 변형과, 유사 성능의 ResNet의 parameter efficiency를 비교한 실험 결과를 보여준다.

<br/>
![Fig.4](/blog/images/DenseNet, Fig.4(removed).png )
>**Fig.4** <br/>C10+에 대한 parameter efficiency 비교
>
>**(Left)**  DenseNet 변형 간의  비교
>
>**(Middle)** DenseNet-BC와 pre-activation ResNet 간의 비교
>
>**(Right)** 0.8M개의 parameter를 갖는 100-layer DenseNet과 10M개의 parameter를 갖는 1001-layer pre-activation ResNet의 training loss 및 test error 비교

<br/>
[AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)이나 [VGG](https://arxiv.org/pdf/1409.1556.pdf)과 같은 다른 대중적인 네트워크 아키텍처에 비해, pre-activation ResNet이 더 적은 수의 parameter를 사용하면서도 일반적으로 더 나은 성능을 보이기 때문에 pre-activation ResNet과 DenseNet($$k = 12$$)의 성능 비교 결과만 제공한다.

<br/>
결과를 요약하면 다음과 같다. DenseNet의 학습 설정은 이전 섹션과 동일하게 유지했다.
- DenseNet-BC가 DenseNet의 변형 중 parameter efficiency가 가장 좋다.

- DenseNet-BC이 ResNet과 유사한 성능을 달성하는데 필요한 parameter는 1/3에 불과하다.
>이는 Fig.3의 ImageNet에 대한 결과와 일치한다.

- Trainable parameter의 수가 0.8M인 DenseNet-BC가 10.2M개의 paramter를 사용하는 1001-layer pre-activation ResNet과 유사한 성능을 달성 할 수 있다.

<br/>
### Implicit Deep Supervision
DenseNet의 성능 개선에 대한 한 가지 이유는, 각 layer들이 shorter connection을 통해 loss function으로부터 추가적인 supervision이 이뤄진다는 것이다. 즉, DenseNet을 일종의 "deep supervision"으로 볼 수 있다.

<br/>
Deep supervision의 이점은, 모든 hidden layer에 classifier를 부착함으로써 intermediate layer들이 식별력 있는 feature를 익히도록 구성한 [Deeply-Supervised Net(DSN)](https://arxiv.org/pdf/1409.5185.pdf)에서 보였다.

<br/>
DenseNet도 내부적으로는 유사한 deep supervision을 포함하는 방식으로 수행된다.
- 네트워크의 끝에 위치한 single classifier는 최대 2~3개의 transition layer를 통해 모든 layer에 대한 direct supervision을 제공한다.

<br/>
그럼에도, DenseNet은 모든 layer간에 동일한 loss와 gradient를 공유하기 때문에 훨씬 덜 복잡하다.

<br/>
### Stochastic vs. deterministic connection
DenseNet과 [stochastic depth regulaization of ResNet](https://arxiv.org/pdf/1603.09382.pdf) 간에는 흥미로운 관계가 있다.

<br/>
Stochastic depth은 ResNet의 layer를 무작위로 drop하여, 주변 layer 간의 direct connection을 만든다.

<br/>
여기서 pooling layer는 drop에 포함되지 않기 때문에, 네트워크는 DenseNet과 비슷한 connectivity pattern을 갖는다.
>모든 intermediate layer가 임의로 drop 된 경우에는, 동일한 pooling layer간에 direct connection이 생길 가능성도 있다.

<br/>
두 방법이 완전히 다름에도 stochastic depth에 대한 DenseNet의 해석은, stochastic regularizer의 효과에 대한 insight를 제공할 수 있다.

<br/>
### Feature Reuse
네트워크 디자인 상, DenseNet의 layer들은 모든 선행 layer의 feature-map에 액세스 할 수 있다.
>때로는 transition layer를 통해 액세스하기도 함.

<br/>
이번엔 학습 된 네트워크의 경우에도 이러한 이점을 잘 취하는지 조사하는 실험을 진행한다.

<br/>
먼저 $$L = 40$$, $$k = 12$$인 DenseNet을 C10+에 대해 학습한다.

<br/>
Dense block 내의 각 conv layer $$\ell$$에 대해, layer $$s$$와의 connection에 할당 된 average absolute weight를 계산한다.

<br/>
**Fig.5**는 3개의 dense block 모두에 대한 heat-map을 보여준다.

<br/>
![Fig.5](/blog/images/DenseNet, Fig.5(removed).png )
>**Fig.5** <br/>The average absolute filter weights of convolutional layers in a trained DenseNet.
>
>Pixel $$(s, \ell)$$의 color는 dense block 내의 conv layer $$s$$와 $$\ell$$을 연결하는 weight의 average L1 norm으로 인코딩 한 것이다.
>>Feature-map의 개수에 따라 수행.
>
>검정색 사각형으로 highlight한 3개의 열은, 두 개의 transition layer와 classification layer에 해당한다.
>
>첫 번째 행은 dense block의 input layer에 연결 된 weight를 인코딩 한 것이다.

<br/>
**Fig.5**는 선행 layer들의 conv layer에 대한 dependency를 average absolute weight로 대신 보여준다.

<br/>
$$(\ell, s)$$ 위치의 pixel이 빨간색인 것은, $$\ell$$이 평균적으로 선행하는 s-layer들이 생성한 feature-map을 강력하게 사용한다는 것을 나타낸다.

<br/>
**Fig.5**로부터 몇 가지 현상을 관찰을 할 수 있다.

<br/>
- 모든 layer는 동일한 block 내의 많은 input에 weight를 분산시킨다.
>이는 매우 초반부의 layer에 의해 추출 된 feature가, 실제로 동일한 dense block 내의 전체 layer에 의해 직접 사용됨을 나타낸다.

- Transition layer의 weight 또한, 선행하는 dense block 내의 모든 layer에 weight를 분산시킨다.
>이는 information이 몇 개의 간접적인 방향을 통해 first-to-last layer로 이동 함을 나타낸다.

- 두 번째 및 세 번째 dense block 내의 layer는 transition layer의 출력(첫 번째 행)에 최소한의 weight만 일관되게 할당한다.
>Transition layer가 중복된 feature(평균 weight가 낮은)를 많이 출력한다는 것을 나타내며, 이는 이러한 출력이 compression되는 DenseNet-BC의 결과와 정확하게 일치한다.

- 맨 오른쪽에 표시된 final classification layer도 전체 dense block의 weight를 사용하긴 하지만, 최종 feature-map에 집중하는 것으로 보인다.
>이는 네트워크에서 늦게 생성되는 high-level feature들이 더 있을 수 있음을 나타낸다.

---
## 6. Conclusion
이 논문에서는 새로운 convolutional network architecture인 Dense Convolutional Network(DenseNet)를 제안한다.

<br/>
이 네트워크는 동일한 feature-map size를 가진 두 layer 사이에 direct connection을 도입한다.

<br/>
DenseNet은 자연스럽게 수백 개의 layer로 확장되는 반면, optimization difficulty는 없음을 보여줬다.

<br/>
실험에서는 DenseNet이 overfitting이나 성능 저하의 징후없이, parameter가 증가할수록 일관되게 성능이 향상하는 경향이 있었다.

<br/>
여러 설정에 대해, 경쟁이 치열한 여러 dataset에서 state-of-the-art 성능을 달성했다.

<br/>
또한, DenseNet은 SOTA 성능을 달성하기 위해, 훨씬 적은 수의 parameter와 계산량을 요구한다.

<br/>
본 연구에서는 ResNet에 최적화 된 hyperparameter를 채택했으므로, hyperparameter 및 learning rate scheduling을 보다 세부적으로 조정하면 DenseNet을 더욱 향상시킬 수 있을거라 생각된다.

<br/>
간단한 connectivity rule을 따르는 동안, DenseNet은 identity mapping과 deep supervision 및 diversified depth의 특성을 자연스럽게 통합한다.
>Identity mapping - [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
>
>Deep supervision - [DSN](https://arxiv.org/pdf/1409.5185.pdf)
>
>Diversified depth - [Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)

<br/>
이는 feature reuse를 네트워크 전체에서 가능하게 해주며, 결과적으로 더 compact한 모델을 학습 할 수 있게 된다. 또한, 실험에 따르면 모델의 성능도 올라간다.

<br/>
Compact한 internal representation과 감소 된 feature 중복성으로 인해, DenseNet은 convolution feature를 기반으로 하는 다양한 computer vision task에서 적합한 feature extractor가 될 것이다.

<br/>
향후에는 feature transfer with DenseNet에 대한 연구를 진행할 예정이다.

---

<br/>
<br/>
{% include disqus.html %}
