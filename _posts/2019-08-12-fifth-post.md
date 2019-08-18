---
title: "(Xception) Xception: Deep Learning with Depthwise Separable Convolutions 번역 및 추가 설명과 Keras 구현"
date: 2019-08-12 16:29:11 -0400
tags: AI ComputerVision Paper Xception SepConv2D
categories:
  - Paper
toc: true
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Paper Information

CHOLLET, François. **"Xception: Deep learning with depthwise separable convolutions"**. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2017. p. 1251-1258. 2017.
> a.k.a. [Xception paper](https://arxiv.org/pdf/1610.02357.pdf)


---
## Abstract
이 논문에서는 Inception module을, 일반적인 convolution과 **depthwise separable convolution**의 중간 단계로 해석하는 것을 제시한다.
>**Depthwise separable convolution**은 depthwise convolution 뒤에 pointwise convolution이 뒤따르는 형태이다.

<br/>
이러한 관점에서 **depthwise separable convolution**은 가장 많은 수의 tower가 있는 inception module로 볼 수 있다.
>Tower는 inception module 내부에 존재하는 conv layer와 pooling layer 등을 말한다. 즉,  **depthwise separable convolution**은 확장 된 버전의 Inception module로 볼 수 있다.

<br/>
이러한 통찰은 Inception 기반의 새로운 CNN 구조를 제안하는데 기여됐다. 새로운 구조에서는 Inception module들이 **depthwise separable convolution**으로 대체된다.

<br/>
**Xception**이라 부르는 이 아키텍처는, ImageNet dataset에서 **Inception-V3**보다 성능이 약간 좋았으며, 3.5억개의 이미지와 17000개의 class로 구성된 dataset에서는 **Inception-V3**보다 훨씬 뛰어난 성능을 보였다.

<br/>
**Xception** 구조는 **Inception-V3**과 동일한 수의 parameter를 가지므로, 성능의 향상은 capacity의 증가가 아닌, 모델 parameter의 효율적인 사용으로부터 얻어진 것이다.

---
## 1. Introduction
CNN은 최근 computer vision 분야에서 가장 중요한 알고리즘으로 알려져 있으며, 이를 설계하기 위한 기법의 개발에 상당한 관심을 기울이고 있다.

<br/>
CNN 설계의 역사는 feature extraction을 위한 간단한 **convolution**과 spatial sub-sampling을 위한 max-pooling의 stack으로 이루어진 [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) 스타일의 모델로 시작됐다.

<br/>
이 아이디어는 [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)에서 개선됐다. Max-pooling 사이에는 convolution이 여러 번 반복되어, 네트워크가 모든 spatial scale에 대해 풍부한 feature를 학습 할 수 있게 됐다.

<br/>
이후로는 이 스타일의 네트워크를 점점 더 깊게하는 경향이 있었으며, 대부분 매년 열리는 ILSVRC competition에 의해 주도됐다.
>2013년의 [Zeiler와 Fergus의 연구](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)와 2014년의 [VGG](https://arxiv.org/pdf/1409.1556.pdf)부터 깊어지기 시작했다.

<br/>
이 시점에서 Szegedy 등이 도입한 새로운 스타일인 Inception 구조가 등장했다.
>Inception 자체는 이전의 구조인 [Network-In-Network](https://arxiv.org/pdf/1312.4400.pdf)에서 영감을 받았다.
>
>2014년의 [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)(Inception V1)을 시작으로, [Inception-v2](https://arxiv.org/pdf/1502.03167.pdf)와 [Inception-v3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf), 가장 최근에는 [Inception-ResNet](https://arxiv.org/pdf/1602.07261.pdf)으로 개선됐다.

<br/>
Inception은 처음 소개 된 이후로 [ImageNet dataset](https://arxiv.org/pdf/1409.0575.pdf)과 Google에서 사용하는 internal dataset(JFT)에 대해 가장 성능이 우수한 모델 군 중 하나였다.
>특히 [JFT](https://arxiv.org/pdf/1503.02531.pdf)에서 성능이 좋았다고 함.

<br/>
Inception 스타일 모델의 핵심적인 building block은, 여러 버전이 존재하는 Inception module이다. Fig.1에서는 [Inception-v3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)에서 볼 수 있는 표준 형태의 Inception module을 보여준다.

<br/>
![Fig.1](/blog/images/Xception, Fig.1(removed).png )
>**Fig.1** <br/>A canonical Inception module. (Inception-v3)
>
>Inception 모델은 이러한 모듈들의 stack으로 이해할 수 있다.

<br/>
Inception module은 개념적으로 convolution과 유사하지만, 실험에서는 적은 parameter로도 풍부한 표현을 학습할 수 있는 것으로 나타났다.

<br/>
이들은 어떻게 작동하며, 일반적인 convolution과 어떻게 다른 것일까? Inception 이후에는 어떤 디자인 전략이 나올까?

<br/>
### 1.1. The Inception hypothesis
Conv layer는 2개의 spatial dimension(width, height)과 channel dimension로 이루어진 3D 공간에 대한 filter를 학습하려고 시도한다.
>따라서 single convolution kernel은 **cross-channel correlation**과 **spatial correlation**을 동시에 mapping하는 작업을 수행한다.

<br/>
Inception module의 기본 아이디어는 **cross-channel correlation**과 **spatial correlation**을 독립적으로 볼 수 있도록 일련의 작업을 명시적으로 분리함으로써, 이 프로세스를보다 쉽고 효율적으로 만드는 것이다.
>일반적인 Inception module의 경우, 우선 **1x1 convolution**으로 **cross-correlation**을 보고, input보다 작은 3~4개의 분리 된 공간에 mapping한다. 그 다음 보다 작아진 3D 공간에 **3x3 혹은 5x5 convolution**을 수행하여 **spatial correlation**을 mapping한다. (**Fig.1** 참조)

<br/>
실제로 Inception의 핵심 가설은, **cross-channel correlation**과 **spatial correlation**이 함께 매핑되지 않도록 충분히 분리하는 것이 좋을 것이란 가설이다.

<br/>
Convolution의 size를 하나만 사용하고, average pooling을 포함하지 않는 단순화 된 버전의 Inception module을 고려하자. (**Fig.2** 참조)

<br/>
![Fig.2](/blog/images/Xception, Fig.2(removed).png )
>**Fig.2** <br/>A simplified Inception module.

<br/>
이 Inception module은 **large 1x1 convolution**을 수행하고, output channel들의 겹치지 않는 부분(segment) 대한 **spatial convolution**이 오는 형태로 reformulation 될 수 있다. (**Fig.3** 참조)

<br/>
![Fig.3](/blog/images/Xception, Fig.3(removed).png )
>**Fig.3** <br/>A strictly equivalent reformulation of the simplified Inception module.

<br/>
이 관찰은 자연스럽게 다음의 의문을 제기한다.
- Partition 내의 segment 개수나 크기가 어떤 영향을 갖는가?

- Inception의 가설보다 훨씬 강력한 가설을 세우고, **cross-channel correlation**과 **spatial correlation**을 완전히 분리시켜 mapping 할 수 있다고 가정하는 것이 합리적인가?

<br/>
### 1.2 The continuum between convolutions and separable convolutions
1.1절의 강력한 가설에 기반한 **extreme version**의 Inception module은, 먼저 1x1 convolution으로 **cross-channel correlation**을 mapping하고, 모든 output channel들의 **spatial correlation**을 따로 mapping한다. (**Fig.4** 참조)

<br/>
![Fig.4](/blog/images/Xception, Fig.4(removed).png )
>**Fig.4** <br/>An “extreme” version of our Inception module, with one
spatial convolution per output channel of the 1x1 convolution.

<br/>
저자들은 이러한 extreme form의 Inception module이 **depthwise separable convolution**와 거의 동일하다고 언급한다.
>**Depthwise separable convolution**은 [2014년에 발표 된 연구](https://www.di.ens.fr/data/publications/papers/phd_sifre.pdf)에서 설계한 네트워크에 사용됐었으며, [TensorFlow framework](https://arxiv.org/pdf/1603.04467.pdf)에 포함 된 이후로 더 대중화 됐다.

<br/>
**Depthwise separable convolution**을 TensorFlow나 Keras와 같은 deep learning framework에서는 일반적으로 '**separable convolution**'이라 부르며, **depthwise convolution**으로 구성된다.
>즉, input의 **각 channel에 대해 독립적으로 spatial convolution이 수행**되고, 그 뒤로는 **pointwise convolution**이 이어지는 형태다.
>
>**Pointwise convolution(1x1 convolution)**은 **depthwise convolution**의 output channel을 새로운 channel space에 projection하는 작업이다.

<br/>
이것을 **spatially separable convolution**과 혼동해서는 안 된다.
>Image processing 커뮤니티에서는 **Spatially separable convolution**도 "**separable convolution**"이라 부른다고 한다.

<br/>
**Extreme version**의 Inception module과 **separable convolution** 간의 두 가지 차이점은 다음과 같다.
- **Operation의 순서**
>**Inception**에서는 **1x1 convolution**을 먼저 수행하는 반면, TensorFlow에서와 같이 일반적으로 구현된 **depthwise separable convolution**은 **channel-wise spatial convolution**을 먼저 수행한 뒤에 **1x1 convolution**을 수행한다.

- **첫 번째 operation 뒤의 non-linearity 여부**
>**Inception**에서는 두 operation 모두 non-linearity로 ReLU가 뒤따르는 반면, **separable convolution**은 일반적으로 non-linearity 없이 구현된다.

<br/>
이 둘을 Keras code로 비교를 하면 다음과 같다.
``` python
## Extreme version of an Inception module
x = Conv2D(32, (1, 1), use_bias=False)(x)
x = DepthwiseConv2D(3, activation='relu')(x)

## Separable convolution
x = DepthwiseConv2D(3, use_bias=False)(x)
x = Conv2D(32, (1, 1), activation='relu')(x)

```
>Conv2D에서 (1, 1) convolution filter를 사용하면 pointwise convolution이 된다.
>
>본 절의 후반에 있는 코드에서 설명하겠지만, 선행하는 layer에서는 bias를 사용하지 않는다.

<br/>
첫 번째 차이점은 중요하지 않지만, 두 번째 차이점은 중요할 수 있으므로 실험 파트에서 조사한다. (**Fig.10** 참조)
>저자들은 이러한 operation들이 stacked setting에서 사용되기 때문에, 첫 번째 차이점은 중요하지 않다고 주장한다.

<br/>
또한, 일반 Inception module과 depthwise separable convolution 사이의 intermediate formulation도 가능함에 주목한다.
>사실상, 이들 사이에는 별개의 스펙트럼이 존재한다.
>
>Depthwise separable convolution은 spatial convolution 수행에 사용 되는 **independent channel-space segment**의 개수에 의해 parameterize 된다. 즉, input channel의 수에 따라 parameter 개수가 결정된다는 말이다.

<br/>
이 스펙트럼을 양 극단으로 구분하면 다음과 같다.
- **일반적인 convolution**
>1x1 convolution이 선행하는 경우.
>
>일반적인 convolution은 전체 channel을 하나의 segment로 다루기 때문에, single-segment case에 해당한다.

- **depthwise separable convolution**
>Channel 당 하나의 segment로 다룬다.

<br/>
**Inception module**은 수백 개의 channel을 3~4개의 segment로 다루기 때문에, 둘 사이에 위치한다.

<br/>
저자들은 이러한 관측으로부터, **Inception module**을 **depthwise separable convolution**으로 대체함으로써 Inception 계열의 구조를 개선하는 것이 가능할 것이라 제안한다.
>즉, depthwise separable convolution을 stacking한 모델이 된다.

<br/>
이는 TensorFlow에서 사용할 수 있는 효율적인 depthwise separable convolution으로 실현되며, Keras에서도 **SeparableConv2D**라는 이름의 layer로 제공된다.

<br/>
아까 separable convolution의 선행하는 layer가 bias를 사용하지 않는다고 언급했는데, Kears code로 정확히 알아보자. 우선 **separable convolution layer**를 사용하는 방법은 다음과 같다.
``` python
# Separable Convolution
x = DepthwiseConv2D(3, use_bias=False)(x)
x = Conv2D(32, (1, 1), activation='relu')(x)

# Separable Convolution using SeparableConv2D
x = SeparableConv2D(32, 3, activation='relu')(x)
```
>Random seed와 weight를 고정하여, 둘의 동작이 완전히 동일함을 직접 확인했다.

<br/>
이번에는 선행하는 layer가 bias를 사용하지 않는다는 것을 weight dimension으로 확인해보자. d우선 다음과 같이 간단한 모델을 구현한다.
``` python
model_input = Input( shape=input_shape )

x = Conv2D(16, 3, activation='relu')(model_input)
x = SeparableConv2D(24, 3, activation='relu')(x)
x = SeparableConv2D(32, 3, activation='relu')(x)
x = SeparableConv2D(32, 3, activation='relu')(x)
x = SeparableConv2D(48, 3, activation='relu')(x)
x = SeparableConv2D(64, 3, activation='relu')(x)

x = GlobalAveragePooling2D()(x)
x = Dense(10, activation='softmax')(x)

model = Model(model_input, x)
```
>목적 달성만을 위해 기능 활용을 최소화 한 모델이다.

<br/>
위의 모델에서 **SeparableConv2D**의 weight를 찍어서 확인해보면 다음과 같다.

<br/>
![Extra.1](/blog/images/Xception, Extra.1(removed).png )
>SeparableConv2D layer의 weight dimensions
>
>왼쪽에서부터 depthwise convolution, pointwise convolution, bias 순서이다. 확인해보면 bias는 하나 뿐이며, dimension이 pointwise convolution에 맞춰져 있음을 알 수 있다.


<br/>
다음은 이 아이디어를 기반으로하면서, Inception-v3과 비슷한 수의 parameter를 사용하는 CNN 구조를 제시하고, 두 개의 large-scale image classification 작업에서 성능을 평가한다.

---
## 2. Prior work
이 연구는 다음 분야의 이전 연구들에 크게 의존한다.

<br/>
### Convolutional neural networks
CNN 구조[[1](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf), [2](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), [3](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)], 특히 제안하는 구조는 몇 가지 측면에서 [VGG-16](https://arxiv.org/pdf/1409.1556.pdf)의 구조와 유사하다.

<br/>
### The Inception architecture family
Convolution을 여러 개의 branch로 factoring하여, channel들과 그 공간에서 연속적으로 동작하는 것에 대한 이점을 보여줬다.
>[Inception-v1](https://arxiv.org/pdf/1409.4842.pdf), [Inception-v2](https://arxiv.org/pdf/1502.03167.pdf), [Inception-v3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf), [Inception-v4](https://arxiv.org/pdf/1602.07261.pdf)

<br/>
### Depthwise separable convolution
네트워크에서 spatially separable convolution이 사용됐던건 2012년 이전이지만, depthwise version은 더 최근에 사용됐다.

<br/>
Laurent Sifre는 2013년 Google Brain internship 과정에서 depthwise separable convolution을 개발했으며, 이를 AlexNet에 적용하여 큰 수렴 속도 향상과 약간의 성능 향상, 모델 크기의 감소 등의 성과를 얻었다.
>[Laurent Sifre의 작업에 대한 개요](https://www.youtube.com/watch?v=VhLe-u0M1a8)는 ICLR에서 처음 발표됐다.
>
>자세한 실험 결과는 [Sifre의 논문](https://www.di.ens.fr/data/publications/papers/phd_sifre.pdf)의 6.2절에 보고됐다.

<br/>
Depthwise separable convolution에 대한 초기 연구는, Sifre와 Mallat의 사전 연구인 [transformation-invariant scattering](https://www.di.ens.fr/data/publications/papers/cvpr_13_sifre_mallat_final.pdf)에서 영감을 받았으며, 이후에는 depthwise separable convolution이 [Inception-v1](https://arxiv.org/pdf/1409.4842.pdf)과 [Inception-v2](https://arxiv.org/pdf/1502.03167.pdf)의 첫 번째 레이어로 사용됐다.

<br/>
Google의 Andrew Howard는 depthwise separable convolution을 사용하여 [MobileNets](https://arxiv.org/pdf/1704.04861.pdf)이라는 효율적인 모바일 모델을 도입했다.
>상당한 저용량의 모델임.

2014년 [Jin 등의 연구](https://arxiv.org/pdf/1412.5474.pdf) 및 2016년 [Wang 등의 연구](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w10/Wang_Factorized_Convolutional_Neural_ICCV_2017_paper.pdf)에서도 separable convolution을 사용하여 CNN의 크기 및 계산 비용을 줄이는 연구를 수행했다.

<br/>
### Residual connections
[ResNet](https://arxiv.org/pdf/1512.03385.pdf)에서 소개됐다. 이 논문에서 제안하는 구조에도 residual connection이 광범위하게 사용된다.

---
## 3. The Xception architecture
이 장에서는 전적으로 **depthwise separable convolution**에 기반한 CNN 구조를 제안한다.

<br/>
다음과 같은 가설을 세운다.
- CNN의 feature map에서 **cross-channels correlation**과 **spatial correlation**의 mapping은 완전히 분리될 수 있다.

<br/>
이는 Inception 구조에 기반한 강력한 버전의 가설이므로, 제안하는 구조를 "**Extreme Inception**"을 나타내는 이름인 **Xception**으로 명명한다.
>**Xception** 구조는 36개의 conv layer로 feature extraction을 수행한다. 네트워크에 대한 자세한 설명은 **Fig.5**를 참조하자.

<br/>
![Fig.5](/blog/images/Xception, Fig.5(removed).png )
>**Fig.5** <br/>Xception architecture
>
>Entry flow를 시작으로, 8회 반복되는 middle flow, 마지막에는 exit flow를 거치는 구조이다.
>
>모든 Convolution과 SeparableConvolution의 뒤에는 [BN](https://arxiv.org/pdf/1502.03167.pdf)이 뒤따른다.
>
>SeparableConvolution layer의 depth multiplier는 1이다.

<br/>
실험 평가에서는 image classification에 대해서만 조사하므로, 제안하는 CNN에는 logistic regression layer가 뒤따른다.
>Logistic regression layer 앞에 FC layer를 삽입 할 수도 있으며, 이는 실험 평가 섹션에서 알아본다. (**Fig.7** 및 **Fig.8** 참조)

<br/>
36개의 conv layer는 14개의 모듈로 구성되며, 첫 번째와 마지막 모듈을 제외한 나머지의 주위에는 linear residual connection이 있다.

<br/>
요약하자면, **Xception** 구조는 **residual connection**이 있는 **depthwise separable convolution**의 linear stack으로 볼 수 있다.

<br/>
따라서 **Xception** 구조는 정의와 수정이 매우 쉽게 이루어질 수 있다.
>Keras나 TensorFlow-Slim과 같은 high-level library를 사용하면 30~40줄의 코드로 구현이 가능하다.
>
>[VGG-16](https://arxiv.org/pdf/1409.1556.pdf)이나 훨씬 복잡한 [Inception-v3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)과는 차이가 있다.

---
## 4. Experimental evaluation
규모의 유사성을 고려해, Xception을 Inception-v3 아키텍처와 비교한다.
>Xception과 Inception-v3의 parameter 수는 거의 동일하므로, 성능 차이는 네트워크 capacity의 차이에서 비롯된 것이 아니다. (**Table.3** 참조)

<br/>
성능은 두 개의 image classification task에 대해 비교한다.
- **ImageNet dataset, 1000-class, signle-label classification**

- **Large-scale JFT dataset, 17000-class, multi-label classification**

<br/>
### 4.1. The JFT dataset
[JFT](https://arxiv.org/pdf/1503.02531.pdf)는 large-scale image classification을 위한 Google의 internal dataset으로, Hinton 등이 처음 도입했다. 데이터는 17000-class label로 분류 된 3.5억개의 고해상도 이미지로 구성된다.
>ImageNet은 1000-class label로 분류 된 120만개의 고해상도 이미지로 구성되므로, ImageNet의 약 300배에 해당하는 학습 데이터가 주어진다.

<br/>
JFT에 대해 학습 된 모델의 성능 평가를 위해, 보조 dataset인 **FastEval14k**를 사용한다.
>약 6000개의 class로 분류 된 14000개의 이미지로 구성된다.
>
>각 이미지는 평균적으로 36.5개의 label에 해당한다. (multi-label)

<br/>
이 dataset에서는 **top-100 prediction에 대한 Mean Average Precision**(이하 **MAP@100**)으로 성능을 평가하고, **MAP@100**의 각 class에 기여도를 가중치로 부여하여, 소셜 미디어 이미지에서 해당 class가 얼마나 흔하게 사용되는지 추정한다.
>흔하게 사용되는 label을 중요한 label로 본다.

<br/>
이 평가 절차는 소셜 미디어 이미지에서 자주 발생하는 label의 성능을 측정하기 위한 것으로, Google의 프로덕션 모델에 중요하다고 한다.

<br/>
### 4.2. Optimization configuration
ImageNet과 JFT에서는 서로 다른 optimization configuration을 사용했다.

### On ImageNet
 - Optimizer : SGD

 - Momentum : 0.9
 
 - Initial learning rate : 0.045
 
 - Learning rate decay : decay of rate 0.94 every 2 epochs

### On JFT
 - Optimizer : RMSprop

 - Momentum : 0.9

 - Initial learning rate : 0.001

 - Learning rate decay : decay of rate 0.9 every 3M samples

<br/>
두 dataset 모두, Xception과 Inception-v3에 동일한 optimization configuration이 사용됐다.
>이 configuration은 Inception-v3의 최고 성능에 맞춘 것이며, Xception에 최적화 된 hyperparameter로 조정하려는 시도를 하지 않았다. 두 네트워크의 학습 결과가 다르기 때문에, 특히 ImageNet dataset에서는 Inception-v3에 맞춰진 configuration이 최적값이 아닐 수 있다. (**Fig.6** 참조)

<br/>
모든 모델은 inference time에 Polyak averaging을 사용하여 평가된다.
>Exponential moving average에 해당한다.

<br/>
### 4.3. Regularization configuration

### Weight decay
Inception-v3은 rate가 4e-5인 weight decay(L2 regularization)를 사용하여, ImageNet의 성능에 맞게 신중하게 조정됐다. Xception에서는 이 rate가 매우 부적합하기 때문에, 1e-5를 사용한다.

<br/>
Optimal weight decay rate에 대한 광범위한 탐색은 하지 않았으며, ImageNet과 JFT에 대한 실험 모두에서 동일한 weight decay rate가 사용됐다.

<br/>
### Dropout
ImageNet 실험의 경우, 두 모델 모두 rate가 0.5인 dropout layer를 logistic regression layer의 앞에 포함한다.

<br/>
JFT 실험의 경우, dataset의 크기를 고려하면 적절한 시간 내에 overfitting 될 수가 없으므로 dropout이 포함되지 않는다.

<br/>
### Auxiliary loss tower
Inception-v3 아키텍처에서는 네트워크의 초반부에 classification loss를 역전파하여, 추가적인 regularization 메커니즘으로 사용하기 위한 auxiliary loss tower를 선택적으로 포함할 수 있다.

<br/>
단순화를 위해, auxiliary tower를 모델에 포함하지 않기로 했다.

<br/>
### 4.4. Training infrastructure
모든 네트워크는 [TensorFlow의 ditributed learning framework](https://arxiv.org/pdf/1603.04467.pdf)를 사용하여 구현됐으며, 각각 60개의 NVIDIA K80 GPU에서 학습됐다.

<br/>
ImageNet 실험의 경우, 최상의 classification 성능을 달성하기 위해 synchronous gradient descent과 data parallelism을 이용했으며, JFT의 경우에는 학습 속도를 높이기 위해 asynchronous gradient descent를 사용했다.

<br/>
ImageNet에 대한 실험은 각각 약 3일이 걸렸고, JFT에 대한 실험은 1개월이 넘게 걸렸다.
>JFT dataset에 대해 full convergence로 학습하려면 3개월씩 걸리기 때문에, 이렇게까지 학습하진 않았다고 한다.

<br/>
필요한 건 대충 나왔으니, Kears로 구현해보자. **Xception** 모델을 정의하면 다음과 같다.
``` python
def conv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu', weight_decay=1e-5):
    x = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    
    if activation:
        x = Activation(activation='relu')(x)
    
    return x

def sepconv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu', weight_decay=1e-5, depth_multiplier=1):
    x = SeparableConv2D(filters, kernel_size, padding=padding, strides=strides, depth_multiplier=depth_multiplier, depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    
    if activation:
        x = Activation(activation='relu')(x)
    
    return x

def Xception(model_input, classes):
    ## Entry flow
    x = conv2d_bn(model_input, 32, (3, 3), strides=2) # (299, 299, 3) -> (150, 150, 32)
    x = conv2d_bn(x, 64, (3, 3))

    for fliters in [128, 256, 728]: # (75, 75, 64) -> (75, 75, 128) -> (38, 38, 256) -> (19, 19, 728)
        residual = conv2d_bn(x, fliters, (1, 1), strides=2, activation=None)
        
        x = Activation(activation='relu')(x)
        x = sepconv2d_bn(x, fliters, (3, 3))
        x = sepconv2d_bn(x, fliters, (3, 3), activation=None)
        x = MaxPooling2D((3, 3), padding='same', strides=2)(x)
        
        x = Add()([x, residual])
        
        
    ## Middle flow
    for i in range(8): # (19, 19, 728)
        residual = x
        
        x = Activation(activation='relu')(x)
        x = sepconv2d_bn(x, 728, (3, 3))
        x = sepconv2d_bn(x, 728, (3, 3))
        x = sepconv2d_bn(x, 728, (3, 3), activation=None)
        
        x = Add()([x, residual])
        
        
    ## Exit flow
    residual = conv2d_bn(x, 1024, (1, 1), strides=2, activation=None) # (19, 19, 728) -> (10, 10, 1024)
        
    x = Activation(activation='relu')(x)
    x = sepconv2d_bn(x, 728, (3, 3))
    x = sepconv2d_bn(x, 1024, (3, 3), activation=None) # (19, 19, 728) -> (19, 19, 1024)
    x = MaxPooling2D((3, 3), padding='same', strides=2)(x) # (19, 19, 1024) -> (10, 10, 1024)
    
    x = Add()([x, residual])
    
    x = sepconv2d_bn(x, 1536, (3, 3))
    x = sepconv2d_bn(x, 2048, (3, 3))

    x = GlobalAveragePooling2D()(x)
    
    ## Optinal fully-connected layers
    '''
    x = Dense(4096)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    
    x = Dense(4096)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    '''
    
    x = Dropout(0.5)(x)
    x = Dense(classes, activation=None)(x)
    
    model_output = Dense(classes, activation='softmax')(x)

    model = Model(model_input, model_output, name='Xception')
    
    return model

```
>각 convolutional block에는 weight decay가 L2 regularizer로 적용되어 있다.

<br/>
Optimization configuration을 적용한 학습 코드는 다음과 같다.
``` python
from keras.models import Model, Input
from keras.layers import Conv2D, SeparableConv2D, Dense, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, BatchNormalization, Dropout
from keras.layers import Add
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.datasets import cifar10

import numpy as np
import keras.backend as K

def Upscaling_Data(data_list, reshape_dim):
    ...

def conv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu', weight_decay=1e-5):
    ...
    
def sepconv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu', weight_decay=1e-5, depth_multiplier=1):
	...

def Xception(model_input, classes):
	...
    
class LearningRateSchedule(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%2 == 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr*0.94)

input_shape = (299, 299, 3)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = Upscaling_Data(x_train, input_shape)
x_test = Upscaling_Data(x_test, input_shape)

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

classes = 10

model_input = Input( shape=input_shape )

model = Xception(model_input, 10)

optimizer = SGD(lr=0.045, momentum=0.9)
callbacks_list = [LearningRateSchedule()]

model.compile(optimizer, loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=callbacks_list)

```
>논문에 설명되지 않은 paramter 값은 적당히 설정했다.
>
>역시 CIFAR-10 dataset을 resize하여 학습하는 코드다.

<br/>
여기서 **depth multiplier**가 1이라는 것이 뭔지 알아보기 위해, 아래와 같이 정의한 모델에서 각 SeparableConv2D의 weight dimension을 출력해보자.
``` python
model_input = Input( shape=input_shape )

x = Conv2D(16, 3, activation='relu')(model_input)
x = SeparableConv2D(24, 3, activation='relu')(x)
x = SeparableConv2D(32, 3, depth_multiplier=2, activation='relu')(x)
x = SeparableConv2D(32, 3, activation='relu')(x)
x = SeparableConv2D(48, 3, depth_multiplier=3, activation='relu')(x)
x = SeparableConv2D(64, 3, activation='relu')(x)

x = GlobalAveragePooling2D()(x)
x = Dense(10, activation='softmax')(x)

model = Model(model_input, x)
```

<br/>
![Extra.2](/blog/images/Xception, Extra.2(removed).png )
>확인해보면 depth_multiplier 값에 따라 depthwise convolution에 사용되는 filter의 개수가 달라지며, 이에 따라 pointwise convolution의 input channel도 달라지는 것을 알 수 있다.

<br/>
### 4.5. Comparison with Inception V3

### 4.5.1 Classification performance
모든 평가는 single crop과 single model에 대해 수행됐다.

<br/>
ImageNet에 대한 결과는 test set이 아닌 validation set에 대해 측정됐다. (**Table.1**, **Fig.6** 참조)
>ILSVRC 2012 validation set 중 non-blacklisted image에 해당함.

<br/>
![Table.1](/blog/images/Xception, Table.1(removed).png )
>**Table.1** <br/>**ImageNet**에 대한 classification 성능 비교.
>
>각 성능은 single-crop, single model에 대해 측정됐다.
>
>Inception-v3는 auxiliary classifier가 포함되지 않은 경우의 성능이다.

<br/>
![Fig.6](/blog/images/Xception, Fig.6(removed).png )
>**Fig.6** <br/>Training profile on ImageNet.

<br/>
JFT에 대한 결과는 full convergence 후가 아닌, 3000만 회의 iteration 후에 측정됐다. (**Table.2**, **Fig.7**, **Fig.8** 참조)
>약 1개월 정도 학습한 상태에 해당한다.

<br/>
JFT에서는 FC layer가 포함되지 않은 버전과, logistic regression layer 이전에 4096개의 노드로 이루어진 FC layer가 2개 포함 된 버전을 테스트했다.

<br/>
![Table.2](/blog/images/Xception, Table.2(removed).png )
>**Table.2** <br/>**JFT**에 대한 classification 성능 비교.
>
>마찬가지로 single-crop, single model에 대해 측정됐다.

<br/>
![Fig.7](/blog/images/Xception, Fig.7(removed).png )
>**Fig.7** <br/>Training profile on JFT, without fully-connected layers.

<br/>
![Fig.8](/blog/images/Xception, Fig.8(removed).png )
>**Fig.8** <br/>Training profile on JFT, with fully-connected layers.

<br/>
두 결과를 요약하자면, 다음과 같다.
- **ImageNet**에서의 Xception은 Inception-v3보다 약간 더 나은 결과를 보여줬다.

- **JFT**에서의 Xception은 FastEval14k에 대한 MAP@100에서 4.3%의 상대적 개선을 보여줬다.

<br/>
또한 Xception은, [ResNet](https://arxiv.org/pdf/1512.03385.pdf)에서 보고 된 ResNet-50, ResNet-101, ResNet-152의 ImageNet classification 성능보다 우수하다.

<br/>
Xception 아키텍처는 ImageNet dataset보다 JFT dataset에 대해 더 큰 성능 향상을 보여줬다. 이는 Inception-v3가 ImageNet dataset에 중점을 두고 개발됐기 때문에, 디자인이 특정 작업에 과적합 된 것으로 볼 수 있다.
>반면, JFT dataset에 중점을 두고 조정 된 아키텍처는 없다.

<br/>
즉, ImageNet dataset에 더 적합한 hyperparameter를 찾는다면 상당한 추가적 성능 향상을 얻을 수 있을거라 볼 수 있다.

<br/>
### 4.5.2 Size and speed
**Table.3**에서는 Inception-v3와 Xception의 크기 및 속도를 비교한다.
>크기는 trainable parameter의 개수로 측정된다.

<br/>
![Table.3](/blog/images/Xception, Table.3(removed).png )
>**Table.3** <br/>Size and training speed comparison.

<br/>
Parameter의 개수는 ImageNet에 대한 학습 모델에서 측정됐으며, 초당 training step(gradient update)은 ImageNet dataset에 대한 synchronous gradient descent 횟수를 측정했다.
>ImageNet에 대한 학습 모델은 FC layer가 없는 경우이다.
>
>Synchronous gradient descent는 60개의 K80 GPU로 수행됐다.

<br/>
두 모델의 크기 차이는 약 3.5% 이내로 거의 같으며, 학습 속도는 Xception이 약간 느린 것으로 나타났다.
>Depthwise convolution operation 수준에서의 engineering optimization이 이루어진다면, Inception-v3보다 Xception이 더 빠르게 될 것으로 기대된다고 한다.

<br/>
두 모델이 거의 동일한 수의 parameter를 가지고 있다는 사실은, ImageNet과 JFT에 대한 성능 향상이 capacity의 증가가 아니라, 모델의 parameter를 보다 효율적으로 사용함으로써 이뤄졌다는 것을 나타낸다.

<br/>
### 4.6. Effect of the residual connections
Xception 구조에서의 residual connection 효과를 정량화하기 위해, residual connection을 포함하지 않는 버전의 Xception을 ImageNet에 대해 벤치마크했다. (**Fig.9** 참조)

<br/>
![Fig.9](/blog/images/Xception, Fig.9(removed).png )
>**Fig.9** <br/>Training profile with and without residual connections.

<br/>
수렴 속도나 최종 classification 성능 측면에 있어, residual connection은 반드시 필요한 것으로 보여진다.
>물론, residual 버전과 동일한 optimization configuration으로 non-residual 버전을 벤치마킹했기 때문에 차이가 더 크게 나온 것일 수도 있다.

<br/>
이 결과는 Xception에서 residual connection의 중요성을 보여준 것일 뿐이지, depthwise separable convolution의 stack인 이 모델의 구축에 필수 조건은 아니다.

<br/>
또한, non-residual VGG-style 모델의 모든 conv layer를 depthwise separable convolution으로 교체했을 때, 동일한 parameter 수를 가지고도 JFT dataset에서 Inception-v3보다 뛰어난 결과를 얻었다.
>이 때, depthwise separable convolution의 depth multiplier는 1이다.

<br/>
### 4.7. Effect of an intermediate activation after pointwise convolutions
1.2절에서 **depthwise separable convolution**과 **Inception module**의 다른 점이, depthwise separable convolution의 **depthwise operation**과 **pointwise operation** 사이에 non-linearity를 포함해야할 수도 있음을 시사한다고 언급했었다.
>지금까지의 실험 파트에서는 non-linearity가 포함되지 않았다.

<br/>
이 절에서는 intermediate non-linearity로 ReLU 또는 ELU의 포함 여부에 따른 효과를 실험적으로 테스트한 결과를 보인다.

<br/>
ImageNet에 대한 실험 결과는 **Fig.10**에 보이듯, non-linearity가 없으면 수렴이 빨라지고 최종 성능도 향상됐다.

<br/>
![Fig.10](/blog/images/Xception, Fig.10(removed).png )
>**Fig.10** <br/>Training profile with different case of applying activations.
>
>각 activation은 separable convolution layer에서 depthwise operation과 pointwise operation 사이에 위치한다.

<br/>
이는 Inception module에 대한 [Szegedy의 연구 결과](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)와 반대되는 놀라운 결과다.
>Inception-v3 논문임. [이전 포스트](https://sike6054.github.io/blog/paper/third-post/) 참조.

<br/>
이는 spatial convolution이 적용되는 intermediate feature space의 depth가 non-linearity의 실용성에 매우 중요하게 작용되는 것으로 볼 수 있다.

<br/>
즉, Inception module에 있는 deep feature space의 경우에는 non-linearity가 도움이 되지만, shallow feature space의 경우에는 오히려 정보 손실로 인해 해로울 수가 있다.
>Depthwise separable convolution의 1-channel deep feature space가 shallow feature space에 해당한다.
>
>즉, 일반적인 convolution의 경우에는 입력의 전체 channel에 대한 spatial convolution이 이루어지지만, depthwise separable convolution의 경우에는 각 channel에 대한 1-channel spatial convolution이 이루어지기 때문에, non-linearity로 인한 정보 손실이 크리티컬하게 작용할 수 있다는 말이다.

---
## 5. Future directions
1.2절에서 일반적인 convolution과 depthwise separable convolution 간에는 별개의 스펙트럼이 존재한다고 언급했었다. Inception module은 이 스펙트럼의 한 지점에 해당한다.

<br/>
실험적인 평가에서 Inception module의 extreme form인 **depthwise separable convolution**이 일반적인 Inception module보다 이점을 가질 수 있음을 보여줬지만, **depthwise separable convolution**이 optimal이라고 믿을 이유는 없다.

<br/>
일반적인 Inception module과 **depthwise separable convolution**의 스펙트럼 사이에 위치한 아키텍처가 추가적인 이점을 가질 수도 있다. 이는 향후 연구를 위해 남겨졌다.

---
## 6. Conclusions
일반 Convolution과 depthwise separable convolution이 어떻게 양 극단의 스펙트럼에 놓여 있는지를 보여줬다. Inception module은 둘의 중간 지점에 위치한다.

<br/>
이러한 관찰에 따라, 컴퓨터 비전 아키텍처에서 Inception module을 **depthwise separable convolution**으로 대체 할 것을 제안했다.

<br/>
이 아이디어를 기반으로 **Xception**이라는 새로운 아키텍처를 제시했으며, 이는 Inception-v3과 유사한 수의 parameter를 갖는다.

<br/>
Xception을 Inception-v3과 성능을 비교했을 때, ImageNet dataset에 대해서는 성능이 조금 향상됐으며, JFT dataset에 대해서는 성능이 크게 향상됐다.

<br/>
Depthwise separable convolution은 Inception module과 유사한 속성을 지녔음에도 일반 conv layer만큼 사용하기가 쉽기 때문에, 향후 CNN 아키텍처 설계의 초석이 될 것으로 기대된다.

---

<br/>
<br/>
{% include disqus.html %}
