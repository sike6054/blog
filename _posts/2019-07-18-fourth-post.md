---
title: "Inception-v4 논문 정리"
date: 2019-07-18 08:29:11 -0400
tags: AI ComputerVision Paper Inception-v4
categories:
  - Paper
toc: true
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<br/>
keras 코드 삽입 예정

## Paper Information

SZEGEDY, Christian, et al. "Inception-v4, inception-resnet and the impact of residual connections on learning". In: Thirty-First AAAI Conference on Artificial Intelligence. 2017.
> a.k.a. [Inception-v4 paper](https://arxiv.org/pdf/1602.07261.pdf)


---
## Abstract
매우 깊은 CNN 구조는, 최근 몇 년 동안의 image recognition 성능 발전에 중요한 역할을 했다. 비교적 낮은 계산 비용으로 상당히 우수한 성능을 달성한 **Inception** 구조도 그 중 하나에 해당한다.

<br/>
최근에는 더 전통적인 CNN 구조에, **residual connection**을 도입한 [연구](https://arxiv.org/pdf/1512.03385.pdf)가 있었으며, 이는 ILSVRC 2015에서 state-of-the-art 성능을 달성했다.
>[ResNet](https://arxiv.org/pdf/1512.03385.pdf)을 말한다. 이는 Inception 구조의 최신 버전인 [Inception-v3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)와 유사한 성능에 해당하며, 이로부터 **residual connection**과 **Inception** 구조를 결합했을 때의 이점에 대한 의문이 생겨나게 된다.

<br/>
이 논문에서는 **residual connection**을 통한 학습이, **Inception network**의 학습을 크게 가속화한다는 경험적 증거를 명확히 제시한다. 또한, **residual Inception network**가 유사한 비용의 **non-residual Inception network**보다 성능이 약간 좋다는 증거도 제시한다.
>**non-residual**은 **Inception-v4**를, **residual**은 **Inception-ResNet-v1**과 **Inception-ResNet-v2**를 말한다.

<br/>
논문에서는 **residual**과 **non-residual** 두 버전의 Inception network를 위한 새로운 구조를 몇 가지 제시한다. 이러한 구조적 변화가 ILSVRC 2012 classification 분야에서 single-frame recognition 성능을 크게 향상시켰다.

<br/>
또한, 적절한 activation scaling이 **very wide residual Inception network**의 학습 안정화에 얼마나 도움되는지 알아본다.

<br/>
실험에서는 3개의 Inception-ResNet-v2와 1개의 Inception-v4를 ensemble하여 ImageNet classification의 test set에서 3.08%의 top-5 error를 얻어냈다.

---
## 1. Introduction
2012 ImageNet competion에서 [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)이 우승한 이래로, 이 네트워크는 컴퓨터 비전의 다양한 분야에 적용되어 성공적인 성과를 거뒀다.

- [Object detection (R-CNN)](https://arxiv.org/pdf/1311.2524.pdf)

- [Segmentation (FCN)](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

- [Human pose estimation (DeepPose)](https://arxiv.org/pdf/1312.4659.pdf)

- [Video classification](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/42455.pdf)

- [Object tracking](https://papers.nips.cc/paper/5192-learning-a-deep-compact-image-representation-for-visual-tracking.pdf)

- [Super-resolution](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.642.1999&rep=rep1&type=pdf)

>이는 CNN 구조가 성공적으로 적용된 분야의 일부에 불과하다.

<br/>
본 연구에서는, 최근에 발표 된 [ResNet](https://arxiv.org/pdf/1512.03385.pdf)과 [Inception-v3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)의 두 아이디어를 결합하는 연구를 진행했다.

<br/>
[ResNet](https://arxiv.org/pdf/1512.03385.pdf)에서, deep architecture의 학습에는 residual connection이 본질적으로 중요한 것이라 주장했다.

<br/>
Inception network는 일반적으로 very deep한 구조를 갖는다. 따라서 [ResNet](https://arxiv.org/pdf/1512.03385.pdf)의 주장에 근거하면, Inception 구조의 filter contatenation 단계를 residual connection으로 교체하는 것이 자연스럽다.
>이로써, Inception은 계산 효율성을 유지하면서도 residual approach의 이점을 모두 취할 수 있게 됐다.

<br/>
또한, 둘의 단순한 결합 외에도, Inception 자체를 더 깊고 넓게 만들어서 보다 효율적이게 만들 수 있는가에 대한 연구도 진행했다. 이전 버전인 Inception-v3보다, 더 많은 Inception 모듈을 사용하면서도 단순하고 획일화 된 구조의 Inception-v4를 설계했다.

<br/>
Inception-v3에서는, Inception 초기 버전의 baggage를 많이 이어받았었다. 기술적인 제약 사항은 주로 [DistBelief](https://www.cs.toronto.edu/~ranzato/publications/DistBeliefNIPS2012_withAppendix.pdf)로 분산 학습을 하기 위한 model partitioning에서 발생했다.
>Baggage는 단점 정도로 보면 된다.

<br/>
현재는 학습 환경을 [TensorFlow에서 자체 개발한 분산 학습 프레임워크](https://arxiv.org/pdf/1603.04467.pdf)로 옮겼고, 위의 제약 사항들이 풀렸기 때문에 구조를 크게 단순화 할 수 있었다. 이 구조에 대해서는 3장에서 설명한다.

<br/>
이 논문에서는 순수한 Inception의 변형인 **Inception-v3**와 **Inception-v4**, 그리고 유사한 비용의 하이브리드 버전인 **Inception-ResNet**을 비교한다.
>Parameter나 computational complexity가 non-residual 모델과 유사해야한다는 주된 제약 사항을 고려한 임시적인 방법으로 만들어진 모델들이다.

<br/>
실제로 더 크고 넓은 버전의 **Inception-ResNet**을 ImageNet classification dataset에서 테스트했다.

<br/>
마지막 실험은 논문에서 제안한 최고 성능의 모델들을 모두 ensemble하여 평가하고 있다.

<br/>
Inception-v4와 Inception-ResNet-v2가 유사한 성능을 보였으며, ImageNet validation dataset의 single frame에 대한 평가에서 state-of-the-art의 성능을 능가했다. 따라서, ImagaeNet과 같이 잘 연구 된 데이터에서, 이러한 조합들이 어떻게 state-of-the-art 성능을 달성하는지 알아보려 했다.

<br/>
Ensemble 성능에서는 single frame에서의 성능 이득만큼 차이나지 않았다.
>Table.2에 나온 **single frame 성능**에서, **Inception-v3**와 **Inception-ResNet-v2**의 차이는 top-5 error와 top-1 error에서 각각 **0.7%**, **1.3%**씩 난다. 반면, Table.5의 enssemble 성능에서는 **Inception-v3를 4개 ensemble**한 것과 **Inception-v4를 1개, Inception-ResNet-v2를 3개 ensemble**의 차이는 각각 **0.4%**, **0.8%**만 난다.

<br/>
그럼에도, ensemble에서의 best 성능은 ImageNet validation set에 대한 top-5 error가 3.1%를 달성한다.

<br/>
마지막 장에서는 classification에 실패한 경우의 일부에 대해 알아보고, ensemble이 여전히 dataset에 대한 label noise까지 도달하진 못했으며, 예측을 위한 개선 여지가 여전히 존재한다고 결론 내린다.

---
## 2. Related Work
CNN은 [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 이후부터 large scale image recognition 분야에서 널리 사용됐다.

<br/>
그 다음으로 중요한 구조 몇 가지는 다음과 같다.
- [Network-in-network](https://arxiv.org/pdf/1312.4400.pdf)

- [VGGNet](https://arxiv.org/pdf/1409.1556.pdf)

- [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf) (Inception-v1)

<br/>
Residual connection은 [ResNet](https://arxiv.org/pdf/1512.03385.pdf)에서 소개됐다. 여기서는 image recognition과 object detection 분야에서, signal의 additive merging을 활용함으로써 얻는 이점에 대해, 설득력 있는 이론적 및 실용적인 증거를 제시하고 있다.

<br/>
![Fig.1](/blog/images/Inception-v4, Fig.1(removed).png )
>**Fig.1** <br/>[ResNet](https://arxiv.org/pdf/1512.03385.pdf)에서 도입한 residual connection.

<br/>
![Fig.2](/blog/images/Inception-v4, Fig.2(removed).png )
>**Fig.2** <br/>[ResNet](https://arxiv.org/pdf/1512.03385.pdf)에서 제안한 비용이 최적화 버전의 residual connection.

<br/>
[ResNet](https://arxiv.org/pdf/1512.03385.pdf)의 저자들은 residual connection이 본질적으로 매우 깊은 CNN의 학습에 필요하다고 주장하고 있다. 하지만, 이 논문의 연구 결과에서는 최소한 image recognition에 대해서는 이 주장을 지지하지 못하는 것으로 보인다.
>이러한 residual connection이 이득이 되는 범위를 이해하려면, deeper architecture에 대한 measurement point가 더 많이 필요해 보인다.

<br/>
실험에서는 residual connection을 활용하지 않고도, 경쟁력 있는 성능의 very deep network를 학습 시키는 것이 그리 어렵지 않다는 것을 보여준다.
>**Inception-v4**를 말함.

<br/>
하지만, residual connection을 사용하면 학습 속도가 크게 향상되는 것으로 보이며, 이는 residual connection의 활용에 대한 큰 이유가 된다.

<br/>
Inception deep convolutional architecture는 [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)에서 소개됐으며, 이는 **Inception-v1** 이라고도 칭했다. 이후의 Inception 구조는 다양한 방법으로 개선 됐으며, 처음에는 [batch normalization](https://arxiv.org/pdf/1502.03167.pdf)을 이용한 **Inception-v2** 구조가 제안됐다. 그 다음으로는 factorization 기법의 추가를 통해 개선 된 **Inception-v3** 구조가 제안됐다.

---
## 3. Architectural Choices

### 3.1 Pure Inception blocks
이전의 Inception 모델은 분할 방식으로 학습됐다.
>각 복제본(replica)은, 전체 모델이 메모리에 올라갈 수 있도록 여러 개의 sub-network로 분할됐다.

<br/>
Inception 구조는 고도로 튜닝될 수 있다. 즉, 학습 후의 네트워크 성능에 영향을 미치지 않는 layer들의 filter 개수에 많은 변화를 줄 수 있다.

<br/>
여러 sub-network 간의 계산적인 균형을 위해 layer의 크기를 조심스럽게 튜닝했으며, 이를 통해 학습 속도를 최적화시켰다.
>이전 버전에선 그랬다고 함.

<br/>
이와 대조적으로, [TensorFlow 분산 학습 프레임워크](https://arxiv.org/pdf/1603.04467.pdf)를 도입하면, 복제본의 분할하지 않고도 가장 최근의 모델을 학습시킬 수 있다.
>이는 gradient 계산에 필요한 tensor를 신중하게 고려하고, 이러한 tensor를 줄이기 위한 계산적인 구조화를 통해 부분적으로 가능해진다. 즉, backpropation에 사용된 메모리의 recent optimization으로 가능하게 된다.

<br/>
저자들의 과거 연구에서는 구조적인 변경에 대해 보수적이었으며, 제한 된 실험을 했었다.
>네트워크의 안정을 유지하면서, 일부 구성 요소들에 변화를 주기 위함.

<br/>
또한, 네트워크 초반부의 구조를 단순화하지 않으면 필요 이상으로 복잡해보였다고 한다. 그래서 이번 Inception-v4의 실험에서는, 이런 불필요한 baggage를 버리고 각 grid size에 대한 Inception block의 구조를 획일화 시켰다.

<br/>
**Inception-v4**의 전체적인 구조는 **Fig.3**를, 각 Incepiton modlue의 자세한 구조는 **Fig.4 ~ Fig.9**을 참조하자.
>Inception module의 구조에서, 'V'라고 표시 된 convolution은 `padding='valid'`를 뜻한다.

<br/>
![Fig.3](/blog/images/Inception-v4, Fig.9(removed).png )
>**Fig.3** <br/>**Inception-v4**의 전체 구조에 대한 개요다.

<br/>
![Fig.4](/blog/images/Inception-v4, Fig.3(removed).png )
>**Fig.4** <br/>**Inception-v4**와 **Inception-ResNet-v2**의 입력 부분에 사용되며, Fig.3의 **Stem**에 해당한다. 우측의 shape은 해당 layer의 출력 shape을 뜻한다.

<br/>
![Fig.5](/blog/images/Inception-v4, Fig.4(removed).png )
>**Fig.5** <br/>**Inception-v4**에서 grid size가 $$35\times 35$$일 때 사용되는 Inception block이다. Fig.3의 **Inception-A**에 해당한다.

<br/>
![Fig.6](/blog/images/Inception-v4, Fig.5(removed).png )
>**Fig.6** <br/>**Inception-v4**에서 grid size가 $$17\times 17$$일 때 사용되는 Inception block이다. Fig.3의 **Inception-B**에 해당한다.

<br/>
![Fig.7](/blog/images/Inception-v4, Fig.6(removed).png )
>**Fig.7** <br/>**Inception-v4**에서 grid size가 $$18\times 8$$일 때 사용되는 Inception block이다. Fig.3의 **Inception-C**에 해당한다.

<br/>
![Fig.8](/blog/images/Inception-v4, Fig.7(removed).png )
>**Fig.8** <br/>**Inception-v4**에서 grid size를 $$35\times 35$$에서 $$17\times 17$$로 줄일 때 사용하는 reduction module이며, Fig.3과 Fig.10의 **Reduction-A**에 해당한다. 각 conv layer에 사용되는 filter 수는 모델마다 다르며, 각 conv filter 수인 $$k, l, m, n$$은 Table.1을 따른다.

<br/>
![Table.1](/blog/images/Inception-v4, Table.1(removed).png )
>**Table.1** <br/>각 Inception 버전에 사용되는 **Reduction-A**의 filter 개수를 나타낸다.

<br/>
![Fig.9](/blog/images/Inception-v4, Fig.8(removed).png )
>**Fig.9** <br/>**Inception-v4**에서 grid size를 $$17\times 17$$에서 $$8\times 8$$로 줄일 때 사용하는 reduction module이며, Fig.3의 **Reduction-B**에 해당한다.

<br/>
### 3.2 Residual Inception Blocks
Residual 버전의 Inception network에서는, 기존의 Inception에서 사용된 것보다 더 저렴한 비용의 Inception block을 사용한다.

<br/>
각 Inception block 뒤에는, filter bank의 dimension을 입력의 depth에 맞추기 위한 filter-expansion layer가 사용된다.
>Activation이 없는 1x1 conv layer에 해당하며, 이는 Inception block에 의한 dimensionality reduction을 보완하기 위함이다. [Inception-v3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)에서 언급한 **Avoid Representational Bottlenecks**이 목적인 것으로 보인다. [이전 포스트](https://sike6054.github.io/blog/paper/third-post)의 2장 참조.

<br/>
Residual Inception에 대한 여러 버전을 시도했으며, 그 중 2가지에 대해서만 자세하게 설명한다. **Inception-ResNet-v1**의 비용은 대략 **Inception-v3**과 유사하며, **Inception-ResNet-v2**의 비용은 3.1절의 **Inception-v4**와 일치한다.
>실제로 **Inception-v4**는 더 많은 layer의 수로 인해, 학습 속도가 더 느린 것으로 판명됐다.

<br/>
두 버전의 전체적인 구조는 **Fig.10**를, **Inception-ResNet-v1**과 **Inception-ResNet-v2**에 사용 된 Inception module은 각각 **Fig.11 ~ Fig.15**와 **Fig.16 ~ Fig.19**를 참조하자.
>**Inception-ResNet-v1**의 **Reduction-A**는 Fig.8을 따르며, **Inception-ResNet-v2**의 **Stem**은 Fig.4를, **Reduction-A**는 Fig.8을 따른다.

<br/>
![Fig.10](/blog/images/Inception-v4, Fig.15(removed).png )
>**Fig.10** <br/>**Inception-ResNet-v1**과 **Inception-ResNet-v2**의 전체 구조에 대한 개요다.

<br/>
**Inception-ResNet-v1**에서 사용 된 각 모듈의 자세한 구조는 다음과 같다.

<br/>
![Fig.11](/blog/images/Inception-v4, Fig.14(removed).png )
>**Fig.11** <br/>**Inception-ResNet-v1**의 입력 부분에 사용되며, Fig.10의 **Stem**에 해당한다. 우측의 shape은 해당 layer의 출력 shape을 뜻한다.

<br/>
![Fig.12](/blog/images/Inception-v4, Fig.10(removed).png )
>**Fig.12** <br/>**Inception-ResNet-v1**에서 grid size가 $$35\times 35$$일 때 사용되는 Inception block이다. Fig.10의 **Inception-ResNet-A**에 해당한다.

<br/>
![Fig.13](/blog/images/Inception-v4, Fig.11(removed).png )
>**Fig.13** <br/>**Inception-ResNet-v1**에서 grid size가 $$17\times 17$$일 때 사용되는 Inception block이다. Fig.10의 **Inception-ResNet-B**에 해당한다.

<br/>
![Fig.14](/blog/images/Inception-v4, Fig.12(removed).png )
>**Fig.14** <br/>**Inception-ResNet-v1**에서 grid size를 $$17\times 17$$에서 $$8\times 8$$로 줄일 때 사용하는 reduction module이며, Fig.10의 **Reduction-B**에 해당한다.

<br/>
![Fig.15](/blog/images/Inception-v4, Fig.13(removed).png )
>**Fig.15** <br/>**Inception-ResNet-v1**에서 grid size가 $$8\times 8$$일 때 사용되는 Inception block이다. Fig.10의 **Inception-ResNet-C**에 해당한다.

<br/>
<br/>
**Inception-ResNet-v2**에서 사용 된 각 모듈의 자세한 구조는 다음과 같다.

<br/>
![Fig.16](/blog/images/Inception-v4, Fig.16(removed).png )
>**Fig.16** <br/>**Inception-ResNet-v2**에서 grid size가 $$35\times 35$$일 때 사용되는 Inception block이다. Fig.10의 **Inception-ResNet-A**에 해당한다.

<br/>
![Fig.17](/blog/images/Inception-v4, Fig.17(removed).png )
>**Fig.17** <br/>**Inception-ResNet-v2**에서 grid size가 $$17\times 17$$일 때 사용되는 Inception block이다. Fig.10의 **Inception-ResNet-B**에 해당한다.

<br/>
![Fig.18](/blog/images/Inception-v4, Fig.18(removed).png )
>**Fig.18** <br/>**Inception-ResNet-v2**에서 grid size를 $$17\times 17$$에서 $$8\times 8$$로 줄일 때 사용하는 reduction module이며, Fig.10의 **Reduction-B**에 해당한다.

<br/>
![Fig.19](/blog/images/Inception-v4, Fig.19(removed).png )
>**Fig.19** <br/>**Inception-ResNet-v2**에서 grid size가 $$8\times 8$$일 때 사용되는 Inception block이다. Fig.10의 **Inception-ResNet-C**에 해당한다.

<br/>
Residual 버전과 non-residual 버전 간의 또 다른 기술적 차이는, [BN](https://arxiv.org/pdf/1502.03167.pdf)을 traiditional layer에서만 사용됐으며, summation에는 사용하지 않았다.

<br/>
[BN](https://arxiv.org/pdf/1502.03167.pdf)을 충분히 사용하는 것이 이득이긴 하지만, 각 모델의 복제본을 single GPU 상에서 유지하기 위함이다.
>큰 activation size를 가진 layer가 차지하는 메모리 공간은 GPU memory를 불균형하게 소비하는 것으로 밝혀졌다고 한다. 이러한 layer들 위의 [BN](https://arxiv.org/pdf/1502.03167.pdf)을 생략함으로써, Inception block의 수를 크게 늘릴 수 있었다.
>
>컴퓨팅 리소스를 보다 효율적으로 활용해서, 이런 trade-off가 필요 없어졌으면 하는 바램이 있다더라.

<br/>
### 3.3 Scaling of the Residuals
Filter 개수가 1000개를 초과하게 되면 residual variant가 불안정해지기 시작하며, 네트워크가 학습 초기에 죽어버리는 것으로 나타났다. 이는 learning rate를 낮추거나, [BN](https://arxiv.org/pdf/1502.03167.pdf)을 추가하는 것으로는 예방할 수 없다.
>수만 번의 iteration 이후부터 average pooling 이전의 마지막 layer에서는 0만을 출력했다고 한다.
>
>Residual variant는 **Inception-ResNet** 버전을 말하는 것으로 보인다.

<br/>
Residual을 누적 된 layer activation에 추가하기 전에 **scaling down**을 하는 것이 학습의 안정화에 도움되는 것처럼 보였다. 이를 위한 scaling factor는 0.1에서 0.3 사이의 값을 사용했다. 구조는 Fig.20을 참조하자.

<br/>
![Fig.20](/blog/images/Inception-v4, Fig.20(removed).png )
>**Fig.20** <br/>**Inception-ResNet module**의 **scaling**을 위한 general schema이다.
>
>Scaling block은 마지막 linear activation을 적절한 상수에 대해 scaling한다. 일반적으로 적절한 scaling factor는 약 0.1 정도라 한다.
>
>Inception block 대신, 임의의 subnetwork를 사용하는 일반적인 ResNet의 경우에도 이 아이디어가 유용할 것으로 보인다고 한다.

<br/>
[ResNet](https://arxiv.org/pdf/1512.03385.pdf)에서는 very deep residual network의 경우에도 비슷한 불안정성을 관찰했고, 이를 위해 2단계로 learning rate를 스케쥴링 했다.
>첫 단계인 "warm-up" 단계에서는 매우 낮은 learning rate로 학습하다가, 두 번째 단계에서는 높은 learning rate로 학습한다.
>
>[이전 포스팅](https://sike6054.github.io/blog/paper/first-post)의 4.2절 CIFAR-10에 대한 실험에서 110-layer로 학습하는 경우에 해당한다.

<br/>
저자들은 filter의 수가 매우 많은 경우에는 0.00001의 매우 낮은 learning rate조차도 불안정성에 대처하기에 충분하지 않으며, 높은 learning rate로 학습한다면 이를 제거할 기회를 가지게 된다는 것을 알아냈다.

<br/>
저자들은 또한, 이 방법 대신 residual을 scaling하는 것이 훨씬 더 안정적이라는 것을 알아냈다.이러한 scaling이 엄밀히 꼭 필요한 것은 아니며, 최종 성능에 해를 끼치지 않으면서 학습의 안정화에 도움이 되는 것이라 한다.

---
## 4. Training Methodology
20개의 복제본(replica)이 각각 NVidia Kepler GPU에서 수행되도록 [TensorFlow 분산 학습 시스템]을 사용하여 stochastic gradient로 학습했다.

<br/>
실험 초기에는 decay가 0.9인 [momentum](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)을 사용했었고, best 성능은 decay가 0.9, $$\epsilon$$이 1.0인 [RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)을 사용하여 달성했다.

<br/>
Learning rate는 0.045부터 시작하여, 2번의 epoch마다 0.94를 곱했다.

<br/>
모델 평가는 시간이 지남에 따라 계산 된 parameter들의 running average로 수행됐다.

---
## 5. Experimental Results
우선 4가지 버전의 학습 중 top-1 및 top-5 validation error를 관찰한다.

<br/>
Bounding box의 질이 좋지 못한 약 1700개의 blacklisted entity를 생략한 validation set의 subset에 대해 지속적으로 평가했었음을 실험 후에 발견했다.
>Blacklisted entity의 생략은 CLS-LOC benchmark에 대해서만 이뤄졌어야 했었다.

<br/>
그럼에도, 저자들의 이전 연구를 포함한 다른 연구들과는 비교할 수 없을 정도의 긍정적 수치를 얻었다.
>성능의 차이는 top-1와 top-5 error에서 각각 0.3%, 0.15% 정도였으며, 그 차이가 일관적이었기 때문에 성능 그래프 간의 비교가 공정한 것으로 본다고 한다.

<br/>
![Fig.21](/blog/images/Inception-v4, Fig.22(removed).png )
>**Fig.21** <br/>계산 비용이 거의 동일한 **Inception-v3**와 **Inception-ResNet-v1**의 top-5 error 비교. 각 성능은 ILSVRC 2012의 non-blacklisted validation set에 대한 single-crop으로 측정됐다.
>
>Residual 버전의 학습 속도가 빠르며, **Inception-v3**보다 성능이 약간 좋은 것으로 나타났다.

<br/>
![Fig.22](/blog/images/Inception-v4, Fig.21(removed).png )
>**Fig.22** <br/>계산 비용이 거의 동일한 **Inception-v3**와 **Inception-ResNet-v1**의 top-1 error 비교. 각 성능은 ILSVRC 2012의 non-blacklisted validation set에 대한 single-crop으로 측정됐다.
>
>Residual 버전의 학습 속도가 빠르며, **Inception-v3**보다 성능이 약간 좋지 않은 것으로 나타났다.

<br/>
![Fig.23](/blog/images/Inception-v4, Fig.24(removed).png )
>**Fig.23** <br/>계산 비용이 거의 동일한 **Inception-v4**와 **Inception-ResNet-v2**의 top-5 error 비교. 각 성능은 ILSVRC 2012의 non-blacklisted validation set에 대한 single-crop으로 측정됐다.
>
>Residual 버전의 학습 속도가 빠르며, **Inception-v4**보다 성능이 약간 좋은 것으로 나타났다.

<br/>
![Fig.24](/blog/images/Inception-v4, Fig.23(removed).png )
>**Fig.24** <br/>계산 비용이 거의 동일한 **Inception-v4**와 **Inception-ResNet-v2**의 top-1 error 비교. 각 성능은 ILSVRC 2012의 non-blacklisted validation set에 대한 single-crop으로 측정됐다.
>
>Residual 버전의 학습 속도가 빠르며, **Inception-v4**보다 성능이 약간 좋은 것으로 나타났다.

<br/>
![Fig.25](/blog/images/Inception-v4, Fig.25(removed).png )
>**Fig.25** <br/>**Inception-v3**, **Inception-v4**, **Inception-ResNet-v1**, **Inception-ResNet-v2** 4가지 모델의 top-5 error 비교.
>
>Residual 버전의 학습 속도가 빠르며, 모델의 크기에 따라 최종 성능이 달라지는 것으로 보인다.

<br/>
![Fig.26](/blog/images/Inception-v4, Fig.26(removed).png )
>**Fig.26** <br/>**Inception-v3**, **Inception-v4**, **Inception-ResNet-v1**, **Inception-ResNet-v2** 4가지 모델의 top-5 error 비교.
>
>Fig.26의 top-5 evaluation과 흡사한 결과를 보인다.

<br/>
반면, 50000개의 이미지로 구성 된 validation set에 대해, multi-crop 및 ensemble 결과는 재수행했다. 또한, 최종 ensemble 결과는 test set에 대해 수행된 후, 검증을 위해 ILSVRC test server에 전송하고 overfitting이 일어나지 않았는지 확인했다.
>저자들은 최종 검증을 한 번만 수행했었으며, 작년에는 결과를 두 번만 제출했다는 점을 강조하고 싶다고 함. 테스트의 수에 따라, 제안하는 모델의 일반적인 성능을 추정할 수 있다고 믿기 때문.
>
>각 두 번의 결과는 각각 BN-Inception과 ILSVRC 2015 CLS-LOC 대회에 해당한다.

<br/>
마지막으로, **Inception**과 **Incepion-ResNet**의 다양한 버전에 대한 성능 비교를 보여준다.

<br/>
**Inception-v3**와 **Inception-v4**는 residual connection을 활용하지 않는 deep convolutional network이며, **Inception-ResNet-v1**과 **Inception-ResNet-v2**는 filter concatenation 대신 residual connection을 이용하는 Inception style network이다.

<br/>
Table.2는 validation set에 대한 다양한 구조들의 single model, single-crop 성능을 보여준다.

<br/>
![Table.2](/blog/images/Inception-v4, Table.2(removed).png )
>**Table.2** <br/>Single-crop, single model에 대한 실험 결과이다.
>
>각 성능은 ILSVRC 2012의 validation set에서 non-blacklisted subset에 대해 측정됐다.

<br/>
아래 Table.3 ~ Table.5의 각 crop 수는 [ResNet](https://arxiv.org/pdf/1512.03385.pdf)과 [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)을 따른다.

<br/>
Table.3은 다양한 모델들이 적은 수의 crop을 사용한 경우의 성능을 보여준다.
<br/>
![Table.3](/blog/images/Inception-v4, Table.3(removed).png )
>**Table.3** <br/>10/12-crop evaluation, single model에 대한 실험 결과이다.
>
>각 성능은 ILSVRC 2012의 validation set인 50000개 이미지에 대해 측정됐다.

<br/>
Table.4는 다양한 모델에 대한 multi-crop, single model 성능을 보여준다.

<br/>
![Table.4](/blog/images/Inception-v4, Table.4(removed).png )
>**Table.4** <br/>144-crop/dense evaluation, single model에 대한 실험 결과이다.
>
>각 성능은 ILSVRC 2012의 validation set인 50000개 이미지에 대해 측정됐다.

<br/>
Table.5는 ensemble 결과를 비교한다.

<br/>
![Table.5](/blog/images/Inception-v4, Table.5(removed).png )
>**Table.5** <br/>144-crop/dense evaluation, single model에 대한 실험 결과이다.
>
>각 성능은 ILSVRC 2012의 validation set인 50000개 이미지에 대해 측정됐다.
>
>Inception-v4(+Residual) ensemble은 **Inception-v4** 1개와 **Inception-ResNet-v2** 3개로 구성되며, validation set과 test set에 대해 평가됐다. 테스트 성능은 top-5 error가 3.08%이며, validation set에 overfitting되지 않았다.

---
## 6. Conclusions
이 논문에서는 3가지의 새로운 network architecture를 보였다.

- **Inception-ResNet-v1** : [Inception-v3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)과 비슷한 계산 비용을 가진 하이브리드 버전

- **Inception-ResNet-v2** : Recognition 성능이 크게 향상 된 비싼 비용의 하이브리드 버전

- **Inception-v4** : **Inception-ResNet-v2**와 거의 동일한 recognition 성능을 가진 non-residual, pure Inception 버전

<br/>
Residual connection의 도입으로 Inception 구조의 학습 속도가 얼마나 향상되는지 알아봤다. 또한, 제안하는 모델들은 모델의 크기가 커짐에 따라, residual connection의 유무에 상관없이 이들의 모든 이전 네트워크 성능을 능가했다.

---

<br/>
<br/>
{% include disqus.html %}
