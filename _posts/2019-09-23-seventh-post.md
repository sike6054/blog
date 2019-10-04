---
title: "(SENet) Squeeze-and-Excitation Networks 번역 및 추가 설명과 Keras 구현"
date: 2019-09-23 11:38:11 -0400
tags: AI ComputerVision Paper Attention SENet
categories:
  - Paper
toc: true
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Paper Information

HU, Jie; SHEN, Li; SUN, Gang. **"Squeeze-and-excitation networks"**. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2018. p. 7132-7141.
> a.k.a. [SENet paper](https://arxiv.org/pdf/1709.01507.pdf)

Reference link 수정 및 keras 코드 삽입 예정

---
## Abstract
CNN의 가장 중요한 구성 요소는, **각 layer에서 local receptive field 내부의 spatial 및 channel-wise information을 융합함으로써, 네트워크가 유익한 feature를 구성**할 수 있도록 해주는 convolution operator다.

<br/>
이에 대한 광범위한 선행 연구들에서는, **이러한 relationship의 spatial component를 조사하고, feature 계층 전반에 걸친 spatial encoding의 품질을 향상시킴으로써 CNN의 representational power를 강화**하고자 했다.
>이전 문단에서 bold체로 표시한 부분을 말하는 것으로 보인다.

<br/>
본 연구에서는 선행 연구들과 달리 channel relationship에 중점를 두어, **channel 간의 inter-dependency를 명시적으로 모델링함으로써 channel-wise feature response를 adaptive하게 recalibration**해주는 unit인 **"Squeeze-and-Excitation"(SE) block**을 제안한다.
>Feature response는 layer(이 경우는 conv layer에 해당)의 출력이다.

<br/>
본문에서는 이 block들을 쌓아, 여러 dataset에서 매우 효과적으로 일반화 되는 **SENet** 아키텍처를 형성할 수 있음을 보여준다.

<br/>
또한 SE block을 위한 약간의 추가적인 계산 비용을 들이면, 기존의 SOTA CNN 아키텍처 성능이 크게 향상될 수 있음을 보여준다.

<br/>
Squeeze-and-Excitation Network는 ILSVRC 2017 classification에서 top-5 error를 2.251%로 줄여서 1위를 차지했으며, 2016년도의 우승 성적에 비해 25%의 상대적 개선을 이뤘다.

---
## 1. Introduction
CNN은 다양한 분야의 visual task에서 유용한 모델로 입증됐다.

네트워크의 각 conv layer의 filter들은 입력 채널을 따라 인근의 spatial connectivity pattern을 나타낸다.
>Local receptive field 내부의 spatial 및 channel-wise information을 융합.

<br/>
CNN은 일련의 conv layer에 non-linear activation과 downsampling operator를 끼워넣음으로써 다음의 목적을 달성하는 image representation을 생성할 수 있다.
- Hierarchical pattern의 capture

- Global theoretical receptive field

<br/>
Computer vision 연구의 중심 주제는, 주어진 작업에 가장 중요한 이미지의 속성만을 캡처하고 성능을 향상시킬 수 있는 powerful representation을 찾는 것이다.

<br/>
Vision task에서 인공 신경망 알고리즘은 널리 사용되고 있으며, 인공 신경망 아키텍처의 design에 관한 연구는 이 분야의 핵심 영역이 됐다.

<br/>
최근의 연구에 따르면, 학습 메커니즘을 feature 간의 spatial correlation을 캡처하는데 도움이 되는 네트워크에 통합함으로써, CNN에 의해 생성된 representation을 강화할 수 있다.
- Inception architecture에서는 multi-scale process를 네트워크 모듈에 통합함으로써 성능을 향상시켰다. [[1](https://arxiv.org/pdf/1409.4842.pdf)][[2](https://arxiv.org/pdf/1502.03167.pdf)]

- 이후에는 Spatial dependency를 더 잘 모델링하고, [[1](https://arxiv.org/pdf/1512.04143.pdf)][[2](https://arxiv.org/pdf/1603.06937.pdf)]

- 네트워크 구조에 spatial attention을 포함시켰다. [[1](https://arxiv.org/pdf/1506.02025.pdf)]

<br/>
이 논문에서는 다른 측면으로, channel 간의 relationship을 조사한다.

<br/>
본문에서는 convolutional feature의 channel 간의 상호 의존성을 명시적으로 모델링하는 Squeeze-and-Excitation(SE) block을 소개한다.
>네트워크에서 생성된 representation의 질을 향상시키기 위함이 주 목적인 new architectural unit이다.

<br/>
이를 위해, 네트워크가 feature recalibration을 수행할 수 있도록 해주는 메커니즘을 제안한다. 이 메커니즘을 통하면, **global information**으로부터 **유익한 feature를 선택적으로 강조**하면서 **덜 유용한 feature를 억제**하는 방법을 배울 수 있다.

<br/>
![Fig.1](/blog/images/SENet, Fig.1(removed).png )
>**Fig.1** <br/>A Squeeze-and-Excitation block.

<br/>
**Fig.1**의 SE building block을 보자. 임의로 주어진 $$F_{tr}$$은 입력 $$X$$는 feature map $$U$$로 mapping되고, $$U$$는 다음의 두 operation을 순서대로 통과한다.
>이 때, $$U$$는 $$U\in \Bbb{R}^{H\times W\times C}$$.

<br/>
**Squeeze**
- Spatial dimension(HxW)에 걸친 feature map을 aggregation함으로써 channel descriptor를 생성한다.

- 이 descriptor는 channel-wise feature response의 global distribution(information)을 임베딩하여, 네트워크의 모든 layer에서 global receptive field로부터 얻을 수 있는 information을 사용할 수 있도록 해준다.
>**Fig.1**의 $$F_{sq}(\cdot)$$ 에 해당.

<br/>
**Excitation**
- 임베딩을 input으로 하여 per-channel modulation weight를 생성하는 simple self-gating 메커니즘 형태다.

- 이 weight들은 feature map $$U$$에 적용되며, 이렇게 생성된 SE block의 출력은 후속 layer에 바로 사용될 수 있다.
>**Fig.1**의 $$F_{ex}(\cdot , W)$$ 및 $$F_{scale}(\cdot ,\cdot)$$ 에 해당.

<br/>
이러한 SE block의 stack으로 **SENet**을 구성할 수 있다.

<br/>
또한, SE block은 다양한 depth의 네트워크 아키텍처에서 original block의 대체로도 사용될 수 있다. (6.4절 참조)

<br/>
Building block의 형태는 generic하지만, 수행하는 역할들은 네트워크 내의 block의 위치(depth)에 따라 다르다.

<br/>
**In earlier layers**
- Class에 구애받지 않는 방식으로 유익한 featrue를 자극하여 shared low-level representation을 강화한다.

<br/>
**In later layers**
- 점점 더 specialised 되며, class에 특화 된 방식으로 다른 input에 응한다. (7.2절 참조)

<br/>
즉, SE block에 의해 수행 된 feature recalibration의 이점은, 네트워크를 거쳐가며 누적될 수 있다.

<br/>
새로운 CNN 아키텍처의 설계나 개발은 어려운 engineering task에 속하며, 일반적으로 새로운 hyperparameter나 layer configuration에 대한 선택도 요구된다.

<br/>
반대로, **SE block의 구조는 간단하면서도 성능을 효과적으로 향상시킬 수 있으며, 기존의 SOTA 아키텍처에서 SE block에 대응되는 component만 교체하는 것으로 직접적인 사용이 가능하다. 또한, SE block은 계산적으로도 가볍기 때문에, SE block을 사용할 시에도 model complexity와 computational cost를 약간만 증가시킬 뿐이다.**

<br/>
이러한 주장에 대한 증거를 제공하기 위해, 여러 버전의 SENet을 개발하여 ImageNet dataset에 대해 광범위한 평가를 수행한다.

<br/>
또한 이 approach의 이점이 특정 task나 dataset에 제한되지 않음을 나타내는 ImageNet 외의 결과도 제시한다.

<br/>
SENet을 활용하여 ILSVRC 2017 classification 분야에서 1위를 차지했으며, best model ensemble은 test set에 대해 2.251%의 top-5 error를 달성했다.
>이는 전년도의 1위와 비교할 때, 약 25%의 상대적 개선에 해당한다. (2.991%, top-5 error)

---
## 2. Related Work

### Deeper architectures
[VGGNet](https://arxiv.org/pdf/1409.1556.pdf)과 [Inception](https://arxiv.org/pdf/1409.4842.pdf)은 네트워크의 depth를 늘리면 학습할 수 있는 representation의 품질을 크게 높일 수 있음을 보여줬다.

<br/>
[BN](https://arxiv.org/pdf/1502.03167.pdf)은 각 layer에 대한 입력의 distribution을 정규화함으로써, deep network의 학습 프로세스에 안정성을 더해주며, 보다 부드러운 optimisation surface를 생성한다.

<br/>
이를 바탕으로 ResNet에서는 identity-based skip connection을 사용하면 상당히 깊고 강력한 네트워크를 학습할 수 있음을 보여줬다.
>ResNet [[1](https://arxiv.org/pdf/1512.03385.pdf)][[2](https://arxiv.org/pdf/1603.05027.pdf)]

<br/>
[Highway network](https://arxiv.org/pdf/1507.06228.pdf)는 shortcut connetion을 따라 information flow를 조절하는 gating mechanism을 도입했다.

<br/>
이러한 연구에 이어, layer 간의 connection에 대한 추가적인 reformulation이 추가로 이루어진 연구가 있었으며, 이는 deep network의 학습과 representational property에 대한 개선 가능성을 보여줬다.
>그 연구 [[1](dual path network)][[2](densenet)]

<br/>
대안적이면서 밀접하게 관련된 다음의 연구들은, 네트워크 내에 포함 된 computational unit의 개선 방법에 중점을 두고 있다.

<br/>
Grouped convolution은 학습 된 transformation의 cardinality를 증가시키는 방법으로 입증되어, 널리 사용되고 있다.
>[Deep root](18), [ResNeXt](19)

<br/>
Multi-branch convolution을 사용하면 operator를 보다 유연하게 구성할 수 있으며, 이는 grouping operator의 자연스러운 확장 버전으로 볼 수 있다.

<br/>
이전 연구에서, cross-channal correlation은 일반적으로 spatial structure에 독립적이거나, 1x1 convolution를 함께 사용한 새로운 feature 조합으로 mapping 된다.
>[[1](https://arxiv.org/pdf/1405.3866.pdf)], [Xception](https://arxiv.org/pdf/1610.02357.pdf), [NIN](https://arxiv.org/pdf/1312.4400.pdf)

<br/>
본 연구의 대부분은 model/computational complexity를 줄이는 것에 집중했으며, channel relationship이 local receptive field를 가진 instance-agnostic function으로 formulation 될 수 있다는 가정을 반영했다.
>Channel relationship을 원하는 방식으로 formulation 할 수 있다는 가정.

<br/>
반대로, 이 논문에서는 **global information**을 채널 간 non-linear dependency의 dynamic한 모델링을 명시적으로 해주는 메커니즘의 **unit**에 제공한다면, 학습 프로세스를 용이하게하고 네트워크의 representational power을 크게 향상시킬 수 있다고 주장한다.
>Representational power : 네트워크의 표현력

<br/>
### Algorithmic Architecture Search
위에서 설명한 연구들 외에도, manual한 디자인 대신 네트워크 구조 자체를 학습 목표로 한 연구에 대한 많은 역사가 있다.

<br/>
이 분야의 초기 연구 대부분은 진화론적으로 network topology를 탐색하는 방법을 확립한 neuro-evolution community에서 수행됐다.
>네트워크 망의 구성 방식에 대한 탐색이다.
>
>Designing neural networks using genetic algorithms.
>Evolving neural networks through augmenting topologies.

<br/>
진화론적인 탐색은 계산적 부담이 큰 경우가 많지만, sequence model에 좋은 memory cell을 찾는 것과 largescale image classification을 위한 정교한 아키텍처의 학습 등에서 주목할만한 결과를 얻었다.
>Sequence model [[1](http://julian.togelius.com/Bayer2009Evolving.pdf)] [[2](http://proceedings.mlr.press/v37/jozefowicz15.pdf)]
>
>Largescale image classification [[1](http://openaccess.thecvf.com/content_ICCV_2017/papers/Xie_Genetic_CNN_ICCV_2017_paper.pdf)] [[2](https://arxiv.org/pdf/1802.01548.pdf)]

<br/>
이러한 방법들의 계산적 부담을 줄이기 위한 효율적인 대안으로, [Lamarckian inheritance](Efficient multi-objective neural architecture search via lamarckian evolution)과 [differentiable architecture search](DARTS: Differentiable architecture search) 기반의 방법들이 제안됐다.

<br/>
Architecture search는 hyperparameter optimization을 공식화하고, [random search]나 [보다 정교한 model-based optimization technique]을 사용하면 문제를 해결할 수 있다.
>Random search (Random search for hyper-parameter optimization)
>
>Model-based optimization technique (Progressive neural architecture search) (Deeparchitect: Automatically designing and training deep architectures)

<br/>
가능한 네트워크 구조들로부터의 [topology selection]이나 [direct architecture prediction]은 추가로 수행 가능한 architecture search tool로써 제안되어 왔으며, 이들은 특히 reinforcement learning으로부터 상당한 결과를 얻어냈다.
>Topology selection [37]
>
>Direct architecture prediction [38] [39]
>
>From reinforcement learning [40] [41] [42] [43] [44]

<br/>
SE 블록은 이러한 검색 알고리즘을위한 원자 적 빌딩 블록으로 사용될 수 있으며, 동시 작업에서이 용량에서 매우 효과적인 것으로 입증되었습니다 [45].

<br/>
### Attention and gating mechanisms
Attention은 이용 가능한 computational resource를, signal 중 가장 유익한 요소에 할당하는 성향을 갖게끔 유도할 수단으로써 해석될 수 있다.
>[46], [47], [48], [49], [50], [51].

<br/>
Attention 메커니즘은 다음의 분야를 포함한 많은 분야에서 유용성을 입증했다.
- Sequence learning [52], [53]

- Localization and understanding in image [9], [54]

- Image captioning [55][56]

- Lip reading [57]

<br/>
이러한 응용 연구들에서는, 일련의 기법들 간의 적응을 위해, 고차원의 abstraction을 나타내는 1개 이상의 layer로 이루어진 하나의 operator로써 통합될 수 있다.

<br/>
일부는 spatial 및 channel attention의 결합에 대한 흥미로운 연구를 제공한다.
>[58], [59]

<br/>
Deep residual network의 내부에 삽입되는 [hourglass module](8)에 기반한, 강력한 trunk-and-mask attention 메커니즘을 도입했다.
>[58]

<br/>
이와 대조적으로, 제안하는 **SE block**은 **computationally efficient한 channel-wise relationship을 모델링**하여, **네트워크의 representational power를 향상**시키는 것에 중점을 둔 **lightweight gating 메커니즘**으로 구성된다.

---
## 3. Squeeze-and-Excitation Blocks
Squeeze-and-Excitation block은 입력 $$X\in \Bbb{R}^{H'\times W'\times C'}$$를 feature-map $$U\in \Bbb{R}^{H\times W\times C}$$에 mapping하는 transformation $$F_{tr}$$ 위에 구축될 수 있는 computational unit이다.

<br/>
위 notation에서 $$F_{tr}$$은 convolution operator를, $$V = [v_1, v_2, ... , v_C]$$은 학습 된 filter kernel set을 나타낸다.
>$$v_c$$는 $$c$$-th filter의 parameter에 해당한다.

<br/>
그런 다음, 출력은 $$U = [u_1, u_2, ... , u_C]$$로 나타낼 수 있다. 여기서 각 $$u_c$$는 **Eqn.1**에 따라 계산된다.

<br/>
>**Eqn.1**
>
>$$u_c = v_c\ast X = \sum_{s=1}^{C'} {v_c^s \ast x^s}$$

<br/>
**Eqn.1**의 각 notation은 다음을 의미하며, 표기의 단순화를 위해 bias는 생략된다.
- $$*$$ is convolutional operation

- $$vc = [v_c^1, v_c^2, ... , v_c^{C'}]$$.

- $$X = [x^1, x^2, ... , x^{C'}]$$.

- $$u_c\in \Bbb{R}^{H\times W}$$.

<br/>
여기서 $$v_c^s$$는 $$v_c$$의 $$s$$-th channel에 해당하는 2D spatial kernel이며, 이들은 입력 $$X$$의 각 $$s$$-th channel에서 동작한다.

<br/>
출력은 모든 channel을 통한 합산에 의해 생성되므로 channel dependency가 $$v_c$$ 안에 함축되지만, filter에 의해 캡처 된 local spatial correlation과 얽혀있게 된다.
>모든 channel을 통한 합산은 convolution 연산에 해당한다.
>
>Channel dependency와 local spatial correlation을 한꺼번에 고려하여 하나의 결과로 내는 것은 어려우면서도 다소 신뢰성이 떨어질 수 있는 동작으로 생각될 수 있다.

<br/>
Convolution으로 모델링 된 channel relationship은, 본질적으로 implicit(암시적)하고 local한 특징이 있다.
>최상위 layer의 경우(classifier를 말하는 듯)는 예외이다.

<br/>
Channel의 상호 의존성을 명시적으로 모델링하게 되면, convolutional feature에 대한 학습이 향상될 것이며, 이는 후속되는 transformation들이 활용할 수 있는 유익한 feature에 대한 sensitivity를 증가시킬 수 있을 것으로 기대된다.

<br/>
결과적으로, 다음의 두 가지의 목적을 따른다.
- **Global information에 대한 access** 제공

- Next transformation 이전에, **squeeze**와 **excitation**으로 이루어진 2-step operation을 통한 **filter response의 recalibration**

<br/>
SE block의 구조는 **Fig.1**을 참조하자.

<br/>
### 3.1. Squeeze: Global Information Embedding
Channel dependency를 이용하기 위해, 먼저 출력 feature들의 각 channel에 대한 signal을 고려한다.

<br/>
학습 된 각 filter들은 local receptive field에서 동작하기 때문에, transformation output인 $$U$$의 각 unit들은 해당 영역 외의 contextual information을 이용할 수 없다.
>Transformation이 일어나는 각 layer input 상의 local receptive field 정보만을 가지고 해당 layer output을 만들어 내기 때문이다.

<br/>
이 문제의 완화를 위해, global spatial information을 channel descriptor로 **squeeze**하는 것을 제안한다.
>**Squeeze**는 쥐어 짜낸다는 의미를 가지고 있다.

<br/>
이 목적은 GAP를 통해 channel-wise statistic을 얻음으로써 달성된다.
>Global Average Pooling

<br/>
공식적으로 statistic $$z\in \Bbb{R}^C$$는, $$U$$를 spatial dimension $$H\times W$$에 대해 축소하여 얻어내므로, $$z$$의 $$c$$-th element는 **Eqn.2**에 따라 계산된다.

<br/>
>**Eqn.2**
>
>$$z_c = F_{sq}(u_c) = \frac{1}{H\times W}\sum_{i=1}^H \sum_{j=1}^W {u_{c}(i,j)}$$

<br/>
**(Discussion)**
- Transformation의 출력인 $$U$$는, 전체 이미지에 대한 statistic이 표현되는 local descriptor의 collection으로 해석될 수 있다.

- 이러한 정보를 이용하는 것은, 이전의 feature engineering task에서 널리 사용됐었다.
>[60], [61], [62]

- 우리는 가장 단순한 aggregation 기법인 GAP를 택했으며, 보다 정교한 전략이 여기에서 사용될 수 있음을 지적한다.

<br/>
### 3.2. Excitation: Adaptive Recalibration
**Squeeze**에서 얻은 aggregated information을 사용하기 위해, channel-wise dependency를 모두 캡처하는 두 번째 operation을 수행한다.

<br/>
이 목표를 달성하기 위해선, 이 operation은 다음의 두 가지 기준을 충족해야 한다.
- **Flexible**해야 한다.
>특히, channel 간의 non-linear interaction을 학습할 수 있어야 한다.

- 여러 channel을 강조할 수 있도록, **non-mutually-exclusive relationship을 학습**해야 한다.
>Mutually-exclusive relationship은 one-hot activation에 해당한다.

<br/>
이러한 기준들을 충족시키기 위해, sigmoid activation을 사용하는 간단한 gating mechanism을 채택한다. Operation은 **Eqn.3**에 따라 동작된다.

<br/>
>**Eqn.3**
>
>$$s = F_{ex}(z, W) = \sigma(g(z, W)) = \sigma(W_2\delta(W_1z))$$.

<br/>
**Eqn.3**의 각 notation은 다음을 의미한다.
- $$\delta$$ is ReLU

- $$W_1 \in \Bbb{R}^{\frac{C}{r}\times C}$$.

- $$W_2 \in \Bbb{R}^{C\times \frac{C}{r}}$$.

<br/>
Model complexity를 제한하면서 일반화를 돕기 위해, non-linearity 주위에 2개의 FC layer로 bottleneck을 형성함으로써 gating mechanism을 parameter화 한다.
>여기서 bottleneck은 dimensionality-reduction layer에 해당한다.

<br/>
즉, reduction ratio가 $$r$$ **bottleneck**과 **ReLU**, 그리고 $$U$$의 channel dimension으로 되돌아가기 위한 **dimensionality-increasing layer**로 이루어진다.
>3.3절의 **Fig.2**와 **Fig.3**을 참조하면 쉽게 알 수 있다.

<br/>
SE block의 최종 출력은 activation $$s$$로 $$U$$를 rescaling하여 얻는다. 이 operation은 **Eqn.4**를 따른다.

<br/>
>**Eqn.4**
>
>$$\widetilde{x}_c = F_{scale}(u_c, s_c) = s_c u_c $$.

<br/>
**Eqn.4**의 각 notation은 다음을 의미한다.
- $$\widetilde{X} = [\widetilde{x}_1, \widetilde{x}_2, ... , \widetilde{x}_C]$$.

- $$F_{scale}(u_c, s_c)$$ is channel-wise multiplication

- $$s_c$$ is scalar

- $$u_c \in \Bbb{R}^{H\times W}$$ (feature-map)

<br/>
결국 요약하자면 **Squeeze-and-Excitation**은 다음의 연산들이 순서대로 동작하는 것이다.
- **[GAP] - [FC-ReLU-FC-Sigmoid-Scale]**
>각각 **[Squeeze] - [Excitation]** operation에 해당하며, Excitation 초반의 FC와 ReLU은 bottleneck이므로 reduction ratio $$r$$의 비율만큼 channel이 감소된 layer이다.

<br/>
**(Discussion)**
- Excitation operator는 input-specific descriptor인 $$z$$를 channel weight set에 mapping한다.
>여기서 input-specific descriptor ** $$z$$ **는 **squeeze** operation의 결과에 해당한다. 즉, GAP를 통과한 ** $$z$$ **를 channel-wise weight로써 사용할 수 있게 해주는 것이 **excitation** operation으로 볼 수 있다.

<br/>
- SE block은 본질적으로 입력에 따라 조절되는 dynamic한 성질을 도입하는데, 이는 convolution filter가 동작되는 local receptive field에만 국한되지 않은 relationship을 갖는 channel 상에서의 self-attention 동작으로 볼 수 있다.
>다소 복잡하게 설명하지만, 결국 3.1절의 도입부에 설명한 문제점을 극복하는 동작에 해당하는 것으로 볼 수 있다는 말이다.
>
>그 문제점은, convolution filter가 학습될 때, local receptive field 상의 정보만을 이용하기 때문에 발생하는 한계점에 해당한다.

<br/>
### 3.3. Instantiations
SE block은 각 convolution에 따라오는 non-linearity 뒤에 삽입하는 방식으로 VGGNet과 같은 기존의 아키텍처에 통합될 수 있다.

<br/>
또한, SE block의 flexibility는 standard convolution 외의 transformation에도 직접 적용될 수 있음을 의미한다.

<br/>
이 점을 설명하기 위해, SE block을 보다 복잡한 아키텍처들에 통합하여 SENet을 구성한다.

<br/>
먼저 [Inception network](https://arxiv.org/pdf/1409.4842.pdf)를 위한 SE block의 구성을 고려하자.

<br/>
여기서는 단순히 Inception module 전체에 대해 transformation $$F_{tr}$$을 취하며, Inception network 내의 각 Inception module에 이를 적용함으로써 SE-Inception network를 얻는다. (**Fig.2** 참조)

<br/>
![Fig.2](/blog/images/SENet, Fig.2(removed).png )
>**Fig.2** <br/>The schema of the original Inception module (left) and the SE-Inception module (right).

<br/>
또한, SE block은 residual network에도 직접 사용될 수 있다. (**Fig.3** 참조)

<br/>
![Fig.3](/blog/images/SENet, Fig.3(removed).png )
>**Fig.3** <br/>The schema of the original Residual module (left) and the SEResNet module (right).

<br/>
여기서, SE block의 transformation $$F_{tr}$$은 residual module의 non-identity branch로 사용된다. 즉, **Squeeze**와 **Excitation**은 모두 identity branch와 summation 되기 전에 수행된다.

<br/>
[ResNeXt](19), [Inception-ResNet](21), [MobileNet](64), [ShuffleNet](65)에 SE block을 통합한 버전도 비슷한 방식으로 구성할 수 있다.

<br/>
SENet architecture의 구체적인 예시로, SE-ResNet-50과 SE-ResNeXt-50 구조에 대한 자세한 설명이 **Table.1**에 나와 있다.

<br/>
![Table.1](/blog/images/SENet, Table.1(removed).png )
>**Table.1** <br/>(Left) ResNet-50. (Middle) SE-ResNet-50. (Right) SE-ResNeXt-50 with a 32x4d template.
>
>$$f_c$$는 excitation operation의 두 FC layer에 해당한다.

<br/>
SE block의 flexible한 특성에 따른 하나의 결과는, 이러한 아키텍처들에 통합할 수 있는 방법이 여러 개 존재한다는 것이다.

<br/>
따라서 SE block을 아키텍처에 통합할 때 사용하는 통합 전략(integration strategy)의 sensitivity를 평가하기 위해, SE block의 포함을 위한 다양한 디자인 탐색의 ablation experiment도 제공한다. (6.5절 참조)


---
## 4. Model and Computational Complexity
제안 된 SE block design이 실용화 되려면, 성능과 model complexity 간의 trade-off를 고려해야 한다.

<br/>
모듈과 관련된 계산 부담을 설명하기 위해, ResNet-50과 SE-ResNet-50를 비교한다.

<br/>
ResNet-50은 224x224 pixel input image에 대한 single forawd pass에 ~3.86 GFLOPs가 필요하다.

<br/>
각 SE block은 **squeeze** 단계에서 GAP, **excitation** 단계에서는 2개의 작은 FC layer와 그 뒤에는 비용이 저렴한 channel-wise scaling이 수행된다.

<br/>
Aggregation에서 reduction ratio $$r$$을 16으로 설정할 때, SE-ResNet-50은 ~3.87 GFLOPs를 필요로 한다.
>이는 original ResNet-50에 비해, 0.26%의 상대적 증가에 해당한다.

<br/>
약간의 추가적인 계산 부담을 감수한 SE-ResNet-50의 정확도는 ResNet-50의 정확도를 능가하며, 실제로 ~7.58 GFLOPs를 요구하는 ResNet-101의 정확도에 근접한다. (5.1절의 **Table.2** 참조)

<br/>
실질적으로, ResNet-50의 single pass forward 및 backward에는 190ms가 소요됐으며, SE-ResNet-50의 경우에는 209ms가 소요됐다.
>NVIDIA Titan X GPU가 8개 있는 서버에서, traning minibatch를 256으로 한 경우에 측정됐다.

<br/>
이는 GPU library에서 GAP와 small inner-product operation에 대한 추가 최적화를 진행한다면, 합리적인 runtime overhead가 될 수 있다고 한다.

<br/>
Embedded device appliction의 중요성에 따라, 각 모델의 CPU inference time도 벤치마크 한다.
- 224x224 pixel input image의 경우, ResNet-50은 164ms, SE-ResNet-50은 167ms 소요됐다고 한다.

<br/>
SE block에 의해 발생하는 작은 계산 비용은, 모델 성능에 대한 contribution에 의해 정당화된다고 생각한다.

<br/>
다음으로는 SE block에 의해 도입 된 추가 parameter를 고려한다.

<br/>
추가되는 parameter들은 gating mechanism의 두 FC layer에서만 발생되므로, 전체 네트워크 capacity의 작은 일부에 해당한다.

<br/>
FC layer에 의해 추가된 parameter의 수는 구체적으로 **Eqn.5**를 따른다.

<br/>
>**Eqn.5**
>
>$$\frac{2}{r}\sum_{s=1}^S N_s\cdot {C_s}^2$$
>
>GAP와 bottleneck FC layer 간의 parameter와 두 FC layer 간의 parameter에 해당한다.

<br/>
**Eqn.5**의 각 notation은 다음을 의미한다.
- $$r$$ : reduction ratio

- $$S$$ : stage
>Stage는 동일한 크기의 feature map에서 동작되는 block collection의 단위를 말한다. 즉, 56x56 feature map을 출력하는 3개의 block은 동일한 stage에 해당한다.

- $$C_s$$ : dimension of the output channels

- $$N_s$$ : number of repeated blocks for stage $$s$$
>Bias가 사용될 때 추가되는 parameter의 및 계산 비용은 무시할만한 수준이다.

<br/>
SE-ResNet-50은 ResNet-50에 필요한 ~25M parameter에 ~2.5M parameter가 추가로 도입되며, 이는 ~10% 증가에 해당한다.

<br/>
실제로 이 parameter들의 대부분은 네트워크의 final stage로부터 나오며, **excitation** operation은 channel이 가장 많은 곳에서 수행된다.
>Stage 2인 경우에는 channel이 256인 위치에서 수행 됨.

<br/>
상대적으로 비용이 큰 final stage의 SE block을 제거한 경우에는, parameter 증가율을 ~4%로 줄이면서도 성능 손실은 미미하다는 사실도 발견했다. 이는 parameter usage가 주 고려사항인 경우에 유용할 수 있는 방법이다. 이에 대한 설명은 6.4절 및 7.2절을 참조하자.
>Final stage의 SE block을 제거한 경우, ImageNet에 대한 top-5 error에서 0.1% 미만의 작은 성능 손실만 일어났다.

---
## 5. Experiments
이 장에서는 다양한 task, dataset, architecture 상에서 SE block의 효과를 알아보는 실험을 수행한다.

<br/>
### 5.1. Image Classification
SE block의 영향을 평가하기 위해, 먼저 1.28M개의 training image와 50K개의 validation image로 구성된 1000-class ImageNet 2012 dataset에 대해 실험한다.

<br/>
Training set에서 네트워크를 학습시키고, validation set에 대한 top-1/top-5 error를 측정한다.

<br/>
각 baseline 아키텍처와, 이와 대응되는 SE 아키텍처는 동일한 optimization setting에서 학습한다.

<br/>
실험에서는 표준 관행들을 따르며, data augmentation을 수행한다.
- Random cropping using scale and aspect ratio to a size of 224x224 pixels
>[Inception-ResNet-v2]와 [SE-Inception-ResNet-v2]의 경우, 299x299 pixels

- Random horizontal flipping

<br/>
각 input image는 mean RGB-channel substraction을 통해 정규화한다.

<br/>
모든 모델은 *ROCS*에서 학습한다.
>대규모 네트워크의 효율적인 parallel training을 위해 설계된 distributed learning system이다.

<br/>
학습 관련 hyperparameter는 다음과 같다.
- Synchronous SGD with momentum 0.9

- Minibatch size : 1024

- Initial learning rate : 0.6

- 30 epoch마다 learning rate를 10으로 나눔

- Weight initializer : He initialization

- Trained for 100 epochs

<br/>
SE block의 reduction ratio $$r$$는 16을 기본 값으로 사용하며, 따로 명시한 경우에는 해당 값을 적용한다.

<br/>
모델 evaluation에서는 shorter edge가 256이 되도록 resize한 후, 224x224 pixel로 centre-cropping 한다.
>[Inception-ResNet-v2]와 [SE-Inception-ResNet-v2]의 경우에는 shorter edge가 352가 되도록 resize한 후, 299x299 pixel로 centre-cropping 한다.

<br/>
### Network depth
SE-ResNet과 ResNet 아키텍처를, 여러 depth에 대해 비교한 결과를 **Table.2**에서 보인다.

<br/>
![Table.2](/blog/images/SENet, Table.2(removed).png )
>**Table.2** <br/>(Left) Single-crop error rates (%) on the ImageNet validation set and complexity comparisons.
>
>Original은 [여기](https://github.com/Kaiminghe/deep-residual-networks)를 따른다.
>
>공정한 비교를 위해 baseline 모델을 re-implementation하여 학습한 결과와, 대응하는 SENet 버전의 결과를 모두 보여준다.
>
>괄호 안의 숫자는 re-implementation에 대한 성능 향상도를 나타낸다.
>
>$$\dagger$$는 validaion set 중, non-blacklisted subset에 대해 측정된 성능이다.
>
>VGG-16과 SE-VGG-16은 BN을 사용하여 학습됐다.

<br/>
실험을 통해, SE block은 계산량이 매우 작게 증가하면서 여러 depth에서 일관적으로 성능을 향상시키는 것을 관찰할 수 있었다.

<br/>
놀랍게도, SE-ResNet-50은 single-crop top-5 validation error가 6.62%로 측정됐으며, 이는 ResNet-50의 7.48%보다 0.86% 낮고, 훨씬 더 깊은 ResNet-101의 6.52%와 비슷한 결과이다.
>SE-ResNet-50은 3.87 GFLOPs의 계산이 필요하며, 이는 ResNet-101의 7.58 GFLOPs보다 절반 정도밖에 되지 않는다.

<br/>
또한, 이 성능 향상 패턴은 더 깊은 네트워크에서도 반복됐다.
- SE-ResNet-101의 6.07% top-5 error는 ResNet-152의 6.34%보다 0.27% 높은 성능을 보인다.

<br/>
**SE block** 자체가 depth를 추가한다는 점은 유의해야하지만, 매우 효율적인 계산 방식으로 동작하며, 기본 아키텍처의 depth가 늘어남에 따른 return이 감소하는 지점에서도 좋은 결과를 얻을 수 있다.

<br/>
또한, 다양한 depth의 네트워크에서 성능 향상이 일관되게 이뤄졌으며, 이는 SE block에 의해 유도 된 개선이 네트워크의 depth를 단순히 증가시킨 것과 보완적일 수 있음을 시사한다.

<br/>
### Integration with modern architectures
다음은 SE block을 두 개의 최신 아키텍처인 [Inception-ResNet-v2]()과 [ResNeXt](19)(32x4d로 setting)에 통합한 효과에 대해 알아본다. 두 아키텍처는 모두 추가적인 computational building block을 기존 네트워크에 도입한다.

<br/>
위 두 아키텍처에 대응하는 SENet 버전인 SE-Inception-ResNet-v2과 SE-ResNeXt를 구성하고, 성능 측정 결과를 **Table.2**에 보인다.
>SE-ResNeXt-50의 구성은 **Table.1**에 제공된다.

<br/>
이전 실험에서와 마찬가지로, 두 아키텍처는 모두 SE block을 도입함으로써 성능이 크게 향상됐다.

<br/>
특히 SE-ResNeXt-50의 경우 top-5 error는 5.49%로 측정됐으며, 이는 직접 대응되는 ResNeXt-50(5.90%)과 더 깊은 ResNeXt-101(5.57%) 보다 우수하다.
>ResNeXt-101은 SE-ResNeXt-50에 비해, 총 parameter의 수와 computational overhead가 거의 두 배에 해당한다.

<br/>
실험에서는 [Inception-ResNet-v2]()을 직접 re-implementation하여 측정했으며, 이는 논문에서 측정된 결과와 약간의 성능 차이가 있다.

<br/>
SE-Inception-ResNet-v2은 top-5 error가 4.79%로 측정됐으며, 이는 re-implemented Inception-ResNet-v2의 5.21% 보다 0.42% 우수한 성능이다.

<br/>
또한 [VGG-16]()과 [BN-Inception]()에 대한 실험을 통해, non-residual 네트워크에서 동작할 때의 SE block 영향을 평가한다.

<br/>
VGG-16을 scratch로부터 쉽게 학습할 수 있도록, 각 convolution 뒤에 BN을 추가한다.
>Learned weight를 load하지 않고, 각종 initializer를 통해 초기화 한 weight를 scratch라고 한다.

<br/>
마찬가지로, VGG-16과 SE-VGG-16는 동일한 setting에서 학습됐다. 성능 비교는 **Table.2**를 참조하자.

<br/>
Residual network에 대한 결과와 유사하게, SE block이 non-residual setting에서 성능을 개선함을 관찰했다.

<br/>
SE block이 이러한 모델들의 최적화에 미치는 영향에 대한 insight를 제공하기 위해, baseline 아키텍처와 해당 SE 버전의 training curve를 **Fig.4**에서 보인다.

<br/>
![Fig.4](/blog/images/SENet, Fig.4(removed).png )
>**Fig.4** <br/>Training baseline architectures and their SENet counterparts on ImageNet.

<br/>
SE block은 최적화 과정 전반에 걸쳐 꾸준한 개선을 가져오며, 이 추세는 다양한 baseline 네트워크 아키텍처에서 상당히 일관된다.

<br/>
### Mobile setting
마지막으로는 mobile-optimized network 중 대표적인 [MobileNet](64)과 [ShuffleNet](65) 두 가지 아키텍처를 고려합니다.

<br/>
이 실험에서는 minibatch size를 256으로 하며, [ShuffleNet](65)에서와 같이 data augmentation과 regulization을 약간 덜 적극적으로 사용한다.

<br/>
8개의 GPU에서 학습됐으며, 학습 관련 hyperparameter는 다음과 같다.
- SGD with momentum 0.9

- Initial learning rate : 0.1

- Validation loss가 줄어들지 않을 때마다 10으로 나눔

- ~400 epochs
>[ShuffleNet](65)의 baseline 성능을 재현하기에 이정도 필요했다고 함.

<br/>
**Table.3**에 나온 결과는, SE block이 계산 비용을 최소로 증가시키면서도 일관된 큰 성능 향상도를 보여준다.

<br/>
![Table.3](/blog/images/SENet, Table.3(removed).png )
>**Table.3** <br/>Single-crop error rates (%) on the ImageNet validation set and complexity comparisons.
>
>[MobileNet](64)은 해당 논문의 "1.0 MobileNet-224"를 따르며, [ShuffleNet](65)은 해당 논문의 "ShuffleNet $$1\times (g=3)$$을 따른다.
>
>괄호 안의 숫자는 re-implementation에 대한 성능 향상도다.

<br/>
### Additional datasets
다음은 SE block의 이점이 ImageNet 외의 dataset까지 일반화되는지 알아본다.

<br/>
CIFAR-10과 CIFAR-100에서 널리 사용된 몇 가지 baseline 아키텍처와 테크닉으로 실험한다.
- [ResNet](14) (ResNet-110, ResNet-164)

- [WideResNet](67) (WideResNet-16-8)

- [Shake-Shake](68)

- [Cutout](69)

<br/>
**CIFAR-10 / CIFAR-100**
- 32x32 RGB image

- 10-class / 100-class

- 50K training set

- 10K test set

<br/>
이러한 네트워크와 SE block의 통합은, 3.3절과 동일한 접근 방식을 따른다.

<br/>
전처리는 다음과 같다.
- 각 baseline과 해당하는 SENet 버전은 표준 data augmentation을 이용
>Data augmentation [24], [71]

- Randomly horizontally flip

- 32x32로 random crop 하기 전, 각 면에 4 pixel만큼 zero-padding

- Mean/std normalization

<br/>
학습 hyperparameter 설정은 원본 논문에서 제안한 것과 동일하다.
>Minibatch size, initial learning rate, weight decay 등

<br/>
각 baseline 및 해당 SENet 버전의 성능은 **Table.4**과 **Table.5**에서 보인다.

<br/>
![Table.4](/blog/images/SENet, Table.4(removed).png )
>**Table.4** <br/>Classification error (%) on CIFAR-10.

<br/>
![Table.5](/blog/images/SENet, Table.5(removed).png )
>**Table.5** <br/>Classification error (%) on CIFAR-100.

<br/>
모든 성능 비교에서 SENet 버전이 baseline 아키텍처보다 성능이 뛰어났으며, 이는 SE block의 이점이 ImageNet dataset에만 국한되지 않음을 시사한다.

<br/>
### 5.2. Scene Classification
또한, Scene classification을 위한 [Places365-Challenge](73) dataset에 대한 실험도 수행한다.
- 365-class

- 8M training set

- 36.5K validation set

<br/>
Scene understanding 작업은 모델의 일반화 성능과 추상화 처리 능력에 대한 대안적인 평가를 제공한다.
>종종 모델이 더 복잡한 data association을 다뤄야 하며, 더 다양한 수준의 변화에 강인해야하기 때문이다.

<br/>
SE block의 효과를 평가하기 위한 baseline으로 ResNet-152를 채택했으며, 학습은 두 논문에서 사용한 프로토콜을 따른다.
>두 논문 [72], [74]

<br/>
이 실험도 scratch로부터 학습했으며, **Table.6**에서 성능 비교 결과를 보여준다.

<br/>
![Table.6](/blog/images/SENet, Table.6(removed).png )
>**Table.6** <br/>Single-crop error rates (%) on Places365 validation set.

<br/>
SE-ResNet-152의 top-5 error는 11.01%로 측정됐으며, 이는 ResNet-152의 11.61% 보다 낮다.
>SE block이 scene classification 성능도 개선할 수 있다는 증거가 된다.

<br/>
또한, 이 성능은 이 dataset에서의 SOTA 모델인 [Places-365-CNN](72)의 11.48%을 능가한다.

<br/>
### 5.3. Object Detection on COCO
이번에는 [COCO dataset](75)을 사용하여, object detection 작업에 대한 SE block의 효과를 추가로 평가한다.

<br/>
이 실험에서는 [ResNext](19)에서와 동일하게 **minival** 프로토콜을 따른다.
- 80K training set과 vallidation set의 35K subset를 모델 학습에 이용하고, 나머지 validation set인 5K subset에 대해 평가한다.

<br/>
Weight는 ImageNet dataset에 대해 학습 된 weight로 초기화 한다.

<br/>
모델의 평가를 위한 base detection framework로는 [Faster R-CNN](4)을 사용하고, [여기](76)에서 사용한 hyperparameter를 사용한다.
>즉, 학습 스케쥴이 2배에 해당하는 ene-to-end 학습 과정이 된다.

<br/>
실험의 목표는 object detector의 trunk architecture인 ResNet을 SE-ResNet으로 대체한 것을 평가하여, 성능의 변화가 더 나은 representation에 의한 것일 수 있도록 한다.

<br/>
**Table.7**은 ResNet-50, ResNet-101과 대응하는 SE 버전을 trunk architecture로 사용하여 validation set에 대한 성능을 보여준다.

<br/>
![Table.7](/blog/images/SENet, Table.7(removed).png )
>**Table.7** <br/>Faster R-CNN object detection results (%) on COCO minival set.

<br/>
- 표준 AP metric에 대해 SE-ResNet-50은 2.4% 향상(6.3%의 상대 개선)됐으며, AP@IoU=0.5에 대해서는 3.1% 향상됐다.

- 더 깊은 SE-ResNet-101의 경우도 AP metric에 대해 2.0% 향상(5.0% 상대 개선)됐다.

<br/>
즉, 이 실험은 SE block의 일반성을 보여준 것으로 요약될 수 있다.

<br/>
다양한 architecture/task/dataset에서 성능 향상을 이룰 수 있다.

<br/>
### 5.4. ILSVRC 2017 Classification Competition
SENet는 ILSVRC classification에서 1위를 차지하게 되는 토대가 됐다.

<br/>
Winning entry는 표준 multi-scale 및 multi-crop을 사용한 small ensemble이며, test set에 대한 top-5 error가 2.251%를 달성했다.

<br/>
이 submission의 일부로, SE block을 수정한 [ResNeXt](19)와 통합하여 SENet-154를 추가로 구성했다.
>SENet-154의 자세한 아키텍처 설명은 Appendix에 제공된다.

<br/>
SENet-154의 성능은 224x224 및 320x320 pixel로 crop하여 측정했으며, **Table.8**에서는 ImageNet validation set에 대한 성능 비교를 보여준다.

<br/>
![Table.8](/blog/images/SENet, Table.8(removed).png )
>**Table.8** <br/>Single-crop error rates (%) of state-of-the-art CNNs on ImageNet validation set with crop sizes 224x224 and 320x320 / 299x299.


<br/>
SENet-154는 성능은 224x224 centre crop evaluation에서 측정된 것이며, top-1 error는 18.68%, top-5 error는 4.47%로 나왔다.
>224x224 centre crop evaluation이 성능이 가장 좋았다고 함.

<br/>
대회에 이어서 ImageNet 벤치마크에는 많은 진전이 있었으며, 가장 성능이 좋은 것으로 알고 있는 모델들과의 비교를 **Table.9**에서 보여준다.

<br/>
![Table.9](/blog/images/SENet, Table.9(removed).png )
>**Table.9** <br/>Comparison (%) with state-of-the-art CNNs on ImageNet validation set using larger crop sizes/additional training data.
>
>SENet-154는 320x320 pixel로 crop하여 학습한 성능이다.

<br/>
ImageNet data만 사용한 best performance는 [최근 연구](79)에서 보였다.
>이 연구는 [searched architecture](31)의 성능을 올리기 위해, reinforcement learning을 사용한 data augmentation 정책을 개발한다.

<br/>
또 최근의 [다른 연구](80)는 ResNeXt-101 32x48d 아키텍처를 사용하여, 전체에서의 best performance를 보였다.
>이 연구는 모델을 약 10억 개의 weakly labeled image에서 pre-training하고, ImageNet에 대한 fine-tuning을 수행한다.

<br/>
보다 정교한 data augmentation과 대규모 pre-training에 의한 이러한 개선 사항들은, SENet 아키텍처를 보완할 수 있을 것으로 보인다.

---
## 6. Ablation Study
이 장에서는 ablation 실험를 수행하여, SE block을 구성하는 여러 configuration의 효과를 잘 이해하고자 한다.
>Ablation study는 복잡한 네트워크 상에서 제안하는 방법의 동작을 이해하기 위해, 특정 모듈에 대한 변경 혹은 제거하는 등의 접근 방식이다. [여기](https://stats.stackexchange.com/questions/380040/what-is-an-ablation-study-and-is-there-a-systematic-way-to-perform-it?answertab=votes#tab-top)를 참고하자.

<br/>
모든 ablation 실험은 8개의 GPU를 사용하는 단일 머신을 사용해서 ImageNet dataset에 대해 수행하며, 백본 아키텍처로는 ResNet-50을 사용한다.

<br/>
ResNet 아키텍처에 적용되는 **excitation** operation에서는, FC layer의 bias를 제거했을 때 channel dependency 모델링이 쉬워진다는 것을 발견했으며, 이 실험에서는 이러한 구성을 따른다.

<br/>
기타 학습 관련 세팅은 다음과 같다.
- Data augmentation은 5.1절의 방식을 따른다.

- 각 버전에서의 성능 상한을 확인하기 위해 초기 learning rate는 0.1로 하여, validation loss가 더 이상 줄어들지 않을 때까지 ~300 epoch 동안 학습한다.

- Learning rate가 10으로 나눠지는 시점이 총 3회 일어난다.

- 학습 중에는 [Label smoothing regularization](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)을 사용한다.

<br/>
>Validation loss plateau에 관계없이 [125/200/250]번 째 epoch에서 learning rate를 감소시키고 총 270 epoch 동안 학습한 경우, ResNet-50과 SE-ResNet-50의 top-1/top-5 error가 각각 23.21%/6.53%와 22.20%/6.00%로 측정됐다고 한다.

<br/>
### 6.1. Reduction ratio
**Eqn.5**에서 도입 된 reduction ratio **$$r$$**은, **SE block의 capacity 및 computational cost 변화에 관련된 hyperparameter**이다.

<br/>
이 hyperparameter에 따른 [ performance / computational cost ] 간의 trade-off를 알아보기 위해, 다양한 $$r$$ 값에 대한 SE-ResNet-50을 실험한다.

<br/>
**Table.10**은 다양한 reduction ratio에 대해서도 강력한 성능을 얻을 수 있음을 보여준다.

<br/>
![Table.10](/blog/images/SENet, Table.10(removed).png )
>**Table.10** <br/>Single-crop error rates (%) on ImageNet and parameter sizes for SE-ResNet-50 at different reduction ratios.

<br/>
$$r$$이 작을수록 모델의 parameter 수는 크게 늘어나지만, complexity 증가에 따른 성능 향상이 단조롭진 않았으며, $$r$$ = 16인 경우가 performance/complexity의 균형이 잘 맞았다.

<br>
실제로는 모든 네트워크에서 동일한 $$r$$ 값을 사용하는 것이 최적이 아닐 수 있으므로, base 아키텍처에 맞춰 tuning하면 추가적인 개선이 이루어질 수 있다.

<br/>
### 6.2. Squeeze Operator
**Squeeze** operator로써, global max pooling 대신 global aveage pooling을 사용하는 것에 대한 중요성을 알아본다.
>이 부분에 대해서는 잘 작동했기 때문에, 보다 정교한 대안을 고려하지 않았다.

<br/>
**Table.11**의 결과에 따르면 GMP과 GAP 모두 효과적이었으며, GAP가 약간 더 나은 성능을 보였으므로 **squeeze** operator로 선택된다.

<br/>
![Table.11](/blog/images/SENet, Table.11(removed).png )
>**Table.11** <br/>Effect of using different squeeze operators in SE-ResNet-50 on ImageNet (error rates %).

<br/>
SE block의 성능은 특정 aggregation operator의 선택에 크게 달라지지 않는다.

<br/>
### 6.3. Excitation Operator
**Excitation** 메커니즘의 non-linearity 선택에 따른 평가를 수행한다.

<br/>
**Table.12**는 sigmoid를 ReLU나 tanh로 대체한 경우와 성능을 비교한다.

<br/>
![Table.12](/blog/images/SENet, Table.12(removed).png )
>**Table.12** <br/>Effect of using different non-linearities for the excitation operator in SE-ResNet-50 on ImageNet (error rates %).

<br/>
Sigmoid를 tanh로 대체한 경우에는 성능이 약간 저하됐으며, ReLU의 경우에는 크게 저하됐다.
>ReLU를 사용한 경우, SE-ResNet-50의 성능이 ResNet-50보다 낮다.

<br/>
이는 SE block이 효과적이기 위해선, **excitation** operator를 신중하게 구성할 필요가 있다는 것을 암시한다.

<br/>
### 6.4. Different stages
SE block을 ResNet-50에 통합할 때, SE block의 영향을 여러 stage에서 알아본다.
>한 번에 한 stage씩 진행.
>
>같은 feature-map 크기를 가진 block들을 하나의 stage로 정의한다.

<br/>
특히, intermediage stage에 SE block을 추가한다.
- [stage 2 / stage 3 / stage 4]에 대한 결과가 **Table.13**에 나와있다.

<br/>
![Table.13](/blog/images/SENet, Table.13(removed).png )
>**Table.13** <br/>Effect of integrating SE blocks with ResNet-50 at different stages on ImageNet (error rates %).

<br/>
SE block은 아키텍처의 각 stage에서 도입될 때마다 성능이 개선됐다.

<br/>
또한, 서로 다른 stage의 SE block에 의해 얻어지는 이득들은, 네트워크 성능 향상을 위해 효과적으로 결합될 수 있다는 점에서 보완 적이다.

<br/>
### 6.5. Integration strategy
SE block을 기존 아키텍처에 통합 할 때, SE block의 위치가 성능에 미치는 영향을 평가하는 실험을 수행한다.

<br/>
제안 된 SE 디자인 외에도, 다음의 3가지 변형을 고려한다.
- SE-PRE block : SE block이 residual unit 전에 위치하는 경우

- SE-POST block : SE unit이 identity branch의 summation 뒤에 위치하는 경우.
>ReLU activation을 거친 뒤다.

- SE-Identity block : SE unit이 residual unit과 병렬로 수행되는 identity connection에 위치하는 경우.

<br/>
위 변형 버전들은 **Fig.5**에 나와 있으며, 각 변형들의 성능은 **Table.14**에 보인다.

<br/>
![Fig.5](/blog/images/SENet, Fig.5(removed).png )
>**Fig.5** <br/>SE block integration designs explored in the ablation study.

<br/>
![Table.14](/blog/images/SENet, Table.14(removed).png )
>**Table.14** <br/>Effect of different SE block integration strategies with ResNet-50 on ImageNet (error rates %).

<br/>
제안 된 SE와 SE-PRE, SE-Identity 버전의 block들은 각각 비슷한 성능을 보인 반면, SE-POST block을 사용하는 경우엔 성능이 저하됐다.

<br/>
이 실험은 SE unit으로부터의 성능 향상이, branch aggregation 이전에 적용되는 경우에 상당히 강력하다는 것을 나타낸다.

<br/>
또한, SE block을 residual unit 내로 이동시켜, 3x3 conv layer 바로 뒤에 배치한 변형도 구성한다.
>상기 실험에서의 각 SE block은 residual unit의 외부에 위치한다. **Fig.5**의 (b) 참조.

<br/>
3x3 conv layer는 더 적은 channel이 있기 때문에, SE block에 의해 추가되는 parameter의 수도 감소된다.
>ResNet-50 아키텍처의 residual block은 1x1/3x3/1x1 conv layer로 구성되며, 1~2번 째 conv layer가 bottleneck layer이기 때문이다.

<br/>
**Table.15**의 성능 비교는, **SE 3x3** 버전이 **standard SE**보다 적은 수의 parameter로도 유사한 성능을 달성하는 것을 보여준다.

<br/>
![Table.15](/blog/images/SENet, Table.15(removed).png )
>**Table.15** <br/>Effect of integrating SE blocks at the 3x3 convolutional layer of each residual branch in ResNet-50 on ImageNet (error rates %).

<br/>
즉, 특정 아키텍처에 대해 SE block을 어떻게 사용하냐에 따라 추가적인 효율성도 기대할 수 있다.
>이 실험의 범위를 벗어나는 추측이긴 함.

---
## 7. Role of SE Blocks
제안 된 SE block이 다양한 visual task에서 성능을 향상시킨 것으로 나타났지만, **squeeze** operation의 상대적 중요성과 **excitation** 메커니즘이 실제로 어떻게 작동하는지 이해하고자 한다.

<br/>
Deep network에서 학습 된 representation에 대한 철저한 이론적 분석은 여전히 어려운 과제이다. 따라서 practical function에 대한 최소한의 기본적 이해를 얻는 것을 목표로, SE block의 역할을 조사하기 위한 실험을 진행한다.

<br/>
### 7.1. Effect of Squeeze
**Squeeze**로부터 생성 된 global embedding이 실제로 성능 면에서 중요한 역할을 수행하는지 여부를 평가하기 위해, GAP를 수행하지 않으면서 동일한 수의 parameter를 추가한 SE block에 대해 실험한다.

<br/>
Pooling operation을 제거하고, **excitation**에서 두 개의 FC layer를 동일한 channel dimention을 가진 1x1 convolution으로 대체하여, 이를 **NoSqueeze**로 칭한다.

<br/>
SE block의 경우와 대조적으로, 이러한 point-wise convolution들은 local operator의 출력 함수로써 channel을 remapping 할 수 있다.

<br/>
일반적으로 deep network의 후반 layer에서는 theoretical한 global receptive field를 가지지긴 하지만, global embedding에 대한 direct access는 **NoSqueeze** 버전에서는 더 이상 할 수 없게 된다.

<br/>
두 모델의 performance 및 computational complexity는 **Table.16**에서 표준 ResNet-50 모델과 비교된다.

<br/>
![Table.16](/blog/images/SENet, Table.16(removed).png )
>**Table.16** <br/>Effect of Squeeze operator on ImageNet (error rates %).

<br/>
실험에서는 global information의 사용이 모델의 성능에 큰 영향을 미치며, **squeeze** operation의 중요성을 강조한다는 것을 알았다.

<br/>
또한, SE block을 사용하면 global information을 NoSqueeze 디자인에 비해 저렴하게 사용할 수 있다.

<br/>
### 7.2. Role of Excitation
이 장에서는 SE block에서 **excitation** operator의 기능을 명확한 그림으로 보여주기 위한 조사를 진행한다.

<br/>
SE-ResNet-50 모델의 activation example에 대해 연구하고, 네트워크의 다양한 depth 상에서 다른 class/image들에 대한 distribution을 조사한다.

<br/>
특히 **excitation**이 다른 class의 이미지들과, 같은 class 내의 이미지들에서 어떻게 다른지 이해하고자 한다.

<br/>
우선 다른 class에 대한 **excitation**의 distribution를 고려하자.

<br/>
먼저 ImageNet dataset으로부터 의미와 모양이 다양한 4개의 class를 sampling한다.
>goldfish, pug, plane, cliff에 해당한다. 아래 이미지 참조.
>
>![Extra.1](/blog/images/SENet, Extra.1(removed).png )

<br/>
그런 다음 validation set에서 각 class에 대한 50개의 sample을 추출하여, 각 stage의 마지막 SE block(downsampling 직전)에서 50개의 uniformly sampled channel에 대한 average activation을 계산한다. 계산 된 distribution은 **Fig.6**에 보인다.
>**Fig.6**에는 모든 class에 걸친 mean activation의 distribution도 포함된다.

<br/>
![Fig.6](/blog/images/SENet, Fig.6(removed).png )
>**Fig.6** <br/>Activations induced by the Excitation operator at different depths in the SE-ResNet-50 on ImageNet.

<br/>
**Excitation**의 역할에 대해 다음의 3가지 관찰했다.
- 네트워크 초반부의 layer에서는, 다른 class에 걸친 distribution이 매우 유사하다. **(SE_2_3)**
>네트워크의 초반부에서는, feature channel의 중요성이 다른 class에 의해 공유될 가능성이 있음을 시사한다.

<br/>
- 보다 후반부에서는, 각 channel의 가치가 class에 따라 훨씬 크게 달라진다. **(SE_4_6) 및 (SE_5_1)**
>서로 다른 class 간에는 feature들에 대한 선호도가 다르기 때문이다.
>
>이러한 결과는 이전 연구[[81](81)][[82](82)]의 결과와 일치한다. 즉, 보다 하위 layer 일수록 더 general(class-agnostic for classification task)한 feature를 학습하는 반면, 보다 상위 layer에서는 더 고차원의 feature를 학습한다[[83](83)].

<br/>
- 네트워크의 마지막 stage에서는 약간 다른 현상을 관찰했다.
>**(SE_5_2)**에서는 대부분의 activation이 1에 가깝게 포화된 상태의 경향을 보였다.
>>모든 activation이 1을 취하는 시점에서는 SE block이 identity operator로 줄어든다.
>
><br/>**SE_5_3**에서는 다른 class 간에도 약간의 scale 변화 외에는 거의 유사한 패턴을 보였다.
>>Classifier에서 tuning 될 수 있다.
>
><br/>이는 **SE_5_2**와 **SE_5_3**이, 상위의 block들 보다 네트워크 recalibration에 덜 중요하다는 것을 나타낸다.
>>마지막 stage의 SE block을 제거한 경우, 추가되는 parameter의 수를 크게 줄이면서도 미미한 성능 저하만 생긴다는 것을 보여준 4장의 내용과 일치한다.

<br/>
마지막으로, **Fig.7**에서는 두 sample class인 *goldfish*와 *plane*에 대해, 동일한 class 내의 이미지에 대한 activation의 mean/std를 보여준다.

<br/>
![Fig.7](/blog/images/SENet, Fig.7(removed).png )
>**Fig.7** <br/>Activations induced by Excitation in the different modules of SE-ResNet-50 on image samples from the goldfish and plane classes of ImageNet.

<br/>
여기서도 **Fig.6**의 inter-class visualization 결과와 일치하는 경향을 관찰했으며, 이는 SE block의 dynamic한 동작이 class/instance 둘 다에 따라 달라짐을 나타낸다.

<br/>
특히, 단일 class 내의 representation 다양성이 상당히 높은 네트워크의 상위 layer에서는, 식별 성능을 향상시키기 위해 feature recalibration의 이점을 취하는 것을 배운다[[84](84)].

<br/>
요약하자면, SE block은 instance-specific response를 생성하지만, 아키텍처의 여러 layer에서 모델의 class-specific needs를 지원한다.

---
## 8. Conclusion
본 논문에서는 **dynamic한 channel-wise feature recalibration을 수행**함으로써, **네트워크의 representational power를 향상**시키는 architectural unit인 **SE block**을 제안했다.

<br/>
다양한 실험에서 SENet의 효과가 입증됐으며, 여러 종류의 dataset/task에서 SOTA 성능을 달성했다.

<br/>
또한, SE block은 이전 아키텍처들이 channel-wise feature dependency를 적절하게 모델링할 수 없다는 점을 설명한다.

<br/>
이러한 insight가 강력한 discriminative feature가 요구되는 다른 task에서도 유용할 수 있기를 바란다.

<br/>
마지막으로, SE block에 의해 생성 된 feature importance(특징 중요도) 값은, model compression을 위한 pruning과 같은 다른 task에서도 사용될 수 있다.

---

<br/>
<br/>
{% include disqus.html %}
