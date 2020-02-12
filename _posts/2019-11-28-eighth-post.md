---
title: "Shake-shake regularization 번역 및 추가 설명과 Keras 구현"
date: 2019-11-28 13:19:11 -0400
tags: AI ComputerVision Paper Shake-Shake Regularization
categories:
  - Paper
toc: true
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Paper Information

GASTALDI, Xavier. **"Shake-shake regularization"**. arXiv preprint arXiv:1705.07485, 2017.
>[Paper](https://arxiv.org/pdf/1705.07485.pdf)

---
## Abstract
이 논문에서 소개하는 방법은 overfitting 문제에 직면한 딥 러닝 실무자를 돕기 위함이다.

<br/>
아이디어는 multi-branch network에서 **standard summation of parallel branches**를 **stochastic affine combination**으로 대체하는 것이다.

<br/>
Shake-shake regularization이 3-branch resnet에 적용되었을 때, CIFAR-10/CIFAR-100에서 2.86%/15.85%의 테스트 오류를 달성했다.
>Best single shot published results (당시 기준)

<br/>
Skip connection이나 BN이 없는 아키텍처에 대한 실험에서는 고무적인 결과를 보였으며, 수많은 응용 가능성을 열어줬다.

<br/>
저자의 코드는 [여기](https://github.com/xgastaldi/shake-shake)에서 제공된다.
>[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)를 기반으로 구현된 Lua 코드다.

---
## 1. Introduction
[ResNet](https://arxiv.org/pdf/1512.03385.pdf)은 ILSVRC&COCO 2015 competition에서 처음 소개됐으며, ImageNet detection/localization 및 COCO detection/segmentation 분야에서 1위를 차지했다.

<br/>
이후로도 성능 향상을 위해 많은 노력이 있었으며, depth/width/cardinality에 따른 영향을 조사한 연구가 있었다.
>Depth : [ResNet-v2](https://arxiv.org/pdf/1603.05027.pdf) / [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)<br/>
>Width : [Wide ResNet](https://arxiv.org/pdf/1605.07146.pdf)<br/>
>Cardinality : [ResNeXt](https://arxiv.org/pdf/1611.05431.pdf) / [Inception-ResNet-v2](https://arxiv.org/pdf/1602.07261.pdf) / [multi-resnet](https://arxiv.org/pdf/1609.05672.pdf)

<br/>
ResNet은 강력한 모델이지만, 소규모 데이터 셋에 대해서는 overfitting 문제가 발생할 수 있다.

<br/>
이 문제를 해결하기 위해 [weight decay](http://www.cs.toronto.edu/~fritz/absps/sunspots.pdf), early stopping, [dropout](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf) 등을 포함한 많은 기법들이 제안됐다.

<br/>
Regularization method로써 제시되지는 않았지만, [BN](https://arxiv.org/pdf/1502.03167.pdf)은 각 mini-batch 단위로 변동하는 통계에 따라 네트워크를 regularize한다.

<br/>
이와 유사하게 SGD는 noisy gradient를 이용하는 gradient descent로 해석될 수 있으며, [mini-batch 크기에 따른 네트워크의 일반화 성능에 대한 연구](https://arxiv.org/pdf/1609.04836.pdf)도 있었다.

<br/>
2015년 이전에는 대부분의 분류 아키텍처에서 overfitting을 방지하기 위해 dropout을 사용했었지만, BN의 도입으로 인해 그 효율성이 떨어졌다.
>그렇다고 한다 : [BN](https://arxiv.org/pdf/1502.03167.pdf) / [Wide ResNet](https://arxiv.org/pdf/1605.07146.pdf) / [Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)

<br/>
다른 regularization method를 찾기 위해 연구자들은 multi-branch network에서 특별히 제공되는 가능성을 살펴보기 시작했으며, 일부 연구에서는 학습 중 information path의 일부를 무작위로 drop 시킬 수 있음을 발견했다.
>올바른 조건(right condition)이 주어졌을 때에 한함 : [Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf) / [FractalNet](https://arxiv.org/pdf/1605.07648.pdf)

<br/>
위 마지막 두 연구처럼, 본 연구에서 제안하는 방법은 **standard summation of parallel branches**를 **stochastic affine combination**으로 대체함으로써 **multi-branch network의 일반화 능력을 향상**시키는 것을 목표로 한다.
>Affine combination은 linear combination에서의 계수 합을 1로 제한하는 것을 말하며, 논문에서는 2개의 branch에 곱해지는 scaling 계수가 이에 해당한다. ([참고](https://wikidocs.net/17412))
>
>2개의 branch가 있는 모델로 제안하는 regularization 기법의 성능을 검증하고 있다.

<br/>
### 1.1. Motivation
전통적으로 data augmentation 기법은 input image에만 적용이 되어왔지만, 컴퓨터에게는 input image와 intermediate representation 간에 실질적인 차이가 없다.

<br/>
따라서 data augmentation 기법을 internal representation에도 적용할 수 있을 것이다.

<br/>
Shake-Shake regularization은 **2개의 tensor를 stochastic하게 blending**함으로써 이러한 효과를 생성하려는 시도로 만들어졌다.
>Stochastic한 blend는 확률적인 혼합으로 직역될 수 있다.

<br/>
### 1.2. Model description on 3-branch ResNets
각 notation은 다음을 뜻한다.
- $$x_i$$ : residual block $$i$$의 input tensor.

- $$\mathcal{W}_i^{(1)}$$, $$\mathcal{W}_i^{(2)}$$ : 2개의 residual unit에 연관된 weight set.

- $$\mathcal{F}$$ : residual function.
>e.g. a stack of two 3x3 convolutional layers.

- $$x_{i+1}$$ : residual block $$i$$의 output tensor.


<br/>
2개의 residual branch를 가진 typical pre-activation ResNet은 **Eqn.1**을 따른다.

<br/>
>**Eqn.1**
>
>$$x_{i+1} = x_i + \mathcal{F}(x_i, \mathcal{W}_i^{(1)}) + \mathcal{F}(x_i, \mathcal{W}_i^{(2)})$$

<br/>
본 논문에서 제안하는 modification은 $$\alpha_i$$가 uniform distribution을 따르는 [0, 1] 범위의 랜덤 값인 경우, 학습 시에 **Eqn.2**를 따른다.

<br/>
>**Eqn.2**
>
>$$x_{i+1} = x_i + \alpha_{i}\mathcal{F}(x_i, \mathcal{W}_i^{(1)}) + (1 - \alpha_{i})\mathcal{F}(x_i, \mathcal{W}_i^{(2)})$$

<br/>
Dropout에서와 동일한 이유로, test time에는 모든 $$\alpha_{i}$$가 0.5로 설정된다.

<br/>
이 방법은 residual branch들이 완전하게 drop 되는 대신, scale-down이 수행되는 형태의 drop-path로 볼 수 있다.
>Drop-path는 [FractalNet](https://arxiv.org/pdf/1605.07648.pdf)에서 제안한 방법이며, 완전하게 drop 된다는 것은 0을 곱하는 것과 같다. 아래의 그림을 참고하자.
><br/>
>![Extra.1](/blog/images/Shake-Shake, Extra.1(removed).png )

<br/>
Binary variable 대신 향상/감소 계수로 대체하는 것은 [Shakeout](https://arxiv.org/pdf/1904.06593.pdf)이나 [Whiteout](https://arxiv.org/pdf/1612.01490.pdf)과 같은 dropout 변형 연구에서도 제안됐다.
>Binary variable은 dropout이나 drop-path와 같이 drop의 여부에 따른 두 케이스만 있는 경우를 말한다. 두 dropout 변형 연구의 핵심을 아래의 그림으로 요약한다.
>
>**Shakeout**
><br/>
>![Extra.2](/blog/images/Shake-Shake, Extra.2(removed).png )
>
><br/>
>**Whiteout**
><br/>
>![Extra.3](/blog/images/Shake-Shake, Extra.3(removed).png )

<br/>
위와 같은 기법들이 input tensor와 noise tensor 간에 element-wise multiplication을 수행할 때, Shake-Shake regularization은 전체 image tensor에 하나의 scalar $$\alpha_{i}$$(or $$1-\alpha_{i}$$)를 곱한다.

<br/>
### 1.3. Training procedure
<br/>
![Fig.1](/blog/images/Shake-Shake, Fig.1(removed).png )
>**Fig.1** <br/>연산 시점에 따른 scaling 계수 적용.
>
>**Left** : Forward traning pass.<br/>
>**Center** : Backward training pass.<br/>
>**Right** : At test time.

<br/>
**Fig.1**에 나와있듯, 모든 scaling 계수는 각 forward pass 전에 새로운 random 값으로 갱신한다.

<br/>
이 작업의 핵심은 각각의 backward pass 전에 scaling 계수의 갱신을 반복하는 것이다.

<br/>
이를 통해, 학습 중의 forward/backward flow가 stochastic하게 blend 된다.

<br/>
이 아이디어와 관련된 이전 연구에서는, 학습 중 gradient에 noise를 추가하면 복잡한 네트워크의 학습 및 일반화에 도움이 된다는 것을 보여줬다.
>연구 1,  [연구 2](https://arxiv.org/pdf/1511.06807.pdf)
>
>하나는 내용이 오픈 된 자료가 쉽게 보이지 않으므로 링크 생략.

<br/>
Shake-Shake regularization는 gradient noise를 **gradient augmentation** 형태로 대체하는 확장된 개념으로 볼 수 있다.

---
## 2. Improving on the best single shot published results on CIFAR

### 2.1. CIFAR-10

### 2.1.1. Implementation details
네트워크는 총 26-layer로 이루어져 있으며, 구조는 다음과 같다.
- 첫 번째 layer는 16개의 filter가 있는 3x3 conv.

- 각각 4개의 residual block이 포함된 3개의 stage가 뒤따라옴.
>각 stage의 feature map size는 32/16/8 이다.
>
>Downsampling 시에 width는 2배가 된다. (해당 layer의 filter 채널 수를 뜻함)

- 8x8 average pooling과 FC layer로 끝난다.

<br/>
Residual path는 다음의 구조를 따른다.
- ReLU - Conv(3x3) - BN - ReLU - Conv(3x3) - BN - Mul

<br/>
Skip connection은 기본적으로 identity function에 해당하며, downsampling이 필요한 경우에는 약간의 커스텀 된 구조를 사용한다.

<br/>
Downsampling 시에 사용되는 구조는 2개의 concatenated flow로 이루어져 있으며, 각 flow는 다음의 구성을 따른다.
- AvgPool(1x1) - Conv(1x1)
>이 경우에는 BN이나 ReLU를 사용하지 않는다.

<br/>
두 flow 중 하나는 다른 위치에서 average pooling 샘플을 만들기 위해, 입력을 우측 하단으로 1 pixel 만큼 shift 한다.
>예를 들어, stride가 2인 경우에는 각 flow가 입력 feature map에서 홀수/짝수 번째의 pixel만을 입력으로 취하게 되며, 1x1 average pooling은 각 flow가 pixel을 선택적으로 취하기 위한 일종의 연산 트릭으로 볼 수 있다.
>
>즉, 1x1 average pooling이 동작에 꼭 필요한 것은 아니다.

<br/>
두 flow를 concatenate하면 width가 2배로 된다.
>즉, 각 1x1 conv layer의 filter 개수는 입력과 동일하다.

<br/>
모델은 CIFAR-10 50k training set으로 학습됐으며, 10k test set에 대해 평가한다.
>32x32 크기의 RGB 이미지로 이루어짐.

<br/>
학습 방법에 관련된 내용은 다음과 같다.
- Standard tanslation/flipping data augmentation 적용.

- 1800 epoch 동안 학습.
>제안하는 기법에서 도입된 stochasticity 때문에 학습 시간을 길게 잡았다고 한다.

- Initial learning rate : 0.2

- [Cosine annealing](https://arxiv.org/pdf/1608.03983.pdf) without restart
>원 논문에서 제안한 cosine annealing을 적용하면 learning rate가 아래의 그래프와 같이 변화한다.
>
>![Extra.4](/blog/images/Shake-Shake, Extra.4(removed).png )
>
>Default에 해당하는 빨간색과 파란색을 제외한 그래프가 cosine annealing에 해당한다. 각자의 hyperparamter 값에 따라 주기는 다르지만 초기 learning rate로 되돌아가는 패턴이 반복되는데, **restart**는 이를 의미한다.
>
>즉, **cosine annealing without restart**는 전체 학습에 걸쳐 한 번의 감소 주기를 갖도록 한다. 이를 적용했을 때의 learning rate 변화는 아래의 그래프와 같다.
>
>![Extra.5](/blog/images/Shake-Shake, Extra.5(removed).png )

- Mini-batch size : 128

- 2 GPUs

- 기타 구현 세부 사항은 [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)와 동일.

<br/>
### 2.1.2. Influence of Forward and Backward training procedures
Base network는 26 2x32d ResNet이다.
>Network depth(26) / residual branch(2) / 첫 번째 residual block의 width(32).

<br/>
- **"Shake"** : 모든 scaling 계수가 각 pass 전에 새로운 random 값으로 갱신.

- **"Even"** : 모든 scaling 계수가 각 pass 전에 0.5로 설정.

- **"Keep"** : forward pass에서 사용 된 scaling 계수를 backward pass에서도 사용.

- **"Batch"** : 각 residual block $$i$$의 scaling 계수를 mini-batch 내의 모든 이미지에 대해 동일하게 적용.

- **"Image"** : 각 residual block $$i$$의 scaling 계수를 mini-batch 내의 각 이미지에 대해 다른 scaling 계수를 적용.
>하단의 image level update procedure 참조.

<br/>
**Image level update procedure**
- $$x_0$$ : dimension이 128x3x32x32인 original input mini-batch tensor.
>Mini-batch size(128) / CxHxW (3x32x32)
>
>예를 들어, 26 2x32d 모델의 두 번째 stage에서는 dimension이 128x64x16x16인 mini-batch tensor $$x_i$$로 변환 된다.

<br/>
- 첫 번째 dimension(mini-batch)을 따라 tensor를 slicing하고, 각 $$j^{th}$$ slice에 scalar $$\alpha_{i.j}$$(or $$1 - \alpha_{i.j}$$)를 곱한다.
>위의 예제에서는 slice가 128개, 각 slice의 dimension은 64x16x16이 된다.

<br/>
**Table.1**의 각 성능은 3회 측정 결과의 평균이며, 96d 모델의 경우에는 5회 측정 결과의 평균이다.

<br/>
![Table.1](/blog/images/Shake-Shake, Table.1(removed).png )
>**Table.1** <br/>Error rates (%) on CIFAR-10.
>
>모든 이전 결과들보다 0.1% 이상 성능이 좋은 경우는 bold체, 가장 좋은 결과는 파란색으로 표시.

<br/>
**Table.1**과 **Fig.2**에서 **"Shake-Keep"**은 오류율에 특히 큰 영향을 미치지 않는 것으로 나타났다.
>**Fig.2**에서는 머리만 따서 **"S-K"**와 같이 표기 함.
>
>"Shake" -> Forward -> "Keep" -> Backward 동작을 의미.
>
>즉, forward pass 전에 새로운 random 값으로 $$\alpha$$을 정하고, 이 값을 backward pass 시의 $$\beta$$로도 사용.

<br/>
![Fig.2](/blog/images/Shake-Shake, Fig.2(removed).png )
>**Fig.2** <br/>Regularization 방법에 따른 학습 그래프
>
>**Left** : Training curves of a selection of 32d models.<br/>
>**Right**: Training curves (dark) and test curves (light) of the 96d models.

<br/>
**"Even-Shake"**는 "Image" 레벨에서 적용된 경우에만 효과가 있었다.

<br/>
**"Shake-Even"**과 **"Shake-Shake"**는 모두 32d 모델에서 강력한 결과를 보였지만, **"Shake-Shake"**의 경우에는 첫 번째 residual block의 filter 수가 64d로 증가할 때 차이를 만들기 시작했다.
>모델의 capacity가 어느 정도 받쳐줘야 **"Shake-Shake"**이 더 효과적으로 작용한다는 결과로 볼 수 있다.

<br/>
Scaling 계수를 **"Image"** 레벨에서 적용하는 것이 regularization의 효과가 더 좋았다.

<br/>
이번에는 CIFAR-10에 대한 성능 평가 모델을 keras로 구현한다.
>구현은 아래 링크들을 참고했다.
>
>[Link-1](https://github.com/xgastaldi/shake-shake), [Link-2](http://research.sualab.com/practice/review/2018/06/28/shake-shake-regularization-review.html), [Link-3](https://github.com/jonnedtc/Shake-Shake-Keras)

``` python
n_blocks = 4
d = 32 # Width of the first shake_block.

def Shake_ResNet26(model_input, classes=10):
    x = Conv2D(16, (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(model_input) # (32, 32, 16)
    x = BatchNormalization()(x)
    
    x = shake_stage(x, d, n_blocks) # (32, 32, 32)
    x = shake_stage(x, d*(2**1), n_blocks) # (16, 16, 64)
    x = shake_stage(x, d*(2**2), n_blocks) # (8, 8, 128)

    x = GlobalAveragePooling2D()(x)
    
    model_output = Dense(classes, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x) # 'softmax'
    
    model = Model(inputs=model_input, outputs=model_output, name='Shake-ResNet26')
        
    return model

```
>3개의 shake_stage로 구성, 각 stage의 feature map size는 32/16/8이며, filter의 개수는 32/64/128으로 구현됐다.
>
>분명 본문에서는 downsampling 시에 filter 개수가 2배로 늘어난다고 했는데, 저자의 구현 코드를 보면 입력 filter의 개수와 출력 filter의 개수가 다르게 입력됐을 때 filter 개수를 2배로 늘리고 있다.
>
>이 부분 때문에 코드의 일부가 꼬여버렸다.

<br/>
``` python
def shake_stage(x, filters, blocks=4):
    strides = 2 if filters != d else 1
    
    x = shake_block(x, filters, strides) # projection layer

    for i in range(blocks-1):
        x = shake_block(x, filters, 1)
    
    return x
```
>첫 번째 shake_stage일 경우에만 downsampling을 수행하지 않도록 되어있으며, 각 shake_stage는 4개의 shake_block을 가진다.

<br/>
``` python
def shake_block(x, filters, strides=1):
    if strides == 1 and filters != d:
        residual = x  
    else:
        residual = shake_projection(x, filters, strides)
    
    branch_1 = Activation('relu')(x)
    branch_1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(branch_1)
    branch_1 = BatchNormalization()(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branch_1 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(branch_1)
    branch_1 = BatchNormalization()(branch_1)
    
    branch_2 = Activation('relu')(x)
    branch_2 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(branch_2)
    branch_2 = BatchNormalization()(branch_2)
    branch_2 = Activation('relu')(branch_2)
    branch_2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(branch_2)
    branch_2 = BatchNormalization()(branch_2)
    
    shaked_branches = ShakeShake()([branch_1, branch_2])
    
    return Add()([residual, shaked_branches])
    
```
>각 shake_block은 **ReLU - Conv(3x3) - BN - ReLU - Conv(3x3) - BN - Mul** 구조의 residual branch를 2개 가진다.
>
>본문에서는 downsampling이 일어날 시(strides != 1)에 커스텀 구조를 사용한다고 해놓고는, 위에서 말했듯이 저자의 구현 코드에서는 입력 filter의 개수와 출력 filter의 개수가 다르다는 이유로 첫 번째 stage에서도 커스텀 구조를 거친다.
>
>따라서 각 stage의 첫 번째 block의 경우에는 stride에 관계없이 shake_projection을 수행한 후에 addition하며, 나머지 block에서는 입력을 그대로 addition하는 identity mapping을 따른다.
>
>**Mul**에 해당하는 ShakeShake 부분은 아래의 shake_projection 다음에 설명한다.

<br/>
``` python
def shake_projection(x, filters, strides):
    x = Activation('relu')(x)
    
    proj_1 = Lambda(lambda y: y[:, 0::strides, 0::strides, :])(x)
    proj_1 = Conv2D(filters//2, (1, 1), strides=1, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(proj_1)
    
    if strides == 1:
        proj_2 = ZeroPadding2D( ((1, 0), (1, 0)) )(x)
        proj_2 = Lambda(lambda y: y[:, :-1, :-1, :])(proj_2)
        
    elif strides == 2:
        if K.int_shape(x)[1]%2 == 0:
            proj_2 = Lambda(lambda y: y[:, 1::strides, 1::strides, :])(x)
        else:
            proj_2 = ZeroPadding2D( ((1, 0), (1, 0)) )(x)
            proj_2 = Lambda(lambda y: y[:, 0::strides, 0::strides, :])(proj_2)
            
    proj_2 = Conv2D(filters//2, (1, 1), strides=1, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(proj_2)
    
    '''
    proj_1 = AveragePooling2D((1, 1), strides=strides)(x)
    proj_1 = Conv2D(filters//2, (1, 1), strides=1, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(proj_1)
    
    proj_2 = ZeroPadding2D( ((1, 0), (1, 0)) )(x)
    proj_2 = Lambda(lambda y: y[:, :-1, :-1, :])(proj_2)
    proj_2 = AveragePooling2D((1, 1), strides=strides)(proj_2)
    proj_2 = Conv2D(filters//2, (1, 1), strides=1, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(proj_2)
    '''
    
    concat = Concatenate()([proj_1, proj_2])
    
    return BatchNormalization()(concat)

```
>본문에서 설명하는 아키텍처는 하단에 주석처리한 부분에 해당한다.
>
>굳이 다른 방법으로 구현한 이유는 두 가지다. 우선 1x1 AvgPooling의 의도를 생각하면 굳이 사용할 필요가 없기 때문이고, 약간의 information loss가 발생하기 때문이다.
>
>본문의 방법대로 구현한 경우에 각 projection branch가 취하는 pixel은 아래의 그림과 같다.
>
>![Extra.9](/blog/images/Shake-Shake, Extra.9(removed).png )
>
>입력 feature map의 length가 짝수인 경우에는 proj_2가 zero padded pixel을 취하고, 우측과 하단의 1 pixel 만큼은 버리는 현상이 생긴다.
>
>이러한 information loss를 보완하기 위해, 위의 shake_projection 구현에서는 feature map 길이의 odd/even 여부에 따라 다르게 구현했다. 위와 같이 구현했을 때, 각 projection branch가 취하는 pixel은 아래의 그림과 같이 바뀐다.
>
>![Extra.10](/blog/images/Shake-Shake, Extra.10(removed).png )
>
>strides가 1인 경우도 작성한 이유는, 위에서 언급한 저자의 거짓말로 인해 꼬여버린 부분이다. 이 부분에는 stride가 1이기 때문에, 취하는 pixel의 차이를 유지하기 위해 우측 하단의 pixel을 버리는 현상을 보완하지 않았다.

<br/>
``` python
class ShakeShake(Layer):
    def __init__(self, **kwargs):
        super(ShakeShake, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ShakeShake, self).build(input_shape)

    def call(self, x):
        # unpack x1 and x2
        assert isinstance(x, list)
        x1, x2 = x
        
        forward, backward, level = shake_type.split('-')
        
        # create alpha and beta
        batch_size = K.shape(x1)[0] # K.int_shape(x1)[0]
        
        if level == 'B':
            alpha = K.random_uniform((1, 1, 1, 1))
            beta = K.random_uniform((1, 1, 1, 1))
            
            alpha = K.tile(alpha, (batch_size,1,1,1))
            beta = K.tile(beta, (batch_size,1,1,1))
            
        elif level == 'I':
            alpha = K.random_uniform((batch_size, 1, 1, 1))
            beta = K.random_uniform((batch_size, 1, 1, 1))
            
        def on_train():
            # Forward
            if forward == 'E':
                scaled_forward = 0.5*x1 + 0.5*x2
                
            elif forward in ['K', 'S']:
                scaled_forward = alpha*x1 + (1-alpha)*x2
            
            # Backward
            if backward == 'E':
                scaled_backward = 0.5*x1 + 0.5*x2
                
            elif backward == 'K':
                return scaled_forward
            
            elif backward == 'S':
                scaled_backward = beta*x1 + (1-beta)*x2
            
            return scaled_backward + K.stop_gradient(scaled_forward - scaled_backward)
        
        # E-E during testing phase
        def on_test():
            return 0.5*x1 + 0.5*x2
        
        return K.in_train_phase(on_train, on_test)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]

```
>shake_block의 **Mul**에 해당하는 부분이다. **E-E-B** ~ **S-S-I**에 해당하는 모든 타입들을 구현해뒀으며, 각 barnch의 출력에 scaling 계수를 곱한 후에 둘을 더하는 것까지 포함하여 출력으로 return 한다.
>
>K.stop_gradient()는 네트워크의 forward 연산 시에 identity로 동작하며, backward 연산 시에는 통째로 무시된다.
>
>즉, forward 시에는 **scaled_backward + scaled_forward - scaled_backward**가 되어, **scaled_forward**만 남으며, backward 시에는 **scaled_backward**만 남아있게 된다.
>
>K.in_train_phase()는 training 시에 첫 번째로 넘겨받은 인자를 수행하며, inference 시에는 두 번째로 넘겨받은 자를 수행한다.
>
>즉, inference 시에는 **Even-Even**으로 동작.

<br/>
``` python
class LearningRateSchedule(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = K.get_value(self.model.optimizer.lr)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / epochs))
        K.set_value(self.model.optimizer.lr, lr*cosine_decay)
```
>Cosine annealing을 callback 함수로 구현한다.

<br/>
``` python
datagen = ImageDataGenerator(horizontal_flip=True)

def data_generator(generator, X, Y, crop_shape=(32, 32), pad_length=4, batch_size=128):
    gen_X_Y = generator.flow(X, Y, batch_size=batch_size)
    
    while True:
        batch_x, batch_y = gen_X_Y.next()

        cropped_batch = []
        
        for img in batch_x:
            # zero padding
            padded_img = np.pad(img, ((pad_length, pad_length), (pad_length, pad_length), (0,0)), mode='constant')
            
            # random crop
            delta_h = np.random.randint(0, padded_img.shape[0] - crop_shape[0] + 1)
            delta_w = np.random.randint(0, padded_img.shape[1] - crop_shape[1] + 1)
            
            cropped_batch.append(padded_img[delta_h:(delta_h+crop_shape[0]), delta_w:(delta_w+crop_shape[1]), :])
            
        yield (np.stack(cropped_batch), batch_y)
```
>data_generator 호출 시에 generator로 datagen을 넘겨받는다. 
>
>gen_X_Y는 randomly horizontal flip이 적용된 입력을 batch 단위(shape=(128, 32, 32, 3))로 return 받는다. 
>
>각 이미지에 대해 4 pixel만큼 zero-padding을 수행하고 32x32 크기로 random crop을 수행한다.

<br/>
``` python
from keras.models import Model, Input
from keras.layers import Conv2D, GlobalAveragePooling2D, Activation, Dense, BatchNormalization, ZeroPadding2D
from keras.layers import Add, Concatenate, Layer, Lambda
from keras.optimizers import SGD
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from keras.utils import to_categorical
from keras.datasets import cifar10

import keras.backend as K
import tensorflow as tf

import numpy as np
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class ShakeShake(Layer):
    ...

def shake_projection(x, filters, strides):
    ...
    
def shake_block(x, filters, strides=1):
    ...
 
def shake_stage(x, filters, blocks=4):
    ...
    
def Shake_ResNet26(model_input, classes=10):
    ...

class LearningRateSchedule(Callback):
    ...

def data_generator(generator, X, Y, crop_shape=(32, 32), pad_length=4, batch_size=128):
    ...
    
shake_type = 'S-S-I' # 'Forward-Backward-Level' # Forward in {E,K,S} / Backward in {E,K,S} / Level in {B,I}
n_blocks = 4
d = 32 # Width of the first shake_block.

input_shape = (32, 32, 3)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')/255.
x_test =  x_test.astype('float32')/255.

mean_train = np.mean(x_train, axis=(0, 1, 2))
std_train = np.std(x_train, axis=(0, 1, 2))

x_train = (x_train - mean_train) / std_train
x_test = (x_test - mean_train) / std_train

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model_input = Input( shape=input_shape )

model = Shake_ResNet26(model_input, 10)

batch_size = 128
epochs = 1800
optimizer = SGD(lr=0.2, decay=1e-4, momentum=0.9)

model.compile(optimizer, 'categorical_crossentropy', ['acc'])

datagen = ImageDataGenerator(horizontal_flip=True)

filepath = 'weights/' + model.name + '.{epoch:02d}-{acc:.2f}-{val_acc:.2f}.hdf5'
callbacks_list = [ModelCheckpoint(filepath, 
                                  monitor='val_acc',
                                  verbose=1, 
                                  save_weights_only=True, 
                                  save_best_only=True, 
                                  mode='auto', 
                                  period=1),
                  ReduceLROnPlateau(monitor='val_loss', patience=epochs+1),
                  CSVLogger('logs/' + model.name + '.log'),
                  LearningRateSchedule()]

history = model.fit_generator(data_generator(datagen, x_train, y_train, (input_shape[0], input_shape[1]), 4, batch_size), 
                                      steps_per_epoch=50000//batch_size, 
                                      epochs=epochs, 
                                      callbacks=callbacks_list, 
                                      validation_data=(x_test, y_test))    

```
>학습을 수행하는 main 코드다.
>
>Callback 함수 중, ReduceLROnPlateau()의 patience를 epochs+1로 준 이유는 CSVLogger에 learning rate가 찍혀 나오도록 하기 위함이다. (cosine annealing으로 learning rate scheduling을 수행하기 때문에 동작할 필요가 없음.)

<br/>
### 2.2. CIFAR-100
CIFAR-100에 대한 평가에 사용된 아키텍처는 pre-activation을 제외한 [ResNeXt](https://arxiv.org/pdf/1611.05431.pdf) 모델이다.
>CIFAR-10에 사용 된 모델보다 약간 더 나은 결과를 보였다고 한다.

<br/>
각종 hyperparameter는 cosine annealing에 관련된 learning rate 및 epoch(1800)을 제외하고는 [ResNeXt](https://arxiv.org/pdf/1611.05431.pdf)의 원문과 동일하다.

<br/>
**Table.2**의 base network는 ResNeXt-29 2x4x64d 에 해당한다.
>Network depth(29) / residual branches(2) / cardinality(4) / bottleneck depth(64)
>
>Grouped convolution으로 구현하는 경우에는 bottleneck depth(64) 대신 group conv(256)이 된다. ([ResNeXt](https://arxiv.org/pdf/1611.05431.pdf)의 type C에 해당)
>
>즉, 하나의 residual branch는 아래의 그림을 따른다.
>
>![Extra.6](/blog/images/Shake-Shake, Extra.6(removed).png )

<br/>
![Table.2](/blog/images/Shake-Shake, Table.2(removed).png )
>**Table.2** <br/>Error rates (%) on CIFAR-100.
>
>모든 이전 결과들보다 0.5% 이상 성능이 좋은 경우는 bold체, 가장 좋은 결과는 파란색으로 표시.

<br/>
모델이 커지면서 길어진 학습 시간으로 인해, CIFAR-10에서보다 적은 횟수만 테스트한 결과다.

<br/>
흥미롭게도 CIFAR-100의 주요 hyperparameter는 batch size였으며, CIFAR-10에서와 달리 batch size를 128에서 32로 줄여야 했다.
>줄이지 않았을 때는 **E-E-B**의 성능이 비교 대상으로 쓸 정도가 아니었다고 한다.
>
>Batch size를 줄여야 한다는걸 GPU를 2개 사용한 경우라고 조건을 걸어두었는데, 이게 큰 상관이 있는진 모르겠다.

<br/>
**Table.2**에서는 batch size가 작아지면서 regularization 효과가 증가하면 training procedure의 선택에 영향을 줄 수 있음을 볼 수 있다.
>**Table.2**에서는 **S-E-I**의 성능이 조금 더 좋은 것으로 나타남.
>
>작은 batch size는 noisy하며, regularization 효과가 있다고 한다. [링크](https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/) 참조

<br/>
### 2.3. Comparisons with state-of-the-art results
논문 작성 당시의 CIFAR-10 best single shot model는 다음과 같다.
- DenseNet-BC, k = 40
>Top-1 error : 3.46% / 25.6M parameters

- ResNeXt-29, 16x64d
>Top-1 error : 3.56% / 68.1M parameters

<br/>
제안하는 regularization 기법을 적용한 결과는 다음과 같다.
- 26 2x32d **"Shake-Even-Image"** ResNet
>Top-1 error : ? % / 2.9M paramter
>
>DenseNet-BC / ResNeXt-29 에 비해 각각 9/23 배 적은 parameter를 가졌음에도 성능이 유사했다고 함.

- 26 2x96d "Shake-Shake-Image" ResNet
>Top-1 error : 2.86% / 26.2M paramters
>
>총 3번 수행한 평균이며, Median/Min/Max는 2.87%/2.72%/2.95%로 측정 됨.

<br/>
CIFAR-100에 대해서는 standard ResNeXt-29 8x64d 모델의 hyperparameter를 일부 수정하여 측정했으며, top-1 error는 16.34%로 측정됐다. (**Table.2**의 **E-E-B**)
>Batch size / no pre-activation / longer training time / cosine annealing

<br/>
"Shake-Even" regularization을 추가했을 때 top-1 error가 15.85%로 줄어들었다.
>총 3번 수행한 평균이며, Median/Min/Max는 15.85%/15.66%/16.04%로 측정 됨.

<br/>
결과는 **Table.3**과 같다.

<br/>
![Table.3](/blog/images/Shake-Shake, Table.3(removed).png )
>**Table.3** <br/>Test error (%) and model size on CIFAR.
>
>가장 좋은 결과는 파란색으로 표시.

---
## 3. Correlation between residual branches
Regularization을 통한 residual branch 간의 correlation 증감 여부를 확인하기 위해, 다음의 테스트를 수행했다.

<br/>
각 residual block에 대해 다음과 같이 수행한다.
1. Mini-batch tensor $$x_i$$를 각 residual branch 1/2를 통과하고, 각 output tensor를 $$y_i^{(1)}$$/$$y_i^{(2)}$$에 저장한다.
>각 residual branch는 다음의 구조를 따름.
>
>ReLU - Conv(3x3) - BN - ReLU - Conv(3x3) - BN - Mul(0.5)

2. 두 output tensor를 flatten하여 vector $$flat_i^{(1)}$$/$$flat_i^{(2)}$$로 만든다.

3. Online algorithm으로 variance/covariance를 계산한다.
>각 vector의 variance / 두 vector 간의 covariance

<br/>
4. Test set의 모든 이미지가 forward 될 때까지 반복하고, 결과 covariance/variance로부터 correlation을 계산한다.

<br/>
**Fig.3**은 **E-E-B**와 **S-S-I**을 각각 26 2x32d 모델에서 3번씩 CIFAR-10에 대해 측정한 결과다.

<br/>
![Fig.3](/blog/images/Shake-Shake, Fig.3(removed).png )
>**Fig.3** <br/>Correlation results on **E-E-B** and **S-S-I** models.

<br/>
**Fig.3**에서는 regularization을 통해 두 output tensor 간 correlation이 감소된 것으로 나타났으며, 이는 regularization이 두 branch가 다른 것을 배우도록 강요한다는 가설을 뒷받침하는 결과다.

<br/>
위 측정 방법은 residual block의 끝에서 이뤄지는 summation이 좌/우 residual branch 내 layer들의 alignment를 강제한다는 가정이 포함되므로, alignment issue를 염두에 둬야 한다.
>Alignment issue는 동일한 조건으로 같은 모델을 하더라도, 동일한 위치의 layer가 항상 같은 표현을 학습한다고 보장할 수 없기 때문에 발생하는 문제다. 즉, 두 vector 간의 covariance 계산에서 이러한 issue가 고려될 수 있다.
>
>본문에서 레퍼 달아둔 [논문](https://arxiv.org/pdf/1511.07543.pdf)에 따르면, core representation은 공유되지만, rare feature들은 그렇지 않다고 한다. 아래의 두 그림을 참조하자.
>
><br/>
>![Extra.7](/blog/images/Shake-Shake, Extra.7(removed).png )
>
><br/>
>![Extra.8](/blog/images/Shake-Shake, Extra.8(removed).png )

<br/>
**Fig.4**에서는 각 block의 처음 3개 layer에 대한 layer-wise correlation을 계산하여 alignment가 강제되는 효과가 있는지 확인있다.

<br/>
![Fig.4](/blog/images/Shake-Shake, Fig.4(removed).png )
>**Fig.4** <br/>Layer-wise correlation between the first 3 layers of each residual block.
>
>예를 들어, L1R3은 $$y_i^{(1)}$$(left branch)의 첫 번째 layer와 $$y_i^{(2)}$$(right branch)의 세 번째 layer의 activation 간 correlation을 의미한다.

<br/>
**Fig.4**에서 좌/우 branch의 동일한 layer 간 correlation이 다른 layer들보다 높으며, 이는 summation이 alignment를 강제한다는 가정과 일치하는 결과다.
>L1R1 / L2R2 / L3R3 에 해당.

---
### 4. Regularization strength
이 장에서는 forward pass에서 small weight가 부여 된 branch에다가 backward pass 시 큰 weight를 부여하면 어떻게 될지 살펴본다. (반대의 경우도 포함)
>Forward/backword pass에 대한 weight는 각각 $$\alpha$$/$$\beta$$를 의미.

<br/>
이미지 $$j$$가 residual block $$i$$에서 forward/backward pass 할 때 사용 된 scaling 계수를 $$\alpha_{i.j}$$/$$\beta_{i.j}$$ 하자.

<br/>
첫 번째 테스트(method 1)는 $$\beta_{i.j}$$를 $$1-\alpha_{i.j}$$로 사용한다.

<br/>
이 섹션의 모든 테스트는 26 2x32d 모델에 **"Image"** 레벨로 regularization을 사용하여 CIFAR-10에 대해 수행됐다.

<br/>
이 모델들은 26 2x32d **"Shake-Keep-Image"** 모델과 비교된다.

<br/>
M1(method 1)의 결과는 **Fig.5**의 왼쪽 그래프에서 볼 수 있다.
>파란색 그래프에 해당.

<br/>
![Fig.5](/blog/images/Shake-Shake, Fig.5(removed).png )
>**Fig.5** <br/>Layer-wise correlation between the first 3 layers of each residual block.
>
>**Left** : Training curves (dark) and test curves (light) of models M1 to M5.<br/>
>**Right** : Illustration of the different methods in Table 4.

<br/>
M1으로 세팅한 영향이 크게 나타났으며, training error가 매우 높게 측정됐다.

<br/>
**Table.4**의 테스트 M2 ~ M5는 M1(method 1)이 왜 그렇게 큰 영향을 미치는지 이해할 수 있도록 설계된 케이스들이다.

<br/>
![Table.4](/blog/images/Shake-Shake, Table.4(removed).png )
>**Table.4** <br/>Update rules for $$\beta_{i.j}$$.

<br/>
**Fig.5**의 오른쪽은 **Table.4**에서 M1 ~ M5 케이스를 그래프로 나타낸 것이며, 이를 통해 알 수 있는 사실은 다음과 같다.
- Regularization 효과는 $$\alpha_{i.j}$$에 대한 $$\beta_{i.j}$$의 상대적 위치와 관련이있는 것으로 보인다.

- $$\beta_{i.j}$$가 $$\alpha_{i.j}$$로부터 멀어질수록 regularization 효과가 더 강해진다.

- 차이가 0.5보다 커지면 그 강도가 급증하는 것으로 보인다.

>**Fig.5** 오른쪽 그림의 위/아래의 그래프는 $$\alpha_{i.j}$$가 각각 0.5보다 크거나 작은 경우로 나눠서 나타낸다.
>
>그래프를 읽는 방법은 다음과 같다. 각 그래프의 x축에서(그래프 상단) $$\alpha_{i.j}$$가 해당 위치의 값일 때, 각 method로 계산되어 나올 수 있는 $$\beta_{i.j}$$ 값의 범위를 나타낸다. 즉, $$\alpha_{i.j}$$ < 0.5의 케이스를 예로 들면, M2는 [$$0$$, $$\alpha_{i.j}$$], M5는 [$$1-\alpha_{i.j}$$, $$1$$] 값으로 세팅된다.
>
>즉, **Fig.5** 오른쪽 그림의 그래프에서 M2는 $$\alpha_{i.j}$$와 가깝고, M5는 멀다고 볼 수 있다. **Fig.5**의 왼쪽 그래프를 보면 $$\alpha$$와 $$\beta$$가 큰 차이가 나지 않는 M2와 큰 차이가 나는 M5의 training curve를 비교하면 그 효과를 쉽게 알 수 있다.

<br/>
이러한 insight들은 regularization의 강도를 보다 정확하게 제어하고자 할 때 유용할 수 있다.

---
## 5. Removing skip connections / Removing Batch Normalization
또 하나의 흥미로운 의문은 skip connection이 제 역할을 하는지에 대한 여부다.

<br/>
많은 딥 러닝 시스템이 ResNet을 사용하지 않으며, skip connection 없이도 이러한 유형의 regularization이 동작한다면 잠재적인 응용 가능성이 커질 수 있다.

<br/>
**Table.5**와 **Fig.6**은 skip connection을 제거한 결과를 나타낸다. 각 결과에 해당하는 모델 A/B/C는 다음과 같이 변형됐다.
- **(A)** : CIFAR-10에 사용 된 26 2x32d에서 skip connection만 제거함.
>ReLU - Conv(3x3) - BN - ReLU - Conv(3x3) - BN - Mul (2 branches).

- **(B)** : A와 동일하지만, branch 당 conv layer를 1개만 사용하고, block의 수를 두 배로 늘림.
>ReLU - Conv(3x3) - BN - Mul (2 branches).

<br/>
![Table.5](/blog/images/Shake-Shake, Table.5(removed).png )
>**Table.5** <br/>Error rates (%) on CIFAR-10.

<br/>
![Fig.6](/blog/images/Shake-Shake, Fig.6(removed).png )
>**Fig.6** <br/>Training curves (dark) and test curves (light).
>
>**Left** : Architecture **(A)**.<br/>
>**Center** : Architecture **(B)**.<br/>
>**Right** : Architecture **(C)**.

<br/>
아키텍처 **(A)**와 **(B)**는 각각 1/2회 테스트했다.

<br/>
**(A)**의 결과는 skip connection 없이도 shake-shake regularization이 작동할 수 있음을 분명히 보여준다.

<br/>
특정 아키텍처의 26 2x32d 모델에서는 **S-S-I**가 너무 강하게 작용하여 underfit 됐으며, 보다 soft한 **S-E-I**의 성능이 더 좋았다.
>이는 model capacity가 64d나 96d와 같이 증가했을 때 달라질 수 있다.

<br/>
**(B)**에서는 주목할만한 결과들이 있었다.
- Regularization이 더 이상 작동하지 않았다.
>이는 자체적으로 각 branch에 있는 2개의 convolution 간 interaction으로 인해 regularization 효과가 발생했음을 나타낸다.

- **S-E-I**와 **E-E-B**의 train/test curve가 완전히 동일하게 나타났다.
>이는 아키텍처 **(B)**의 경우 forward pass의 shake operation이 loss에 영향을 미치지 않음을 나타낸다.

- Training curve가 매우 다름에도, **S-S-I**와 **E-E-B**/**S-E-I**의 test curve는 거의 동일했다.
>Variance는 달랐음.

<br/>
마지막으로 BN의 유무에 따른 작동 여부를 확인한다.

<br/>
BN은 일반적으로 computer vision dataset에 사용되지만, NLP 등 다른 유형의 문제의 경우에는 그렇지 않다.

<br/>
**(C)**는 **(A)**에서 BN을 제외한 아키텍처다.
>즉, skip connection도 없으며, 2개의 branch는 ReLU - Conv(3x3) - ReLU - Conv(3x3) - Mul 구조를 따름.

<br/>
**E-E-B** 모델이 수렴할 수 있도록 depth를 26에서 14로 감소했으며, initial learning rate는 0.025로 1 epoch 동안 warm start 한 후에 0.05로 세팅했다.

<br/>
BN이 없으면 모델이 훨씬 더 sensitive 해지고, 이전과 동일한 방법을 적용했을 때 모델이 발산했다.

<br/>
더 soft한 regularization 효과를 위해, **S-E-I** 모델을 선택했고, $$\alpha_{i.j}$$의 범위를 [0,1]에서 [0.4,0.6]으로 줄였다.

<br/>
CIFAR-10에 대해 아키텍처 **(C)**와 다른 범위의 값을 사용한 모델들도 테스트했다.

<br/>
**Table.5**와 **Fig.6**에서는 제안하는 regularization은 매우 효과적이지만, 모델을 발산시키기도 매우 쉬워진다는 것을 알 수 있다.
>**S-E-I v3** (14 2x32d) 참조.

---
## 6. Conclusion
본문의 실험들에서는 multi-branch network의 branch들을 decorrelate 시킴으로써, overfitting 현상을 완화시키는 것을 볼 수 있다.

<br/>
제안하는 방법을 사용하면 CIFAR dataset에서 state-of-the-art 성능을 얻을 수 있었으며, ResNet 또는 Batch Normalization을 사용하지 않는 아키텍처의 성능을 잠재적으로 향상시킬 수 있음을 보여줬다.

<br/>
결과는 고무적이었지만, 정확한 동작 원리에 대한 의문은 아직 남아있다.

<br/>
이러한 원리를 이해한다면, 더 크고 복잡한 아키텍처에서도 응용할 수 있을 것이다.


---

### 2020-02-11 수정
2.2절에서 ResNeXt-29 2x4x64d 아키텍처 설명이 잘못된 부분을 수정

<br/>
<br/>
{% include disqus.html %}
