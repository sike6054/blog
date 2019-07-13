---
title: "[Ubuntu 18.04] Anaconda 가상 환경에 Keras (with Tensorflow, CUDA 10.0) 세팅하기."
date: 2019-07-14 05:09:11 -0400
tags: AI Keras Install Environment
categories:
  - Keras
toc: true
---

# 0. Intro

이번에 개인용 딥 러닝 서버를 새로 구매했다. 주요 스펙은 다음과 같다.
- MainBoard : **Z390**
- CPU : **i7-8700**
- GPU : **RTX 2080 Ti**
- RAM : **64G (16G x 4)**

<br/>
기존에 많이 이용한 Ubuntu 16.04에서 CUDA 9.0 버전을 이용하고 싶었으나, 보드랑 호환 문제가 있었는지 OS 설치 단계부터 아래 사진과 같은 문제들을 만났다.

<br/>
![Fig.1](/blog/images/Keras_Instll, 0.boot error.png )

<br/>
Secure boot 비활성화나 standard mode로 변경하는 등, 다양한 해결 방법들을 사용해봤지만 별 효과가 없었다. 어차피 천년만년 구버전을 이용할 수도 없으니 겸사겸사 **Ubuntu 18.04**에 **CUDA 10.0** 버전을 설치하기로 했다. 주요 에러들은 없어졌지만 nouveau에 관련된 메시지는 계속 나타났으며, 다음의 방법으로 해결할 수 있다.
>위 사진에는 nouveau 문제를 해결한 후에 찍은 메시지라 보이지 않는다.

<br/>
우선 Ubuntu가 설치 된 USB로 부팅을 하면 아래와 같은 모습을 볼 수 있다.

<br/>
![Fig.2](/blog/images/Keras_Instll, 0.nouveau_0.png )
>이건 사진을 깜빡하는 바람에 [여기](https://kldp.org/node/159690)서 가져왔다.

<br/>
위 화면일 때, 빠르게 **e** 키를 눌러주면 아래와 같은 화면에 진입하게 된다.
>약간 여유는 있지만, 너무 지체하면 화면이 넘어 가버리더라.

<br/>
![Fig.3](/blog/images/Keras_Instll, 0.nouveau_1.png )

<br/>
여기서 맨 오른쪽에 `splash ---`라고 된 부분을 `splash nomodeset`으로 바꿔주면 된다.
>방향키로 커서를 옮기고, 키보드로 타이핑하여 수정하면 된다. 타이핑 결과는 아래 그림 참조.

<br/>
![Fig.3](/blog/images/Keras_Instll, 0.nouveau_2.png )
>올리고 보니 손이 흔들렸더라.

<br/>
이후에 F10을 누르면, 현재 세팅으로 부팅된다. 하지만, 이는 현재 부팅에만 적용하는 방법이며, 영구적으로 적용하려면 부팅 후에 터미널을 열고 다음의 동작을 따르자.

 1. `sudo vi /etc/default/grub`
 
 2. 파일 내용 중, `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"` 라인에서 `"quiet splash"`부분을  `"quiet splash nomodeset"`으로 수정하고 저장

 3. `sudo update-grub`

<br/>
nouveau 관련 해결 방법은 [여기](http://ejklike.github.io/2017/03/05/install-ubuntu-16.04-with-nvidia-gpu.html)를 참조했다. 물론 vi 대신 다른 편집기를 이용해도 된다.

<br/>
또한, 본 글에서는 Ubuntu 18.04를 설치하는 부분은 생략하고 있다. 모든 PATH가 CUDA 10.0 버전을 기준으로 작성됐으므로, 다른 버전을 사용할 경우에는 주의해서 수정할 필요가 있다.

---
# 1. ssh 세팅
서버니까 원격으로 작업하고싶다. 필자는 putty를 주로 이용하기 때문에 ssh로 연결할 수 있게 준비한다.

### 1-1. IP 확인
`sudo apt-get install net-tools`<br/>
`ifconfig -a`

### 1-2. putty로 해당 IP에 접근해보면
![Fig.4](/blog/images/Keras_Instll, 1-2.putty.png )
>당연히 접속이 불가능하다. 이건 따라하지 말자.

### 1-3. ssh 설치
`sudo apt-get install ssh`
![Fig.5](/blog/images/Keras_Instll, 1-3.login.png )
>잘 된다.

<br/>
# 2. Anaconda 가상 환경에 Keras(with Tensorflow) 설치

`sudo apt install python3-dev python3-pip`
  
### 2-1. Anaconda 설치
`wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh`
>다운로드가 안 되면 [링크](https://www.anaconda.com/distribution/#linux)에 가서 원하는 버전으로 다운로드하면 된다.

<br/>
`export PATH="/your_path/anaconda3/bin:$PATH"`<br/>
`conda create -n your_name pip python=3.7`
>Python 버전은 자기가 원하는 것으로 설정하면 된다. PATH는 절대경로로 설정해주자.

<br/>
### 2-2. Anaconda 가상 환경에 Tensorflow 설치하기
`conda activate your_name`
>최초로 activate 하는 시점에는 `conda activate your_name`로 수행하면 에러가 날 수 있다. 처음에만 `source activate`와 `source deactivate`를 사용해주자.

<br/>
`(your_name) pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.14.0-cp37-cp37m-linux_x86_64.whl`
>URL은 하단의 그림에 표시한 것이다. 버전이 다를 경우, [링크](https://www.tensorflow.org/install/pip)에서 본인에게 맞는 URL을 확인할 수 있다.

<br/>
![Fig.5](/blog/images/Keras_Instll, 2-2.tensorflow-gpu.PNG )

<br/>
`(your_name) pip install --upgrade tensorflow-gpu`<br/>
`(your_name) conda deactivate`

<br/>
### 2-3. NVIDIA Graphics Driver 설치
[링크](https://www.nvidia.com/Download/index.aspx?lang=en-us)로 가서 다운로드 받고 설치한다.
>GUI 환경에서 설치한 바람에 CLI 환경에서 설치하는 내용이 누락됐다.

<br/>
### 2-4. CUDA 설치
`wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64`
>혹은 [링크](https://developer.nvidia.com/cuda-toolkit-archive)로 가서 설치하자.

<br/>
`sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64`<br/>
`sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub`
>key가 다르다면 위의 `dpkg`라인의 수행 결과 마지막 부분에 나온 key 등록 명령어를 복사해서 수행하면 된다. (아래 수행 결과 참조)

<br/>
![Fig.6](/blog/images/Keras_Instll, 2-4.CUDA.png )

<br/>
`sudo apt-get install cuda`

<br/>
### 2-5. cuDNN 설치
[링크](https://developer.nvidia.com/cudnn)로 가서 CUDA 버전에 맞는 cuDNN을 다운로드 한다. (로그인이 필요하기 때문에, wget으로 다운받으려 하면 403 에러가 난다.)
>`cuDNN Library for Linux`를 다운로드 했기 때문에, tar 파일 기준으로 설명한다.

<br/>
`tar -xzvf cudnn-10.0-linux-x64-v7.6.1.34.tgz`<br/>
`sudo cp cuda/include/cudnn.h /usr/local/cuda/include`<br/>
`sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64`<br/>
`sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*`

<br/>
`export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH`<br/>
`export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`

<br/>
### 2-6. Keras 설치
`conda activate your_name`<br/>
`(your_name) conda install h5py`<br/>
`(your_name) conda install graphviz`<br/>
`(your_name) conda install pydot`<br/>
`(your_name) conda install keras`

<br/>
### 2-7. 예제 실행
[ResNet 포스팅](https://sike6054.github.io/blog/paper/first-post/)에서 구현한 코드를 일부 수정한 코드이다.

``` python
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dense, BatchNormalization, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau

from keras.utils import to_categorical
from keras.datasets import mnist, cifar10
import numpy as np

def conv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu', name='default'):
    x = Conv2D(filters, kernel_size, kernel_initializer='he_normal', padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation='relu')(x)

    return x

def plain18(model_input, classes=10):
    conv1 = conv2d_bn(model_input, 64, (7, 7), strides=2, padding='same') # (112, 112, 64)

    conv2_1 = MaxPooling2D((3, 3), strides=2, padding='same')(conv1) # (56, 56, 64)
    conv2_2 = conv2d_bn(conv2_1, 64, (3, 3))
    conv2_3 = conv2d_bn(conv2_2, 64, (3, 3))

    conv3_1 = conv2d_bn(conv2_3, 128, (3, 3), strides=2) # (28, 28, 128)
    conv3_2 = conv2d_bn(conv3_1, 128, (3, 3))

    conv4_1 = conv2d_bn(conv3_2, 256, (3, 3), strides=2) # (14, 14, 256)
    conv4_2 = conv2d_bn(conv4_1, 256, (3, 3))

    conv5_1 = conv2d_bn(conv4_2, 512, (3, 3), strides=2) # (7, 7, 512)
    conv5_2 = conv2d_bn(conv5_1, 512, (3, 3))


    gap = GlobalAveragePooling2D()(conv5_2)

    model_output = Dense(classes, activation='softmax', kernel_initializer='he_normal')(gap) # 'softmax'

    model = Model(inputs=model_input, outputs=model_output, name='Plain18')

    return model

def ResNet18(model_input, classes=10):
    conv1 = conv2d_bn(model_input, 64, (7, 7), strides=2, padding='same') # (112, 112, 64)

    conv2_1 = MaxPooling2D((3, 3), strides=2, padding='same')(conv1) # (56, 56, 64)
    conv2_2 = conv2d_bn(conv2_1, 64, (3, 3))
    conv2_3 = conv2d_bn(conv2_2, 64, (3, 3), activation=None) # (56, 56, 64)

    shortcut_1 = Add()([conv2_3, conv2_1])
    shortcut_1 = Activation(activation='relu')(shortcut_1) # (56, 56, 64)


    conv3_1 = conv2d_bn(shortcut_1, 128, (3, 3), strides=2)
    conv3_2 = conv2d_bn(conv3_1, 128, (3, 3)) # (28, 28, 128)

    shortcut_2 = conv2d_bn(shortcut_1, 128, (1, 1), strides=2, activation=None) # (56, 56, 64) -> (28, 28, 128)
    shortcut_2 = Add()([conv3_2, shortcut_2])
    shortcut_2 = Activation(activation='relu')(shortcut_2) # (28, 28, 128)


    conv4_1 = conv2d_bn(conv3_2, 256, (3, 3), strides=2)
    conv4_2 = conv2d_bn(conv4_1, 256, (3, 3)) # (14, 14, 256)

    shortcut_3 = conv2d_bn(shortcut_2, 256, (1, 1), strides=2, activation=None) # (28, 28, 128) -> (14, 14, 256)
    shortcut_3 = Add()([conv4_2, shortcut_3])
    shortcut_3 = Activation(activation='relu')(shortcut_3) # (14, 14, 256)


    conv5_1 = conv2d_bn(conv4_2, 512, (3, 3), strides=2)
    conv5_2 = conv2d_bn(conv5_1, 512, (3, 3)) # (7, 7, 512)

    shortcut_4 = conv2d_bn(shortcut_3, 512, (1, 1), strides=2, activation=None) # (14, 14, 256) -> (7, 7, 512)
    shortcut_4 = Add()([conv5_2, shortcut_4])
    shortcut_4 = Activation(activation='relu')(shortcut_4) # (7, 7, 512)


    gap = GlobalAveragePooling2D()(shortcut_4)

    model_output = Dense(classes, activation='softmax', kernel_initializer='he_normal')(gap) # 'softmax'

    model = Model(inputs=model_input, outputs=model_output, name='ResNet18')

    return model

input_shape = (32, 32, 3)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = np.float32(x_train / 255.)
x_test = np.float32(x_test / 255.)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model_input = Input( shape=input_shape )

model = plain18(model_input, 10)

optimizer = SGD(lr=0.1, decay=0.0001, momentum=0.9)
callbacks_list = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1)]

model.compile(optimizer, 'categorical_crossentropy', ['acc'])

model.fit(x_train, y_train, batch_size=32, epochs=6, validation_split=0.2, callbacks=callbacks_list)
```

<br/>
코드를 저장하고 다음의 명령어로 실행해본다.

<br/>
`conda activate your_name`<br/>
`(your_name) python test_resnet.py`

<br/>
최신 버전을 설치하다 보면, 각종 warning을 마주하게 된다.
- **successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero**
>[링크](https://hiseon.me/data-analytics/tensorflow/tensorflow-numa-node-error/)를 참조하면 간단히 해결할 수 있다.

<br/>
- **The name tf.is_variable_initialized is deprecated. Please use tf.compat.v1.is_variable_initialized instead.**류의 Warning
>비슷한 Warning이 다수 목격될 수도 있다. 무시하도록 하는 방법도 있지만, 손이 빠르다면 직접 소스파일을 수정하는 무식한 방법도 약이 된다.

<br/>
![Fig.7](/blog/images/Keras_Instll, 2-7.Example.PNG )

<br/>
# 3. 끝
<br/>
