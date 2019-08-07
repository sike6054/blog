---
title: "Keras 코드 수행 시에 출력되는 Warning을 포함한 기타 메시지 제거."
date: 2019-08-07 12:57:11 -0400
tags: AI Keras TroubleShooting
categories:
  - Keras
comments: true
toc: true

---

<br/>
# 0. Intro
Keras를 설치하고 코드를 수행하면 아래와 같은 끔찍한 메시지를 볼 수 있다.

![Fig.1](/blog/images/Warning, 0.Warnings.png )
>매우 끔찍하다. 물론 사용자의 환경이나 코드에 따라, 메시지의 종류나 양이 다를 수 있다.
>
>이 중에는 Warning도 있지만, 단순히 정보를 보여주는 메시지도 있다.

<br/>
물론 동작에는 문제가 없지만, 각종 Warning 메시지들이 딱히 보고싶진 않을 것이다.

<br/>
따라서 본 포스트에서는 이러한 메시지를 출력하지 않도록 하는 방법을 다룬다.

<br/>
짧게 요약하자면, 아래의 커맨드 한 줄 수행과 코드 삽입이 해결 방법이다.

<br/>
`export TF_CPP_MIN_LOG_LEVEL=2`

``` python
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
```

<br/>
---
# 1. Warning 제거 방법
## 1.1 Information
아래와 같은 메시지들은 거의 Warning이 아닌 기본 정보들이다.

![Fig.2](/blog/images/Warning, 1.1.Info.png )

<br/>
아래의 명령어를 수행하여 해결할 수 있다.

`export TF_CPP_MIN_LOG_LEVEL=2`
>만약 **successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero** 메시지가 사라지지 않는다면, [링크](https://hiseon.me/data-analytics/tensorflow/tensorflow-numa-node-error/)를 참조하여 간단히 해결할 수 있다.

<br/>
각 레벨에 대한 정보는 아래의 그림을 참조하자.

![Fig.3](/blog/images/Warning, 1.1.Level.png )

<br/>
아래의 코드로도 해결할 수 있지만 매번 코드에 삽입해야한다. Parameter로 받아서 제어하고 싶다면 이를 사용하자.

``` python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

<br/>
## 1.2 Tensorflow
아래와 같은 메시지들은 Tensorflow 버전이 바뀌면서 발생하는 Future Warning 류이다. 

![Fig.4](/blog/images/Warning, 1.2.Tensorflow.png )

<br/>
아래의 코드를 삽입하면 해결할 수 있다.

``` python
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.Error)
```
>검색해보면 `tf.logging.set_verbosity(tf.logging.Error)`라고 나오는데, 이를 삽입하면 이 코드에 대한 Warning은 계속 출력된다. 따라서, 이 코드는 새로운 버전에 맞추어 작성해줘야 한다.

<br/>
## 1.3 Numpy
아래와 같은 메시지는 설치된 Tensorflow에서 권장하는 것보다 상위 버전의 numpy를 사용하는 경우에 생기는 메시지라고 한다.

![Fig.5](/blog/images/Warning, 1.3.Numpy_future_warning.png )

<br/>
이 때의 numpy 버전은 다음과 같다.

![Fig.6](/blog/images/Warning, 1.3.Numpy_version.png )

<br/>
반면, 위 메시지가 출력되지 않는 numpy의 버전을 확인하면 다음과 같다.

![Fig.6](/blog/images/Warning, 1.3.Numpy_version_conda.png )

<br/>
다음 명령어를 수행하여, 메시지가 출력되지 않는 버전으로 numpy를 다운그레이드하자.

`pip3 install --upgrade numpy==1.16.4`
>Anaconda에서는 다운그레이드 명령어가 다를 수 있다.

<br/>
---
# 2. Warning 제거 결과

<br/>
![Fig.6](/blog/images/Warning, 2.Result.png )

<br/>
각 해결 방법들이 ERROR 메시지는 출력하도록 작성되어 있는데, 이는 GPU 관련 에러가 발생 시에는 중지되지 않고 CPU로 수행하는 경우가 있기 때문이다.

<br/>
물론 Warning이 가지는 의미를 생각한다면, 무작정 무시하기보다는 어떤 내용이 있는지 가끔씩 체크해주는 것이 좋다.

<br/>
---
# 3. 참고 자료
[Link-1](https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)<br/>
[Link-2](https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error)<br/>
[Link-3](https://insightcampus.co.kr/tensorflow15/)<br/>
[Link-4](https://unix.stackexchange.com/questions/369361/downgrading-numpy-1-12-1-to-1-10-1)<br/>

<br/>
<br/>
{% include disqus.html %}
