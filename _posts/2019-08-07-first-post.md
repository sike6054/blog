---
title: "대용량 데이터를 빠르게 받아보자."
date: 2019-08-07 15:42:11 -0400
tags: AI Install Environment
categories:
  - etc
comments: true
toc: true

---

<br/>
# 0. Intro
<br/>
작업을 하다보면, 대용량인 데이터를 다운받을 일이 생길 수 있다.

<br/>
환경에 따른 일반적인 다운로드 방법은 다음과 같다.

- (GUI) 사이트에 접속하여 직접 다운로드

- (CLI) **wget**을 통한 다운로드

<br/>
하지만, 데이터를 제공하는 서버가 해외면서 상태까지 좋지 않다면, 다운로드에만 상당한 시간이 소요될 수 있다.
>대표적으로 **ImageNet**이 있다.

<br/>
본 포스트에서는, 이러한 경우에 사용할 만한 명령어를 소개한다.
>물론, 대용량이 아닌 경우라도 사용할 수 있으며, 보다 좋은 방법이 있을 수 있음.

<br/>
---
# 1. Axel
<br/>
**wget**을 사용하는 대신 **axel**을 사용한 방법이다. 

<br/>
**axel**은 다수의 connection을 통한 분할 다운로드 방식으로 가속화 된 다운로드 속도를 제공한다. 단순한 사용법에는 큰 차이가 없으며, 제공되는 기능이 많다.

<br/>
우선 사용하려면 설치부터 해야한다. 다음의 명령어로 설치할 수 있다.

`sudo apt-get install axel`

<br/>
대략 쓸만한 옵션은 다음과 같다.

- `--max-speed=x` : 다운로드에서 소요되는 리소스를 제어하기 위해, 최고 속도를 Byte 단위로 제한할 수 있다. `-s x`로도 사용할 수 있으며, default는 속도에 제한을 걸지 않는다.

- `--num-connections=x` : 다운로드할 때 사용될 connection의 수를 정한다. `-n x`로도 사용할 수 있다.

- `--search[=x]` : 미러 사이트를 선택할 수 있다. 복수의 미러 사이트를 선택할 수 있으며, `-S[x]`로도 사용할 수 있다.

<br/>
---
# 2. 속도 비교
<br/>
**wget**과 **axel**의 다운로드 속도 차이를 비교해보자.

<br/>
비교에 사용 된 데이터는 ILSVRC 2012 classification dataset 중 validation set으로, 용량은 6.3GB 이다.

<br/>
![Fig.1](/blog/images/1.Axel, wget.PNG)
>**wget**의 다운로드 속도는 최대 3.02MB/s 정도로 측정됐다.

<br/>
![Fig.2](/blog/images/1.Axel, axel.PNG)
>**axel**의 다운로드 속도는 한도 끝도 없이 올라가더니, 최대 15.79MB/s 정도로 측정됐다.

<br/>
ImageNet 데이터를 다운로드한다고 몇 차례 사용했었는데, 통상적으로 **wget**의 4~7배 정도의 속도로 다운로드 됐었다.
>물론 ImageNet 사이트가 워낙에 불안정하다보니, 오차가 크다.

<br/>
다운로드가 완료될 때는 다음과 같다.

![Fig.3](/blog/images/1.Axel, result.PNG)
>각 connection들의 다운로드가 비동기적으로 완료된다.

<br/>
이후에 GUI로 사이트에서 직접 다운로드도 테스트 해봤을 때, 측정 된 속도는 다음과 같았다.

 - (GUI) 200~400KB/s
 
 - (wget) 340KB/s

 - (axel) 1MB/s

>마찬가지로 ImageNet에 대한 다운로드 속도이므로, 비교에 대한 100% 신뢰는 못한다. 어찌됐건, **axel**이 가장 빠른 것만은 눈에 보일 정도였다.

<br/>
**axel**은 위의 이미지들에서 보이듯이, 진행 상황을 지저분하게 출력한다. `verbose` 옵션을 off 시킬 수는 있지만, 진행 상황을 전혀 볼 수 없게 된다.

<br/>
이러한 대용량 데이터들은 [링크](https://academictorrents.com/collection/imagenet-2012)와 같이, 토렌트로 제공되는 경우도 있다.

<br/>
---
# 3. 참고 자료
<br/>
[Link-1](http://manpages.ubuntu.com/manpages/trusty/man1/axel.1.html)
[Link-2](https://deviantcj.tistory.com/290)
[Link-3](https://zetawiki.com/wiki/%EC%9A%B0%EB%B6%84%ED%88%AC_axel_%EC%84%A4%EC%B9%98)

<br/>
<br/>
{% include disqus.html %}
