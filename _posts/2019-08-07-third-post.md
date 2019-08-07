---
title: "Tensorflow에서 참조하는 CUDA와 설치되는 CUDA 간의 버전 꼬이는 문제 해결 방법."
date: 2019-08-07 14:53:11 -0400
tags: Keras Environment TroubleShooting
categories:
  - Keras
comments: true
toc: true

---

<br/>
# 1. 사건과 해결
사건의 전말은 다음과 같다.

**1. Tensorflow를 CUDA 10.0 버전에 맞춰 세팅했었는데, 무슨 바람이 불었는지 CUDA 10.1 버전을 설치하게 됐다.**

**2. 그러나, 이후에도 Tensorflow는 계속해서 `libcublas.so.10.0` 류의 cudnn library를 참조하려 한다.**

**3. 당연히 library path에는 `libcublas.so.10.1`와 같이 버전이 모두 바뀌어 있는 상황.**

**4. Tensorflow가 참조하는 버전을 바꾸고 싶었지만 재설치로도 효과가 없었다.**

**5. 다시 CUDA 10.0 버전을 설치하기로 한다.**

**6. 하지만, CUDA 10.1의 key를 입력해서인지 계속해서 CUDA 10.1 버전이 설치된다.**

<br/>
위의 상황은 다음의 한 줄로 정리할 수 있다.

- **Tensorflow는 CUDA 10.0 버전의 library를 참조하지만, 설치를 하면 CUDA 10.1 버전만 설치된다.**

<br/>
끔찍한 데드락이다. 꼬여도 제대로 꼬였다. 구글링을 통해 4번의 방법을 찾아보긴 했지만, 잘 나오지 않았다.

<br/>
[이전 포스트](https://sike6054.github.io/blog/keras/first-post/)에도 있지만, CUDA 설치 가이드에서는 [링크](https://developer.nvidia.com/cuda-toolkit-archive)에서 deb 파일을 다운받고, 아래의 명령어를 수행하는 것이다.

`sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64`
<br/>`sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub`
<br/>`sudo apt-get update`
<br/>`sudo apt-get install cuda`

<br/>
하지만, 위의 과정으로는 CUDA 10.1 버전이 설치되는 무한의 데드락에서 헤어나올 수 없다.
>키보드가 박살나지 않도록 주의하자.

<br/>
다행히도 다음의 명령어로 CUDA 10.0 버전을 설치할 수 있다.

`sudo apt-get install cuda-10-0`

<br/>
---
# 2. 참고 자료
[Link-1](https://devtalk.nvidia.com/default/topic/1050914/cuda-setup-and-installation/cuda-remove-10-1-and-install-10-0-ubuntu-18-04/)<br/>

<br/>
<br/>
{% include disqus.html %}
