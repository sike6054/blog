---
title: "Anaconda 가상 환경 (base) 자동 활성화 해제."
date: 2019-08-07 16:51:11 -0400
tags: Environment TroubleShooting
categories:
  - etc
comments: true
toc: true

---

<br/>
# 1. (base)

Anaconda의 가상 환경에다가 세팅을 한 탓인지, CLI 환경으로 접속하면 base가 활성화 된 상태로 시작된다.
<br/>
![Fig.1](/blog/images/Base, 1.base.PNG)
>Anaconda의 default 가상 환경이다.

<br/>
자동 활성화를 해제하려면, 다음의 명령어를 수행하면 된다.

`conda activate base`
<br/>`conda config --set auto_activate_base false`

<br/>
---
# 2. 참고 자료
<br/>
[Link-1](https://stackoverflow.com/questions/54429210/how-do-i-prevent-conda-from-activating-the-base-environment-by-default)



<br/>
<br/>
{% include disqus.html %}
