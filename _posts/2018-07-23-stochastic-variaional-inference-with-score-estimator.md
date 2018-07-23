---
layout: post
title: "Stochastic Variaional Inference with Score Estimator"
description: ""
date: 2018-07-23
tags: [Machine Learning]
comments: true
---

For following optimization problem,

$$ argmin_{\theta} \text{KL} \left[ q_{\theta}(x) || p(x|y_0)  \right] $$

Using stochastic gradient descent with following gradient (score estimator):

$$ \nabla_{\theta} \text{KL} [ q_{\theta}(x) || p(x|y_0) ]$$

## 1. Basis

이것은 $$ \mathbb{E}_{x \sim q_{\theta}(x)} \left[
          \left( \nabla_{\theta}  \log(q_{\theta}(x)) \right)
          \cdot \log \frac{q_{\theta}(x)}{p(x,y_0)}
        \right] $$ 와 같은데, 그 증명은 다음과 같다.

$$\eqalign{
      & \nabla_{\theta} \text{KL} [ q_{\theta}(x) || p(x|y_0) ] \\
      & = \nabla_{\theta} \left(
        \sum_{x} q_{\theta}(x) \log \frac{q_{\theta}(x)}{p(x,y_0)}
        + \sum_{x} q_{\theta}(x) \log p(y_0)
      \right) \\
      & = \nabla_{\theta} \left(
        \sum_{x} q_{\theta}(x) \log \frac{q_{\theta}(x)}{p(x,y_0)}
      \right)
      + \nabla_{\theta} \left( \sum_{x} q_{\theta}(x) \log p(y_0) \right) \\
      & = \nabla_{\theta} \left(
        \sum_{x} q_{\theta}(x) \log \frac{q_{\theta}(x)}{p(x,y_0)}
      \right) & \because \text{Lemma 1} \\
      & =
      \sum_{x} \nabla_{\theta} \left( q_{\theta}(x) \right) \log \frac{q_{\theta}(x)}{p(x,y_0)} + 
      \sum_{x} q_{\theta}(x) \nabla_{\theta} \left( \log \frac{q_{\theta}(x)}{p(x,y_0)} \right) \\
      & = \sum_{x} \nabla_{\theta} \left( q_{\theta}(x) \right) \log \frac{q_{\theta}(x)}{p(x,y_0)}
        & \because \text{Lemma 2} \\
      & = \sum_{x} q_{\theta}(x) \cdot
      \left( \nabla_{\theta}  \log(q_{\theta}(x)) \right) \cdot
      \log \frac{q_{\theta}(x)}{p(x,y_0)}
        & \because \text{Lemma 3} \\
      & =  \mathbb{E}_{x \sim q_{\theta}(x)} \left[
          \left( \nabla_{\theta}  \log(q_{\theta}(x)) \right)
          \cdot \log \frac{q_{\theta}(x)}{p(x,y_0)}
        \right]
}$$


### Lemma 1

$$\eqalign{
      \nabla_{\theta} \left( \sum_{x} q_{\theta}(x) \log p(y_0) \right)
      = \log p(y_0) \nabla_{\theta} \left( \sum_{x} q_{\theta}(x) \right)
      = \log p(y_0) \nabla_{\theta} (1)
      = 0
}$$

### Lemma 2

$$\eqalign{
      \sum_{x} q_{\theta}(x) \nabla_{\theta} \left( \log \frac{q_{\theta}(x)}{p(x,y_0)} \right)
      & = \sum_{x} q_{\theta}(x) \nabla_{\theta} \left( \log q_{\theta}(x) \right) \\
      & = \sum_{x} q_{\theta}(x) \frac{\nabla_{\theta}q_{\theta}(x)}{q_{\theta}(x)} & \because \text{Lemma 3} \\ 
      & = \sum_{x} \nabla_{\theta} q_{\theta}(x) \\
      & = \nabla_{\theta} \sum_{x} q_{\theta}(x) \\
      & = \nabla_{\theta}(1) = 0
}$$

### Lemma 3

$$\eqalign{
      \nabla_{\theta} \left( \log q_{\theta}(x) \right) = \frac{\nabla_{\theta}q_{\theta}(x)}{q_{\theta}(x)}
}$$

여기서 Lemma 3을 Log trick 이라고 부르시던데, 딱히 trick이라고 부를만큼 특별하지는 않은 것 같다. 고등학교 때 배웠던 log 함수 미분의 다변수 버전일 뿐.

## 2. Reduce Variance (Control Variate)

이를 estimate하는 sampling 알고리즘은 다음과 같다.

$$q_{\theta}(x)$$에서 뽑힌 $$x1, ... x_N$$에 대해,

$$\eqalign{
      \frac{1}{N} \sum_{i=1}^{N}
      \left( \nabla_{\theta}  \log(q_{\theta}(x_i)) \right)
      \cdot \left( \log \frac{q_{\theta}(x_i)}{p(x_i,y_0)} \right)
}$$

이 알고리즘를 그대로 사용하면 variance가 매우 큰 결과를 얻게 된다. variance를 줄이는 방법으로, 상수 $$B$$에 대한 아래의 estimator를 사용한다. 이 때 B를 control variate이라고 부른다.

$$\eqalign{
      \frac{1}{N} \sum_{i=1}^{N}
      \left( \nabla_{\theta}  \log(q_{\theta}(x_i)) \right)
      \cdot \left( \log \frac{q_{\theta}(x_i)}{p(x_i,y_0)} -B \right)
}$$

이것이 $$ \nabla_{\theta} \text{KL} [ q_{\theta}(x) \mid \mid p(x \mid y_0) ] $$ 와 같은 이유는 다음과 같다.

$$\eqalign{
      & \mathbb{E}_{x \sim q_{\theta}(x)} \left[
        \left( \nabla_{\theta}  \log(q_{\theta}(x)) \right)
        \cdot \left( \log \frac{q_{\theta}(x)}{p(x,y_0)} - B \right)
      \right] \label{q3_b_equation} \\
      & = \mathbb{E}_{x \sim q_{\theta}(x)} \left[
        \left( \nabla_{\theta}  \log(q_{\theta}(x)) \right)
        \cdot \left( \log \frac{q_{\theta}(x)}{p(x,y_0)} \right)
      \right] -
      \mathbb{E}_{x \sim q_{\theta}(x)} \left[
        \left( \nabla_{\theta}  \log(q_{\theta}(x)) \right) \cdot B
      \right] \\
      & = \mathbb{E}_{x \sim q_{\theta}(x)} \left[
        \left( \nabla_{\theta}  \log(q_{\theta}(x)) \right)
        \cdot \left( \log \frac{q_{\theta}(x)}{p(x,y_0)} \right)
      \right] -
      B \cdot \mathbb{E}_{x \sim q_{\theta}(x)} \left[
        \left( \nabla_{\theta}  \log(q_{\theta}(x)) \right)
      \right] \\
      & = \mathbb{E}_{x \sim q_{\theta}(x)} \left[
        \left( \nabla_{\theta}  \log(q_{\theta}(x)) \right)
        \cdot \left( \log \frac{q_{\theta}(x)}{p(x,y_0)} \right)
      \right] -
      B \cdot \sum_{x} q_{\theta}(x) \nabla_{\theta} \left( \log q_{\theta}(x) \right) \\
      & = \mathbb{E}_{x \sim q_{\theta}(x)} \left[
        \left( \nabla_{\theta}  \log(q_{\theta}(x)) \right)
        \cdot \left( \log \frac{q_{\theta}(x)}{p(x,y_0)} \right)
      \right] & \because \text{Lemma 2} \\
      & = \nabla_{\theta} \text{KL} [ q_{\theta}(x) || p(x|y_0) ]  & \because \text{(1)}
}$$

## 3. Optimze Control Variate

2에서 주어진 식의 variance를 최소화하는 control variate $$B$$의 값을 구할 수 있다.

$$\eqalign{
      B^{*} = 
      & \frac
        {\mathbb{E}_{x \sim q_{\theta}(x)} \left[
          \left( \nabla_{\theta} \log(q_{\theta}(x)) \right) ^2
          \cdot \left( \log \frac{q_{\theta}(x)}{p(x,y_0)} \right)
        \right]}
        {\mathbb{E}_{x \sim q_{\theta}(x)} \left[
          \left( \nabla_{\theta} \log(q_{\theta}(x)) \right) ^2
        \right]}
}$$

증명은 다음과 같다.

For the simplicity, let $$C$$ be $$ \left( \nabla_{\theta} \log(q_{\theta}(x)) \right) \cdot \left( \log \frac{q_{\theta}(x)}{p(x,y_0)} - B \right)$$.

In this problem, we have to find $$B$$ that minimizes the variance of the estimate for $$N=1$$, which is $$\text{Variance}(C) = \mathbb{E}_{x \sim q_{\theta(x)}}[C^2] - \left( \nabla_{\theta} \text{KL} [ q_{\theta}(x) \mid \mid p(x \mid y_0 ) ] \right)^2$$.

Because $$\left( \nabla_{\theta} \text{KL} [ q_{\theta}(x) \mid \mid p(x \mid y_0) ] \right)^2$$ is just a constant, we can conclude that $$\text{argmin}_{B} [ \text{Variance}(C) ] = \text{argmin}_{B} [ \mathbb{E}_{x \sim q_{\theta}(x)}[C^2] ]$$.

$$\eqalign{
      \mathbb{E}_{x \sim q_{\theta}(x)}[C^2] =
      &\ \mathbb{E}_{x \sim q_{\theta}(x)} \left[
        \left( \nabla_{\theta} \log(q_{\theta}(x)) \right) ^2
        \cdot \left( \log \frac{q_{\theta}(x)}{p(x,y_0)} \right) ^2
      \right] \\
      & - 2B \cdot \mathbb{E}_{x \sim q_{\theta}(x)} \left[
        \left( \nabla_{\theta} \log(q_{\theta}(x)) \right) ^2
        \cdot \left( \log \frac{q_{\theta}(x)}{p(x,y_0)} \right)
      \right] \\
      & + B^2 \cdot \mathbb{E}_{x \sim q_{\theta}(x)} \left[
        \left( \nabla_{\theta} \log(q_{\theta}(x)) \right) ^2
      \right]
}$$

Its derivative with respect to $$B$$ is,

$$\eqalign{
      \frac{d}{dB} \mathbb{E}_{x \sim q_{\theta}(x)}[C^2] = 
      & - 2 \cdot \mathbb{E}_{x \sim q_{\theta}(x)} \left[
        \left( \nabla_{\theta} \log(q_{\theta}(x)) \right) ^2
        \cdot \left( \log \frac{q_{\theta}(x)}{p(x,y_0)} \right)
      \right] \\
      & + 2B \cdot \mathbb{E}_{x \sim q_{\theta}(x)} \left[
        \left( \nabla_{\theta} \log(q_{\theta}(x)) \right) ^2
      \right]
}$$

$$B^{*}$$ that makes the derivative zero will be,
$$\eqalign{
      B^{*} = 
      & \frac
        {\mathbb{E}_{x \sim q_{\theta}(x)} \left[
          \left( \nabla_{\theta} \log(q_{\theta}(x)) \right) ^2
          \cdot \left( \log \frac{q_{\theta}(x)}{p(x,y_0)} \right)
        \right]}
        {\mathbb{E}_{x \sim q_{\theta}(x)} \left[
          \left( \nabla_{\theta} \log(q_{\theta}(x)) \right) ^2
        \right]}
}$$