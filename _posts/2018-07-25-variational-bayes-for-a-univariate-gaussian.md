---
layout: post
title: "Variational Bayes for a Univariate Gaussian"
description: ""
date: 2018-07-25
tags: [Machine Learning]
comments: true
---

Murphy 21.5.1과 Bishop 10.1.3를 합친 내용이다. Notation은 Murphy의 것을 사용했다.

Univariate Gaussian $$p(\mu, \lambda \mid \mathcal{D})$$에 Vairational Bayes (VB)를 적용하고 싶다. (mean $$\mu$$, precision $$\lambda = 1/\sigma^2$$)

여기서는 아래와 같은 conjugate prior를 사용한다.

$$ p(\mu, \lambda) = \mathcal{N} (\mu \mid \mu_0, (\kappa_0 \lambda)^{-1}) Ga(\lambda \mid a_0, b0) $$

Murphy에서는 위 식만을 소개하나 (Murphy 21.65), Bishop에서는 각 요소에 관한 식을 모두 기술해놓았다 (Bishop 10.21 - 10.23).

Likelihood function인 $$p(\mathcal{D} \mid \mu, \lambda)$$는,

$$p(\mathcal{D} \mid \mu, \lambda) = \left( \frac{\lambda}{2 \pi} \right)^{N/2} exp \left( - \frac{\lambda}{2} \sum_{i=1}^N (x_i - \mu)^2 \right) $$

$$\mu$$ 와 $$\lambda$$에 대한 prior는,

$$p(\mu \mid \lambda) = \mathcal{N} (\mu \mid \mu_0, (\kappa_0 \lambda)^{-1} ) = \sqrt{\frac{\kappa_0 \lambda}{2 \pi}}e^{-\frac{\kappa_0 \lambda}{2} (\mu - \mu_0)^{2}}$$

$$p(\lambda) = Ga(\lambda \mid a_0, b_0) = \frac{b_0^{a_0}}{\Gamma(a_0)} \lambda^{a_0-1}e^{-b_0 \lambda} $$

그리고, approximate factored posterior로서 다음을 사용한다.

$$ q(\mu, \lambda) = q_{\mu} (\mu) q_{\lambda} (\lambda) $$

## Target Distribution

unnormalized log posterior는,

$$\eqalign{
    log \tilde{p} (\mu, \lambda)
    & = log p(\mu, \lambda, \mathcal{D}) \\
    & = log p(\mathcal{D} \mid \mu, \lambda) + log p(\mu \mid \lambda) + log p(\lambda)
}$$

여기서 각 로그 항은,

$$\eqalign{
    & log p(\mathcal{D} \mid \mu, \lambda) = \frac{N}{2} log \lambda - \frac{\lambda}{2} \sum_{i=1}^N (x_i - \mu)^2 + C \\
    & log p(\mu \mid \lambda) = \frac{1}{2}log(\kappa_0 \lambda) - \frac{\kappa_0 \lambda}{2} (\mu - \mu_0)^2 +C \\
    & log p(\lambda) = (a_0 - 1) log \lambda - b_0 \lambda + C
 }$$

## Update $$q_{\mu} (\mu)$$

Variational Inference에서 가장 중요한 식은 Murphy 21.35, Bishop 10.9다.

$$ log q_j (\textbf{x}_j) = \mathbb{E}_{-q_j} \left[ log \tilde(p) (\textbf{x}) \right] + C $$

의미는, $$L(q_j)$$를 최대화하고, $$ KL(q_j \mid \mid exp ( \mathbb{E} \left[ log \tilde(p) (\textbf{x}) \right])) $$ 를 최소화 하기위한 $$q_j$$의 optimal solution이다.

아래 첨자 $$-q_j$$는 $$q_j$$를 제외한 다른 factor에 대해서라는 뜻인데, 이 경우에서는 $$q_{\mu}$$와 $$q_{\lambda}$$ 밖에 없으므로,

$$\eqalign{
    log q_{\mu} (\mu)
    & = \mathbb{E}_{-q_{\mu}} \left[ log \tilde{p} (\mu) \right] + C \\
    & = \mathbb{E}_{q_{\lambda}} \left[ log p(\mathcal{D} \mid \mu, \lambda) + log p(\mu \mid \lambda)  \right] + C \\
    & = - \frac{\mathbb{E}_{q_{\lambda}}[\lambda]}{2} \left( \sum_{i=1}^N (x_i - \mu)^2  + \kappa_0 (\mu - \mu_0)^2 \right) + C \\
    & = - \frac{\mathbb{E}_{q_{\lambda} }[\lambda] (\kappa_0 + N)}{2} \left(
             \mu - \frac{\kappa_0 \mu_0 + N \bar{x}}{\kappa_0 + N}
         \right)^2 + C \\
    & = - \frac{\kappa_N}{2} (\mu - \mu_N)^2 + C
}$$

마지막 식은, $$q_{\mu} (\mu)$$가 $$ \mathcal{N} (\mu \mid \mu_N, \kappa_N^{-1}) $$일 때의 식인데, 마지막에서 두 번째 식이 이 형태를 가짐을 확인할 수 있다.

따라서, $$q_{\mu} (\mu)$$는 아래와 같은 $$\mu_N, \kappa_N$$에 대한 $$ \mathcal{N} (\mu \mid \mu_N, \kappa_N^{-1}) $$ 를 따른다 (Murphy 21.71, Bishop 10.26-27).

$$ \kappa_N = (\kappa_0 + N) \mathbb{E}_{q_{\lambda}}[\lambda] $$

$$ \mu_N = \frac{\kappa_0 \mu_0 + N \bar{x}}{\kappa_0 + N} $$

$$q_{\lambda} (\lambda)$$도 비슷한 형태의 update를 할 수 있다. 차이점은 Gaussian이 아닌 Gamma 분포의 형태로 맞추어 $$a_N, b_N$$을 구한다는 것이다.

## References
- Murphy, Kevin P. "Machine learning: a probabilistic perspective." (2012).
- Bishop, Christopher M. "Pattern Recognition and Machine Learning (Information Science and Statistics)."