---
layout: post
title: "Variational Bayes Expectation Maximization for Mixtures of Gaussians"
description: ""
date: 2018-08-01
tags: [Machine Learning]
comments: true
---

EM과 VBEM을 함께 공부하면서 Technical한 부분을 중심으로 정리했다. EM은 Murphy 11.4, Bishop 9.2, VBEM은 Murphy 21.6, Bishop 10.2를 참고했으며, Notation은 Murphy의 것을 주로 쓰되 일부는 Bishop의 것을 사용했다.

## Expectation Maximzation

Data point $$i$$에 대해, $$x_i$$를 observed variable, $$z_i$$를 hidden variable이라고 하자. Expectation Maximization (EM)의 목표는 다음의 log likelihood $$l(\theta)$$를 최대화하는 것이다.

$$ l(\theta) = \sum_{i=1}^N \log p(x_i \vert \theta) = \sum_{i=1}^N \log \left[ \sum_{z_i}p(x_i, z_i \vert \theta) \right] $$

이것을 계산하는 것이 어려우니까, 다음의 함수 $$Q$$를 최대화하도록 한다.

$$ \textbf{[E-step]}\qquad  Q(\theta, \theta^{t-1}) = \mathbb{E}\left[ \sum_{i=1}^N \log p(x_i, z_i \vert \theta) \middle| \mathcal{D}, \theta^{t-1}\right] $$

여기서, $$\theta^{t-1}$$은 $$t-1$$번째 파라미터, $$\mathcal{D}$$은 관측된 데이터다. E-step에서 $$ Q(\theta, \theta^{t-1}) $$을 계산하고 나면, 이어지는 M-step에서 $$ \theta^t $$를 업데이트한다.

$$ \textbf{[M-step]}\qquad \theta^t = \text{argmax}_{\theta}Q(\theta, \theta^{t-1}) $$

### EM for Mixtures of Gaussians

Gaussian Mixture Model (GMM)은 Gaussian distribution이 선형으로 결합한 형태다.

$$ p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \vert \mu_k, \Sigma_k) $$

$$ z $$는 Mix된 Gaussian 중에서 어떤 분포에 해당하는지를 나타낸다. $$z_i = k$$라면, 해당 Data point $$i$$는 $$k$$번째 Gaussian 분포에 속함을 나타낸다.

Mixing coefficient $$\pi_k$$는 다음과 같다.

$$ \pi_k = p(z = k) $$

먼저 $$p(x, z \vert \theta)$$는,

$$\eqalign{
    p(x, z \vert \theta)
    &= p(z \vert \theta) p(x \vert z, \theta) \\
    &= p(z) p(x \vert z, \theta) \\
    &= \prod_{k=1}^K \pi^{I(z=k)}\cdot \prod_{k=1}^K p(x \vert \theta_k)^{I(z=k)}& (\text{Bishop 9.10, 9.11})\\
    &= \prod_{k=1}^K (\pi_k p(x \vert \theta_k))^{I(z = k)}\\
}$$

$$Q(\theta, \theta^{t-1})$$를 구하면,

$$\eqalign{
    Q(\theta, \theta^{t-1})
    &= \mathbb{E}\left[ \sum_{i=1}^N \log p(x_i, z_i \vert \theta) \right] \\
    &= \sum_{i=1}^N \mathbb{E}\left[ \log p(x_i, z_i \vert \theta) \right] \\
    &= \sum_{i=1}^N \mathbb{E}\left[ \log \left[ \prod_{k=1}^K (\pi_k p(x_i \vert \theta_k))^{I(z_i = k)}\right] \right] \\
    &= \sum_{i}^N \sum_{k}^K \mathbb{E}[I(z_i=k)] \cdot \log [\pi_k p(x_i \vert \theta_k)] \\
    &= \sum_{i}^N \sum_{k}^K p(z_i=k \vert x_i, \theta^{t-1}) \cdot \log [\pi_k p(x_i \vert \theta_k)] \\
    &= \sum_{i}^N \sum_{k}^K r_{ik}\cdot \log [\pi_k p(x_i \vert \theta_k)] \\
}$$

여기서 $$r_{ik}$$는 $$p(z_i=k \vert x_i, \theta^{t-1})$$으로 정의된다.

#### E-step

$$Q(\theta, \theta^{t-1})$$를 구하기 위해서는 $$r_{ik}$$를 구해야 한다.

$$\eqalign{
    r_{ik}
    &= p(z_i=k \vert x_i, \theta^{t-1}) \\
    &= \frac{p(z_i=k \vert \theta^{t-1})p(x_i \vert z_i = k, \theta^{t-1})}{\sum_{k'=1}^K p(z_i=k' \vert \theta^{t-1})p(x_i \vert z_i = k', \theta^{t-1})}\\
    &= \frac{\pi_k p(x_i \vert \theta_k^{t-1})}{\sum_{k'}\pi_{k'}p(x_i \vert \theta_{k'}^{t-1})}
}$$

#### M-step

앞서 말했듯이, M-step에서는 $$Q(\theta, \theta^{t-1})$$를 최대화하는 값으로 파라미터를 update한다. GMM에서 파라미터는 $$\pi_k$$와 더불어, $$\mu_k, \Sigma_k$$ ($$\theta_k$$로 표현됨) 가 있다.

$$\mu_k, \Sigma_k$$에 대해 $$Q(\theta, \theta^{t-1})$$을 미분해서, 이를 최대화하는 $$\mu_k, \Sigma_k$$를 구하면 된다.

$$\eqalign{
    \frac{d}{d \theta}Q(\theta, \theta^{t-1})
    &= \frac{d}{d \theta}\sum_{i}^N \sum_{k}^K r_{ik}\cdot \log [\pi_k p(x_i \vert \theta_k)] \\
    &= \frac{d}{d \theta}\sum_{i}^N \sum_{k}^K r_{ik}\cdot \log [p(x_i \vert \theta_k)] \\
    &= \frac{d}{d \theta}\left[ -\frac{1}{2}\sum_{i}^N r_{ik}\left[ \log |\Sigma_k| + (x_i - \mu_k)^T \Sigma_k^{-1}(x_i - \mu_k) \right] \right] \\
    & \because p(x \vert \theta_k) = \mathcal{N}(x \vert \mu_k, \Sigma_k) = \frac{1}{\sqrt{(2 \pi)^D |\Sigma|}} \exp \left( -\frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1}(x - \mu_k) \right)
}$$

먼저 $$\mu_k$$에 대해,

$$ \frac{d}{d \mu_k}Q(\theta, \theta^{t-1})
    = \sum_{i}^N r_{ik}\Sigma^{-1}(x_i - \mu_k)
    = \Sigma^{-1}\left( \sum_{i}^N r_{ik}x_i - \mu_k \sum_{i}^N r_{ik}\right)
    = 0
$$

$$\mu_k = \frac{\sum_{i}^N r_{ik}x_i}{\sum_{i}^N r_{ik}}$$

$$ \Sigma_k $$는 계산의 편의를 위해 $$ \Lambda_k = \Sigma_k^{-1}$$에 대해 미분을 한다.

$$\eqalign{
    \frac{d}{d \Lambda_k}Q(\theta, \theta^{t-1})
    &= \frac{d}{d \Lambda_k}\left[ -\frac{1}{2}\sum_{i}^N r_{ik}\left[ - \log |\Lambda_k| + (x_i - \mu_k)^T \Lambda_k (x_i - \mu_k) \right] \right]\\
    &= -\frac{1}{2}\sum_{i}^N r_{ik}\left[ - \frac{1}{|\Lambda_k|}\frac{d |\Lambda_k|}{d \Lambda_k}+ (x_i - \mu_k) (x_i - \mu_k)^T \right] \\
    &= -\frac{1}{2}\sum_{i}^N r_{ik}\left[ - \frac{1}{|\Lambda_k|}|\Lambda_k| (\Lambda_k)^{-1}+ (x_i - \mu_k) (x_i - \mu_k)^T \right] \\ 
    &\qquad \because \frac{d|S|}{dS}= |S|S^{-1}\ \text{for a symmetric matrix}\\
    &= -\frac{1}{2}\sum_{i}^N r_{ik}\left[ - (\Lambda_k)^{-1}+ (x_i - \mu_k) (x_i - \mu_k)^T \right]\\
    &= 0
}$$

$$ \therefore \Sigma_k = \Lambda_k^{-1}
= \frac{\sum_{i}^N r_{ik}(x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i}^N r_{ik}}$$

마지막으로 $$Q(\theta, \theta^{t-1})$$를 최대화하는 $$ \pi_k $$를 구한다. Murphy는 "we obviously have ..."라고 적으셨으니, Bishop의 방법론을 적는다. 개인적으로 'obvious'할 정도로 자명해보이지는 않는다고 생각한다.

$$\eqalign{
    \frac{d}{d \pi_k}Q(\theta, \theta^{t-1})
    &= \frac{d}{d \pi_k}\sum_{i}^N \sum_{j}^K r_{ij}\cdot \log [\pi_j p(x_i \vert \theta_j)] \\
    &= \frac{d}{d \pi_k}\sum_{i}^N r_{ik}\cdot \log [\pi_k] \\
    &= \sum_{i}^N r_{ik}\cdot \frac{1}{\pi_k}\\
}$$

Mixing coefficient $$\pi_k$$는 모든 $$k$$에 대한 합이 1이어야 하는 제약($$ \sum_k^K \pi_k = 1 $$)이 있음을 이용한다. 즉, Lagrange multiplier를 이용한다.

$$\eqalign{
    \frac{d f(\lambda, \pi_k)}{d \pi_k}
    &= \frac{d}{d \pi_k}\left[ Q(\theta, \theta^{t-1}) + \lambda (\sum_j^K \pi_j - 1) \right] \\
    &= \sum_{i}^N r_{ik}\cdot \frac{1}{\pi_k} + \lambda \\
    &= 0 \\
    \therefore \lambda \pi_k &= - \sum_{i}^N r_{ik}
}$$

그런데 제약조건으로부터,

$$\eqalign{
    \lambda = \lambda \sum_j^K \pi_j = - \sum_j^K \sum_{i}^N r_{ij}= - \sum_{i}^N \sum_j^K p(z_i=j \vert x_i, \theta^{t-1}) = - \sum_{i}^N 1 = -N
}$$

$$ \therefore \pi_k = \frac{\sum_{i}^N r_{ik}}{N}$$

정리하면,

$$\eqalign{
    \pi_k &= \frac{r_k}{N}\\
    \mu_k &= \frac{\sum_{i}^N r_{ik}x_i}{r_k}\\
    \Sigma_k &= \frac{\sum_{i}^N r_{ik}(x_i - \mu_k)(x_i - \mu_k)^T}{r_k}\\
    &\text{where}\ r_k = \sum_{i}^N r_{ik}
}$$

#### EM Algorithm

1. $$ \mu_k, \Sigma_k, \pi_k $$와 log likelihood $$ \log p(x \vert \mu, \Sigma, \pi) $$의 값을 초기화 한다.
2. log likelihood의 값이 수렴할 때까지 다음 두 단계(E, M)를 반복한다.
    
$$ \textbf{[E-step]}\qquad r_ik = \frac{\pi_k p(x_i \vert \theta_k^{t-1})}{\sum_{k'}\pi_{k'}p(x_i \vert \theta_{k'}^{t-1})}$$
    
$$\eqalign{
    \textbf{[M-step]}\quad
    &\pi_k^{new}= \frac{r_k}{N}\\
    &\mu_k^{new}= \frac{\sum_{i}^N r_{ik}x_i}{r_k}\\
    &\Sigma_k^{new}= \frac{\sum_{i}^N r_{ik}(x_i - \mu_k)(x_i - \mu_k)^T}{r_k}\\
    &\text{where}\ r_k = \sum_{i}^N r_{ik}
}$$

## Variational Bayes EM

Latent variable인 $$z$$가 섞여 있는 $$ p(\mathbf{\theta}, \mathbf{z}_{1:N}) $$은 일반적으로 계산하기가 어렵다. 따라서 Latent variable ($$\mathbf{z}_{1:N}$$)과 parameter ($$\mathbf{\theta}$$)에 관한 식이 나뉘어질 수 있다는 믿음에서 시작한다. 즉, 다음과 같은 approximate factored posterior를 가정한다.

$$ p(\mathbf{\theta}, \mathbf{z}_{1:N}) \approx q(\mathbf{\theta}) q(\mathbf{z}) = q(\mathbf{\theta}) \prod_{i}q(\mathbf{z}_i) $$

Variational EM Algorithm도 EM Algorithm과 마찬가지로, E-step과 M-step을 반복한다. E-step에서는 $$q(\mathbf{z}_i \vert \mathcal{D})$$를, M-step에서는 $$q(\mathbf{\theta}\vert \mathcal{D})$$를 업데이트한다. 

> 
- The variational E step is similar to a standard E step, except instead of plugging in a MAP estimate of the parameters, we need to average over the parameters. (posterior mean of the parameters and $$ p(\mathbf{z}_i \vert \mathcal{D}, \bar{\mathbf{\theta}}) $$)
- The variational M step is similar to a standard M step, except instead of computing a point estimate of the parameters, we update the hyper-parameters, using the expected sufficient statistics.
- Details on how to do this depend on the form of the model.

### VBEM for Mixtures of Gaussians

![gaussian-mixture-graphical-model](/images/gaussian-mixture-graphical-model.png)

$$\eqalign{
    p(\mathbf{X}, \mathbf{Z}, \mathbf{\theta})
    &= p(\mathbf{X}, \mathbf{Z}, \mathbf{\pi}, \mathbf{\mu}, \mathbf{\Lambda}) \\
    &= p(\mathbf{X}\vert \mathbf{Z}, \mathbf{\pi}, \mathbf{\mu}, \mathbf{\Lambda}) p(\mathbf{Z}\vert \mathbf{\pi}, \mathbf{\mu}, \mathbf{\Lambda}) p(\mathbf{\pi}\vert \mathbf{\mu}, \mathbf{\Lambda}) p(\mathbf{\mu}\vert \mathbf{\Lambda}) p(\mathbf{\Lambda}) \\
    &= p(\mathbf{X}\vert \mathbf{Z}, \mathbf{\mu}, \mathbf{\Lambda}) p(\mathbf{Z}\vert \mathbf{\pi}) p(\mathbf{\pi}) p(\mathbf{\mu}\vert \mathbf{\Lambda}) p(\mathbf{\Lambda}) & \text{(See Graphical Model)}
}$$

$$\mathbf{Z}, \mathbf{X}$$는 각각 $$ \mathbf{Z}= \left\{\mathbf{z}_1, \mathbf{z}_2, ..., \mathbf{z}_N \right\}, \mathbf{X}= \left\{\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_N \right\}$$을 의미한다.

$$\mathbf{Z}, \mathbf{X}$$에 대한 Likelihood는,

$$ p(\mathbf{Z}\vert \mathbf{\pi}) = \prod_{i}^N \prod_{k}^K \pi_k^{z_{ik}}$$

$$ p(\mathbf{X}\vert \mathbf{Z}, \mathbf{\mu}, \mathbf{\Lambda}) = \prod_{i}^N \prod_{k}^K \mathcal{N}(\mathbf{x}_i \vert \mu_k, \mathbf{\Lambda}_k^{-1})^{z_{ik}}$$

여기서 $$z_{ik}$$는 $$ I(z_i = k)$$ ($$1$$ if data point $$i$$ belongs to cluster $$k$$, and $$0$$ otherwise)와 같다.

$$\mathbf{\theta}= (\mathbf{\pi}, \mathbf{\mu}, \mathbf{\Lambda})$$에 대한 Prior는 다음과 같다.

$$ p(\mathbf{\pi}) = \text{Dir}(\mathbf{\pi}\vert \mathbf{\alpha}_0) = C(\mathbf{\alpha}_0) \prod_{k=1}^K \pi_k^{\alpha_{0}- 1}\ \text{(Dirichlet Dist.)}$$

$$ p(\mathbf{\mu}, \mathbf{\Lambda}) = p(\mathbf{\mu}\vert \mathbf{\Lambda}) p(\mathbf{\Lambda}) = \prod_{k = 1}^{K}\mathcal{N}\left( \mathbf{\mu}_{k}| \mathbf{m}_{0}, \left( \beta_{0}\mathbf{\Lambda}_{k}\right)^{- 1}\right) \mathcal{W}\left( \mathbf{\Lambda}_{k}| \mathbf{W}_{0}, \nu_{0}\right)
\ \text{(Gaussian-Wishart Dist.)}$$

이 둘은 conjugate prior distributions으로 계산을 편하게 만들어준다.

Factored posterior는 다음과 같다.

$$
q ( \mathbf{Z}, \mathbf{\pi}, \mathbf{\mu}, \mathbf{\Lambda}) = q ( \mathbf{Z}) q ( \mathbf{\pi}, \mathbf{\mu}, \mathbf{\Lambda})
$$

#### Update $$q(\mathbf{Z})$$

$$L(q_j)$$를 최대화하고, $$ KL(q_j \vert \vert \exp ( \mathbb{E}\left[ \log \tilde{p}(\textbf{x}) \right])) $$ 를 최소화 하기위한 $$q_j$$의 optimal solution을 상기하자 (Murphy 21.35, Bishop 10.9).

$$ \log q_j (\textbf{x}_j) = \mathbb{E}_{-q_j}\left[ \log \tilde{p}(\textbf{x}) \right] + C $$

$$\eqalign{
    \log q(\mathbf{Z})
    &= \mathbb{E}_{-q_{\mathbf{Z}}}\left[ \log p(\mathbf{X}, \mathbf{Z}, \mathbf{\theta}) \right] + C \\
    &= \mathbb{E}_{q_{\mathbf{\theta}}}\left[ \log p(\mathbf{X}, \mathbf{Z}, \mathbf{\theta}) \right] + C \\
    &= \mathbb{E}_{q_{\mathbf{\theta}}}\left[ \log p(\mathbf{X}\vert \mathbf{Z}, \mathbf{\mu}, \mathbf{\Lambda}) +  \log p(\mathbf{Z}\vert \mathbf{\pi}) \right] + C \\
    &= \mathbb{E}_{q_{\mathbf{\theta}}}\left[ 
            \sum_{i}\sum_{k}z_{ik}\log \mathcal{N}(\mathbf{x}_i \vert \mathbf{\mu}_k, \mathbf{\Lambda}_k^{-1})
            + \sum_{i}\sum_{k}z_{ik}\log \mathbf{\pi}_k
        \right] + C \\
    &= \mathbb{E}_{q_{\mathbf{\theta}}}\left[ 
            \sum_{i}\sum_{k}z_{ik}\log \left[
                \sqrt{\frac{\vert \mathbf{\Lambda}_k \vert}{(2 \pi)^D}}
                \exp \left(
                    - \frac{1}{2}(\mathbf{x}_i - \mathbf{\mu}_k)^T \mathbf{\Lambda}_k (\mathbf{x}_i - \mathbf{\mu}_k)
                \right)
            \right]
            + \sum_{i}\sum_{k}z_{ik}\log \mathbf{\pi}_k
        \right] + C \\
    &= \sum_{i}\sum_{k}z_{ik}\left[
            \frac{1}{2}\mathbb{E}_{q_{\mathbf{\theta}}}\left[ \log \vert \mathbf{\Lambda}_k \vert \right]
            - \frac{D}{2}\log 2\pi
            - \frac{1}{2}\mathbb{E}_{q_{\mathbf{\theta}}}\left[ (\mathbf{x}_i - \mathbf{\mu}_k)^T \mathbf{\Lambda}_k (\mathbf{x}_i - \mathbf{\mu}_k) \right]
            + \mathbb{E}_{q_{\mathbf{\theta}}}\left[ \mathbf{\pi}_k \right]
        \right] + C \\
    &= \sum_{i}\sum_{k}z_{ik}\log \rho_{ik}+ C \\
    &\quad \text{where}\ \log \rho_{ik}=
            \frac{1}{2}\mathbb{E}_{q_{\mathbf{\theta}}}\left[ \log \vert \mathbf{\Lambda}_k \vert \right]
            - \frac{D}{2}\log 2\pi
            - \frac{1}{2}\mathbb{E}_{q_{\mathbf{\theta}}}\left[ (\mathbf{x}_i - \mathbf{\mu}_k)^T \mathbf{\Lambda}_k (\mathbf{x}_i - \mathbf{\mu}_k) \right]
            + \mathbb{E}_{q_{\mathbf{\theta}}}\left[ \log \mathbf{\pi}_k \right]
}$$

여기서 $$ \log \rho_{ik}$$를 구성하는 식을 계산하면,

$$\eqalign{

\log \tilde{\pi}_{k}\stackrel{\Delta}{=}\mathbb{E}\left[ \log \pi_{k}\right]
&= \psi \left( \alpha_{k}\right) - \psi \left( \sum_{k ^{\prime}}\alpha_{k ^{\prime}}\right) \\
&\quad \text{where}\ \psi = \text{digamma function} \\

\log \tilde{\Lambda}_{k}\stackrel{\Delta}{=}\mathbb{E}\left[ \log \left| \mathbf{\Lambda}_{k}\right| \right]
&= \sum_{j = 1}^{D}\psi \left( \frac{\nu_{k} + 1 - j}{2}\right) + D \log 2 + \log \left| \Lambda_{k}\right| \\

\mathbb{E}\left[ \left( \mathbf{x}_{i}- \mathbf{\mu}_{k}\right) ^{T}\mathbf{\Lambda}_{k}\left( \mathbf{x}_{i}- \mathbf{\mu}_{k}\right) \right]
&= D \beta_{k}^{- 1}+ \nu_{k}\left( \mathbf{x}_{i}- \mathbf{m}_{k}\right) ^{T}\mathbf{\Lambda}_{k}\left( \mathbf{x}_{i}- \mathbf{m}_{k}\right) \\
}$$

정리하면,

$$\eqalign{
    &\therefore q(\mathbf{Z}) \propto \prod_{i=1}^N \prod_{k=1}^K \rho_{ik}^{z_{ik}}\\
    &\Rightarrow \  q(\mathbf{Z}) = \prod_{i=1}^N \prod_{k=1}^K r_{ik}^{z_{ik}} \\
    &\quad \text{where}\ r_{ik} \propto
        \tilde{\pi}_{k} \tilde{\Lambda}_{k} ^{\frac{1}{2}} \exp \left( - \frac{D}{2 \beta_{k}} - \frac{\nu_{k}}{2} \left( \mathbf{x}_{i} - \mathbf{m}_{k} \right) ^{T} \mathbf{\Lambda}_{k} \left( \mathbf{x}_{i} - \mathbf{m}_{k} \right) \right)
}$$

#### Update $$q(\mathbf{\theta})$$

$$\eqalign{
    \log q(\mathbf{\theta})
    &= \mathbb{E}_{-q_{\mathbf{\theta}}}\left[ \log p(\mathbf{X}, \mathbf{Z}, \mathbf{\theta}) \right] + C \\
    &= \mathbb{E}_{q_{\mathbf{Z}}} \left[ \log p(\mathbf{X}, \mathbf{Z}, \mathbf{\theta}) \right] + C \\
    &= \mathbb{E}_{q_{\mathbf{Z}}} \left[ \log p(\mathbf{X}\vert \mathbf{Z}, \mathbf{\mu}, \mathbf{\Lambda}) + \log p(\mathbf{Z}\vert \mathbf{\pi}) + \log p(\mathbf{\pi}) + \log p(\mathbf{\mu}, \mathbf{\Lambda}) \right] + C \\
    &= \mathbb{E}_{q_{\mathbf{Z}}} \left[ \log p(\mathbf{Z}\vert \mathbf{\pi}) + \log p(\mathbf{\pi}) \right]
    + \mathbb{E}_{q_{\mathbf{Z}}} \left[ \log p(\mathbf{X} \vert \mathbf{Z}, \mathbf{\mu}, \mathbf{\Lambda}) + \log p(\mathbf{\mu}, \mathbf{\Lambda}) \right] + C \\
}$$

우변을 잘 살펴보면, $$ \mathbf{\pi} $$와 $$ \mathbf{\mu}, \mathbf{\Lambda} $$에 관한 식으로 나눌 수 있음을 알 수 있다. 이 말은 즉슨 variational posterior $$ q(\mathbf{\pi}, \mathbf{\mu}, \mathbf{\Lambda} ) $$가 다음과 같이 factorizied 될 수 있다는 뜻이다.

$$
q ( \mathbf { \pi } , \mathbf { \mu } , \mathbf { \Lambda } ) = q ( \mathbf { \pi } ) \prod _ { k = 1 } ^ { K } q \left( \mathbf { \mu } _ { k } , \mathbf { \Lambda } _ { k } \right)
$$

먼저 $$ \mathbf{\pi} $$에 대해,

$$\eqalign{
    \log q(\mathbf{\pi})
    &= \mathbb { E } _ { q_{\mathbf { Z }} } \left[
        \log p ( \mathbf { Z } | \mathbf { \pi } )
        + \log p ( \mathbf { \pi } )
    \right] + C \\
    &= \mathbb { E } _ { q_{\mathbf { Z }} } \left[
        \sum_{i}\sum_{k}z_{ik}\log \mathbf{\pi}_k
        + \sum_{k} (\alpha_0 - 1) \log \mathbf{\pi}_k
    \right] + C \\
    &= \sum_{i}\sum_{k} \mathbb { E } _ { q_{\mathbf { Z }} } \left[ z_{ik} \right] \log \mathbf{\pi}_k
    + (\alpha_0 - 1) \sum_k \log \pi_k + C \\
    &= \sum_{i} r_{ik} \sum_{k} \log \mathbf{\pi}_k
    + (\alpha_0 - 1) \sum_k \log \pi_k + C \\
        &\quad \text{where}\ \mathbb { E } _ { q_{\mathbf { Z }} } \left[ z_{ik} \right] = r_{ik} \\
    &= \left( \sum_{i} r_{ik} + \alpha_0 - 1 \right) \sum_k \log \pi_k + C
}$$

이를 $$ \log \text{Dir} (\pi \vert \mathbf{\alpha}) = (\alpha_k - 1) \sum_{k} \log \mathbf{\pi}_k $$와 비교하면,

$$\eqalign{
    q ( \mathbf { \pi } ) & = \operatorname { Dir } ( \pi | \alpha ) \\
    \alpha _ { k } & = \alpha _ { 0 } + N _ { k } \\
    N _ { k } & = \sum _ { i } r _ { i k }
}$$

$$ \mathbf{\mu, \Lambda} $$도 마찬가지로,

$$\eqalign{
    q \left( \mathbf { \mu } _ { k } , \mathbf { \Lambda } _ { k } \right) & = \mathcal { N } \left( \mathbf { \mu } _ { k } | \mathbf { m } _ { k } , \left( \beta _ { k } \mathbf { \Lambda } _ { k } \right) ^ { - 1 } \right) \mathrm { Wi } \left( \mathbf { \Lambda } _ { k } | \mathbf { L } _ { k } , \nu _ { k } \right) \\
    \beta _ { k } & = \beta _ { 0 } + N _ { k } \\
    \mathbf { m } _ { k } & = \left( \beta _ { 0 } \mathbf { m } _ { 0 } + N _ { k } \overline { \mathbf { x } } _ { k } \right) / \beta _ { k } \\
    \mathbf { L } _ { k } ^ { - 1 } & = \mathbf { L } _ { 0 } ^ { - 1 } + N _ { k } \mathbf { S } _ { k } + \frac { \beta _ { 0 } N _ { k } } { \beta _ { 0 } + N _ { k } } \left( \overline { \mathbf { x } } _ { k } - \mathbf { m } _ { 0 } \right) \left( \overline { \mathbf { x } } _ { k } - \mathbf { m } _ { 0 } \right) ^ { T } \\
    \nu _ { k } & = \nu _ { 0 } + N _ { k } + 1 \\
    \overline { \mathbf { x } } _ { k } & = \frac { 1 } { N _ { k } } \sum _ { i } r _ { i k } \mathbf { x } _ { i } \\
    \mathbf { S } _ { k } & = \frac { 1 } { N _ { k } } \sum _ { i } r _ { i k } \left( \mathbf { x } _ { i } - \overline { \mathbf { x } } _ { k } \right) \left( \mathbf { x } _ { i } - \overline { \mathbf { x } } _ { k } \right) ^ { T }
}$$

EM이 $$\theta$$에 대한 MAP estimate을 구하는 것과 달리, VBEM은 posterior를 계산하고 있음이 차이점이다.

### To be updated
- 세부적인 계산 과정.
- Lower bound of EM and VBEM
