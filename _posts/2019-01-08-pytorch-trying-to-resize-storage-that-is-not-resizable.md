---
layout: post
title: "Pytorch: Trying to resize storage that is not resizable"
description: ""
date: 2019-01-08
tags: [Machine Learning, Pytorch]
comments: true
---

`torch.tensor(x)`를 실행했을 때 "Trying to resize storage that is not resizable" 에러 해결법.

```bash
RuntimeError: Trying to resize storage that is not resizable at /Users/administrator/nightlies/pytorch-1.0.0/wheel_build_dirs/wheel_3.6/pytorch/aten/src/TH/THStorageFunctions.cpp:70
```

`x`가 `numpy.ndarray`일 때, 차원에 0이 들어가 있는 경우 이런 Error를 띄우는 것 같다. 첫 번째 차원이 0이거나 1인 경우는 이런 Error가 발생하지 않는다.

```bash
>>> torch.tensor(np.zeros((2, 0)))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Trying to resize storage that is not resizable at /Users/administrator/nightlies/pytorch-1.0.0/wheel_build_dirs/wheel_3.6/pytorch/aten/src/TH/THStorageFunctions.cpp:70
```
```bash
>>> torch.tensor(np.zeros((1, 0)))
tensor([], size=(1, 0), dtype=torch.float64)
```
```bash
>>> torch.tensor(np.zeros((0, 3)))
tensor([], size=(0, 3), dtype=torch.float64)
```
