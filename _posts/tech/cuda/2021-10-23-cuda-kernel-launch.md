---
layout: post
date: 2021-10-23 03:00:00 AM
title: "CUDA Kernel Launch"
toc: true
toc_sticky: true
comments: true
categories: [CUDA]
tags: [CUDA]
---

## C/C++ function call syntax

```cpp
void func_name( int param, … );
for (int i = 0; i < SIZE; ++i) {
    func_name( param, … );
}
```

<br>

## CUDA kernel launch syntax

```cpp
__global__ void kernel_name( int param, … );
kernel_name <<< 1, SIZE >>>( param, … );
``` 

___`<<< >>>` 는 쿠다 컴파일러가 책임진다(C, Cpp 문법에 존재하지 않는 연산자)___

<br>

## CUDA 에서의 kernel launch
- many threads(ex. 1,000,000) on many core(ex. 1,000)가 일반적인 상황
    - 쓰레드 관리를 위한 모델 
    - 계층구조
    - launches are hierarchical, (grid - block - thread)
    - 커널이 grid를 만들어서 grid가 실행되는 구조
    - grid는 많은 block들을, block들은 많은 thread들을 가짐.
    - thread가 묶여서 block, block이 묶여서 grid가 된다
- thread 내부는 sequential execution.
    - 프로그래머들이 sequential programming에 워낙 익숙하기 때문
    - but, 모든 thread는 병렬로 실행되므로 병렬처리의 이점을 누릴 수 있음
- grid, block 구조는 최대 3차원
    - `kernel_func<<<dimGrid, dimBlock>>>(...);`
    - `kernelFunc<<<3, 4>>>(...);`
    - `kernelFunc<<<dim(3), dim(4)>>>(...);`
    - `kernelFunc<<<dim(3, 1, 1), dim(4, 1, 1)>>>(...);`

<br> 