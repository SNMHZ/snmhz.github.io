---
layout: post
date: 2025-05-07 00:00:00 +0900
title: "FFT 압축 결과는 왜 Latent Vector가 아닐까?"
toc: true
toc_sticky: true
comments: true
categories: [ Tech, Machine Learning ]
tags: [ FFT, Data Compression, Deep Learning ]
---

## 서론

**FFT(Fast Fourier Transform)** 기반 압축과 **Latent Vector**는 <br>
모두 데이터를 효율적으로 표현한다는 공통점이 있지만, 그 메커니즘과 목적에서 근본적인 차이가 있습니다. 

마치, FFT 압축을 수행한 이미지를 쭉 나열하면 Latent Vector로써 기능할 수 있을 것 같은 착각을 불러옵니다.

이 글에서는 두 방법의 작동 원리를 비교하고, **왜 FFT 압축이 Latent Vector가 아닌지** 설명합니다.

## 목차
- [1. FFT 압축이란?](#1-fft-압축이란)
- [2. Latent Vector란?](#2-latent-vector란)
- [3. FFT 압축과 Latent Vector의 차이](#3-fft-압축과-latent-vector의-차이)
- [4. 혼동의 원인](#4-혼동의-원인)
- [5. 실제 예시](#5-실제-예시)
- [6. 결론](#6-결론)

## 1. FFT 압축이란?

FFT는 데이터를 주파수 영역으로 변환해 고주파 성분을 제거하는 **손실 압축** 기법입니다.

### 작동 원리

1. **주파수 변환**: 이미지/신호를 FFT로 주파수 영역으로 변환
2. **고주파 제거**: 인간이 인지하기 어려운 고주파 성분을 임계값(threshold)으로 걸러냄
3. **역변환**: 남은 저주파 성분으로 역FFT를 수행해 압축된 데이터 복원

### 한계점

- **의미 무시**: 픽셀/신호의 물리적 특성만 고려하며, 객체 식별이나 맥락 이해와 무관
- **비학습적**: 데이터 분포를 학습하지 않고, 고정된 수학적 규칙(에너지 크기)으로 압축
- **복원 품질**: 고주파 손실로 인해 디테일이 흐릿해짐

## 2. Latent Vector란?

Latent Vector는 신경망이 데이터의 본질적 특징을 **학습해 추출**한 저차원 벡터입니다.

### 작동 원리

1. **학습 단계**: 오토인코더, GAN 등이 데이터 분포를 학습
2. **특징 추출**: 입력 데이터를 본질적 속성(예: 얼굴 표정, 객체 종류)으로 압축
3. **의미 공간**: 유사한 데이터는 벡터 공간에서 가까이 위치

### 장점

- **맥락 이해**: "고양이 vs. 강아지"처럼 의미적 유사성을 포착
- **유연한 생성**: 벡터 연산으로 새로운 데이터 생성 가능 (예: GAN)

## 3. FFT 압축과 Latent Vector의 차이

### 목적의 차이

- **FFT 압축**: 데이터 크기 감소 (물리적 효율성)
- **Latent Vector**: 데이터의 의미적 특징 추출 (고수준 표현)

### 압축 기준

- **FFT 압축**: 주파수 영역의 에너지 크기
- **Latent Vector**: 신경망이 학습한 의미적 중요도

### 복원 메커니즘

- **FFT 압축**: 역FFT (수학적 복원)
- **Latent Vector**: 디코더 네트워크 (학습 기반 복원)

## 4. 혼동의 원인

- **공통점**: 데이터 크기 감소
- **오해 포인트**:
    - `압축한 벡터 == Latent Vector`라고 생각하기 쉽지만
    - Latent Vector는 **의미 공간(latent space)으로의 매핑**이 필수
    - FFT 압축을 수행한 데이터는, 원본 데이터와 개념적으로 동일

## 5. 실제 예시

### FFT 압축 (JPEG)

- **원본**: 1024x768 RGB 이미지 (2.4MB)
- **FFT 압축**: 고주파 제거 → 300KB JPEG 파일
- **결과**: 디테일 손실 있지만, "산"이라는 객체를 이미지로써 식별 가능. 여전히 이미지 데이터

### Latent Vector (오토인코더 등)

- **원본**: 동일 이미지 (2.4MB)
- **Latent Vector**: 128차원 벡터 (2KB)
- **결과**: 디테일 보존 + "산, 나무, 하늘" 등 의미(제공한 레이블 등)적 특징을 갖는 벡터로 매핑

## 6. 결론

FFT 압축은 **데이터의 물리적 신호를 단순히 줄이는 도구**일 뿐, <br>
Latent Vector처럼 **의미를 이해하거나 생성하는 능력**이 없습니다.

- **FFT 압축** 
  - 인간의 시각적 인지에 맞춘 **손실 압축**
  - threshold에 따라 압축자가 용인 가능한 선에서 원본 데이터의 품질 저하를 수반
- **Latent Vector**
  - 의미적 이해를 기반으로 한 **지능적 압축**
  - 원본 데이터를 어떠한 의미 공간 속 벡터로 매핑

따라서 두 방법은 **목적과 메커니즘에서 근본적으로 다릅니다**