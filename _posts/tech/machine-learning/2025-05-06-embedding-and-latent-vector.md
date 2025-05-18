---
title: "Embedding과 Latent Vector, 헷갈리는 개념 정리"
date: 2025-05-06 00:00:00 +0900
categories: [ Tech, Machine Learning ]
tags: [ embedding, Deep Learning ]
toc: true
toc_sticky: true
comments: true
---

# 서론

머신러닝과 딥러닝에서 embedding, embedding vector, latent vector는 <br>
종종 혼용되거나 모호하게 사용되는 개념입니다.

이 세 용어는 서로 연관되어 있지만 엄밀히는 다른 의미를 가지며, <br>
혼동할 경우 모델 해석이나 구현에 실수가 발생할 수 있습니다.

이 글에서는 각 개념의 정의와 차이를 명확히 정리해보겠습니다.

## 목차
- [1. Embedding이란?](#1-embedding이란)
- [2. Embedding Vector란?](#2-embedding-vector란)
- [3. Latent Vector란?](#3-latent-vector란)
- [4. 개념 간 관계](#4-개념-간-관계)
- [5. 실제 적용 사례](#5-실제-적용-사례)
- [6. 결론](#6-결론)

## 1. Embedding이란?

**Embedding**은 **데이터를 의미적 관계를 보존하는 연속 벡터 공간으로 변환하는 과정**이라고 할 수 있으며, <br>
이 과정에서 차원은 목적과 모델에 따라 증가하거나 감소할 수 있습니다.

### 핵심 특징

- **목적**: 데이터를 벡터 공간에 매핑하여 의미적 관계를 수치적으로 표현
- **특성**: 입력 데이터보다 차원이 축소될 수도, 확장될 수도 있음
- **적용**: 자연어 처리, 추천 시스템, 컴퓨터 비전 등 다양한 분야

### 주요 임베딩 방법

- **Word2Vec, GloVe**: 단어 임베딩
- **BERT, RoBERTa**: 문맥 기반 임베딩
- **Node2Vec**: 그래프 임베딩
- **nn.Embedding**: 딥러닝 모델의 임베딩 레이어

## 2. Embedding Vector란?

**Embedding Vector**는 임베딩 과정을 통해 생성된 실제 벡터를 의미합니다.

### 핵심 특징

- **형태**: 고정된 차원의 실수 벡터 (예: 300차원, 768차원)
- **특성**: 유사한 의미의 데이터는 벡터 공간에서 가까이 위치
- **유연성**: 원본 데이터의 차원보다 크거나 작을 수 있음

### 예시

```python
# PyTorch에서 임베딩 벡터 생성
embedding_layer = nn.Embedding(vocab_size=10000, embedding_dim=64)
input_ids = torch.tensor([1, 2, 3])
embedding_vectors = embedding_layer(input_ids)  # 결과: (3, 64) 크기의 텐서
```

## 3. Latent Vector란?

**Latent Vector**는 데이터의 본질적 특징(잠재적 특성)을 압축적으로 표현한 저차원 벡터입니다.

### 핵심 특징

- **목적**: 데이터의 본질적/잠재적 특징 추출 및 압축
- **조건**: 반드시 차원 축소(압축)를 전제로 함
- **의미 공간**: 의미적 관계가 구조화된 공간을 형성

### 주요 생성 방법

- **오토인코더(Autoencoder)**: 인코더-디코더 구조의 중간 벡터
- **VAE(Variational Autoencoder)**: 확률적 잠재 변수
- **GAN(Generative Adversarial Network)**: 생성 모델의 잠재 공간
- **딥러닝 모델의 중간층 출력**: 모델이 추출한 추상적 특징

## 4. 개념 간 관계

### Latent Vector와 Embedding Vector의 관계

- **포함 관계**: Latent Vector ⊂ Embedding Vector
    - 모든 Latent Vector는 Embedding Vector이지만
    - 모든 Embedding Vector가 Latent Vector는 아님
- **차이점**:
    - **Embedding Vector**: 차원 확장/축소 모두 가능
    - **Latent Vector**: 반드시 데이터의 본질적 특징을 압축한 저차원 벡터

### 구분 예시

- **Embedding Vector이지만 Latent Vector가 아닌 경우**:
    - 10개 카테고리를 64차원 벡터로 변환 (차원 확장)
    - 원-핫 인코딩보다 더 큰 차원의 임베딩
- **Latent Vector의 예**:
    - 오토인코더의 bottleneck 벡터 (1024차원 → 128차원)
    - BERT의 [CLS] 토큰 벡터 (문맥 정보 압축)

## 5. 실제 적용 사례

### 자연어 처리

- **Word Embedding**: 단어를 300차원 벡터로 변환 (Embedding Vector)
- **문장 임베딩**: 문장 전체를 768차원으로 압축 (Latent Vector)

### 이미지 처리

- **CNN 특징 맵**: 이미지의 의미적 특징 추출 (Latent Vector)
- **이미지 패치 임베딩**: ViT에서 이미지 패치를 벡터화 (Embedding Vector)

### 추천 시스템

- **사용자/아이템 임베딩**: 사용자와 아이템을 벡터로 표현 (Embedding Vector)
- **협업 필터링 잠재 벡터**: 사용자-아이템 상호작용의 잠재 패턴 추출 (Latent Vector)

## 6. 결론

- **Embedding**은 데이터를 벡터 공간에 매핑하는 과정
- **Embedding Vector**는 임베딩 과정의 결과물로, 차원 확장/축소가 모두 가능
- **Latent Vector**는 Embedding Vector의 특수한 경우로, 데이터를 어떠한 목표로 하는 본질적 특징으로 압축하여 잘 표현하는 저차원 벡터

간간히 헷갈리는 개념이라 정리합니다.
