---
layout: single
title: "PyTorch 기본"
toc: true
toc_sticky: true
comments: true
---

## PyTorch 핵심 요소
- Numpy : Numpy 구조를 가지는 Tensor 객체로 array 표현
- AutoGrad : 자동미분을 지원하여 DL 연산을 지원
- Function : 다양한 형태의 DL을 지원하는 함수와 모델을 지원함

## PyTorch Operations
- 벡터를 다루는 python 기반 연산들은 대부분 numpy 기반(+ AutoGrad)
- Tensor 클래스를 이용하며, numpy like operations 대부분 적용 가능

## 주요 Tensor Handling Method
- view : tensor의 shape을 변환(numpy의 reshape. 단, 메모리 상에서 약간 다르게 동작함)
- squeeze : 차원의 개수가 1인 차원을 압축(삭제)
- unsqueeze : 차원의 개수가 1인 차원을 추가

## 파이토치 프로젝트 템플릿
- 처음엔 대화식 개발 과정(쥬피터 등)이 유리하지만, 개발 용이성 확보 필요(관리 용이, 유지보수 향상 목적)
- OOP 기반으로 모듈을 만들어 프로젝트 템플릿화
- 일반적인 모듈 목록
    - 실행
    - 설정
    - 데이터
    - 모델
    - 학습
    - 로깅, 지표
    - 저장소
    - 유틸리티
    - etc... 필요에 따라

## nn.Module
- 딥러닝을 구성하는 Layer의 base calss
- input, output, forward, backward, parameter 등 정의
    ```python
    class MyLiner(nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weights = nn.Parameter(torch.randn(in_features, out_features))
            self.bias = nn.Parameter(torch.randn(out_features))
            ...
        
        def forward(self, x : Tensor):
            return x @ self.weights + self.bias
        
        ...
    ```

## Dataset
- 데이터의 입력 형태 정의, 입력 방식 표준화
- Image, Text, Audio 등 다양한 형식 정의 가능
- init, len, getitem의 구현이 필수적
    ```python
    from torch.utils.data import Dataset
    class CustomDataset(Dataset):
        def __init__(self, text, labels):
            self.labels = labels
            self.data = text
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            label = self.labels[idx]
            text = self.data[idx]
            sample = {"Text": text, "Class": label}
            return sample
    ```

## DataLoader
- 데이터의 배치를 생성
- 학습 직전의 데이터 변환을 책임
- Tensor로 변환 및 Batch 처리
- 병렬적인 데이터 전처리 코드의 고민 필요
    ```python
    text = ['Happy', 'Amazing', 'Sad', 'Unhapy', 'Glum']
    labels = ['Positive', 'Positive', 'Negative', 'Negative', 'Negative']
    MyDataset = CustomDataset(text, labels)

    MyDataLoader = DataLoader(MyDataset, batch_size=2, shuffle=True)
    for dataset in MyDataLoader:
        print(dataset)
    # {'Text': ['Glum', 'Unhapy'], 'Class': ['Negative', 'Negative']}
    # {'Text': ['Sad', 'Amazing'], 'Class': ['Negative', 'Positive']}
    # {'Text': ['Happy'], 'Class': ['Positive']}
    ```

## model.save()
- 학습의 결과를 저장
- 모델 architecture와 파라미터를 저장
    ```python
    # 모델의 파라미터만 저장
    # 같은 모델의 형태에서 파라미터만 load
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, "model.pt"))

    new_model = TheModelClass()
    new_model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "model.pt")))

    ###################################
    # 모델의 architecture와 함께 저장
    # 모델의 architecture와 함께 load
    torch.save(model, os.path.join(MODEL_PATH, "model.pt"))
    model = torch.load(os.path.join(MODEL_PATH, "model.pt"))
    ```

## Transfer learning
- 다른 데이터셋으로 학습한 모델을 현재 데이터에 적용
- 현재의 DL에서는 가장 일반적인 학습 기법
- backbone이 잘 학습된 모델에서 일부분만 변경하여 학습을 수행
- pretrained model을 활용시 모델의 일부분을 frozen 시킴