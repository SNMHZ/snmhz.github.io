---
layout: single
title: "데이터 시각화"
toc: true
toc_sticky: true
comments: true
---

## 시각화 고려사항
- 목적 : 시각화하는 이유는 무엇인가
- 독자 : 누구를 대상으로 하는가
- 데이터 : 어떤 데이터를 시각화하는가
- 스토리 : 어떤 흐름으로 인사이트를 전달하는가
- 방법 : 전달하고자 하는 내용에 맞는 방법인가
- 디자인 : UI면에서 만족스러운 디자인인가


## 데이터셋의 종류
- 정형 데이터
    - 테이블 형태로 제공
    - row가 1개 item
    - columns는 attribute(feature)
    - 쉽게 시각화 가능(통계적 특성, 상관관계, 비교 등)
- 시계열 데이터
    - 시간 흐름에 따른 데이터
    - 기온, 주가 등 정형 데이터
    - 음성 비디오 등 비정형 데이터
    - 시간 흐름에 따른 추세(trend), 계절성(seasonality), 주기성(cycle) 등 고려
- 지리 데이터
    - 지도 정보와 보고자 하는 정보간의 조화가 중요
    - 거리, 경로, 분포 등 다양하게 활용
- 관계형 데이터
    - 객체(Node)와 객체 간의 관계(Link)를 시각화
    - 크기, 색, 수 등으로 객체와 관계의 가중치 표현
    - 휴리스틱하게 노드 배치하기
- 계층적 데이터
    - `포함 관계`가 분명한 데이터
    - Tree, Treemap, Sunburst 등


## 데이터의 분류
- 수치형(Numerical)
    - 연속형(Continuous) : 길이, 무게 등
    - 이산형(Discrete) : 주사위, 눈금 등
- 범주형(Categorical)
    - 명목형(Norminal): 혈액형, 종교 등
    - 순서형(Ordinal) : 학년, 별점 등


## 시각화 이해하기
- 마크와 채널
    - Mark : 점(Point), 선(Line), 면(Area)으로 이루어진 데이터 시각화
    - Channel : 각 마크를 변경할 수 있는 요소들
        - Position, Color, Shape, Tilt, Size
- 전주의적 속성(Pre-attentive Attribute)
    - 주의를 주지 않아도 인지하게 되는 요소
    - 동시에 사용하면 인지하기 어려움
        - 적절히 사용시, 시각적 분리 효과(visual popout)