--- 
layout: post
date: 2022-02-08 03:00:00 AM
title: "Deep Learning Optimization"
toc: true
toc_sticky: true
comments: true
categories: [ NAVER BoostCamp AI Tech ]
tags: [ NAVER BoostCamp AI Tech, Deep Learning ]
---

딥러닝에서의 최적화(Optimization)란 무엇일까?

단순하게 생각하면 Gradient Descent를 통해 랜덤으로 초기화된 weight들을 loss를 최소화하는 방향으로 변화시키는 것 이라고 답변할 수 있을 것이다.

하지만, 그 과정에서 고민하고 확인해야 할 것이 무엇이 있는지, 이를 위해 알아두면 좋은 것들에 대해 정리하고자 한다.

## 일반화(Generalization)에 대하여
최적화의 목표 중 하나는, `일반화 성능`을 높이는 것이라 할 수 있다.

### 일반화 성능이란 무엇인가?
> 일반화 성능을 높이면 무조건 좋은건가? <br>
> 일반화란 어떤 의미일까?

학습을 시키게 되면 학습 데이터를 점점 더 잘 맞춘다.

계속 학습을 시키면, training set에 과적합되어 test set에 대해선 잘 맞추지 못하는 모습을 보인다.
<img src="/image/boostcamp/dl-optimization/under-over-fitting.png" width="100%"><br>
<sup>[reference](https://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html)</sup>

학습 데이터에 너무 맞추는 것도 아니고, 너무 못 맞추는 것도 아니고.

그 중간 어딘가가 가장 좋은(Balanced) 지점이라 할 수 있다.

하지만 이는 너무 이론적인 말이고, 실제 문제에서 항상 들어맞는다고 할 수는 없다.

우리가 가정하고 있는건, data가 학습하고자 하는 어떤 목적에서 발생된 구조적인 데이터라고 가정하고 있을 뿐이다.

실제 풀고자 하는 문제의 타겟은 저 급격하게 변하는 모양새의 오버피팅된 모습일 수도 있고, 아예 다른 형태일 수도 있다.

이 부분은 디테일한 데이터 분석을 통해 데이터에 대해 깊게 이해하고, 정확하게 문제를 정의하는 것이 가장 중요하다고 생각된다.

### 일반화된 지점을 어떻게 확인할까?
<img src="/image/boostcamp/dl-optimization/generalization-gap.png" width="100%"><br>
test error와 training error의 차이 통해 일반화 갭의 상태를 보고 결정할 수 있다.

단, 이를 통해서는 이 모델의 성능이 학습 성능이랑 비슷할 것이란 사실만 알 수 있다.

학습 데이터에 대해 성능이 좋지 않으면, 일반화 갭이 작다고 해서 잘 학습되었다고 할 수는 없다.

 - `일반화 성능이 좋다` != `테스트 데이터 성능이 좋을 것이다`
 - `일반화 성능이 좋다` == `테스트 데이터 성능이 학습 데이터 성능과 비슷할 것이다`


## Bias와 Variance

 - variance 비슷한 입력을 넣었을 때 출력이 얼마나 일관적으로 나오는가
    - variance가 낮다 -> 간단한 모델이 많이 이럴 것.
    - variance가 높다 -> 비슷한 입력에 출력이 많이 달라진다. overfitting이 날 가능성이 높다.
 - bias. 비슷한 입력에 대해서 true target에 얼마나 접근하는가.

## bias and variance Trade-off
 노이즈가 학습 데이터에 노이즈가 껴 있다고 가정했을 때
 데이터의 cost를 minimize하는 것은 사실 cost가 bias, variance, noise 3가지 파트로 이루어져 있기 때문에 각각을 minimize 하는 것이 아니라서 하나가 줄어들면 하나가 커질 수 밖에 없다.
 trade-off의 관계에 있다.

## Gradient Descent
 - Stocastic Gradient Descent
 - Mini-batch Gradient Descent
 - Batch Gradient Descent

## 배치 사이즈 문제
 - 1개를 쓰면 너무 오래걸리고, 너무 ㅁ낳이 넣으면 GPU 메모리가 터지고.
 - 적절한 수를 찾아야 한다.
 - 라지 배치사이즈를 사용하면 sharp minimizer에 도착한다.
 - 배치 사이즈를 작게 쓰는게 일반적으로 좋다. 
 - sharp minimize보다는 flat minimize에 도착하는 것이 더 좋다.
<img src="/image/boostcamp/dl-optimization/flat-minimum-advantage.png" width="100%"><br>
<sup>[논문: On Large-Batch Training for Deep Learning](https://arxiv.org/abs/1609.04836)</sup>
  - 일반화 성능이 높아진다.
  - flat minimum에 도착하며 testing function에서 조금 멀어져도 괜찮은 성능을 기대할 수 있다.
  - sharp minimum이면 조금만 멀어져도 굉장히 잘 동작하지 않을 수 있다. 실험적으로 보인 것.

## Gradient Desent Methods
<img src="/image/boostcamp/dl-optimization/optimizers-concept.png" width="100%"><br>
<sup>[reference](https://www.slideshare.net/yongho/ss-79607172)</sup>
 - SGD(Stocastic Gradient Descent)
 - Momentum. 관성. 
    - 이전 배치에서 어느 방향으로 흘렀는지에 대한 정보를 활용하자.
    - 한번 흐른 그래디언트를 유지시켜줘서 그래디언트가 너무 왔다갔다해도 잘 훈련되도록 도와준다.
 - NAG(Nestrov Accelerated Gradient)
    - 현재 자리에서 한번 가 보고 간 자리에서 그래디언트를 계산한 걸 가지고 활용한다.
    - local minima에 가지 못하는 모습을 보일 수 있음.봉우리에 더 빨리 닿도록!
 - Adagrad
    - 많이 변한 파라미터는 적게 변화시키고, 적게 변한 파라미터는 많이 변화시킨다.
    - adaptive learning.
    - 문제 : G가 계속 커지기 때문에 분모가 무한대로 갈 수록 점차 학습이 멈추게 된다.
 - Adadelta
    - G가 계속 커지는 현상을 막겠다.
    - 타임스탬프 t를 윈도우 사이즈 만큼의 그래디언트 변화를 보겟다
    - 이전 100개 동안의 g를 들고 잇어야댄다.. 파라미터가 커지면 힘들다.
    - 아다델타 찾아보기
    - lr이 없다.. 그래서 많이 활용되지 않음.
 - RMSprop
    -  
 - Adam
    - 일반적으로 가장 무난하게 사용.
    - 모멘텀을 같이 활용

## Regularization. 규제
 - 학습을 방해한다.
 - 학습을 방해함으로 얻을 수 있는 이점은, 학습 데이터에만 잘 동작하는게 아니라, 테스트 데이터에 대해서도 잘 동작하도록 하기 위한 것.
 - Early stopping
    - 학습을 멈출 때, test data를 활용하면 cheating이다.
    - 보통 validation error를 이용.
 - Parameter norm penalty
    - 네트워크 파라미터가 너무 커지지 않도록 한다.
    - 이왕이면 네트워크 wegiht가 작은 것이 좋다.
    - function space. 뉴럴넷이 만드는 함수의 공간을 최대한 부드러운 함수를 만들자. 부드러운 함수일 수록 일반화 성능이 높을 것이다..!
 - Data augmentation
    - 뉴럴 넷에서 가장 중요한 것 중 하나.
    - 데이터가 적으면, DL보다 일반적인 ML방법론이 더 좋다.
    - 데이터가 많아지면, 이 많은 데이터에 대한 표현력이 ML방법론에선 부족하다. 따라서 DL방법론의 성능이 더 좋아진다.
    - 데이터를 변화시킴에도, 라벨이 변화하지 않는 수준에서 변화를 시킨다.

 - Noise robustness
    - 왜 잘되는지 아직 의문이 있긴 하지만..
    - 노이즈를 입력에만 넣는게 아니라 웨이트에도 넣어줘도 좋다..
    - 노이즈를 중간중간 넣어주면 그 네트워크가 테스트 단계에서 더 잘 될 수 있다.
 - Label Smoothing
    - 이미지들이 있는 공간 속에서 Decision Boundary를 찾는게 목표.
    - 이 경계를 부드럽게 만들어 주는 효과.
    - ex. cutmix
 - Dropout
    - 뉴럴넷의 weight를 0으로 바꿔준다. 얻을 수 있는 것은 각각의 뉴런들이 좀 더 robust한 feature를 잡을 수 있다라고 해석을 한다.. 수학적으로 증명된 것은 아님.
 - Batch normalization
    - 논란이 참 많다.
    - 내가 적용하고자 하는 BN 레이어의 스태티스틱스를 정규화. 레이어가 1000개의 파라미터로 되어 있다. 각각의 값들에 대하여 평균을 0인 정규분포로 만들어버린다.
    - 이거 하면 일반적으로 성능이 많이 올라간다.. 많은 경우에 성능을 올릴 수 있고 활용하는게 좋다.