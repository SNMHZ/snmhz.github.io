---
layout: post
date: 2025-06-06 00:00:00 +0900
title: "NumPy 1.24 이후 np.float 타입 제거 및 대응 전략"
toc: true
toc_sticky: true
comments: true
categories: [ Tech, Python ]
tags: [ numpy, python ]
---

NumPy를 사용하다 보면 가끔 예상치 못한 버전 업데이트 변경점에 당황할 때가 있습니다. <br>
NumPy 1.24 버전에서 `np.float`와 같은 일부 타입들이 완전히 제거된 것이 바로 그런 경우입니다. <br>
많은 기존 코드에 영향을 줄 수 있는 이 변화는 사실 NumPy의 타입 시스템을 더 명확하게 만들고, <br>
오랜 기간 존재했던 혼란을 해결하기 위한 중요한 발전입니다.

이번 글에서는 이 변화가 왜 필요했는지, <br>
그리고 우리의 코드를 어떻게 수정해야 하는지 구체적인 전략을 알아보겠습니다.

## `np.float`는 왜 사라졌을까?

결론부터 말하면, `np.float`가 혼란의 주범이었기 때문입니다.

### 정체성의 혼란

`np.float`의 정체는 사실 **Python 내장 `float`의 또 다른 이름(alias)** 이었습니다. <br>
하지만 이름 때문에 많은 개발자들이 NumPy가 제공하는 고유한 숫자 타입으로 오해하곤 했습니다.

이로 인해 몇 가지 문제가 발생했습니다-
- 개발자들이 `np.float`를 NumPy 고유의 타입으로 오해
- Python `float`와의 관계가 불분명
- 코드의 가독성과 명확성 저하

### 단계적 제거 과정

NumPy 개발팀은 이 문제를 단계적으로 해결했습니다.

- **NumPy 1.20 (2021년 1월)**: `np.float` 등의 별칭이 처음 deprecated됨
- **NumPy 1.20~1.23**: DeprecationWarning 발생하지만 여전히 사용 가능
- **NumPy 1.24 (2023년 6월)**: 완전히 제거되어 AttributeError 발생

실제 에러 메시지는 다음과 같습니다.

```
AttributeError: module 'numpy' has no attribute 'float'. 
`np.float` was a deprecated alias for the builtin `float`. 
To avoid this error in existing code, use `float` by itself. 
Doing this will not modify any behavior and is safe. 
If you specifically wanted the numpy scalar type, use `np.float64` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
```

### 명확성을 위한 결정

이런 혼란을 바로잡기 위해 NumPy 개발팀은 타입 시스템의 일관성을 높이는 방향으로 결정을 내렸습니다.

이제는 모호한 `np.float` 대신, <br>
아래와 같이 의도를 명확하게 드러내는 타입을 사용해야 합니다.

```python
# 혼동을 야기하던 과거 방식 (제거됨)
np.float(3.14)    # Python float? NumPy float?

# 명확하고 직관적인 현재 방식 (권장)
float(3.14)       # Python 내장 타입 사용
np.float64(3.14)  # NumPy 64비트 부동소수점 타입 사용
np.float32(3.14)  # NumPy 32비트 부동소수점 타입 사용
```

## `float` vs `np.float64`

`np.float`가 사라졌으니 이제 Python의 `float`와 NumPy의 `np.float64` 사이의 관계를 명확히 이해하는 것이 중요합니다. <br>
이 둘은 **사실상 거의 동일한 존재**입니다.

Python의 `float`는 C언어의 `double`을 기반으로 구현되어 있으며, <br>
이는 IEEE 754 표준의 64비트 부동소수점 표현을 사용합니다. <br>
NumPy의 `np.float64` 역시 마찬가지입니다.

```python
import numpy as np

# 동일한 정밀도와 범위를 가집니다.
python_float = 5.9975
numpy_float64 = np.float64(5.9975)

# 내부 표현도 동일합니다.
print(python_float.hex())      # '0x1.7fd70a3d70a3dp+2'
print(numpy_float64.hex())     # '0x1.7fd70a3d70a3dp+2'

# 값도 당연히 같습니다.
print(python_float == numpy_float64)  # True
```

### 그래도 차이는 있습니다

메모리 표현 방식은 같지만, Python 타입 시스템의 관점에서 보면 둘은 엄연히 다릅니다.

```python
# 타입 확인 결과는 다릅니다.
isinstance(2.0, float)         # True
isinstance(2.0, np.float64)    # False
isinstance(np.float64(2.0), float) # False

# 사용 가능한 메서드가 다릅니다.
np.float64(5.9975).sum()       # NumPy 객체이므로 NumPy 메서드 사용 가능
(5.9975).sum()                 # Python float에서는 AttributeError 발생
```

이러한 차이점 때문에 `np.float`를 단순히 `float`로 바꿀지, `np.float64`로 바꿀지는 <br>
코드의 맥락에 따라 결정해야 합니다.

## 안전한 마이그레이션 전략

그렇다면 기존 코드를 어떻게 수정해야 할까요? <br>
다행히 마이그레이션은 그리 복잡하지 않습니다.

### 1. 코드 수정 방법

#### 기본 변환

단순한 값 변환에는 Python 내장 `float`를 사용하는 것이 가장 간단합니다. <br>
NumPy 타입이 꼭 필요한 경우에는 `np.float64`를 사용합니다.

```python
# Before
data = np.float(user_input)

# After (간단한 경우)
data = float(user_input)
# 또는 (NumPy 타입이 필요한 경우)
data = np.float64(user_input)
```

#### 배열 타입 지정

`np.array`의 `dtype`을 지정할 때가 가장 흔한 경우입니다. <br>
이 역시 `float`나 `np.float64`로 바꿔주면 됩니다.

```python
# Before
arr = np.array([1, 2, 3], dtype=np.float)

# After
arr = np.array([1, 2, 3], dtype=float)      # 가장 간단하고 일반적인 방법
# 또는
arr = np.array([1, 2, 3], dtype=np.float64) # 의도를 명시적으로 드러내는 방법
```

### 2. 구버전 호환성 유지

만약 작성하는 코드가 구버전 NumPy와도 호환되어야 한다면, <br>
다음과 같이 예외 처리를 활용할 수 있습니다.

```python
import numpy as np

# 이전 버전과의 호환성을 위한 처리
try:
    # NumPy 1.20+ 에서는 numpy.float가 DeprecationWarning을 발생시키며 여전히 존재
    # 1.24에서 완전히 제거됨
    from numpy import float as np_float
except ImportError:
    # np.float가 없는 최신 버전에서는 Python float를 사용
    np_float = float

# 이제 np_float를 안전하게 사용 가능
arr = np.array([1, 2, 3], dtype=np_float)
```
하지만 라이브러리를 개발하는 경우가 아니라면, <br>
코드를 최신 버전에 맞게 수정하는 것을 더 권장합니다.

### 3. 대규모 코드베이스 한번에 바꾸기

프로젝트 전체에 `np.float`가 퍼져있다면, <br>
자동화된 스크립트로 한 번에 수정하는 것이 효율적입니다.

#### 정규표현식 활용 (Linux/macOS)

`sed`와 같은 커맨드라인 도구를 사용하면 빠르게 변경할 수 있습니다.

```bash
# np.float(를 float(로 변경
sed -i 's/np\.float(/float(/g' **/*.py

# dtype=np.float를 dtype=float로 변경
sed -i 's/dtype=np\.float\b/dtype=float/g' **/*.py
```

> **주의-** `sed -i` 명령어는 OS나 버전에 따라 동작이 다를 수 있으니, <br>
> 실행 전 반드시 코드를 백업하세요.

#### Python 스크립트로 안전하게 바꾸기

플랫폼에 상관없이 더 안전하게 코드를 변경하고 싶다면 <br>
Python 스크립트를 작성하는 것이 좋습니다.

```python
import re
import os
from pathlib import Path

def migrate_numpy_float(file_path):
    # 파일을 UTF-8로 안전하게 읽기
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"파일 읽기 오류 {file_path}: {e}")
        return

    # np.float( -> float(
    content_new = re.sub(r'np\.float\(', 'float(', content)
    # dtype=np.float -> dtype=float
    content_new = re.sub(r'dtype=np\.float\b', 'dtype=float', content_new)
    # np.float_ -> np.float64 (NumPy 2.0 대비)
    content_new = re.sub(r'np\.float_', 'np.float64', content_new)

    if content != content_new:
        print(f"마이그레이션 적용- {file_path}")
        file_path.write_text(content_new, encoding='utf-8')

# 현재 디렉토리 및 하위 디렉토리의 모든 .py 파일을 대상으로 실행
for py_file in Path('.').glob('**/*.py'):
    migrate_numpy_float(py_file)
```
이 스크립트는 `np.float_`처럼 곧 사라질 다른 타입들까지 함께 처리해줄 수 있어 더욱 유용합니다.

## 앞으로의 코딩 습관

이번 변화를 계기로 삼아 더 좋은 코딩 습관을 기를 수 있습니다.

### 1. 명시적인 타입 사용하기

가장 중요한 것은 타입을 명시적으로 사용하는 습관입니다. <br>
모호한 별칭 대신 정확한 타입을 사용하면 코드의 의도가 분명해집니다.

```python
# 좋은 예: 의도가 명확함
def process_data(values):
    return np.array(values, dtype=np.float64)

# 나쁜 예: 모호함 (제거된 방식)
def process_data(values):
    return np.array(values, dtype=np.float)
```

### 2. 타입 힌트 적극 활용하기

Python의 타입 힌트를 함께 사용하면 코드의 안정성을 더욱 높일 수 있습니다.

```python
from typing import Union
import numpy as np
from numpy.typing import NDArray

def calculate_mean(data: Union[list, NDArray]) -> np.float64:
    # np.mean은 기본적으로 float64를 반환하지만, dtype을 명시하여 의도를 확실히 할 수 있음
    return np.mean(data, dtype=np.float64)
```

### 3. 성능이 중요하다면 정밀도 선택하기

모든 부동소수점이 `np.float64`일 필요는 없습니다. <br>
데이터의 특성에 따라 적절한 정밀도를 선택하면 메모리를 효율적으로 사용할 수 있습니다.

```python
# 일반적인 정밀도로 충분하고 메모리 사용량이 중요할 때
small_precision_array = np.zeros(100, dtype=np.float32)  # 400 bytes

# 높은 정밀도가 반드시 필요할 때
high_precision_array = np.zeros(100, dtype=np.float64)  # 800 bytes
```

## NumPy 2.0에서는 `np.float_`도 제거됩니다

NumPy 2.0에서는 밑줄이 하나 붙은 `np.float_`도 제거될 예정입니다. <br>
이것 역시 `np.float64`의 별칭이었기 때문입니다.

[NumPy 2.0 마이그레이션 가이드](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html)에 따르면, <br>
메인 네임스페이스에서 약 100개의 멤버가 deprecated, 제거 또는 이동되었습니다.

마이그레이션을 진행할 때 이 부분도 함께 `np.float64`로 변경해주는 것이 좋습니다.

## 결론

NumPy 1.24의 `np.float` 제거는 처음에는 당황스러울 수 있는 'breaking change'지만, <br>
그 본질을 들여다보면 NumPy가 더 나은 방향으로 발전하고 있다는 신호입니다. <br>

코드의 모호성을 줄이고 타입 시스템의 일관성을 높이려는 노력의 일환인 셈이죠.

이제 우리는 `np.float` 대신 Python의 내장 `float`나 명시적인 `np.float64`를 사용하면 됩니다. <br>
이 둘은 메모리 표현이나 정밀도 면에서 거의 동일하기에 마이그레이션 부담도 적습니다. <br>
이번 기회에 코드 베이스를 점검하고 더 명확한 코드로 개선해 보는 것은 어떨까요?

앞으로는 타입을 명시적으로 지정하는 습관을 통해, <br>
NumPy의 발전 방향에 발맞춰 더욱 견고하고 가독성 좋은 코드를 작성해 나가야겠습니다. 