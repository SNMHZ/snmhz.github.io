---
title: "Python 3.10에서 3.11로 마이그레이션 시 str, Enum 문자열 출력 동작 변경과 대응 전략"
date: 2025-05-16 00:00:00 +0900
categories: [ Tech, Python ]
tags: [ Python, Python3.11, Enum ]
toc: true
toc_sticky: true
comments: true
---

## 서론

Python 3.11에서는 `str, Enum` 다중상속 클래스의 문자열 출력 동작이 크게 변경되었습니다. 

이 변화는 열거형 클래스의 일관성을 높이기 위한 의도적인 수정이지만, 

이미 작성된 코드 호환성에 직접적인 영향을 주므로 주의가 필요합니다.

## 목차
- [1. Python 3.10과 3.11의 차이점](#1-python-310과-311의-차이점)
- [2. 영향받는 코드](#2-영향받는-코드)
- [3. 해결책: StrEnum 사용하기](#3-해결책-strenum-사용하기)
- [4. 하위 호환성 유지 방법](#4-하위-호환성-유지-방법)
- [5. 테스트 권장사항](#5-테스트-권장사항)
- [6. 결론](#6-결론)

## 1. Python 3.10과 3.11의 차이점

```python
# Python 3.10
from enum import Enum

class Color(str, Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'

print(f"{Color.RED}")  # 'red'
print(str(Color.RED))  # 'Color.RED'
print("{}".format(Color.RED))  # 'red'
print("%s" % Color.RED)  # 'Color.RED'
```

```python
# Python 3.11
from enum import Enum

class Color(str, Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'

print(f"{Color.RED}")  # 'Color.RED' - 변경됨!
print(str(Color.RED))  # 'Color.RED'
print("{}".format(Color.RED))  # 'Color.RED' - 변경됨!
print("%s" % Color.RED)  # 'Color.RED'
```

이 변경은 Python의 Enum 클래스 동작을 더 일관되게 만들기 위한 의도적인 수정입니다. 

Python 3.10까지는 `str, Enum` 다중상속 클래스가 

f-string이나 str.format()에서 사용될 때 `__format__` 메서드가 값을 반환했지만, 

`str()` 함수나 %-형식 문자열에서는 클래스와 멤버 이름을 반환했습니다. 

3.11부터는 이 불일치가 해소되어 모든 문자열 변환 상황에서 클래스명과 멤버명을 함께 출력하도록 바뀌었습니다[^3][^5].

## 2. 영향받는 코드

이 변경은 다음과 같은 상황에서 문제를 일으킬 수 있습니다:

- f-string에서 Enum 멤버를 직접 사용하는 코드
- `str.format()`을 사용한 포맷팅
- Enum 값이 문자열로 자동 변환되는 API 호출

```python
# 3.10에선 작동하나 3.11에선 실패하는 코드
def make_path(color: Color) -> str:
    return f"/colors/{color}"  # 3.10: "/colors/red", 3.11: "/colors/Color.RED"
```

## 3. 해결책: StrEnum 사용하기

Python 3.11에서는 이 문제를 해결하기 위해 `StrEnum` 클래스가 새로 추가되었습니다[^1][^2].

```python
# Python 3.11
from enum import StrEnum

class Color(StrEnum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'

print(f"{Color.RED}")  # 'red'
print(str(Color.RED))  # 'red'
```

`StrEnum`은 문자열 포맷팅 시 값을 반환하도록 설계된 새로운 클래스입니다. 

기존 Python 3.10의 `str, Enum` 다중상속이 f-string에서만 값을 반환했던 것과 달리, 

`StrEnum`은 모든 문자열 변환 상황에서 일관되게 값을 반환합니다[^3][^5].

## 4. 하위 호환성 유지 방법

Python 3.10 이하 버전과 3.11 이상을 모두 지원하려면:

```python
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    
    class StrEnum(str, Enum):
        """Python 3.10 이하에서 StrEnum 에뮬레이션"""
        def __str__(self):
            return self.value
            
        def __format__(self, format_spec):
            return format(self.value, format_spec)

class Color(StrEnum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'
```

위 코드는 Python 3.11에서는 내장된 `StrEnum`을 사용하고, 

이전 버전에서는 `str, Enum` 다중상속에 `__str__`과 `__format__` 메서드를 직접 구현하여 동일한 동작을 구현합니다[^4].

## 5. 테스트 권장사항

여러 Python 버전에서의 동작을 확인하기 위해 다음 테스트를 권장합니다:

```python
def test_enum_string_format():
    assert f"{Color.RED}" == "red"
    assert str(Color.RED) == "red"
    assert "{}".format(Color.RED) == "red"
    assert "{:>5}".format(Color.RED) == "  red"  # 포맷 지정자 테스트
    assert f"{Color.RED:>5}" == "  red"  # f-string 포맷 지정자 테스트
```

## 6. 결론

이 변경사항은 Python 3.11로 업그레이드하는 많은 프로젝트에 영향을 줄 수 있으므로, 미리 코드를 점검하고 대응하는 것이 중요합니다. 특히 f-string이나 문자열 포맷팅을 사용하는 코드는 반드시 테스트를 통해 동작을 확인해야 합니다.

<hr>

[^1]: https://docs.python.org/3/howto/enum.html
[^2]: https://blog.pecar.me/python-enum
[^3]: https://tsak.dev/posts/python-enum/
[^4]: https://tomwojcik.com/posts/2023-01-02/python-311-str-enum-breaking-change 
[^5]: https://docs.python.org/3/whatsnew/3.11.html#enum 