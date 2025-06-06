---
title: "Airflow DAG 파싱 주기 설정 시 주의할 점과 최적화 전략"
date: 2025-05-24 00:00:00 +0900
categories: [ Tech, Airflow ]
tags: [ Airflow ]
toc: true
toc_sticky: true
comments: true
---

## 서론

Airflow를 운영하다 보면 DAG 변경사항이 언제 반영되는지 답답할 때가 많습니다.

파일을 수정했는데 UI에서 바로 확인이 안 되고, <br>
**"도대체 언제 반영되는 거야?"** 싶어 F5를 연타한 경험, 다들 있으실 거예요.

저도 처음에는 단순히 **"파싱 주기만 짧게 하면 되겠지"**라고 생각했는데, <br>
무작정 적용했다가는 또 다른 문제가 터질 수 있습니다.

오늘은 실무에서 겪었던 **파싱 주기 설정 시 주의할 점**들과 <br>
나름대로 찾아낸 **최적화 팁**들을 공유해보려고 합니다.

## 목차
- [1. Airflow DAG 파싱 과정 이해하기](#1-airflow-dag-파싱-과정-이해하기)
- [2. 주요 옵션별 역할과 동작 방식](#2-주요-옵션별-역할과-동작-방식)
- [3. 왜 파싱만으로는 의미가 없는가?](#3-왜-파싱만으로는-의미가-없는가)
- [4. 설정 시 주의점 & 권장 전략](#4-설정-시-주의점--권장-전략)
- [5. 결론 및 개인적인 소감](#5-결론-및-개인적인-소감)

---

## 1. Airflow DAG 파싱 과정 이해하기

### 1.1 전체 파싱 흐름도

``` 
DAG 파일 변경/추가 
      ↓
① 디렉토리 스캔 (refresh_interval) 
      ↓ 
② DAG 파일 파싱 (min_file_process_interval) 
      ↓
③ 직렬화 & DB 저장(CLI 반영) (min_serialized_dag_update_interval) 
      ↓
④ Webserver 읽기 (min_serialized_dag_fetch_interval) 
      ↓
⑤ UI/REST API에 반영
```

### 1.2 각 단계별 상세 설명

**① 디렉토리 스캔**

- DAG 프로세서가 DAGs 폴더를 주기적으로 스캔하여 새로운 파일이나 삭제된 파일을 감지
- 파일의 last modified date, 크기 등 메타데이터 확인

**② DAG 파일 파싱**

- .py로 작성된 DAG 코드를 실행하여 DAG 객체 파싱
- 외부 임포트, 환경 변수, 동적 구성 요소 모두 처리
- **여기서 중요한 건**, 파일이 변경되지 않아도 외부 의존성 때문에 주기적 파싱이 필요하다는 점입니다

**③ 직렬화 & DB 저장**

- 파싱된 DAG 객체를 JSON 형태로 직렬화
- 메타데이터 DB의 `serialized_dag` 테이블에 저장
- **이 단계가 완료되어야 비로소 다른 컴포넌트들이 변경사항을 인식**합니다

**④ Webserver 읽기**

- Webserver가 DB에서 직렬화된 DAG 정보를 읽어옴
- 메모리에 DagBag 구성하여 UI/REST API에서 사용

### 1.3 왜 풀파싱(전체 파싱)이 필요한가?

Airflow에서 단순히 파일 변경만 감지하지 않고 주기적으로 모든 DAG을 파싱하는 이유는 다음과 같습니다

- **동적 DAG 생성** - 환경 변수, 데이터베이스 조회, API 호출로 DAG 구조가 동적으로 결정
- **외부 의존성** - 임포트된 모듈, 설정 파일이 바뀌면 DAG 파일은 그대로여도 파싱 결과 변경
- **신뢰성 우선** - 변경 감지 실패로 인한 DAG 동기화 문제를 방지

---

## 2. 주요 옵션별 역할과 동작 방식

### 2.1 `[dag_processor] refresh_interval`

- **기능** - DAG 번들에서 새 파일을 찾거나 갱신하는 주기 (초 단위)
- **기본값** - `300` (5분)
- **역할** - 새 파일 추가/삭제 감지
- ⚠️ `min_file_process_interval`보다 짧게 설정하면 불필요한 스캔이 발생합니다

### 2.2 `[dag_processor] min_file_process_interval`

- **기능** - 각 DAG 파일을 파싱하는 최소 간격 (초 단위)
- **기본값** - `30`
- **역할**
    - 파일 변경 여부와 무관하게 주기적으로 전체 DAG 파싱
    - 외부 의존성(임포트 모듈, 환경 변수) 변화 감지
- **Tip**
    - _파싱이 모든 변화의 시작점이므로, 이 옵션이 DAG 반영 속도에 가장 직접적인 영향을 줍니다_

### 2.3 `[core] min_serialized_dag_update_interval`

- **기능** - 직렬화된 DAG을 DB에 저장하는 최소 간격 (초 단위)
- **기본값** - `30`
- **역할** - 파싱 결과를 메타데이터 DB에 저장
- ⚠️ **중요**
    -  이 단계가 완료되어야 CLI가 변경사항을 인식할 수 있습니다

### 2.4 `[core] min_serialized_dag_fetch_interval`

- **기능** - Webserver가 DB에서 직렬화된 DAG을 읽어오는 최소 간격 (초 단위)
- **기본값** - `10`
- **역할** - UI/REST API 반영 속도 제어

---

## 3. 왜 파싱만으로는 의미가 없는가?

### 3.1 파싱 과정과 저장 과정이 분리되어 있음

**문제 시나리오**

```ini
[dag_processor]
min_file_process_interval = 10  # 10초마다 파싱

[core]
min_serialized_dag_update_interval = 300  # 5분마다 DB 저장
```

**결과**

- DAG 프로세서는 10초마다 파싱하여 최신 DAG 구조 인식
- 하지만 DB 저장은 5분마다만 수행
- **Webserver/UI는 5분 후에야 변경사항 확인 가능**

### 3.2 Airflow 2.x/3.x의 직렬화 필수 구조

Airflow 2.0부터 [DAG 직렬화가 필수가 되었고](https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/dag-serialization.html) 비활성화할 수 없습니다.

- **DAG 프로세서/스케줄러** - DAG 파싱 + 직렬화 + DB 저장 담당
- **Webserver** - DAG 파일을 직접 파싱하는 것이 금지되고, DB의 직렬화된 정보만 사용
- **CLI** - 마찬가지로 DB의 직렬화된 정보를 참조

**결론** - 파싱 결과가 DB에 저장되지 않으면, 아무리 자주 파싱해도 시스템 전체에 변경사항이 반영되지 않습니다.


## 4. 설정 시 주의점 & 권장 전략

### 4.1 옵션 간 밸런스가 핵심

**최적화 예제**

```
refresh_interval ≥ min_file_process_interval ≈ min_serialized_dag_update_interval
```

**권장 설정 예시**

```ini
[dag_processor]
refresh_interval = 300
min_file_process_interval = 30

[core]
min_serialized_dag_update_interval = 30
min_serialized_dag_fetch_interval = 10
```

### 4.2 환경별 튜닝 가이드

**소규모 환경 (DAG < 100개)**

- 빠른 반영 우선: `min_file_process_interval = 10`
- 동기화: `min_serialized_dag_update_interval = 10`

**대규모 환경 (DAG 1000+개)**

- 안정성 우선: `min_file_process_interval = 60`
- DB 부하 고려: `min_serialized_dag_update_interval = 120`
- 압축 활성화: `compress_serialized_dags = True` (단, DAG 의존성 뷰가 비활성화됨)

### 4.3 모니터링 추천 지표

**성능 메트릭**

- `dag_processing.total_parse_time`: 전체 파싱 소요 시간
- `dag_processing.last_duration`: 마지막 파싱 소요 시간
- `scheduler.heartbeat`: 스케줄러가 살아있는지 확인

**임계점 판단**
- `total_parse_time > min_file_process_interval`이면 파싱 주기를 늘려야 합니다
- Webserver 메모리 사용량이 급증하면 `min_serialized_dag_fetch_interval`을 늘려야 합니다

## 5. 결론 및 개인적인 소감

### 5.1 우선순위 설정 원칙

1. **`min_file_process_interval`을 기준으로 삼으세요**
    - 파싱이 모든 변화의 시작점이므로, 이 값을 가장 신중하게 관리해야 합니다
2. **DB 저장 주기는 파싱 주기와 동기화**
    - 파싱만 하고 저장하지 않으면 의미가 없습니다
    - `min_serialized_dag_update_interval = min_file_process_interval` 권장
3. **Webserver 반영 속도는 기존 DAG 운영 관점에서는 부차적**
    - UI 지연 허용 범위 내에서 `min_serialized_dag_fetch_interval` 조정
    - 단, RestAPI를 통해 Airflow를 제어하고 있다면 이 값 역시 중요할 수 있음

### 5.2 개인적인 경험담

이런 설정들을 일부 만져본 경험으로는, <br>
모든 주기를 바로 짧게 만들기보다는 **점진적으로 최적화**하는 것을 추천합니다.

특히 파싱 주기를 너무 짧게 설정한다면 CPU 사용량이 급증할 수 있으니 <br>
전체 시스템이 불안정해질 수 있으니, 항상 모니터링을 하면서 조정하는 것을 권장드립니다.

가장 중요한 건, **"파싱은 시작이지만, 저장 없이는 끝나지 않는다"**는 점입니다.

빠른 파싱보다는 일관된 파싱-저장 주기를 유지하는 것이 <br>
안정적인 Airflow 운영의 핵심이라고 생각합니다.

---

**참고 자료**

[Apache Airflow DAG Serialization 공식 문서](https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/dag-serialization.html)

[Apache Airflow Release Notes](https://airflow.apache.org/docs/apache-airflow/stable/release_notes.html)