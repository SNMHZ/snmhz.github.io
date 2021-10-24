---
layout: single
title: "CUDA Architecture"
toc: true
toc_sticky: true
comments: true
---

# CUDA Archtecture

- CUDA hardware의 구조(Tesla GP100 예시)
    - 1GPU에 6GPC(graphics processing cluster)
    - 1GPC에 10Pascal SM -> 1GPU에 60SM
    - 1SM(unit) = 32SP + 16DP + 8SFU + 2Tex
        - SP(streaming processor) : FP32 core, 메인 CUDA core, ALU for a single CUDA thread
        - DP(double precision) : FP64 core
        - SFU(sepcial function unit) : sin, cos, square root 등 특별한 연산 1클락에 해결 가능
        - Tex(texture processor) : for graphics purpose, CUDA로 사용시 사용하지 않기도 하고 메모리로 쓰기도 함

<br>

- CUDA 의 확장성
    - CUDA dedvice는 1~4개의 SM의 저가 모바일 기기부터 1000+의 고가 워크스테이션까지 매우 다양
    - thread block 개념을 도입하여 해결(SM 1개가 thread block 1개 처리)
    - so, grid - block - thread의 계층 구조 필요
    - thread block 들이 SM에 자유롭게 assign 되어서 처리되는 구조
    - Each block can execute in any order relative to other blocks

<br>

- SM에서 CU(control Unit, SM당 1개)의 실행 구조
    - 1개의 CU의 제어를 받아 32 core(SP) 가 물리적으로 동시에 실행
    - 1개의 warp scheduler
    - 32 thread가 같은 instruction을 동시 실행
    - SM 1개는 2048+ thread를 동시 관리 -> memory의 느린 반응 속도 해결

<br>

- Thread와 Warp
    - Thread는 독립적 실행 단위(실)
    - Warp 평행하게 관리되는 여러개의 실(Warp를 만드는 것처럼 여러 실을 평행하게 관리)
    - CUDA에서의 Warp는 32개의 thread(SM이 32개의 SP를 가지므로)
    - lane: Warp 내에서의 thread의 index(0~31)
    - block 에는 1024개의 thread가 있지만, 32개씩 끊어서 warp로 관리
    - 20개 이상의 warp가 대기 상태로 있는 것이 효율적
        - memory access 시간을 고려
        - warp 전환간 거의 zero-overhead. 충분히 많은 register를 확보하고 있기 때문
        - warp scheduler는 HW로 구현되어 오버헤드 거의 없음

<br>

- 2레벨 병렬 처리
    - grid는 thread blocks로 이루어져 있으므로 SM에 병렬 처리
    - thread block은 여러 warp로 갈라져서 병렬 처리
    - warp / block 종료 시 다음 warp / block을 처리
    - 자원 제약에 대한 고려가 필요하지만, thread수를 1024정도로 잡으면 문제없음
    - block의 실행 순서가 정해져 있지 않음

<br>

- warp id, lane id
    - GPU assembly instruction으로 체크 가능
    - warp id : SM 내에서, 특정 warp의 ID number
        ```cpp
        __device__ unsigned warp_id(void) {
            // this is not equal to threadIdx.x / 32
            unsigned ret;
            asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
            return ret;
        }
        ```
    - lane id : warp 내에서, 자신의 lane id
        ```cpp
        __device__ unsigned lane_id(void) {
            unsigned ret;
            asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
            return ret;
        }
        ```

<br>