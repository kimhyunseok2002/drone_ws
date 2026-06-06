# drone_practice — 자율 비행 미션 (PX4 SITL)

배포된 `drone_practice` 워크스페이스에 **자율 비행 솔루션**을 통합한 패키지입니다.
**이륙 → 경로 추종(Pure Pursuit) → 장애물 회피(VFH / DWA, 2D LiDAR 1개) → 정밀 착륙.**

- OS Ubuntu 20.04 / ROS Noetic / Gazebo Classic 11 / PX4 **v1.13.3** SITL / MAVROS
- 기체: stock `iris` + **2D LiDAR 1개**(360°, 최대 10 m, `/scan`) — 카메라 0대
- 경로 추종: **Pure Pursuit** / 장애물 회피: **VFH**(기본) 또는 **DWA** (전환 가능)

---

## 평가 방식 대응 (중요)

평가 당일에는 **`worlds/practice.world` 와 `mission/practice_path.csv` 두 파일만 교체**됩니다.
본 솔루션의 `mission.launch` 는 이 **고정 경로**를 읽으므로, 두 파일만 바꾸면 그대로 평가됩니다.
(배포 파일 이름·위치 변경 없음. 제공 world/path는 수정하지 않습니다.)

- 경로 CSV는 `x,y` 또는 `x,y,z` 모두 처리(여기서 z열은 무시하고 `cruise_altitude` 사용).
- 착륙 패드 = 경로의 **마지막 점**으로 자동 인식.

---

## 빌드 & 실행

```bash
# 1) 빌드
cd ~/Desktop/drone_ws/drone_ws
catkin_make

# 2) 환경 설정 (매 새 터미널)
source devel/setup.bash
source src/drone_practice/launch/setup_px4_env.sh   # PX4 env + 모델 경로
# (Intel GPU/VNC에서 Gazebo GUI가 죽으면) export LIBGL_ALWAYS_SOFTWARE=1

# 3) 실행
roslaunch drone_practice mission.launch                 # GUI, VFH(기본)
roslaunch drone_practice mission.launch avoidance:=dwa  # DWA 사용
roslaunch drone_practice mission.launch gui:=false      # 헤드리스(빠른 확인)
```

띄우면 자동으로: Gazebo(eval world) → iris+2D LiDAR 스폰 → PX4 SITL → MAVROS →
OFFBOARD 진입 → 이륙 2.5 m → 경로추종/회피 → 패드 정밀착륙 → disarm.

> PX4 설치 위치가 다르면 `launch/setup_px4_env.sh` 의 `PX4_DIR` 만 맞춰주세요.

---

## 구조 (추가/통합된 부분)

```
drone_practice/
├── launch/
│   ├── mission.launch        # ★ 자율 미션 실행 (고정 eval 경로 사용)
│   ├── practice.launch        # (배포 원본, 미사용)
│   └── setup_px4_env.sh       # (배포 원본) PX4 env + 모델 경로
├── scripts/mission_node.py    # ★ 상태기계 (TAKEOFF→FOLLOW→LAND, OFFBOARD)
├── src/drone_practice/        # ★ 알고리즘 모듈 (ROS 없이 단위테스트 가능)
│   ├── pure_pursuit.py        #   경로 추종
│   ├── vfh.py                 #   VFH 회피
│   ├── dwa.py                 #   DWA 회피
│   └── path_utils.py
├── config/mission_params.yaml # ★ 모든 튜닝 파라미터
├── models/iris_2d_lidar/      # ★ iris + 2D LiDAR(10 m, /scan)
├── models/{waypoint_marker,landing_pad,start_marker}/   # (배포 원본)
├── worlds/practice.world      # (평가 시 교체)
└── mission/practice_path.csv  # (평가 시 교체)
```

---

## 동작 원리 (요약)

```
 practice_path.csv ─► Pure Pursuit ─► 목표 방향
                                        │
 /scan(2D LiDAR) ─► VFH 또는 DWA ─► 충돌없는 진행 ─► vx,vy (+ 고도 P제어 2.5 m)
                                        ▼
              /mavros/setpoint_raw/local (OFFBOARD) ─► 패드 위 AUTO.LAND
```

- **Pure Pursuit**: 경로 위 lookahead 점으로 향하는 목표 heading 산출.
- **VFH**: `/scan` 극좌표 히스토그램 → 통과 가능한 gap 중 목표에 가장 가까운 방향 선택.
  **위치-캐럿 제어**로 관성 드리프트 제거.
- **DWA**: 도달 가능한 속도들을 예측·평가(목표정렬+여유+속도)해 최적 속도 선택.
  동역학을 직접 고려 → 더 부드럽고 드리프트 없음. (`avoidance:=dwa`)
- **고도**: 회피는 수평으로만, 2.5 m 유지(규정).
- **착륙**: 경로 마지막 점(패드) 위 정렬 후 `AUTO.LAND`.

---

## 규정 준수
- 전 구간 자율(OFFBOARD), 수동 입력 없음 · 카메라 0대 · LiDAR 탐지 10 m
- stock iris 물리 그대로 · 회피는 수평만(고도 유지) · 제공 world/path 무수정(인자로만 사용)

## 주요 파라미터 (`config/mission_params.yaml`)
| 파라미터 | 기본 | 의미 |
|---|---|---|
| `obstacle_avoidance` | vfh | `vfh` 또는 `dwa` |
| `cruise_altitude` | 2.5 | 순항 고도 [m] |
| `mpc_xy_cruise` | 0.8 | 수평 순항 속도 [m/s] |
| `carrot_distance` | 1.0 | (VFH) 위치 캐럿 전방 거리 [m] |
| `vfh_safety_distance` | 0.35 | 장애물 안전 여유 [m] |
| `dwa_max_speed` | 1.0 | (DWA) 속도 상한 [m/s] |

검증(연습 맵 기준): 충돌 0, 장애물 여유 1.1~1.8 m, 착륙 오차 ~0.1 m.
