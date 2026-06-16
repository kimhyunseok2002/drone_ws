# drone_ws

PX4 SITL 자율 비행 기말 프로젝트 워크스페이스 (KAU).
**이륙 → 경로 추종(Pure Pursuit) → 장애물 회피(VFH / DWA, 2D LiDAR 1개) → 정밀 착륙.**

- Ubuntu 20.04 · ROS Noetic · Gazebo Classic 11 · PX4 **v1.13.3** SITL · MAVROS
- 기체: stock `iris` + **2D LiDAR 1개**(360°, 10 m, `/scan`) + **하단 카메라 1개**(`/landing_cam/image_raw`, 비전 정밀착륙용)
- 경로 추종 **Pure Pursuit** · 회피 **VFH / DWA / FGM / MPPI** 4종 전환 (`avoidance:=dwa|fgm|mppi`)

> 코드/문서 전체는 패키지 안에 있습니다 → [`src/drone_practice/`](src/drone_practice/)
> 자세한 설명: [`src/drone_practice/README.md`](src/drone_practice/README.md)

## 빌드 & 실행
```bash
git clone https://github.com/kimhyunseok2002/drone_ws.git
cd drone_ws && catkin_make 
source devel/setup.bash
source src/drone_practice/launch/setup_px4_env.sh      # PX4 env + 모델 경로
roslaunch drone_practice mission.launch                # VFH알고리즘 (기본)
roslaunch drone_practice mission.launch avoidance:=dwa # DWA알고리즘 (vfh|dwa|fgm|mppi)
# roslaunch drone_practice mission.launch gui:=false   # 헤드리스
```

## 평가 방식 대응
평가 당일에는 `src/drone_practice/worlds/practice.world` 와
`src/drone_practice/mission/practice_path.csv` **두 파일만 교체**됩니다.
`mission.launch` 가 이 고정 경로를 읽으므로 두 파일 교체만으로 동작합니다.
(배포 원본 파일명 보존, 제공 world/csv 무수정.)

## 구성
```
src/drone_practice/
├── launch/mission.launch          # 자율 미션 실행 (고정 eval 경로)
├── scripts/mission_node.py        # 상태기계 (TAKEOFF→FOLLOW→LAND, OFFBOARD)
├── src/drone_practice/            # pure_pursuit · vfh · dwa · path_utils
├── config/mission_params.yaml     # 튜닝 파라미터
├── models/iris_2d_lidar/          # iris + 2D LiDAR
├── worlds/practice.world          # (평가 시 교체)
└── mission/practice_path.csv      # (평가 시 교체)
```

검증: 충돌 0, 장애물 여유 1.1~1.8 m, 착륙 오차 ~0.1 m.
