#!/usr/bin/env python3
"""추가 연습 맵 Path CSV 생성기 (map2 / map3)

generate_path.py 와 동일하게 꼭짓점을 직선으로 잇고 0.1m 간격으로 샘플링한다.
각 맵의 장애물(해당 .world 와 동일한 좌표)에 대해 경로와의 최소 여유(clearance)를
함께 출력하여, "회피는 강제되지만 통과는 가능한" 배치인지 검증한다.

생성 파일:  map2_path.csv,  map3_path.csv
대응 월드:  worlds/map2.world,  worlds/map3.world
"""
import numpy as np
import csv

SAMPLE_INTERVAL = 0.1   # [m]
DRONE_RADIUS = 0.30     # vfh_robot_radius 와 동일 (검증용)


def sample_segment(p1, p2, interval):
    p1, p2 = np.array(p1), np.array(p2)
    distance = np.linalg.norm(p2 - p1)
    num_samples = max(1, int(distance / interval))
    return [tuple(p1 + (i / num_samples) * (p2 - p1)) for i in range(num_samples)]


def build_path(keypoints):
    pts = []
    for i in range(len(keypoints) - 1):
        pts.extend(sample_segment(keypoints[i], keypoints[i + 1], SAMPLE_INTERVAL))
    pts.append(keypoints[-1])
    return pts


def write_csv(filename, pts):
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y', 'z'])
        for x, y, z in pts:
            w.writerow([f'{x:.3f}', f'{y:.3f}', f'{z:.3f}'])


def min_path_clearance(pts, ox, oy, half_size):
    """장애물 표면에서 경로 중심선까지 최소 거리 [m]."""
    xy = np.array([(p[0], p[1]) for p in pts])
    d_center = np.min(np.linalg.norm(xy - np.array([ox, oy]), axis=1))
    return d_center - half_size


def report(name, keypoints, obstacles):
    pts = build_path(keypoints)
    length = sum(np.linalg.norm(np.array(keypoints[i + 1]) - np.array(keypoints[i]))
                 for i in range(len(keypoints) - 1))
    fname = f'{name}_path.csv'
    write_csv(fname, pts)
    print(f'\n[{name}] {fname}: {len(pts)} pts, length {length:.2f} m, '
          f'goal/landing = {keypoints[-1][:2]}')
    for o in obstacles:
        clr = min_path_clearance(pts, o['x'], o['y'], o['half'])
        gap = clr - DRONE_RADIUS   # 기체 외곽이 중심선을 지날 때 남는 여유
        flag = 'OK' if 0.0 <= gap <= 0.8 else ('!! ' if gap < 0.0 else '~ ')
        print(f'  {flag} {o["name"]:<22} center=({o["x"]:.2f},{o["y"]:.2f}) '
              f'half={o["half"]:.2f}  surf->path={clr:+.2f}m  drone-margin={gap:+.2f}m')


# ---------------------------------------------------------------------------
# Map2 : 직선 슬라럼(slalom) 코리도 — 동쪽 직진하며 장애물을 좌/우 번갈아 회피
# ---------------------------------------------------------------------------
MAP2_KEYS = [(0.0, 0.0, 2.5), (4.0, 0.0, 2.5), (8.0, 0.0, 2.5), (12.0, 0.0, 2.5)]
MAP2_OBS = [
    {'name': 'obstacle_cyl_1', 'x': 3.0, 'y': 0.45, 'half': 0.40},   # 우측 -> 좌로 회피
    {'name': 'obstacle_box_1', 'x': 6.0, 'y': -0.45, 'half': 0.40},  # 좌측 -> 우로 회피
    {'name': 'obstacle_cyl_2', 'x': 9.0, 'y': 0.45, 'half': 0.40},   # 우측 -> 좌로 회피
]

# ---------------------------------------------------------------------------
# Map3 : ㄱ자 회전 + 게이트(gate) 통과 — 좁은 문을 지나고 코너에서 장애물 회피
# ---------------------------------------------------------------------------
MAP3_KEYS = [(0.0, 0.0, 2.5), (6.0, 0.0, 2.5), (6.0, 6.0, 2.5)]
MAP3_OBS = [
    {'name': 'gate_left',     'x': 3.0, 'y': 0.85, 'half': 0.40},    # 게이트 좌
    {'name': 'gate_right',    'x': 3.0, 'y': -0.85, 'half': 0.40},   # 게이트 우 (통로폭 ~0.9m)
    {'name': 'corner_cyl',    'x': 6.55, 'y': 3.5, 'half': 0.40},    # 회전 후 직선에서 회피
]


if __name__ == '__main__':
    report('map2', MAP2_KEYS, MAP2_OBS)
    report('map3', MAP3_KEYS, MAP3_OBS)
    print('\n완료. worlds/map2.world, worlds/map3.world 와 함께 사용하세요.')
