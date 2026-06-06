#!/usr/bin/env python3
"""연습용 Path CSV 생성기

각 구간을 직선으로 연결하고 0.1m 간격으로 샘플링한다.
실전 맵에서는 곡선 구간을 추가할 수 있다.
"""
import numpy as np
import csv

# 경로의 꼭짓점들 (이륙 후 비행 고도 2.5m 기준)
keypoints = [
    (0.0,  0.0, 2.5),   # 이륙 직후 호버링 지점
    (3.0,  0.0, 2.5),   # WP1
    (5.0,  2.0, 2.5),   # WP2
    (8.0,  5.0, 2.5),   # WP3
    (12.0, 5.0, 2.5),   # 착륙 패드 직상공
]

SAMPLE_INTERVAL = 0.1  # 0.1m 간격

def sample_segment(p1, p2, interval):
    """두 점 사이를 interval 간격으로 샘플링 (끝점 제외)"""
    p1 = np.array(p1)
    p2 = np.array(p2)
    distance = np.linalg.norm(p2 - p1)
    num_samples = int(distance / interval)
    points = []
    for i in range(num_samples):
        t = i / num_samples
        point = p1 + t * (p2 - p1)
        points.append(tuple(point))
    return points

# 전체 경로 생성
path_points = []
for i in range(len(keypoints) - 1):
    path_points.extend(sample_segment(keypoints[i], keypoints[i+1], SAMPLE_INTERVAL))
path_points.append(keypoints[-1])  # 마지막 점 추가

# CSV 저장
with open('practice_path.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y', 'z'])
    for x, y, z in path_points:
        writer.writerow([f'{x:.3f}', f'{y:.3f}', f'{z:.3f}'])

print(f"생성 완료: {len(path_points)}개 점")
print(f"전체 경로 길이: {sum(np.linalg.norm(np.array(keypoints[i+1]) - np.array(keypoints[i])) for i in range(len(keypoints)-1)):.2f} m")