#!/bin/bash
# PX4 경로를 실제 설치 위치로 바꾸세요
export PX4_DIR=$HOME/PX4-Autopilot

source $PX4_DIR/Tools/setup_gazebo.bash $PX4_DIR $PX4_DIR/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$PX4_DIR
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$PX4_DIR/Tools/sitl_gazebo

# 우리 모델 경로 추가
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(rospack find drone_practice)/models

echo "PX4 환경 설정 완료"