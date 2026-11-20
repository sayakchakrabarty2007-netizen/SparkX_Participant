#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source /home/sayak/Desktop/SparkX/sparkx_ws/install/setup.bash
exec ros2 launch sparkx_bringup practice.launch.py world:=practice_small_day gui:=false

