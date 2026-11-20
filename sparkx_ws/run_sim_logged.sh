#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source /home/sayak/Desktop/SparkX/sparkx_ws/install/setup.bash
cp /opt/ros/jazzy/share/nav2_bt_navigator/behavior_trees/navigate_w_replanning_only_if_path_becomes_invalid.xml /tmp/sparkx_nav_tree_36.xml 2>/dev/null || true
exec ros2 launch sparkx_bringup practice.launch.py world:=practice_small_day gui:=false
