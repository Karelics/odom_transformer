#    Odom transformer - ROS 2 Node for performing odometry transformations.
#    Copyright (C) 2023  Karelics Oy
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Example launch file for odom_transformer"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Launch odom_transformer."""

    package_share_path = get_package_share_directory("odom_transformer")
    transformer_config_path = os.path.join(package_share_path, "params", "odom_transformer_params.yaml")

    odom_transformer = Node(
        package="odom_transformer",
        executable="transformer_node.py",
        name="odom_transformer",
        output={"both": {"screen", "log", "own_log"}},
        parameters=[transformer_config_path],
    )

    return LaunchDescription([odom_transformer])
