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

"""Test basic functionality of the Transformer class with a couple simple cases."""

from math import sqrt
from typing import List

import numpy as np
import pytest

from geometry_msgs.msg import Pose, PoseWithCovariance, Transform, Twist, TwistWithCovariance
from odom_transformer.transformer import Transformer


def test_setup() -> None:
    """Test setting up the transformer."""

    # sensor is 2 m in front, 1 m to the left and 1 m up from robot center, looking straight left
    input_to_base_tf = Transform()
    input_to_base_tf.translation.x = 2.0
    input_to_base_tf.translation.y = 1.0
    input_to_base_tf.translation.z = 1.0
    input_to_base_tf.rotation.x = 0.0
    input_to_base_tf.rotation.y = 0.0
    input_to_base_tf.rotation.z = 1 / sqrt(2)
    input_to_base_tf.rotation.w = 1 / sqrt(2)

    expected_twist_tf_matrix = np.array(
        [
            [0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, -1.0, -2.0],
            [0.0, 0.0, 1.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    expected_pose_cov_tf_matrix = np.array(
        [
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    expected_base_pose_in_input_frame = Pose()
    expected_base_pose_in_input_frame.position.x = -1.0
    expected_base_pose_in_input_frame.position.y = 2.0
    expected_base_pose_in_input_frame.position.z = -1.0
    expected_base_pose_in_input_frame.orientation.x = 0.0
    expected_base_pose_in_input_frame.orientation.y = 0.0
    expected_base_pose_in_input_frame.orientation.z = 1 / sqrt(2)
    expected_base_pose_in_input_frame.orientation.w = -1 / sqrt(2)

    transformer = Transformer(input_to_base_tf)
    transformer._source_to_base_tf._update_cov_tf_matrix()

    # allow some float inaccuracy
    for i, row in enumerate(transformer._twist_tf_matrix.tolist()):
        for j, val in enumerate(row):
            assert abs(val - expected_twist_tf_matrix.tolist()[i][j]) < 1e-14, f"twist tf, row {i+1}, col {j+1}"
            cov_tf_matrix = transformer._source_to_base_tf._cov_tf_matrix
            assert (
                abs(cov_tf_matrix.tolist()[i][j] - expected_pose_cov_tf_matrix.tolist()[i][j]) < 1e-14
            ), f"pose cov tf, row {i+1}, col {j+1}"

    base_pos = transformer._base_pose_in_source_frame.position
    base_ori = transformer._base_pose_in_source_frame.to_msg().orientation
    assert abs(base_pos[0] - expected_base_pose_in_input_frame.position.x) < 1e-14
    assert abs(base_pos[1] - expected_base_pose_in_input_frame.position.y) < 1e-14
    assert abs(base_pos[2] - expected_base_pose_in_input_frame.position.z) < 1e-14
    assert abs(base_ori.x - expected_base_pose_in_input_frame.orientation.x) < 1e-14
    assert abs(base_ori.y - expected_base_pose_in_input_frame.orientation.y) < 1e-14
    assert abs(base_ori.z - expected_base_pose_in_input_frame.orientation.z) < 1e-14
    assert abs(base_ori.w - expected_base_pose_in_input_frame.orientation.w) < 1e-14


@pytest.mark.parametrize(
    "sensor_pos,ang_vel,expected_lin_vel",
    [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]),
    ],
)
def test_angular_to_linear_tf(sensor_pos: List[float], ang_vel: List[float], expected_lin_vel: List[float]) -> None:
    """Test computing the effect of angular twist of the sensor to the linear twist of base."""
    # sensor is at the given position laying on its left side (roll = -pi/2)
    input_to_base_tf = Transform()
    input_to_base_tf.translation.x = sensor_pos[0]
    input_to_base_tf.translation.y = sensor_pos[1]
    input_to_base_tf.translation.z = sensor_pos[2]
    input_to_base_tf.rotation.x = -1 / sqrt(2)
    input_to_base_tf.rotation.y = 0.0
    input_to_base_tf.rotation.z = 0.0
    input_to_base_tf.rotation.w = 1 / sqrt(2)

    transformer = Transformer(input_to_base_tf)

    twist = Twist()
    twist.angular.x = ang_vel[0]
    twist.angular.y = ang_vel[1]
    twist.angular.z = ang_vel[2]

    res = transformer.transform_twist(TwistWithCovariance(twist=twist))
    assert abs(res.twist.linear.x - expected_lin_vel[0]) < 1e-14
    assert abs(res.twist.linear.y - expected_lin_vel[1]) < 1e-14
    assert abs(res.twist.linear.z - expected_lin_vel[2]) < 1e-14


def test_transform_twist() -> None:
    """Test transforming twist."""
    # NOTE: Some values in the test are in no way realistic, they are just chosen for "easy" manual checking

    input_twist = TwistWithCovariance()
    input_twist.twist.linear.x = 1.0
    input_twist.twist.linear.y = 2.0
    input_twist.twist.linear.z = 3.0
    input_twist.twist.angular.x = 1.0
    input_twist.twist.angular.y = 2.0
    input_twist.twist.angular.z = 3.0
    
    input_twist.covariance = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
        31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
    ]

    # rotate 90 degrees counter-clockwise (positive yaw), with the sensor being 1.0 m in front of robot base
    # i.e. the sensor is at point (1, 0, 0) from robot base and turned left.
    input_to_base_tf = Transform()
    input_to_base_tf.translation.x = 1.0
    input_to_base_tf.translation.y = 0.0
    input_to_base_tf.translation.z = 0.0
    input_to_base_tf.rotation.x = 0.0
    input_to_base_tf.rotation.y = 0.0
    input_to_base_tf.rotation.z = 1 / sqrt(2)
    input_to_base_tf.rotation.w = 1 / sqrt(2)
    transformer = Transformer(input_to_base_tf)

    expected_twist = TwistWithCovariance()
    expected_twist.twist.linear.x = -2.0
    expected_twist.twist.linear.y = -2.0
    expected_twist.twist.linear.z = 4.0
    expected_twist.twist.angular.x = -2.0
    expected_twist.twist.angular.y = 1.0
    expected_twist.twist.angular.z = 3.0

    expected_twist.covariance = [
        8.0, 5.0, -19.0, 11.0, -10.0, -12.0,
        30.0, 0.0, -60.0, 30.0, -30.0, -30.0,
        -34.0, -10.0, 74.0, -40.0, 38.0, 42.0,
        26.0, 5.0, -55.0, 29.0, -28.0, -30.0,
        -20.0, -5.0, 43.0, -23.0, 22.0, 24.0,
        -32.0, -5.0, 67.0, -35.0, 34.0, 36.0,
    ]

    transformed_twist = transformer.transform_twist(input_twist)

    assert abs(transformed_twist.twist.linear.x - expected_twist.twist.linear.x) < 1e-14
    assert abs(transformed_twist.twist.linear.y - expected_twist.twist.linear.y) < 1e-14
    assert abs(transformed_twist.twist.linear.z - expected_twist.twist.linear.z) < 1e-14
    assert abs(transformed_twist.twist.angular.x - expected_twist.twist.angular.x) < 1e-14
    assert abs(transformed_twist.twist.angular.y - expected_twist.twist.angular.y) < 1e-14
    assert abs(transformed_twist.twist.angular.z - expected_twist.twist.angular.z) < 1e-14
    for val, exp_val in zip(transformed_twist.covariance, expected_twist.covariance):
        assert abs(val - exp_val) < 1e-13


def test_transform_pose() -> None:
    """Test transforming pose."""
    # NOTE: Some values in the test are in no way realistic, they are just chosen for "easy" manual checking

    # sensor is at the point (1, 0, 0) and "looking" straight left
    input_to_base_tf = Transform()
    input_to_base_tf.translation.x = 1.0
    input_to_base_tf.translation.y = 0.0
    input_to_base_tf.translation.z = 0.0
    input_to_base_tf.rotation.x = 0.0
    input_to_base_tf.rotation.y = 0.0
    input_to_base_tf.rotation.z = 1 / sqrt(2)
    input_to_base_tf.rotation.w = 1 / sqrt(2)
    transformer = Transformer(input_to_base_tf)

    sensor_pose = PoseWithCovariance()
    sensor_pose.pose.position.x = 1.0
    sensor_pose.pose.position.y = 2.0
    sensor_pose.pose.position.z = 3.0
    sensor_pose.pose.orientation.x = 0.0
    sensor_pose.pose.orientation.y = 0.0
    sensor_pose.pose.orientation.z = 1 / sqrt(2)
    sensor_pose.pose.orientation.w = 1 / sqrt(2)

    sensor_pose.covariance = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
        31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
    ]

    expected_pose = PoseWithCovariance()
    expected_pose.pose.position.x = -1.0
    expected_pose.pose.position.y = 0.0
    expected_pose.pose.position.z = 3.0
    expected_pose.pose.orientation.x = 0.0
    expected_pose.pose.orientation.y = 0.0
    expected_pose.pose.orientation.z = 1 / sqrt(2)
    expected_pose.pose.orientation.w = 1 / sqrt(2)

    expected_pose.covariance = [
        8.0, -7.0, -9.0, 11.0, -10.0, -12.0,
        -2.0, 1.0, 3.0, -5.0, 4.0, 6.0,
        -14.0, 13.0, 15.0, -17.0, 16.0, 18.0,
        26.0, -25.0, -27.0, 29.0, -28.0, -30.0,
        -20.0, 19.0, 21.0, -23.0, 22.0, 24.0,
        -32.0, 31.0, 33.0, -35.0, 34.0, 36.0,
    ]

    transformed_pose = transformer.transform_pose(sensor_pose)

    # Allow for small float error
    assert abs(transformed_pose.pose.position.x - expected_pose.pose.position.x) < 1e-14
    assert abs(transformed_pose.pose.position.y - expected_pose.pose.position.y) < 1e-14
    assert abs(transformed_pose.pose.position.z - expected_pose.pose.position.z) < 1e-14
    assert abs(transformed_pose.pose.orientation.x - expected_pose.pose.orientation.x) < 1e-14
    assert abs(transformed_pose.pose.orientation.y - expected_pose.pose.orientation.y) < 1e-14
    assert abs(transformed_pose.pose.orientation.z - expected_pose.pose.orientation.z) < 1e-14
    assert abs(transformed_pose.pose.orientation.w - expected_pose.pose.orientation.w) < 1e-14
    for val, exp_val in zip(transformed_pose.covariance, expected_pose.covariance):
        assert abs(val - exp_val) < 1e-13
