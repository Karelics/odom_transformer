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

"""Some simple unit tests to verify functionality of the helper classes."""

import numpy as np
import pytest

from geometry_msgs.msg import Pose, Transform
from odom_transformer.transform_np import PoseNp, TransformNp


@pytest.fixture(name="transform")
def default_transform() -> Transform:
    """Get default ROS transform."""
    transform = Transform()
    transform.translation.x = 1.0
    transform.translation.y = 2.0
    transform.translation.z = 3.0
    transform.rotation.w = 1 / np.sqrt(2)  # yaw = pi / 2
    transform.rotation.z = 1 / np.sqrt(2)
    return transform


def test_create_and_apply_to_pose(transform: Transform) -> None:
    """Test creating transform from message and applying it to a pose."""
    tf_np = TransformNp.from_msg(transform)
    pose = tf_np.apply_to_pose(PoseNp.from_msg(Pose())).to_msg()

    assert abs(pose.position.x - 1.0) < 1e-14
    assert abs(pose.position.y - 2.0) < 1e-14
    assert abs(pose.position.z - 3.0) < 1e-14
    assert abs(pose.orientation.w - 1 / np.sqrt(2)) < 1e-14
    assert abs(pose.orientation.z - 1 / np.sqrt(2)) < 1e-14
    assert abs(pose.orientation.x - 0.0) < 1e-14
    assert abs(pose.orientation.y - 0.0) < 1e-14


def test_create_and_apply_to_cov(transform: Transform) -> None:
    """Test creating transform from message and applying it to a covariance."""
    cov = [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 4.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 5.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 6.0,
    ]
    expected_cov = [
        2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 5.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 6.0,
    ]

    tf_np = TransformNp.from_msg(transform)
    cov = tf_np.apply_to_covariance(cov)

    assert len(cov) == len(expected_cov)
    for idx, (elem, exp_elem) in enumerate(zip(cov, expected_cov)):
        assert abs(elem - exp_elem) < 1e-14, str(idx)


def test_inverse_transform(transform: Transform) -> None:
    """Test computing the inverse transform."""
    pose = Pose()
    pose.position.x = 1.0
    pose.position.y = 2.0
    pose.position.z = 3.0
    pose.orientation.w = 1 / np.sqrt(2)  # yaw = pi / 2
    pose.orientation.z = 1 / np.sqrt(2)

    tf_np = TransformNp.from_msg(transform)
    tf_inv = tf_np.get_inverse()
    pose = tf_inv.apply_to_pose(PoseNp.from_msg(pose)).to_msg()

    assert abs(pose.position.x) < 1e-14
    assert abs(pose.position.y) < 1e-14
    assert abs(pose.position.z) < 1e-14
    assert abs(pose.orientation.w - 1.0) < 1e-14
    assert abs(pose.orientation.z) < 1e-14
    assert abs(pose.orientation.x) < 1e-14
    assert abs(pose.orientation.y) < 1e-14


def test_combine_transforms(transform: Transform) -> None:
    """Test combining two transforms."""
    tf_1 = Transform()
    tf_1.translation.x = 4.0
    tf_1.translation.y = 5.0
    tf_1.translation.z = 6.0
    tf_1.rotation.w = 1 / np.sqrt(2)  # roll = pi / 2
    tf_1.rotation.x = 1 / np.sqrt(2)

    tf_2 = TransformNp.from_msg(transform)
    tf_np = TransformNp.from_msg(tf_1)
    tf_np.combine_post(tf_2)
    pose = tf_np.apply_to_pose(PoseNp.from_msg(Pose())).to_msg()

    assert abs(pose.position.x + 4.0) < 1e-14
    assert abs(pose.position.y - 6.0) < 1e-14
    assert abs(pose.position.z - 9.0) < 1e-14
    assert abs(pose.orientation.w - 0.5) < 1e-14  # roll = pi / 2, yaw = pi / 2
    assert abs(pose.orientation.z - 0.5) < 1e-14
    assert abs(pose.orientation.x - 0.5) < 1e-14
    assert abs(pose.orientation.y - 0.5) < 1e-14
