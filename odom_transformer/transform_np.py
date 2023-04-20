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

"""Helper classes for handling transforms and poses directly as Numpy arrays instead of having to constantly convert
back and forth between Numpy and ROS messages."""


from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Pose, Transform


@dataclass
class PoseNp:
    """Stores a 3D pose as orientation matrix and position vector.

    Orientation as a 3x3 Numpy array. Position a 1x3 Numpy array.
    """

    __slots__ = ("orientation", "position")

    orientation: NDArray[np.float64]
    position: NDArray[np.float64]

    @classmethod
    def from_msg(cls, msg: Pose) -> PoseNp:
        """Get a PoseNp instance from a ROS Pose message."""
        ori_quat = msg.orientation
        orientation = Rotation.from_quat([ori_quat.x, ori_quat.y, ori_quat.z, ori_quat.w]).as_matrix()
        position = np.array([msg.position.x, msg.position.y, msg.position.z])
        return PoseNp(orientation, position)

    def to_msg(self) -> Pose:
        """Return the pose as a ROS Pose message."""
        ori_quat = Rotation.from_matrix(self.orientation).as_quat()
        pose = Pose()
        pose.position.x = self.position[0]
        pose.position.y = self.position[1]
        pose.position.z = self.position[2]
        pose.orientation.w = ori_quat[3]
        pose.orientation.x = ori_quat[0]
        pose.orientation.y = ori_quat[1]
        pose.orientation.z = ori_quat[2]
        return pose


class TransformNp:
    """Stores and handles transforms as in rotation matrix & translation vector form.

    Rotation as a 3x3 Numpy array. Translation as a 1x3 Numpy array.
    """

    __slots__ = ("rotation", "translation", "_cov_tf_matrix", "_cov_tf_ready")

    def __init__(self, rotation: NDArray[np.float64], translation: NDArray[np.float64]) -> None:
        self.rotation = rotation
        self.translation = translation
        self._cov_tf_matrix = np.array([])
        self._cov_tf_ready = False

    @classmethod
    def from_msg(cls, msg: Transform) -> TransformNp:
        """Get a TransformNp instance from ROS Transform message."""
        rot = Rotation.from_quat([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w]).as_matrix()
        transl = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
        return cls(rot, transl)

    def get_inverse(self) -> TransformNp:
        """Returns the inverse of this transform."""
        rotation = np.linalg.inv(self.rotation)
        translation = -rotation @ self.translation
        return TransformNp(rotation, translation)

    def combine_post(self, transform: TransformNp) -> None:
        """Combine another transform with this one, applying this one first.

        That is, if this transform is T and the given transform is S, then T = TS.
        """
        self.rotation = transform.rotation @ self.rotation
        self.translation = transform.rotation @ self.translation + transform.translation
        self._cov_tf_ready = False

    def apply_to_pose(self, pose: PoseNp) -> PoseNp:
        """Apply the transform to a pose."""
        orientation = self.rotation @ pose.orientation
        position = self.rotation @ pose.position + self.translation
        return PoseNp(orientation, position)

    def apply_to_covariance(self, covariance: List[float]) -> List[float]:
        """Apply the transform to a covariance matrix (given as a list)."""
        if not self._cov_tf_ready:
            self._update_cov_tf_matrix()
        cov_matrix = np.reshape(covariance, (6, 6))
        return list((self._cov_tf_matrix @ cov_matrix @ self._cov_tf_matrix.T).flatten())

    def _update_cov_tf_matrix(self) -> None:
        self._cov_tf_matrix = np.block(
            [
                [self.rotation, np.zeros((3, 3))],
                [np.zeros((3, 3)), self.rotation],
            ]
        )
        self._cov_tf_ready = True
