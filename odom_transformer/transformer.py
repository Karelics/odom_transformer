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


import numpy as np
from numpy.typing import NDArray

from geometry_msgs.msg import PoseWithCovariance, Transform, TwistWithCovariance
from odom_transformer.transform_np import PoseNp, TransformNp


class Transformer:
    """Converts odometry describing the movement of a sensor to describe the movement of the base frame."""

    __slots__ = (
        "_source_to_base_tf",
        "_twist_tf_matrix",
        "_base_pose_in_source_frame",
    )

    def __init__(self, source_to_base_tf: Transform) -> None:
        """Converts Odometry messages from a given topic and in given frame to describe the movement of the base_link
        frame.

        :param source_to_base_tf: Transform from sensor frame to base frame.
        """

        self._source_to_base_tf = TransformNp.from_msg(source_to_base_tf)
        self._twist_tf_matrix = self._get_twist_tf_matrix(self._source_to_base_tf)
        base_to_sensor_tf = self._source_to_base_tf.get_inverse()
        self._base_pose_in_source_frame = PoseNp(base_to_sensor_tf.rotation, base_to_sensor_tf.translation)

    @staticmethod
    def _get_twist_tf_matrix(source_to_base_tf: TransformNp) -> NDArray[np.float64]:
        # Gives the arm lengths with which each angular velocity component of sensor twist affects to
        # the linear velocity components of the base frame. For example, if the sensor is located at
        # the point (x, 0, 0) in base coordinates, then a rotation with angular speed w rad/s of the sensor
        # around the base frame z-axis (angular.z of sensor in base_frame) in the positive direction
        # causes the base frame to move in the y-direction with a velocity of -x*w m/s.
        source_pos = source_to_base_tf.translation
        sensor_arm_matrix = np.array(
            [
                [0, -source_pos[2], source_pos[1]],
                [source_pos[2], 0, -source_pos[0]],
                [-source_pos[1], source_pos[0], 0],
            ]
        )

        # Matrix for computing the effect of angular twist of the sensor to the linear twist of the base_frame.
        angular_to_linear_twist_tf_matrix = sensor_arm_matrix @ source_to_base_tf.rotation

        return np.block(
            [
                [source_to_base_tf.rotation, angular_to_linear_twist_tf_matrix],
                [np.zeros((3, 3)), source_to_base_tf.rotation],
            ]
        )

    def transform_pose(self, pose_w_cov: PoseWithCovariance) -> PoseWithCovariance:
        """Apply stored transform to a PoseWithCovariance."""

        source_pose = PoseNp.from_msg(pose_w_cov.pose)
        source_to_odom_tf = TransformNp(source_pose.orientation, source_pose.position)

        # Fix source offset in position (it is assumed that the pose in source odometry starts at the origin).
        source_to_odom_tf.combine_post(self._source_to_base_tf)

        pose_out = PoseWithCovariance()
        pose_out.pose = source_to_odom_tf.apply_to_pose(self._base_pose_in_source_frame).to_msg()
        pose_out.covariance = self._source_to_base_tf.apply_to_covariance(pose_w_cov.covariance)
        return pose_out

    def transform_twist(self, twist_w_cov: TwistWithCovariance) -> TwistWithCovariance:
        """Apply stored transform to a TwistWithCovariance."""
        tw_lin = twist_w_cov.twist.linear
        tw_ang = twist_w_cov.twist.angular
        tw_vector = np.array([tw_lin.x, tw_lin.y, tw_lin.z, tw_ang.x, tw_ang.y, tw_ang.z])
        cov_matrix = np.reshape(twist_w_cov.covariance, (6, 6))

        twist = TwistWithCovariance()
        twist_array = self._twist_tf_matrix @ tw_vector
        twist.twist.linear.x = twist_array[0]
        twist.twist.linear.y = twist_array[1]
        twist.twist.linear.z = twist_array[2]
        twist.twist.angular.x = twist_array[3]
        twist.twist.angular.y = twist_array[4]
        twist.twist.angular.z = twist_array[5]
        twist.covariance = list((self._twist_tf_matrix @ cov_matrix @ self._twist_tf_matrix.T).flatten())
        return twist
