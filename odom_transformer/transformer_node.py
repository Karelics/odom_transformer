#!/usr/bin/env python3

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

from functools import partial
from typing import Dict, List, Tuple

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.time import Duration, Time
from tf2_ros import Buffer, TransformException, TransformListener

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

from odom_transformer.transformer import Transformer


class TransformerNode(Node):
    """Node for transforming odometries from different sources to describe the movements of the robot's base frame."""

    def __init__(self) -> None:
        super().__init__("odom_transformer")

        self._base_frame = self.declare_parameter("base_frame", "base_link").value

        self._inputs: List[Tuple[str, str, str, str]] = []
        odoms_to_transform = self.declare_parameter("odoms_to_transform", Parameter.Type.STRING_ARRAY).value

        for source_name in odoms_to_transform:
            topic_in = self.declare_parameter(f"{source_name}.topic_in", Parameter.Type.STRING).value
            source_frame = self.declare_parameter(f"{source_name}.source_frame", Parameter.Type.STRING).value
            topic_out = self.declare_parameter(f"{source_name}.topic_out", Parameter.Type.STRING).value

            self._inputs.append((source_name, topic_in, source_frame, topic_out))

        self._transformers: Dict[str, Transformer] = {}
        self._pubs: List[Publisher] = []
        self._subs: List[Subscription] = []

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._get_tfs_timer = self.create_timer(1.0, self.tf_timer_cb)

    def tf_timer_cb(self) -> None:
        """Callback for looking up the required transforms and start operations."""
        if self._start_transformers():
            self.destroy_timer(self._get_tfs_timer)
            self._get_tfs_timer = None
            self._tf_listener.unregister()
            self._tf_listener = None
            self._tf_buffer = None

    def _start_transformers(self) -> bool:
        success = True
        for source_name, topic_in, source_frame, topic_out in self._inputs:
            try:
                if source_name not in self._transformers:
                    transformer = self._setup_transformer(topic_in, source_frame, topic_out)
                    self._transformers[source_name] = transformer
            except RuntimeError as e:
                self.get_logger().warning(
                    f"Couldn't set up odom transformer from frame {source_frame} ({str(e)}). Trying again."
                )
                success = False
        return success

    def _setup_transformer(self, topic_in: str, sensor_frame: str, topic_out: str) -> Transformer:
        transform = self._get_tf(sensor_frame, timeout=5)
        transformer = Transformer(transform.transform)

        pub = self.create_publisher(Odometry, topic_out, qos_profile=10)
        self._pubs.append(pub)
        self.create_subscription(
            Odometry,
            topic_in,
            partial(self.odom_cb, transformer, pub),
            qos_profile=10,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(f"Starting to transform odometry from {topic_in} to {topic_out}")
        return transformer

    def _get_tf(self, source_frame: str, timeout: int) -> TransformStamped:
        try:
            return self._tf_buffer.lookup_transform(
                self._base_frame, source_frame, Time(), timeout=Duration(seconds=timeout)
            )
        except TransformException as e:
            raise RuntimeError(f"Unable to find transform from {source_frame} to base_link: {str(e)}") from e

    def odom_cb(self, transformer: Transformer, pub: Publisher, data: Odometry) -> None:
        """Callback transforming an Odometry message and publishing the result."""
        odom = Odometry(header=data.header, child_frame_id=self._base_frame)
        try:
            odom.pose = transformer.transform_pose(data.pose)
            odom.twist = transformer.transform_twist(data.twist)
            pub.publish(odom)
        except ValueError as e:
            self.get_logger().error(f"Failed to transform odometry from frame {data.child_frame_id}: {str(e)}")

    def on_shutdown(self) -> None:
        """Cleanup on shutdown."""
        if self._get_tfs_timer:
            self._get_tfs_timer.cancel()
            self.destroy_timer(self._get_tfs_timer)
            del self._get_tfs_timer
        for sub in self._subs:
            self.destroy_subscription(sub)
        del self._subs
        del self._transformers
        del self._pubs


if __name__ == "__main__":
    rclpy.init()
    transformer_node = TransformerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(transformer_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        transformer_node.get_logger().info("Shutting down.")
        transformer_node.on_shutdown()
    finally:
        transformer_node.destroy_node()
        rclpy.shutdown()
