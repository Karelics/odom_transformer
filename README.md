# odom_transformer
ROS 2 Node for performing odometry transformations.

This package contains a ROS 2 node implementation that
can perform transformations of odometry messages.
It was created as a workaround to an
[issue in isaac_ros_visual_slam package](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam/issues/21),
where the input base frame was not correctly accounted for,
causing the odometry to always be reported for the camera frame instead
of the robot base frame.

## Prerequisites

* This package is based on ROS 2 Galactic.
Modifications may be required to make it work on other versions of ROS 2.
* Dependencies other than base ROS 2 packages and builtin Python libraries:
  * Numpy
  * Scipy

## Usage instructions

The node can be used to transform odometries given for any frame
to describe the odometry of the robot base frame (or any other frame)
with the following restrictions:

* the transform between the source (sensor) frame and the base frame must be static
(it is not updated during operations);
* the transform from source frame to base frame must be available at node launch
(transforming from a particular topic will only start after the required transform is found);
* if multiple odometries are transformed with the same node, the base (target)
frame is the same for all of them
(you need to launch different instances of the node for different base frames).

### Launch parameters

The node offers following parameters for configuration at launch.

* `base_frame`: The name of the base frame whose movement the resulting odometries should describe.
* `odoms_to_transform`: List of names (keys) of the odometries to transform.
* For each `source_name` in `odoms_to_transform` list:
  * `source_name.topic_in`: Name of the topic for receiving odometry messages.
  * `source_name.source_frame`: Name of the frame whose movement the source odometry describes.
  * `source_name.topic_out`: Name of the topic to which the transformed odometry should be published.

An example parameter file is provided in the params directory.

And example launch file is provided in the launch directory.


## Updates and contributions

Since the package is meant as a workaround, it is not under constant development.
Hence, we do not guarantee any scheduled or unscheduled updates to it.
Regardless of this, feedback and ideas for improvements are welcome.

If you would like to contribute to improving this pacakge,
contact us in the form of an issue in this repository and detail your ideas for the contribution.


## License
This piece of software is released under the GPL 3.0 license.

Odom transformer - ROS 2 Node for performing odometry transformations.

Copyright (C) 2023  Karelics Oy

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
