cmake_minimum_required(VERSION 3.5)
project(odom_transformer)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(nav_msgs REQUIRED)

install(DIRECTORY
  launch
  params
  DESTINATION share/${PROJECT_NAME}/
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  odom_transformer/transformer_node.py
  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
    # Automatically add all unit tests (Python files starting with "test_") in the test directory.
    find_package(ament_cmake_pytest REQUIRED)
    file(GLOB_RECURSE TESTS_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/test" test_*.py)
    foreach(test_name ${TESTS_SRCS})
        get_filename_component(name_only ${test_name} NAME_WE)
        get_filename_component(dir_only ${test_name} DIRECTORY)
        string(REPLACE "/" "_" sanatized_dir_name "${dir_only}")
        message(STATUS Adding: ${sanatized_dir_name}_${name_only} )
        ament_add_pytest_test(${PROJECT_NAME}_${sanatized_dir_name}_${name_only} "test/${test_name}")
    endforeach(test_name)
endif()

ament_package()
