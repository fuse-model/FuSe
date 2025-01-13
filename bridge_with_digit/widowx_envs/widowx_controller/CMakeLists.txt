cmake_minimum_required(VERSION 2.8.3)
project(widowx_controller)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

catkin_python_setup()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

## Generate services in the 'srv' folder
 add_service_files(
   FILES
   GotoNeutral.srv
   OpenGripper.srv
   MoveToEEP.srv
   MoveToState.srv
   GetGripperDesiredState.srv
   GetCartesianPose.srv
   GetState.srv
   GetVRButtons.srv
   EnableController.srv
   DisableController.srv
   SetGripperPosition.srv
 )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
 )

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS # add executable python files here
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
