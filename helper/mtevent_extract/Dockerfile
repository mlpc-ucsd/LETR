FROM ubuntu:20.04

RUN apt update
RUN apt install -y lsb-release curl gnupg
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc |  apt-key add -
RUN apt update
RUN apt install -y python3-numpy python3-rosbag 
