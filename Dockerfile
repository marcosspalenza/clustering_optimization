# Set base image to Ubuntu
FROM ubuntu:20.04

# File / Author Maintainer
MAINTAINER Marcos Spalenza

RUN apt-get update -y && \ 
	apt-get install -y python3 python3-pip python3-dev

RUN pip3 install -U numpy scipy 

RUN pip3 install -U sklearn

RUN pip3 install -U scikit-optimize

# Add clustering script
ADD ./clstr/ /app/

# Set path to /app/
WORKDIR /app/

RUN mkdir /data/

RUN mkdir /data/input/

RUN mkdir /data/output/