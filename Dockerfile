#Set base image to Ubuntu
FROM ubuntu 18.04

#File / Author Maintainer
MAINTAINER Marcos Spalenza

RUN apt-get -y install \
	build-essential \
	python3-dev \
	python3-setuptools \
	python3-pip

RUN pip3 install -U numpy scipy 

RUN apt-get -y install \
	libatlas3-base \
	libatlas-dev
	
RUN pip3 install -U matplotlib

RUN pip3 install -U sklearn

RUN pip3 install -U skopt

#Add clustering script
ADD . /app/

#Add clustering script
ADD ./data/UCI/ /app/data/

#set path to /app
WORKDIR /app

#run script script
CMD "python3" "main_clustering.py" "-m" "uci" "data"
