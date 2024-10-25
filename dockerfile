#getting base image
From ubuntu
#Adding the source code
ADD . /home
#building enviroment
RUN apt-get update
RUN apt-get dist-upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get -y dist-upgrade
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install software-properties-common
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3.8
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip
RUN pip3 install -r /home/requirements.txt
#running the python program
CMD ["python3","home/src/main.py"]

