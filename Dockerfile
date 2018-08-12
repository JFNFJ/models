FROM "ubuntu"
RUN apt-get update && yes | apt-get upgrade
RUN mkdir -p /models
RUN apt-get install -y git python-pip
RUN pip install -r requirements.txt
WORKDIR /data
