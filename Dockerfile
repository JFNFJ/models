FROM python:3.6

RUN apt-get update -y
RUN apt-get install -y python3 python3-dev python3-pip vim git

COPY . /app
WORKDIR /app

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

CMD python3.6 app.py
