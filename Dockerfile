FROM python:3.6-stretch

RUN apt-get update -y
RUN apt-get install -y openmpi-bin libopenmpi-dev
RUN apt-get install -y ffmpeg

COPY demo demo
COPY sdepy sdepy
COPY setup.py setup.py

RUN python3 setup.py install

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

WORKDIR demo
ENV PYTHONPATH "${PYHONPATH}:/demo"
