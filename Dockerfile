FROM tensorflow/tensorflow:1.1.0-py3
LABEL maintainer "sckoo@cs.stanford.edu"
Label version 3

ADD requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y ffmpeg
