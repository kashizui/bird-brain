FROM tensorflow/tensorflow:1.1.0
LABEL maintainer "sckoo@cs.stanford.edu"

ADD requirements.txt requirements.txt

RUN pip install -r requirements.txt
