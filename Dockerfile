FROM puckel/docker-airflow:latest

USER root
RUN apt-get update && apt-get -y update \
    && pip3 -q install pip --upgrade

COPY . .

RUN pip3 install -r requirements.txt
