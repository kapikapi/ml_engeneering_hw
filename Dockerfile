FROM ubuntu:latest
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.8 python3-pip python3-dev
RUN pip3 -q install pip --upgrade
RUN mkdir src
WORKDIR src/
COPY . .
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID new_docker_user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID new_docker_user

USER new_docker_user


CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]