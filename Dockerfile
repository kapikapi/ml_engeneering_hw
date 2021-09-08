FROM ubuntu:latest
ARG USER_ID
ARG GROUP_ID

RUN apt-get update && apt-get -y update \
    && apt-get install -y build-essential python3.8 python3-pip python3-dev \
    && pip3 -q install pip --upgrade \
    && mkdir src
WORKDIR src/
COPY . .

RUN pip3 install -r requirements.txt \
    && pip3 install jupyter \
    && addgroup --gid $GROUP_ID new_docker_user \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID new_docker_user \
    && mkdir static \
    && chown -R new_docker_user static

USER new_docker_user


CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.password='sha1:fef991ec5136:f5f6065b8166643d349209f3c27451653c54fcee'"]