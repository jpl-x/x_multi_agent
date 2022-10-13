# parameters
ARG REPO_NAME="x-image"
ARG DESCRIPTION="Docker image for x library"
ARG MAINTAINER="Vincenzo Polizzi (polivicio@gmail.com)"

# ==================================================>
# ==> Do not change the code below this line
ARG ARCH=amd64
ARG DISTRO=x
ARG BASE_TAG=${DISTRO}-${ARCH}
ARG BASE_IMAGE="base-env:${ARCH}"
ARG PHOTOMETRIC_CALI=false
ARG MULTI_UAV=false
ARG REQUEST_COMM=false
ARG LAUNCHER=default

# define base image
# ARG DOCKER_REGISTRY=docker.io
FROM viciopoli/${BASE_IMAGE} as BASE

# recall all arguments
ARG ARCH
ARG DISTRO
ARG REPO_NAME
ARG DESCRIPTION
ARG MAINTAINER
ARG BASE_TAG
ARG BASE_IMAGE
ARG LAUNCHER
ARG PHOTOMETRIC_CALI
ARG MULTI_UAV
ARG REQUEST_COMM
ENV PHOTOMETRIC_CALI=${PHOTOMETRIC_CALI}
ENV MULTI_UAV=${MULTI_UAV}
ENV REQUEST_COMM=${REQUEST_COMM}

# define/create repository path
ARG REPO_PATH="${SOURCE_DIR}"
RUN mkdir -p "${REPO_PATH}"
WORKDIR "${REPO_PATH}"

ENV CATKIN_WS_DIR "${CATKIN_WS_DIR}"

RUN echo "PHOTOMETRIC_CALI: ${PHOTOMETRIC_CALI}" && echo "MULTI_UAV: ${MULTI_UAV}" && echo "REQUEST_COMM: ${REQUEST_COMM}"
# install
COPY . "${REPO_PATH}/"
RUN mkdir build && cd build && cmake -DPHOTOMETRIC_CALI=${PHOTOMETRIC_CALI} -DMULTI_UAV=${MULTI_UAV} -DREQUEST_COMM=${REQUEST_COMM} .. && make package && dpkg -i x_1.2.3_$(dpkg --print-architecture).deb
