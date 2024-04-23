FROM ubuntu:latest

WORKDIR /home/app

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
        
RUN apt-get update
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository universe

ARG DEBIAN_FRONTEND=noninteractive 

RUN apt-get -y install cmake g++ git python3 python3-distutils python3-tk libxi-dev libxi6 libpython3-dev libxmu-dev tk-dev tcl-dev cmake git g++ libglu1-mesa-dev liblapacke-dev libocct-data-exchange-dev libocct-draw-dev occt-misc libtbb-dev libxi-dev libmkl-dev mpi-default-dev python3-mpi4py python3-setuptools pip vim emacs

RUN pip3 install numpy scipy matplotlib

ARG BASEDIR=/home/app/ngsuite
RUN mkdir -p $BASEDIR
WORKDIR $BASEDIR
RUN git clone --recurse-submodules https://github.com/NGSolve/ngsolve.git ngsolve-src

RUN mkdir $BASEDIR/ngsolve-build
RUN mkdir $BASEDIR/ngsolve-install

WORKDIR $BASEDIR/ngsolve-build
RUN cmake -DCMAKE_INSTALL_PREFIX=${BASEDIR}/ngsolve-install -DUSE_MPI=ON -DUSE_MKL=ON -DMKL_INCLUDE_DIR=/usr/include/mkl/ ${BASEDIR}/ngsolve-src
RUN make -j9 install
ENV NETGENDIR="${BASEDIR}/ngsolve-install/bin"
ENV PATH=$NETGENDIR:$PATH
ENV PYTHONPATH=/home/app/ngsuite/ngsolve-install/lib/python3.10/dist-packages

WORKDIR /home/app        
        
RUN git clone https://gitlab.gwdg.de/learned_infinite_elements/ngs_refsol.git
WORKDIR /home/app/ngs_refsol
RUN python3 setup.py install --user

WORKDIR /home/app         
RUN git clone https://github.com/UCL/sign-changing-repro.git
WORKDIR /home/app/sign-changing-repro 

