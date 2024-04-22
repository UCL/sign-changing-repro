FROM ngsxfem/ngsolve:latest

WORKDIR /home/app        
        
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

RUN sed -i -re 's/([a-z]{2}\.)?archive.ubuntu.com|security.ubuntu.com/old-releases.ubuntu.com/g' /etc/apt/sources.list
RUN apt-get update && apt-get dist-upgrade -y
RUN apt-get install -y  \
  texlive-full \
  wget

RUN pip3 install --user pypardiso

RUN git clone https://gitlab.gwdg.de/learned_infinite_elements/ngs_refsol.git
WORKDIR /home/app/ngs_refsol
RUN python3 setup.py install --user

WORKDIR /home/app         
RUN git clone https://github.com/UCL/sign-changing-repro.git
WORKDIR /home/app/sign-changing-repro
                
