FROM ngsxfem/ngsolve:latest

WORKDIR /home/app        

USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository universe
RUN apt-get install -y psmisc texlive-full 

RUN git clone https://gitlab.gwdg.de/learned_infinite_elements/ngs_refsol.git
WORKDIR /home/app/ngs_refsol
RUN python3 setup.py install --user

WORKDIR /home/app         
RUN git clone https://github.com/UCL/sign-changing-repro.git
WORKDIR /home/app/sign-changing-repro



