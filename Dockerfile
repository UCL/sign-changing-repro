FROM ngsxfem/ngsolve:latest

WORKDIR /home/app        

USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

RUN git clone https://gitlab.gwdg.de/learned_infinite_elements/ngs_refsol.git
WORKDIR /home/app/ngs_refsol
RUN python3 setup.py install --user

WORKDIR /home/app         
RUN git clone https://github.com/UCL/sign-changing-repro.git
WORKDIR /home/app/sign-changing-repro



