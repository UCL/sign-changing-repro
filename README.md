# sign-changing-repro
This repository contains software and instructions to reproduce the numerical experiments in the paper
> A hybridized Nitsche method for sign-changing elliptic PDEs
>
> * authors: Erik Burman(1), Alexandre Ern(2) and Janosch Preuss(1)
> * (1): University College London
> * (2): CERMICS and INRIA Paris


The instructions for running the image are geared towards users who have access to a Unix like environment with a `bash` shell.
Windows users may use Linux subsystems or tools like [Git BASH](https://gitforwindows.org/) or [MobaXterm](https://mobaxterm.mobatek.net/) to 
run these commands.

## Pulling the docker image from Docker Hub 
* Please install the `docker` platform for your distribution as described [here](https://docs.docker.com/get-docker/).
* After installation the `Docker daemon` has to be started. This can either be done on boot or manually. In most Linux 
distributions the command for the latter is either `sudo systemctl start docker` or `sudo service docker start`.
* Pull the docker image using the command `docker pull janosch2888/sign-changing-repro:v1`. 
* Run the image with `sudo docker run -it janosch2888/sign-changing-repro:v1`.
* Proceed further as described in [How to reproduce](#repro).

## Downloading the docker image from Zenodo
* For this option the first two steps are the same as above.
* The image can be downloaded [here]( ). 
* Assuming that `interface-uc-repro.tar` is the filename of the downloaded image, please load the image with `sudo docker load < sign-changing-repro.tar`.
* Run the image with `sudo docker run -it janosch2888/sign-changing-repro:v1`.
* Proceed further as described in [How to reproduce](#repro).

## Manual installation

We need to install `NGSolve` and a small extension called `ngs_refsol` manually. For reference: The code has been developed 
using commit `819b0d3da731bb078204fa54293be0d9feb45842` of the former and commit `f3c5d52cae6a8f24a488d94337178956ace07abc` of the latter. 
Installation instructions for `NGSolve` using package managers are available [here](https://ngsolve.org/downloads) and instructions 
to build from source are [here](https://docu.ngsolve.org/latest/install/install_sources.html). Once `NGSolve` has been installed we can 
install `ngs_refsol` as follows: 

    git clone https://gitlab.gwdg.de/learned_infinite_elements/ngs_refsol.git 
    cd ngs_refsol
    python3 setup.py install --user

For compiling the figures you will also need a recent `latex` distribution installed on your machine.
Now we are ready to clone the repository using 

    git clone https://github.com/UCL/sign-changing-repro.git 

and proceed as described in [How to reproduce](#repro).
 

# <a name="repro"></a> How to reproduce

## <a name="Fig2"></a> Figure 2
Change to directory `scripts`. Run

    python3 symmetric_cavity-easy.py 1 
    python3 symmetric_cavity-easy.py 2 
    python3 symmetric_cavity-easy.py 3 

Afterwards, new data files of the form `Cavity-k__i__-unstructured-easy.dat` will be available in the folder `data`. Here, __i__ in [1,2,3] describes the finite element order k as 
defined in the paper. The data in the files is structured in the follwing columns: 

* h: Proportional to the width of the mesh. 
* h1nat: contains the H^1-error for the Galerkin stabilization not shown in the paper. 
* hybridstab: contains the H^1-error for the new method proposed in this paper.

This will gerate the data for the left plot. Now to produce the data for the right plot we run 

    python3 symmetric_cavity-high-contrast.py 1
    python3 symmetric_cavity-high-contrast.py 2
    python3 symmetric_cavity-high-contrast.py 3
  
Afterwards, the data will be available in the file `Cavity-k__i__-unstructured-high-contrast.dat` which have the same structure as above.
Then, to generate Figure 2. switch to the folder `plots` and run 
 
    lualatex -pdf Cavity-easy.tex 

## <a name="Fig2"></a> Figure 3
Change to directory `scripts`. Run

    python3 symmetric_cavity.py 1
    python3 symmetric_cavity.py 2
    python3 symmetric_cavity.py 3
  
Afterwards, the data for the unstructed meshes will be available in the file `Cavity-k__i__-unstructured.dat` (to be found in the `data` folder) and 
the data on the symmetric meshes in the file `Cavity-k1-symmetric.dat`. As above, __i__ in [1,2,3] denotes the polynomial degree.
 
Then, to generate Figure 3. switch to the folder `plots` and run 
 
    lualatex -pdf Cavity-near-critical-contrast.tex


## <a name="Fig2"></a> Figure 4 and 5
Change to directory `scripts`. Run

    python3 SolveMetaMaterial.py

This will generate all the data. 

* The `vtk` data for the plot without(!) claok (Figure 4 (A)) is available in `NoCloak-order3.vtu`.
* The `vtk` data for the plot with claok (Figure 4 (B)) is available in `MetaMaterial-order3.vtu`.
* The data for the convergence plots in Figure 5 is available in the files MetaMaterial-k__i__.dat` where __i__ in [2,3,4] denotes the 
polynomial order of the FEM. These data files contain the following columns 
  * `h` is the mesh width. 
  * `Galerkin-inner` is the H1-error for the Galerkin method in subdomain \Omega_i.  
  * `Galerkin-outer` is the H1-error for the Galerkin method in subdomain \Omega_e.
  * `Hybridstab-inner` is the H1-error for the stabilized method in subdomain \Omega_i. 
  * `Hybridstab-outer` is the H1-error for the stabilized method in subdomain \Omega_e. 

To generate the Figure 5, switch to the folder `plots` and run 

    lualatex -pdf MetaMaterial-conv.tex


## <a name="Fig2"></a> Figure 6
Change to directory `scripts`. Run

    python3 unsymmetric-cavity.py 

Two data files will be created:
* The file `Cavity-nonsymmetric-k2-unstructured-critical.dat` contains in the column `H1` the H1-error and in the column `IF` the error in the H^1/2-norm on the interface.
* The file `Cavity-nonsymmetric-k2-unstructured-critical-log.dat` contains these columns as well, but the column `fh` gives additionally the logarithmic scaling of h (the x-axis in the right plot) and the column `ref` contains the data for the gray reference line.
The data for the `vtk` plot is available in the file `Cavity-nonsymmetric-k2-unstructured-critical.vtu`.

To generate Figure 6, switch to the folder `plots` and run 

    lualatex -pdf Unsymmetric-cavity-critical-k2.tex




