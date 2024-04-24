from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
import sys
from meshes import MakeStructuredCavityMesh,CreateUnstructuredMesh,MakeUnsymmetricStructuredCavityMesh
from SolveProblem import SolveStandardFEM, SolveHybridStabilized

# Problem parameters for the cavity example
problem = { }
sigma = [1 ,-2]
problem["sigma"] = sigma
solution = [ ((x+1)*(x+1)-(1/(sigma[0]+sigma[1]))*(2*sigma[0]+sigma[1])*(x+1))*sin(pi*y),  (1/(sigma[0]+sigma[1]))*sigma[0]*(x-1)*sin(pi*y) ]
problem["solution"] = solution
problem["solution-grad"] = [ CoefficientFunction((solution[i].Diff(x),solution[i].Diff(y))) for i in range(2) ]
problem["rhs"] = [-sigma[i] * (solution[i].Diff(x).Diff(x) + solution[i].Diff(y).Diff(y))  for i in range(2)]

stabs_order = [  {"CIP": 1e-5,"GLS": 1e-5,"Nitsche": 20,"Dual": 1e-3,"IF": 200},                      
               {"CIP": 5e-5,"GLS": 5e-5,"Nitsche": 20,"Dual": 5e-1,"IF": 1},
               {"CIP": 5e-5,"GLS": 5e-5,"Nitsche": 20,"Dual": 1e-1,"IF": 50}
              ]

orders = {"primal-bulk": 1,
          "primal-IF": 1,
          "dual-bulk":1,
          "dual-IF":1}

all_maxh = np.linspace(0.4,0.1,3,endpoint=False).tolist() + np.linspace(0.1,0.01,30,endpoint=False).tolist() + np.linspace(0.01,0.003,25,endpoint=True).tolist()
all_nnstars = np.array( np.arange(1,20).tolist() + np.arange(20,100,5).tolist())

for order in [1,2,3]:

    stabs = stabs_order[order-1] 

    if order == 1:
        maxhs = all_maxh[1:-4]
    elif order == 2:
        maxhs = all_maxh[1:-26]
    else:
        maxhs = all_maxh[:-30]

    orders = {"primal-bulk": order,
              "primal-IF": order,
              "dual-bulk": 1,
              "dual-IF": order-1 }

    h1nat = [ ]
    hybridstab = [ ]

    #print(stabs)

    for maxh in maxhs:
        mesh = CreateUnstructuredMesh(maxh=maxh)
        h1nat.append(SolveStandardFEM(mesh,orders,problem))
        hybridstab.append(SolveHybridStabilized(mesh,orders,stabs,problem,False))
        
    plt.loglog(maxhs,h1nat,label="natural",marker='o')
    plt.loglog(maxhs,hybridstab,label="stabilized",marker='+')
    maxhnp = np.array(maxhs)

    if order == 1:
        plt.loglog(maxhs,5*maxhnp,color="gray",label="$\mathcal{O}(h)$",marker='+')
    elif order == 2:
        plt.loglog(maxhs,5*maxhnp**2 ,color="gray",label="$\mathcal{O}(h^2)$",marker='+')
    else:
        plt.loglog(maxhs,5*maxhnp**3 ,color="gray",label="$\mathcal{O}(h^3)$",marker='+')
    plt.title("H1-error")
    plt.xlabel("h")
    plt.legend()
    plt.show()

    name_str = "Cavity-k{0}-unstructured-easy.dat".format(order)
    results = [maxhnp, np.array(h1nat,dtype=float), np.array(hybridstab,dtype=float) ]
    header_str = "h h1nat hybridstab"
    np.savetxt(fname ="../data/{0}".format(name_str),
                               X = np.transpose(results),
                               header = header_str,
                               comments = '')
