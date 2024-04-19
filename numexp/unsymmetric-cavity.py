from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/janosch/projects//sign_changing_coeff/numexp")
from meshes import MakeStructuredCavityMesh,CreateUnstructuredMesh,MakeUnsymmetricStructuredCavityMesh,CreateUnstructuredMeshSubdom
from SolveProblem import SolveStandardFEM, SolveHybridStabilized, SolveHybridStabilizedModified

# Problem parameters for the cavity example
problem = { }
#sigma = [1 ,-1]
sigma = [1 ,-1]
#sigma = [1 ,-1.1]
contrast = sigma[1]/sigma[0] 
problem["sigma"] = sigma
alpha = contrast + 3
beta =  contrast +6
solution = [ (alpha*(x+1)*(x+1) - beta*(x+1))*sin(pi*y),  (x-3)*sin(pi*y) ]
problem["solution"] = solution
problem["solution-grad"] = [ CoefficientFunction((solution[i].Diff(x),solution[i].Diff(y))) for i in range(2) ]
problem["rhs"] = [-sigma[i] * (solution[i].Diff(x).Diff(x) + solution[i].Diff(y).Diff(y))  for i in range(2)]

stabs_order = [  {"CIP": 1e-5,"GLS": 1e-5,"Nitsche": 20,"Dual": 1e-3,"IF": 200},                      
               {"CIP": 5e-3,"GLS": 5e-3,"Nitsche": 20,"Dual": 1e-3,"IF": 200}, 
                {"CIP": 5e-5,"GLS": 5e-5,"Nitsche": 20,"Dual": 1e-4,"IF": 50}
              ]
orders = {"primal-bulk": 1,
          "primal-IF": 1,
          "dual-bulk":1,
          "dual-IF":1}

all_maxh = np.linspace(0.4,0.1,3,endpoint=False).tolist() + np.linspace(0.1,0.01,10,endpoint=False).tolist() + np.linspace(0.01,0.006,15,endpoint=True).tolist()

for order,stabs in zip([2,3],stabs_order[1:]):
    if order == 1:
        maxhs = all_maxh[1:]
    elif order == 2:
        maxhs = np.linspace(0.4,0.1,6,endpoint=False).tolist() + np.linspace(0.1,0.0125,24,endpoint=True).tolist() 
    else:
        maxhs = all_maxh[:-30]

    orders = {"primal-bulk": order,
              "primal-IF": order,
              "dual-bulk": order,
              "dual-IF": order }

    h1nat = [ ]
    hybridstab = [ ]
    hybridstab_outer = [ ]
    hybrid_IF = [] 

    for maxh in maxhs:
        print("maxh = ", maxh)
        mesh = CreateUnstructuredMesh(maxh=maxh,b=3)
        if maxh == maxhs[len(maxhs)-5]:
            result = SolveHybridStabilizedModified(mesh,orders,stabs,problem,True)
        else:
            result = SolveHybridStabilizedModified(mesh,orders,stabs,problem)

        hybridstab.append(result[0])
        hybrid_IF.append(result[1])

    plt.loglog(maxhs,hybridstab,label="stabilized-$H^1$-bulk",marker='+')
    plt.loglog(maxhs, hybrid_IF,label="stabillized-H^1/2-IF",marker='+')
    maxhnp = np.array(maxhs)
    if order == 1:
        plt.loglog(maxhs,5*maxhnp,color="gray",label="$\mathcal{O}(h)$",marker='+')
        plt.loglog(maxhs,5*maxhnp**2 ,color="gray",label="$\mathcal{O}(h^2)$",marker='+')
    elif order == 2:
        plt.loglog(maxhs,5*maxhnp**2 ,color="gray",label="$\mathcal{O}(h^2)$",marker='+')
    else:
        plt.loglog(maxhs,5*maxhnp**3 ,color="gray",label="$\mathcal{O}(h^3)$",marker='+')
    plt.title("$\sigma_- = {0}, \sigma_+ = {1}, k = {2}$".format( sigma[1],sigma[0],order))
    plt.xlabel("h")
    plt.legend()
    plt.show()
    plt.clf()


    name_str = "Cavity-nonsymmetric-k{0}-unstructured-critical.dat".format(order)
    if True:
        results = [maxhnp,  np.array(hybridstab,dtype=float), np.array(hybrid_IF,dtype=float) ]
        header_str = "h H1 IF"

        np.savetxt(fname ="../data/{0}".format(name_str),
                               X = np.transpose(results),
                               header = header_str,
                               comments = '')
    
    if order == 2:
        idx = len(maxhs)-7
        truncated_h = maxhs[idx:]
        
        xax = 1/np.abs(np.log( truncated_h ))
        serr = hybridstab[idx:] 
        sIF = hybrid_IF[idx:]
        
        print("truncated_h = ", truncated_h )
        print("serr = " , serr)
        xax = 1/np.abs(np.log( truncated_h ))**2
        plt.plot(xax , serr  ,label="stabilized-$H^1$-bulk" ,marker='+')
        #plt.plot(1/np.abs(np.log( truncated_h )), hybridstab[idx]*1/np.abs(np.log( truncated_h ))**0.25  ,label="ref" ,marker='o')
        
        plt.plot(xax, xax * (serr[0] /xax[0])  ,label="ref" ,marker='o')
        plt.xlabel("h")
        plt.legend()
        plt.show()

    if True:
        results = [ np.array(xax,dtype=float),  np.array(serr,dtype=float) , np.array(sIF,dtype=float), np.array( xax * (serr[0] /xax[0]),dtype=float)]
        header_str = "fh H1 IF ref"

        name_str = "Cavity-nonsymmetric-k{0}-unstructured-critical-log.dat".format(order)
        np.savetxt(fname ="../data/{0}".format(name_str),
                               X = np.transpose(results),
                               header = header_str,
                               comments = '')

