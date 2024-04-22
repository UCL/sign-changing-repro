from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/janosch/projects//sign_changing_coeff/numexp")
from meshes import MakeStructuredCavityMesh,CreateUnstructuredMesh,MakeUnsymmetricStructuredCavityMesh
from SolveProblem import SolveStandardFEM, SolveHybridStabilized

if ( len(sys.argv) > 1 and int(sys.argv[1]) in [1,2,3]  ):
    order = int(sys.argv[1])
else:
    raise ValueError('Invalid input!')

# Problem parameters for the cavity example
problem = { }
#sigma = [1 ,-1.001]
sigma = [1 ,-200]
problem["sigma"] = sigma
solution = [ ((x+1)*(x+1)-(1/(sigma[0]+sigma[1]))*(2*sigma[0]+sigma[1])*(x+1))*sin(pi*y),  (1/(sigma[0]+sigma[1]))*sigma[0]*(x-1)*sin(pi*y) ]
problem["solution"] = solution
problem["solution-grad"] = [ CoefficientFunction((solution[i].Diff(x),solution[i].Diff(y))) for i in range(2) ]
problem["rhs"] = [-sigma[i] * (solution[i].Diff(x).Diff(x) + solution[i].Diff(y).Diff(y))  for i in range(2)]

#"IF": 200, Dual: 1e-3
#{"CIP": 1e-5,"GLS": 1e-5,"Nitsche": 20,"Dual": 1,"IF": 1},
#{"CIP": 5e-5,"GLS": 5e-5,"Nitsche": 20,"Dual": 2e-1,"IF": 200}
#{"CIP": 5e-5,"GLS": 5e-5,"Nitsche": 20,"Dual": 1e-1,"IF": 50}
stabs_order = [  {"CIP": 1e-5,"GLS": 1e-5,"Nitsche": 20,"Dual": 1e-3,"IF": 200},                      
               {"CIP": 5e-5,"GLS": 5e-5,"Nitsche": 20,"Dual": 5e-1,"IF": 1},
               {"CIP": 5e-5,"GLS": 5e-5,"Nitsche": 20,"Dual": 1e-1,"IF": 50}
              ]

#{"CIP": 1e-6,"GLS": 1e-6,"Nitsche": 20,"Dual": 1e-4,"IF": 1e-3},{"CIP": 1e-5,"GLS": 1e-5,"Nitsche": 20,"Dual": 1e-4,"IF": 1},
#stabs_order_reduced_dual_stab = [{"CIP": 1e-5,"GLS": 1e-5,"Nitsche": 20,"Dual": 1e-3,"IF": 500},
#               {"CIP": 5e-5,"GLS": 5e-5,"Nitsche": 20,"Dual": 2e-1,"IF": 100},
#               {"CIP": 5e-5,"GLS": 5e-5,"Nitsche": 20,"Dual": 2e-1,"IF": 100}
#              ]

orders = {"primal-bulk": 1,
          "primal-IF": 1,
          "dual-bulk":1,
          "dual-IF":1}


#all_maxh = np.linspace(0.4,0.1,3,endpoint=False).tolist() + np.linspace(0.1,0.01,20,endpoint=False).tolist() + np.linspace(0.01,0.002,20,endpoint=True).tolist()
all_maxh = np.linspace(0.4,0.1,3,endpoint=False).tolist() + np.linspace(0.1,0.01,30,endpoint=False).tolist() + np.linspace(0.01,0.003,25,endpoint=True).tolist()
all_nnstars = np.array( np.arange(1,20).tolist() + np.arange(20,100,5).tolist())

stabs = stabs_order[order-1] 
solver = "umfpack"
if order == 1:
    solver ="pypardiso" 
    # we use pypardiso for order = 1 since umfpack was observed to  have some stability issues for very small h

if order == 1:
    maxhs = all_maxh[1:]
elif order == 2:
    maxhs = all_maxh[1:-23]
else:
    maxhs = all_maxh[:-30]

orders = {"primal-bulk": order,
          "primal-IF": order,
          "dual-bulk": order,
          "dual-IF": order }

h1nat = [ ]
hybridstab = [ ]
#hybridstab_reduced_dual_stab = [ ]

print(stabs)

for maxh in maxhs:
    mesh = CreateUnstructuredMesh(maxh=maxh)
    h1nat.append(SolveStandardFEM(mesh,orders,problem,solver))
    hybridstab.append(SolveHybridStabilized(mesh,orders,stabs,problem,False,solver))
    #if order == 1:
    #    hybridstab_reduced_dual_stab.append( SolveHybridStabilized(mesh,orders,stabs_order_reduced_dual_stab[0],problem))

plt.loglog(maxhs,h1nat,label="natural",marker='o')
plt.loglog(maxhs,hybridstab,label="stabilized",marker='+')
maxhnp = np.array(maxhs)
if order == 1:
    #plt.loglog(maxhs,hybridstab_reduced_dual_stab,label="reduced dual stab",marker='+')
    plt.loglog(maxhs,5*maxhnp,color="gray",label="$\mathcal{O}(h)$",marker='+')
elif order == 2:
    plt.loglog(maxhs,5*maxhnp**2 ,color="gray",label="$\mathcal{O}(h^2)$",marker='+')
else:
    plt.loglog(maxhs,5*maxhnp**3 ,color="gray",label="$\mathcal{O}(h^3)$",marker='+')
plt.title("H1-error")
plt.xlabel("h")
plt.legend()
plt.show()

name_str = "Cavity-k{0}-unstructured-high-contrast.dat".format(order)
#if order == 1:
#    results = [maxhnp, np.array(h1nat,dtype=float), np.array(hybridstab,dtype=float), np.array( hybridstab_reduced_dual_stab , dtype=float) ]
#    header_str = "h h1nat hybridstab reduceddualstab"
#else: 
if True:
    results = [maxhnp, np.array(h1nat,dtype=float), np.array(hybridstab,dtype=float) ]
    header_str = "h h1nat hybridstab"


np.savetxt(fname ="../data/{0}".format(name_str),
                           X = np.transpose(results),
                           header = header_str,
                           comments = '')


