from ngsolve import *
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/janosch/projects//sign_changing_coeff/numexp")
from meshes import MakeStructuredCavityMesh,CreateUnstructuredMesh
#solver = "pardiso"
solver = "umfpack"
from ngsolve.la import EigenValues_Preconditioner
from decimal import Decimal

def SolveStandardFEM(mesh,orders,problem):

    sigma = problem["sigma"]
    solution = problem["solution"]
    sol_gradient = problem["solution-grad"]  
    coef_f = problem["rhs"] 
    
    dX = tuple( [dx(definedon=mesh.Materials("plus")), dx(definedon=mesh.Materials("minus"))] ) 
    
    V = H1(mesh, order=orders["primal-bulk"], dirichlet="outer") 
    u,v = V.TnT()
    a = BilinearForm(V, symmetric=False)
    sigma_coeff = CoefficientFunction( [sigma[0] if mat=="plus" else sigma[1] for mat in mesh.GetMaterials() ] )
    a  += sigma_coeff*grad(u)*grad(v)*dx
    
    rhs_coeff = CoefficientFunction( [coef_f[0] if mat=="plus" else coef_f[1] for mat in mesh.GetMaterials() ] )
    f = LinearForm(V)
    f += rhs_coeff*v*dx

    with TaskManager():
        a.Assemble()
        f.Assemble()

    gfu = GridFunction(V)
    f.vec.data -= a.mat * gfu.vec
    print("Solving linear system")
    gfu.vec.data += a.mat.Inverse(V.FreeDofs() ,inverse=solver  ) * f.vec

    err = sum( [ ( (gfu   - solution[i])**2 +  InnerProduct(grad(gfu) - sol_gradient[i], grad(gfu) - sol_gradient[i] )) *  dX[i]  for i in [0,1] ]  ) 
    h1err = sqrt( Integrate(err, mesh) ) 
    h1norm = sqrt( Integrate( sum( [ ( (solution[i])**2 +  InnerProduct(sol_gradient[i],sol_gradient[i] )) *  dX[i]  for i in [0,1] ]  ) , mesh)   )

    print("Standard FEM")
    print(" H1-error  = ", h1err  )
    print(" Relative H1-error  = ", h1err/h1norm  )
    print("")
    #del psolver
    return h1err/h1norm 


def SolveHybridStabilized(mesh,orders,stabs,problem,plot=False):
    
    sigma = problem["sigma"]
    solution = problem["solution"]
    sol_gradient = problem["solution-grad"]  
    coef_f = problem["rhs"]

    # indicator function for facets on interface
    VG0 = FacetFESpace( mesh, order=0)
    facets_G_indicator = GridFunction(VG0)
    facets_G_indicator.vec[ VG0.GetDofs(mesh.Boundaries("IF")) ]  = 1.0
    
    # Primal and dual FESpaces
    Q = FacetFESpace( mesh, order=orders["primal-IF"], dirichlet="outer") 
    Q_dual = FacetFESpace( mesh, order=orders["dual-IF"], dirichlet="outer") 
    Vp_primal =  Compress(H1(mesh, order=orders["primal-bulk"], dirichlet="outer", definedon=mesh.Materials("plus") ,  dgjumps=True))
    Vm_primal = Compress(H1(mesh, order=orders["primal-bulk"], dirichlet="outer", definedon=mesh.Materials("minus") , dgjumps=True))
    VGamma_primal = Compress(Q, active_dofs = Q.GetDofs(mesh.Boundaries("IF"))  )
    Vp_dual =  Compress(H1(mesh, order=orders["dual-bulk"], dirichlet="outer", definedon=mesh.Materials("plus") ,  dgjumps=True))
    Vm_dual = Compress(H1(mesh, order=orders["dual-bulk"], dirichlet="outer", definedon=mesh.Materials("minus") , dgjumps=True))
    VGamma_dual = Compress(Q_dual, active_dofs = Q_dual.GetDofs(mesh.Boundaries("IF"))  )
    Vh = Vp_primal *  Vm_primal *  VGamma_primal *  Vp_dual *  Vm_dual * VGamma_dual 

    up,um,uG,zp,zm,zG = Vh.TrialFunction()
    vp,vm,vG,wp,wm,wG  = Vh.TestFunction()
    u = [up,um]
    v = [vp,vm]
    z = [zp,zm]
    w = [wp,wm]
    gradu, gradz, gradv, gradw = [[grad(fun[i]) for i in [0, 1]] for fun in [u, z, v, w]]
    jumpu, jumpz, jumpv, jumpw = [ [ fun[i] - fhat  for i in [0, 1]] for fun,fhat in zip([u, z, v, w],[uG,zG,vG,wG]) ]

    nF = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size
    order = orders["primal-bulk"]

    dX = tuple( [dx(definedon=mesh.Materials("plus")), dx(definedon=mesh.Materials("minus"))] ) 
    dF = tuple( [dx(skeleton=True,  definedon=mesh.Materials("plus") ), dx(skeleton=True, definedon=mesh.Materials("minus") ) ] ) 
    ddT = tuple( [dx( element_boundary=True, definedon=mesh.Materials("plus")) , dx( element_boundary=True, definedon=mesh.Materials("minus") )] )

    def calL(fun):
        hesse = [fun[i].Operator("hesse") for i in [0,1]]
        if  mesh.dim == 2:
            return (-sigma[0]*hesse[0][0,0]-sigma[0]*hesse[0][1,1], -sigma[1]*hesse[1][0,0]-sigma[1]*hesse[1][1,1]  )

    aX = BilinearForm(Vh, symmetric=False)

    # a(u_pm,uG;w_pm,wG)
    aX  += sum(  sigma[i] * gradu[i] * gradw[i] * dX[i] for i in [0, 1])
    for i in [0,1]:
        aX += facets_G_indicator * ( (-1)*sigma[i]*gradu[i]*nF*jumpw[i] - sigma[i]*gradw[i]*nF*jumpu[i] 
               + stabs["Nitsche"] * np.sign(sigma[i]) * sigma[i] *order*order/h*jumpw[i]*jumpu[i]) * ddT[i]   

    # a(v_pm,vG;z_pm,zG)
    aX  += sum(  sigma[i] * gradv[i] * gradz[i] * dX[i] for i in [0, 1])
    for i in [0,1]:
        aX += facets_G_indicator * ( (-1)*sigma[i]*gradv[i]*nF*jumpz[i] - sigma[i]*gradz[i]*nF*jumpv[i] 
               + stabs["Nitsche"] * np.sign(sigma[i]) * sigma[i]*order*order/h*jumpz[i]*jumpv[i]) * ddT[i]   

    # primal stabilization s(u_pm,v_pm)
    aX += sum( [ stabs["CIP"] * h * abs(sigma[i]) * InnerProduct( (gradu[i] - gradu[i].Other()) * nF , (gradv[i] - gradv[i].Other()) * nF ) * dF[i] for i in [0,1] ]  )
    aX += sum( [ stabs["GLS"] * h**2 * calL(u)[i] * calL(v)[i] * dX[i] for i in [0, 1] ] )
    
    for i in [0,1]:
        #aX += facets_G_indicator * 2*10*order*order/h*jumpv[i]*jumpu[i] * ddT[i]   
        aX += facets_G_indicator * stabs["IF"]/h*jumpv[i]*jumpu[i] * ddT[i]   

    # dual stabilization 
    for i in [0,1]:
        aX += stabs["Dual"] * (-1)*gradz[i]*gradw[i]*dX[i] 

    # right hand side 
    fX = LinearForm(Vh)
    fX += sum( coef_f[i] * w[i] * dX[i] for i in [0, 1])
    fX += sum( [ stabs["GLS"] * h**2 * coef_f[i] * calL(v)[i] * dX[i] for i in [0, 1] ] )

    # Setting up matrix and vector
    print("assembling linear system")
    with TaskManager():
        aX.Assemble()
        fX.Assemble()    
    
    gfuX = GridFunction(Vh)
    gfuXh = gfuX.components
    fX.vec.data -= aX.mat * gfuX.vec
    print("Solving linear system")
    gfuX.vec.data += aX.mat.Inverse(Vh.FreeDofs() ,inverse=solver  )* fX.vec

    if plot:
        Draw(IfPos(-x,gfuXh[0],gfuXh[1]),mesh,"uh")
        Draw(IfPos(-x,solution[0],solution[1]),mesh,"u")
        Draw(IfPos(-x,(solution[0]-gfuXh[0])**2,(solution[1]-gfuXh[1])**2 ),mesh,"err")
        input("press enter to continue")

    err = sum( [ ( (gfuXh[i]   - solution[i])**2 +  InnerProduct(grad(gfuXh[i]) - sol_gradient[i], grad(gfuXh[i]) - sol_gradient[i] )) *  dX[i]  for i in [0,1] ]  ) 
    h1err = sqrt( Integrate(err, mesh) ) 
    h1norm = sqrt( Integrate( sum( [ ( (solution[i])**2 +  InnerProduct(sol_gradient[i],sol_gradient[i] )) *  dX[i]  for i in [0,1] ]  ) , mesh)   )

    print("HybridStabilized")
    print(" H1-error  = ", h1err  )
    print(" Relative H1-error  = ", h1err/h1norm  )
    #input("")
    #del psolver
    return h1err/h1norm  


def SolveHybridStabilizedModified(mesh,orders,stabs,problem,plot=False):
    
    #plus_str = "plus-outer|plus-inner"
    #minus_str = "minus-inner|minus-outer"
    plus_str = "plus"
    minus_str = "minus"

    sigma = problem["sigma"]
    solution = problem["solution"]
    sol_gradient = problem["solution-grad"]  
    coef_f = problem["rhs"]

    # indicator function for facets on interface
    VG0 = FacetFESpace( mesh, order=0)
    facets_G_indicator = GridFunction(VG0)
    facets_G_indicator.vec[ VG0.GetDofs(mesh.Boundaries("IF")) ]  = 1.0
    

    # Primal and dual FESpaces
    Q = FacetFESpace( mesh, order=orders["primal-IF"], dirichlet="outer") 
    Q_dual = FacetFESpace( mesh, order=orders["dual-IF"], dirichlet="outer") 
    Vp_primal =  Compress(H1(mesh, order=orders["primal-bulk"], dirichlet="outer", definedon=mesh.Materials(plus_str) ,  dgjumps=True))
    Vm_primal = Compress(H1(mesh, order=orders["primal-bulk"], dirichlet="outer", definedon=mesh.Materials(minus_str) , dgjumps=True))
    VGamma_primal = Compress(Q, active_dofs = Q.GetDofs(mesh.Boundaries("IF"))  )
    Vp_dual =  Compress(H1(mesh, order=orders["dual-bulk"], dirichlet="outer", definedon=mesh.Materials(plus_str) ,  dgjumps=True))
    Vm_dual = Compress(H1(mesh, order=orders["dual-bulk"], dirichlet="outer", definedon=mesh.Materials(minus_str) , dgjumps=True))
    VGamma_dual = Compress(Q_dual, active_dofs = Q_dual.GetDofs(mesh.Boundaries("IF"))  )

    Vh = Vp_primal *  Vm_primal *  VGamma_primal *  Vp_dual *  Vm_dual * VGamma_dual 

    up,um,uG,zp,zm,zG = Vh.TrialFunction()
    vp,vm,vG,wp,wm,wG  = Vh.TestFunction()
    u = [up,um]
    v = [vp,vm]
    z = [zp,zm]
    w = [wp,wm]
    gradu, gradz, gradv, gradw = [[grad(fun[i]) for i in [0, 1]] for fun in [u, z, v, w]]
    jumpu, jumpz, jumpv, jumpw = [ [ fun[i] - fhat  for i in [0, 1]] for fun,fhat in zip([u, z, v, w],[uG,zG,vG,wG]) ]

    nF = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size
    order = orders["primal-bulk"]

    dX = tuple( [dx(definedon=mesh.Materials(plus_str)), dx(definedon=mesh.Materials(minus_str))] ) 
    dX_outer = tuple( [dx(definedon=mesh.Materials("plus-outer")), dx(definedon=mesh.Materials("minus-outer"))] ) 
    
    dF = tuple( [dx(skeleton=True,  definedon=mesh.Materials(plus_str) ), dx(skeleton=True, definedon=mesh.Materials(minus_str) ) ] ) 
    ddT = tuple( [dx( element_boundary=True, definedon=mesh.Materials(plus_str)) , dx( element_boundary=True, definedon=mesh.Materials(minus_str) )] )

    def calL(fun):
        hesse = [fun[i].Operator("hesse") for i in [0,1]]
        if  mesh.dim == 2:
            return (-sigma[0]*hesse[0][0,0]-sigma[0]*hesse[0][1,1], -sigma[1]*hesse[1][0,0]-sigma[1]*hesse[1][1,1]  )

    aX = BilinearForm(Vh, symmetric=False)

    # a(u_pm,uG;w_pm,wG)
    aX  += sum(  sigma[i] * gradu[i] * gradw[i] * dX[i] for i in [0, 1])
    for i in [0,1]:
        aX += facets_G_indicator * ( (-1)*sigma[i]*gradu[i]*nF*jumpw[i] - sigma[i]*gradw[i]*nF*jumpu[i] 
               + stabs["Nitsche"] * np.sign(sigma[i]) * sigma[i] *order*order/h*jumpw[i]*jumpu[i]) * ddT[i]   

    # a(v_pm,vG;z_pm,zG)
    aX  += sum(  sigma[i] * gradv[i] * gradz[i] * dX[i] for i in [0, 1])
    for i in [0,1]:
        aX += facets_G_indicator * ( (-1)*sigma[i]*gradv[i]*nF*jumpz[i] - sigma[i]*gradz[i]*nF*jumpv[i] 
               + stabs["Nitsche"] * np.sign(sigma[i]) * sigma[i]*order*order/h*jumpz[i]*jumpv[i]) * ddT[i]   

    # primal stabilization s(u_pm,v_pm)
    aX += sum( [ stabs["CIP"] * h * abs(sigma[i]) * InnerProduct( (gradu[i] - gradu[i].Other()) * nF , (gradv[i] - gradv[i].Other()) * nF ) * dF[i] for i in [0,1] ]  )
    aX += sum( [ stabs["GLS"] * h**2 * calL(u)[i] * calL(v)[i] * dX[i] for i in [0, 1] ] )
    
    #aX += sum( [ 1e-0 * h**(2*orders["primal-bulk"]) * gradu[i] * gradv[i] * dX[i] for i in [0, 1] ] )
    #aX += sum( [ 1e-2 * h**(2*orders["primal-bulk"]) * u[i] * v[i] * dX[i] for i in [0, 1] ] )
    #aX += sum( [ 5e-2 * h**3 *  InnerProduct( ( u[i].Operator("hesse") - u[i].Other().Operator("hesse") ) , ( v[i].Operator("hesse") - v[i].Other().Operator("hesse") )  ) * dF[i] for i in [0,1] ]  )
    
    aX += sum( [ 5e-2 * h**3 *  InnerProduct( ( u[i].Operator("hesse") - u[i].Other().Operator("hesse") ) , ( v[i].Operator("hesse") - v[i].Other().Operator("hesse") )  ) * dF[i] for i in [0,1] ]  )

    for i in [0,1]:
        #aX += facets_G_indicator * 2*10*order*order/h*jumpv[i]*jumpu[i] * ddT[i]   
        aX += facets_G_indicator * stabs["IF"]/h*jumpv[i]*jumpu[i] * ddT[i]   

    # dual stabilization 
    aX += stabs["Dual"] * (-1)*gradz[1]*gradw[1]*dX[1]
    aX += stabs["Dual"] * (-1)*gradz[0]*gradw[0]*dX[0]

    #def P(fun):
    #    return fun - (fun * nF) * nF
    #jump_tangential_u =  P(gradu[0].Trace()) - P(gradu[1].Trace())
    #jump_tangential_v =  P(gradv[0].Trace()) - P(gradv[1].Trace())
    #aX += facets_G_indicator * 1e0*h*jumpv[i]*jumpu[i] * ddT[i]   

    #jump_tangential_u =  gradu[0].Trace() - gradu[1].Trace()
    #jump_tangential_v =  gradv[0].Trace() - gradv[1].Trace()
    #aX += 1.0 *  h * jump_tangential_u * jump_tangential_v * ds(definedon= mesh.Boundaries("IF"))

    # right hand side 
    fX = LinearForm(Vh)
    fX += sum( coef_f[i] * w[i] * dX[i] for i in [0, 1])
    fX += sum( [ stabs["GLS"] * h**2 * coef_f[i] * calL(v)[i] * dX[i] for i in [0, 1] ] )

    # Setting up matrix and vector
    print("assembling linear system")
    with TaskManager():
        aX.Assemble()
        fX.Assemble()
    


    if False:
        freedofs = [] 
        for i in range(len(Vh.FreeDofs())):
            if Vh.FreeDofs()[i]:
                freedofs.append(i)
        #print("freedofs = ", freedofs)
        freedofs = np.array(freedofs) 
        rows,cols,vals = aX.mat.COO()
        A_sp = sp.csr_matrix((vals,(rows,cols)))
        A_dense = A_sp.todense()
        A_free = A_dense[freedofs[:,None], freedofs[None,:] ] 
        #print("A_free = ", A_free)
        print("cond = {:.2E}".format(Decimal(np.linalg.cond(A_free))))
    
        #preI = Projector(mask=Vh.FreeDofs(), range=True)
        #lams = EigenValues_Preconditioner(mat=aX.mat, pre=preI)
        #print("lams = ", lams)
        #print("cond_ngs = ", abs(max(lams))/abs(min(lams)))


    gfuX = GridFunction(Vh)
    gfuXh = gfuX.components
    fX.vec.data -= aX.mat * gfuX.vec
    print("Solving linear system")
    gfuX.vec.data += aX.mat.Inverse(Vh.FreeDofs() ,inverse=solver  )* fX.vec

    if plot:
        Draw(IfPos(-x,gfuXh[0],gfuXh[1]),mesh,"uh")
        Draw(IfPos(-x,solution[0],solution[1]),mesh,"u")
        Draw(IfPos(-x,(solution[0]-gfuXh[0])**2,(solution[1]-gfuXh[1])**2 ),mesh,"err")
        Draw(IfPos(-x,(solution[0]-gfuXh[0]),(solution[1]-gfuXh[1]) ),mesh,"err-noabs")
        
        domain_values = { 'plus': sqrt( (solution[0]-gfuXh[0])**2 ),  'minus': sqrt( (solution[1]-gfuXh[1])**2 ) }    
        values_list = [domain_values[mat]
                       for mat in mesh.GetMaterials()]
        cf_subdom = CoefficientFunction(values_list)
        vtk_str = "Cavity-nonsymmetric-k{0}-unstructured-critical".format(orders["primal-bulk"])
        VTKOutput(ma=mesh, coefs=[cf_subdom ],
                          names=["abserr"],
                          filename=vtk_str, subdivision=2).Do()


        #input("press enter to continue")
    
    rel_errs = [] 
    dom_str = ["full-bulk","reduced-bulk"]
    #for dXs,str_s in zip( [dX,dX_outer],dom_str): 
    for dXs,str_s in zip( [dX],dom_str): 
        
        err = sum( [ ( (gfuXh[i]   - solution[i])**2 +  InnerProduct(grad(gfuXh[i]) - sol_gradient[i], grad(gfuXh[i]) - sol_gradient[i] )) *  dXs[i]  for i in [0,1] ]  ) 
        h1err = sqrt( Integrate(err, mesh) ) 
        h1norm = sqrt( Integrate( sum( [ ( (solution[i])**2 +  InnerProduct(sol_gradient[i],sol_gradient[i] )) *  dXs[i]  for i in [0,1] ]  ) , mesh)   )
        print("HybridStabilized in {0}".format( str_s ))
        #print(" H1-error  = ", h1err  )
        print(" Relative H1-error  = ", h1err/h1norm  )
        rel_errs.append( h1err/h1norm )
    
    #err_IF = sqrt( Integrate( (1/h) * facets_G_indicator * (gfuXh[2]   - solution[0])**2 * ddT[0]  , mesh) ) 
    l2err_IF = sqrt( Integrate(  facets_G_indicator * (gfuXh[2]   - solution[0])**2 * ddT[0]  , mesh) ) 
    l2norm_IF = sqrt( Integrate(  facets_G_indicator * (solution[0])**2 * ddT[0]  , mesh) ) 

    h1err_IF = sqrt( Integrate(  facets_G_indicator * InnerProduct( grad(gfuXh[2])  - sol_gradient[0], grad(gfuXh[2])  - sol_gradient[0]) * ddT[0]  , mesh) ) 
    h1norm_IF = sqrt( Integrate(  facets_G_indicator * InnerProduct(sol_gradient[0], sol_gradient[0]) * ddT[0]  , mesh) )

    h1half_IF = sqrt(l2err_IF * h1err_IF/( l2norm_IF *  h1norm_IF  ))

    #err_IF = sqrt( Integrate( (1/h) * facets_G_indicator * (gfuXh[2]   - solution[0])**2 * ddT[0]  , mesh) ) 
    
    #err_IF = sqrt( Integrate(  facets_G_indicator * (gfuXh[2]   - solution[0])**2 * ddT[0]  , mesh) ) 
    #print("Interface error =", err_IF )
    print("Interface error =", h1half_IF  )

    #input("")
    #return rel_errs[0], rel_errs[1], h1half_IF
    return rel_errs[0], h1half_IF


