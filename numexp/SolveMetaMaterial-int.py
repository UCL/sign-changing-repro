from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
import sys
from meshes import MakeStructuredCavityMesh,CreateUnstructuredMesh,CreateMetamaterialMesh
solver = "umfpack"
from math import pi 
from ngs_refsol import FundamentalSolution,FundamentalSolution_grad
import mpmath as mp 

RPML = 3.75
RSource = 3.5
a = 1.0
b = 1.2
c = 1.44

kappa0 = 2.19*1e9
kappa1 = 0.48*kappa0

rho0 = 998
rho1 = rho0

omega0 = 2*pi/0.5
omega = 2*pi*2963
omega *= 0.5

r = sqrt(x**2+y**2)


# PML parameters
eta = RPML+1.0  # problem is discretized on annulus with radii [a,eta]
mm_PML = 2 # profile of PML -> polynomial of degree mm_PML
#mm_PML = 1 # profile of PML -> polynomial of degree mm_PML
sigma0 = 4.5 # amplitude of PML
rad_min = RPML  # start of PML
rad_max = eta # end of PML

# define the PML
definedon_PML = 1
w = rad_max-rad_min
sigma_PML = IfPos(r-rad_min,IfPos(rad_max-r, sigma0*( (r-rad_min)/w )**mm_PML ,0),0)
sigma_hat = IfPos(r-rad_min,IfPos(rad_max-r, (sigma0/(mm_PML+1))*(w/r)*( (r-rad_min)/w )**(mm_PML+1) ,0),0)
alpha_PML = 1 + 1j*sigma_PML
beta_PML = 1 + 1j*sigma_hat
dr_sigma_PML = IfPos(r-rad_min,IfPos(rad_max-r,sigma0*mm_PML*(r-rad_min)**(mm_PML-1)/w**mm_PML,0),0)
dr_sigma_hat = IfPos(r-rad_min,IfPos(rad_max-r,(sigma0/(mm_PML+1))*( (r-rad_min)**mm_PML/w**mm_PML)*(mm_PML*r+rad_min)/r**2,0),0)
dr_alpha_PML = 1j*dr_sigma_PML
dr_beta_PML = 1j*dr_sigma_hat
A_00 = (x/r)**2*(beta_PML/alpha_PML) + (y/r)**2*(alpha_PML/beta_PML)
A_01 = (x*y/r**2)*( beta_PML/alpha_PML - alpha_PML/beta_PML)
A_11 = (x/r)**2*(alpha_PML/beta_PML) + (y/r)**2*(beta_PML/alpha_PML)
A_coeff = CoefficientFunction( (A_00,A_01,A_01,A_11) , dims=(2,2) )
d_quot = (  (dr_beta_PML*r+beta_PML)*alpha_PML - beta_PML*r*dr_alpha_PML ) / alpha_PML**2
alpha_prefac = sqrt(1+sigma_PML**2+sigma0**2+sigma_PML**2*sigma0**2)
rtilde_abs = r*sqrt(1+sigma_hat**2)
w_PML = IfPos(r-rad_min, alpha_prefac*exp(-omega*r*sigma_hat*sqrt(1-r**2/rtilde_abs**2)), 1.0)

physpml_trafo = CoefficientFunction( (x*beta_PML ,
                                      y*beta_PML ) )
physpml_jac = CoefficientFunction((
                                   beta_PML + 1j*(x**2/r)*dr_sigma_hat,
                                   1j*(x*y/r)*dr_sigma_hat,
                                   1j*(x*y/r)*dr_sigma_hat,
                                   beta_PML + 1j*(y**2/r)*dr_sigma_hat 
                                  ),
                                  dims=(2,2) )   
pml_phys = pml.Custom(trafo=physpml_trafo, jac = physpml_jac ) 
eta_tilde = rad_max*(1+1j*(sigma0/(mm_PML+1))*(rad_max-rad_min)/rad_max)
print("omega_hat =", exp(-omega*eta_tilde.imag*sqrt(1- (rad_min/abs(eta_tilde))**2)))





# Parameters 
sigma_dic = {"object": 1/rho0,
         "cloak-inner": 1/rho1,
         "cloak-outer": -1/rho1,
         "host-inner": 1/rho0,
         "host-outer": 1/rho0,
         "source-buffer-inner": 1/rho0,
         "source-buffer-outer": 1/rho0,
         "PML": 1/rho0
        }

sigma = [sigma_dic["object"], sigma_dic["cloak-inner"]]

k2 =    {"object": omega**2*(b/a)**4/kappa0,
         "cloak-inner": -omega**2*(b/r)**4/kappa1,
         "cloak-outer": omega**2/kappa1,
         "host-inner": omega**2/kappa0,
         "host-outer": omega**2/kappa0,
         "source-buffer-inner": omega**2/kappa0,
         "source-buffer-outer": omega**2/kappa0,
         "PML": omega**2/kappa0,
        }

dirac_source_pos = (-RSource,0.0)
omega_eff = sqrt(k2["host-outer"])/sqrt(sigma_dic["host-outer"]) 

ref_sol = (1/sigma_dic["host-outer"]) * FundamentalSolution( omega_eff , dirac_source_pos[0]  ,  dirac_source_pos[1]  ,False)
ref_sol_grad = (1/sigma_dic["host-outer"]) * CoefficientFunction((FundamentalSolution_grad(omega_eff, dirac_source_pos[0], dirac_source_pos[1]   , False),
                                                                      FundamentalSolution_grad(omega_eff, dirac_source_pos[0], dirac_source_pos[1]   , True) ) )

omega_eff_object = sqrt(k2["object"])/sqrt(sigma_dic["object"]) 
ref_sol_object = (1/sigma_dic["object"]) * FundamentalSolution( omega_eff_object  , dirac_source_pos[0]  ,  dirac_source_pos[1]  , False)
ref_sol_object_grad = (1/sigma_dic["object"]) * CoefficientFunction((FundamentalSolution_grad(omega_eff_object, dirac_source_pos[0], dirac_source_pos[1]   , False),
                                                                      FundamentalSolution_grad(omega_eff_object, dirac_source_pos[0], dirac_source_pos[1]   , True) ) )

error_inner = "host-inner"
error_outer = "host-outer"
error_object = "object"

# ref
#stabs_order = [{"CIP": 1e-6,"GLS": 1e-6,"Nitsche": 20,"Dual": 1e-6,"IF": 1e-2}, 
#               {"CIP": 1e-6,"GLS": 1e-6,"Nitsche": 20,"Dual": 1e-6,"IF": 1e-2},
#               {"CIP": 1e-5,"GLS": 1e-5,"Nitsche": 20,"Dual": 1e-4,"IF": 1e-2},
#               {"CIP": 1e-6,"GLS": 1e-6,"Nitsche": 20,"Dual": 1e-4,"IF": 1e-2},
#               ]

stabs_order = [{"CIP": 1e-6,"GLS": 1e-6,"Nitsche": 20,"Dual": 1e-6,"IF": 1e-2,"Tikh":2e-4}, 
               {"CIP": 1e-6,"GLS": 1e-6,"Nitsche": 20,"Dual": 1e-6,"IF": 1e-2, "Tikh":8e-2},
               {"CIP": 1e-5,"GLS": 1e-5,"Nitsche": 20,"Dual": 1e-4,"IF": 1e-2,"Tikh":5e-1},
               {"CIP": 1e-6,"GLS": 1e-6,"Nitsche": 20,"Dual": 1e-4,"IF": 1e-2,"Tikh":1e-1},
               ]

def SolveMetaMaterialStandardFEM(mesh,order=5):

    #omega = 8 
    #mesh.SetPML(pml.Radial(rad=RPML,alpha=1j,origin=(0,0)), "PML")
    sigma_coeff = CoefficientFunction([sigma_dic[mat] for mat in mesh.GetMaterials()])
    Draw(sigma_coeff,mesh,"sigma")
    k2_coeff = CoefficientFunction([k2[mat] for mat in mesh.GetMaterials()])

    V = H1(mesh, order=order, dirichlet="outer",complex=True) 
    u,v = V.TnT()
    a = BilinearForm(V, symmetric=False)
    #a  += sigma_coeff*grad(u)*grad(v)*dx
    #a  += (-1)*k2_coeff*u*v*dx
    a  += sigma_coeff * A_coeff * grad(u)*grad(v)*dx
    a  += (-1) * k2_coeff * alpha_PML * beta_PML * u * v * dx

    f = LinearForm(V)

    with TaskManager():
        a.Assemble()
        f.Assemble()
    
    mesh_pt = mesh( dirac_source_pos[0] , dirac_source_pos[1] )
    #f.vec[:] = 0.0 
    gfu = GridFunction(V)
    for i in range(V.ndof):
        gfu.vec[:] = 0.0
        gfu.vec[i] = 1.0
        f.vec.FV().NumPy()[i ] = gfu(mesh_pt)

    gfu.vec[:] = 0.0
    #f.vec.data -= a.mat * gfu.vec
    print("Solving linear system")
    gfu.vec.data += a.mat.Inverse(V.FreeDofs() ,inverse=solver  )* f.vec

    #mesh.UnSetPML("PML")
    Draw(gfu,mesh,"uh")
    Draw(ref_sol,mesh,"sol")
    Draw(  IfPos(r-c,  IfPos( RPML -r,  gfu  , 0.0 )   , 0.0 ) , mesh,"sol-cut")
    Draw(  IfPos(r-c,  IfPos( RPML -r,  sqrt( (ref_sol.real - gfu.real)**2 + (ref_sol.imag - gfu.imag)**2)  , 0.0 )   , 0.0 ) , mesh,"err")
     
    relative_h1_errors = [] 
    for err_d in [error_inner,error_outer]:
        err_form =  (  (ref_sol.real - gfu.real)**2 + (ref_sol.imag - gfu.imag)**2)    *  dx(definedon=mesh.Materials(err_d) ) 
        err_form += ( InnerProduct(ref_sol_grad.real - grad(gfu).real, ref_sol_grad.real - grad(gfu).real ) 
                     + InnerProduct(ref_sol_grad.imag - grad(gfu).imag, ref_sol_grad.imag - grad(gfu).imag )     )   *  dx(definedon=mesh.Materials(err_d)) 
        h1err = sqrt( Integrate(err_form, mesh) ) 
        h1norm = sqrt( Integrate( (  (ref_sol.real)**2 + (ref_sol.imag)**2  
                                     + InnerProduct(ref_sol_grad.real, ref_sol_grad.real) 
                                     + InnerProduct(ref_sol_grad.imag, ref_sol_grad.imag ) )   *  dx(definedon=mesh.Materials(err_d)) , mesh) ) 
        relative_h1_errors.append( h1err/ h1norm )    
        print("Relative H1 error in {0} = {1}".format(err_d , h1err/ h1norm ))

    return tuple(relative_h1_errors)


def SolveHybridStabilized(mesh,orders,stabs,plot=False,export_vtk=False,vtk_str="",remove_cloak=False):

    if remove_cloak:
        sigma_dic["cloak-outer"] = sigma_dic["host-outer"] 
        k2["cloak-inner"] =  k2["host-outer"]
        k2["cloak-outer"] =  k2["host-outer"]
        print("sigma_dic = ", sigma_dic)
        print("k2 = ", k2)
     
    #mesh.SetPML(pml_phys,"PML")
    #mesh.SetPML(pml.Radial(rad=RPML,alpha=1j,origin=(0,0)), "PML")

    #g = exp(1j*omega0*x)

    # indicator function for facets on interface
    VG0 = FacetFESpace( mesh, order=0) 
    facets_G_indicator = GridFunction(VG0)
    facets_G_indicator.vec[ VG0.GetDofs(mesh.Boundaries("Gamma-internal|Gamma-outer")) ]  = 1.0

    # Primal and dual FESpaces
    Q = FacetFESpace( mesh, order=orders["primal-IF"], dirichlet="outer",complex=True) 
    Q_dual = FacetFESpace( mesh, order=orders["dual-IF"], dirichlet="outer",complex=True) 
    Vp_primal =  Compress(H1(mesh, order=orders["primal-bulk"], dirichlet="outer", definedon=mesh.Materials("object|cloak-inner|host-inner|host-outer|source-buffer-inner|source-buffer-outer|PML") ,  dgjumps=True,complex=True))
    Vm_primal = Compress(H1(mesh, order=orders["primal-bulk"], dirichlet=[], definedon=mesh.Materials("cloak-outer") , dgjumps=True,complex=True))
    VGamma_primal = Compress(Q, active_dofs = Q.GetDofs(mesh.Boundaries("Gamma-internal|Gamma-outer")))
    Vp_dual =  Compress(H1(mesh, order=orders["dual-bulk"], dirichlet="outer", definedon=mesh.Materials("object|cloak-inner|host-inner|host-outer|source-buffer-inner|source-buffer-outer|PML") ,  dgjumps=True,complex=True))
    #Vp_dual =  Compress(H1(mesh, order=orders["primal-bulk"], dirichlet="outer", definedon=mesh.Materials("object|cloak-inner|host|PML") ,  dgjumps=True,complex=True))
    Vm_dual = Compress(H1(mesh, order=orders["dual-bulk"], dirichlet=[], definedon=mesh.Materials("cloak-outer") , dgjumps=True,complex=True))
    VGamma_dual = Compress(Q_dual, active_dofs = Q_dual.GetDofs(mesh.Boundaries("Gamma-internal|Gamma-outer"))  )


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

    #dX = tuple( [dx(definedon=mesh.Materials("plus")), dx(definedon=mesh.Materials("minus"))] ) 
    dX = { "object": dx(definedon=mesh.Materials("object")),
           "cloak-inner": dx(definedon=mesh.Materials("cloak-inner")),
           "cloak-outer": dx(definedon=mesh.Materials("cloak-outer")),
           "host-inner":  dx(definedon=mesh.Materials("host-inner")),
           "host-outer":  dx(definedon=mesh.Materials("host-outer")),
           "source-buffer-inner":  dx(definedon=mesh.Materials("source-buffer-inner")),
           "source-buffer-outer":  dx(definedon=mesh.Materials("source-buffer-outer")),
           "PML":  dx(definedon=mesh.Materials("PML")),
         }

    dF = { "object": dx(skeleton=True,  definedon=mesh.Materials("object") )  ,
           "cloak-inner": dx(skeleton=True,  definedon=mesh.Materials("cloak-inner") ) ,
           "cloak-outer":  dx(skeleton=True,  definedon=mesh.Materials("cloak-outer") )   ,
           "host-inner":   dx(skeleton=True,  definedon=mesh.Materials("host-inner") ),  
           "host-outer":   dx(skeleton=True,  definedon=mesh.Materials("host-outer") ),  
           "source-buffer-inner":   dx(skeleton=True,  definedon=mesh.Materials("source-buffer-inner") ),  
           "source-buffer-outer":   dx(skeleton=True,  definedon=mesh.Materials("source-buffer-outer") ),  
            "PML":   dx(skeleton=True,  definedon=mesh.Materials("PML") )  
         }
    
    jumping_materials = ["cloak-inner","cloak-outer","host-inner"]
    jumping_materials_idx = [0,1,0]
    ddT = { "cloak-inner" : dx( element_boundary=True, definedon=mesh.Materials("cloak-inner")),
            "cloak-outer" : dx( element_boundary=True, definedon=mesh.Materials("cloak-outer")),
            "host-inner" : dx( element_boundary=True, definedon=mesh.Materials("host-inner"))
          }

    def calL(fun,reg):
        hesse = fun.Operator("hesse")
        if  mesh.dim == 2:
            return -sigma_dic[reg]*hesse[0,0] - sigma_dic[reg]*hesse[1,1] - k2[reg]*fun  
    
    def calL_PML(fun,reg):
        hesse_u = fun.Operator("hesse")
        grad_u = grad(fun)
        if  mesh.dim == 2:
            return (-1) * sigma_dic[reg] * (  (1/r**2)*d_quot*(x*grad_u[0]+y*grad_u[1]) 
                  +(beta_PML/(alpha_PML*r**2))*(x**2*hesse_u[0,0]+y**2*hesse_u[1,1]+2*x*y*hesse_u[0,1])
                  +(alpha_PML/(beta_PML*r**2))*(-x*grad_u[0]-y*grad_u[1]+y**2*hesse_u[0,0]+x**2*hesse_u[1,1]-2*x*y*hesse_u[0,1])     
                ) - k2[reg] * alpha_PML * beta_PML * fun 

    region_list = [ "object", "cloak-inner", "cloak-outer", "host-inner", "host-outer", "source-buffer-inner", "source-buffer-outer", "PML"]
    idx_map = [0,0,1,0,0,0,0,0]
    
    #alpha_PML * beta_PML

    aX = BilinearForm(Vh, symmetric=False)

    for reg,i in zip(region_list,idx_map):
        if reg not in ["PML"]:
            aX  += (sigma_dic[reg] * gradu[i] * gradw[i] - k2[reg]*u[i]*w[i]  ) * dX[reg]
        else:
            aX  += (sigma_dic[reg] * (A_coeff * gradu[i]) * gradw[i] - k2[reg] *  alpha_PML * beta_PML * u[i]*w[i]  ) * dX[reg]

    # a(u_pm,uG;w_pm,wG)
    for reg,i in zip( jumping_materials, jumping_materials_idx ):
        aX += facets_G_indicator * ( (-1)*sigma_dic[reg]*gradu[i]*nF*jumpw[i] - sigma_dic[reg]*gradw[i]*nF*jumpu[i] 
               + stabs["Nitsche"] * np.sign(sigma_dic[reg]) * sigma_dic[reg] *order*order/h*jumpw[i]*jumpu[i]) * ddT[reg]   

    # a(v_pm,vG;z_pm,zG)
    for reg,i in zip(region_list,idx_map):
        if reg not in ["PML"]:
            aX  += (sigma_dic[reg] * gradv[i] * gradz[i]  -k2[reg]*v[i]*z[i] ) * dX[reg]
        else: 
            aX  += (sigma_dic[reg] * (A_coeff * gradv[i]) * gradz[i]  - k2[reg] * alpha_PML * beta_PML * v[i]*z[i] ) * dX[reg]


    for reg,i in zip( jumping_materials, jumping_materials_idx ):
        aX += facets_G_indicator * ( (-1)*sigma_dic[reg]*gradv[i]*nF*jumpz[i] - sigma_dic[reg]*gradz[i]*nF*jumpv[i] 
               + stabs["Nitsche"] * np.sign(sigma_dic[reg]) * sigma_dic[reg]*order*order/h*jumpz[i]*jumpv[i]) * ddT[reg]   

    # primal stabilization s(u_pm,v_pm)
    for reg,i in zip(region_list,idx_map):
        if reg != "PML":
        #if True:
            aX +=  stabs["CIP"] * h * InnerProduct( (gradu[i] - gradu[i].Other()) * nF , (gradv[i] - gradv[i].Other()) * nF ) * dF[reg] 
            #if reg != "host":
            aX +=  stabs["GLS"] * h**2 * calL(u[i],reg) * calL(v[i],reg) * dX[reg]
          
        else: # The above terms do not make sense in the PML region, instead we add a scaled Tikhonov term 
            #aX += 1e-6 * u[i]*v[i]*dX[reg] 
            aX +=  stabs["CIP"] * h * InnerProduct( (A_coeff* gradu[i] - A_coeff * gradu[i].Other()) * nF , (A_coeff * gradv[i] - A_coeff * gradv[i].Other()) * nF ) * dF[reg] 
            aX +=  stabs["GLS"] * h**2 * calL_PML(u[i],reg) * calL_PML(v[i],reg) * dX[reg]
        

    for reg,i in zip( jumping_materials, jumping_materials_idx ):
        #aX += facets_G_indicator * 2*10*order*order/h*jumpv[i]*jumpu[i] * ddT[i]   
        aX += facets_G_indicator * stabs["IF"]/h*jumpv[i]*jumpu[i] * ddT[reg]   

    # dual stabilization 
    for reg,i in zip(region_list,idx_map):
        aX += stabs["Dual"] * (-1)*gradz[i]*gradw[i]*dX[reg]
        if reg != "cloak-inner":
            aX += stabs["Dual"] * (-1)*z[i]*w[i]*dX[reg]
        #else:
        #    aX += 1e-7 * (-1)*gradz[i]*gradw[i]*dX[reg]
    
    # for testing 
    #for reg,i in zip(region_list,idx_map):
    #    if reg == "PML":
    #        aX += 1e-6 * u[i]*v[i]*dX[reg]
    #    #aX += 1e-7 * (-1)*z[i]*w[i]*dX[reg]

    # right hand side 
    fX = LinearForm(Vh)
    
    for reg,i in zip(region_list,idx_map):
        if reg not in ["PML"]:
            fX +=  stabs["GLS"] * h**2 * 0.0 * calL(v[i],reg) * dX[reg]
            
    # Setting up matrix and vector
    print("assembling linear system")
    with TaskManager():
        aX.Assemble()
        fX.Assemble()


    gfuX = GridFunction(Vh)
    gfuXh = gfuX.components
    mesh_pt = mesh( dirac_source_pos[0] , dirac_source_pos[1] )
    
    offset = Vp_primal.ndof + Vm_primal.ndof +  VGamma_primal.ndof 
    for i in range(Vp_dual.ndof):
        
        gfuXh[3].vec[:] = 0.0
        gfuXh[3].vec[i] = 1.0
        #if abs(gfuXh[3](mesh_pt)) > 1e-10:
        #    print("i = {0}, gfuX(mesh_pt)  = {1} ".format(i, gfuXh[3](mesh_pt) ))
        fX.vec.FV().NumPy()[i+offset] += gfuXh[3](mesh_pt)
     
    gfuX.vec[:] = 0.0
    print("Solving linear system")
    gfuX.vec.data += aX.mat.Inverse(Vh.FreeDofs() ,inverse=solver  )* fX.vec
    
    #mesh.UnSetPML("PML")
    if False:
        N_total = 200 
        r_eval = [-a + i*2*a/N_total for i in range(N_total+1)]
        eval_pts = [ (rr, 0.0, 0.0 ) for rr in r_eval ]
        uh_vals = [gfuXh[0](mesh(*pm)).real for pm in eval_pts ]
        plt.plot(r_eval,uh_vals,label="uh")
        fit_val =np.array( [ 1.2*(1/sigma_dic["object"])*(1j/4)*complex(mp.hankel1(0, 2.8 + omega_eff_object * sqrt( (rr - dirac_source_pos[0] )**2 + (0.0 - dirac_source_pos[1] )**2   ) ))  for rr in r_eval  ]) 
        
        plt.plot(r_eval,fit_val.real,label="fit")
        plt.legend()
        plt.show()


        #ref_sol = (1/sigma_dic["host-outer"]) * FundamentalSolution(sqrt(k2["host-outer"])/sqrt(sigma_dic["host-outer"]), dirac_source_pos[0]  ,  dirac_source_pos[1]  ,False)
        #Draw(gfuXh[0]  ,mesh,"u0")
        #Draw(gfuXh[1]  ,mesh,"u1")
        #Draw(ref_sol_object, mesh, "refsol-object")
        
        #Draw(ref_sol_object-gfuXh[0], mesh, "diff")
        #Draw(ref_sol, mesh, "refsol")
        #Draw(IfPos(r-b,  gfuXh[0]  , IfPos(c-r, gfuXh[1], gfuXh[0]   )  ) ,mesh,"u")
        #Draw(IfPos(b-r,  gfuXh[0]  , IfPos(c-r, gfuXh[1],   gfuXh[0]   )  ) ,mesh,"u")
        #Draw(  IfPos(r-c,  IfPos( RPML -r,  sqrt( (ref_sol.real - gfuXh[0].real)**2 + (ref_sol.imag - gfuXh[0].imag)**2)  , 0.0 )   , 0.0 ) , mesh,"err")
        Draw(  IfPos(r-0,  IfPos( a -r,  sqrt( (ref_sol_object.real - gfuXh[0].real)**2 + (ref_sol_object.imag - gfuXh[0].imag)**2)  , 0.0 )   , 0.0 ) , mesh,"err")
        Draw(  IfPos(r-0,  IfPos( a -r,  ref_sol_object.real   , 0.0 )   , 0.0 )  , mesh,"assumed-u-inner-real")
        Draw(  IfPos(r-0,  IfPos( a -r,    gfuXh[0].real , 0.0 )   , 0.0 )  , mesh,"uh-inner-real")
        Draw(  IfPos(r-0,  IfPos( a -r,  ref_sol_object.imag   , 0.0 )   , 0.0 )  , mesh,"assumed-u-inner-imag")
        Draw(  IfPos(r-0,  IfPos( a -r,    gfuXh[0].imag , 0.0 )   , 0.0 )  , mesh,"uh-inner-imag")

        input("")
    


    relative_h1_errors = [] 
    #for err_d, ref_sol_d, ref_sol_grad_d in zip([error_inner,error_outer,error_object], 
    #                                            [ref_sol,ref_sol,ref_sol_object],
    #                                            [ref_sol_grad,ref_sol_grad,ref_sol_object_grad]):

    for err_d, ref_sol_d, ref_sol_grad_d in zip([error_inner,error_outer], 
                                                [ref_sol,ref_sol],
                                                [ref_sol_grad,ref_sol_grad]):
        err_form =  (  (ref_sol_d.real - gfuXh[0].real)**2 + (ref_sol_d.imag - gfuXh[0].imag)**2)    *  dx(definedon=mesh.Materials(err_d)) 
        err_form += ( InnerProduct(ref_sol_grad_d.real - grad(gfuXh[0]).real, ref_sol_grad_d.real - grad(gfuXh[0]).real ) 
                     + InnerProduct(ref_sol_grad_d.imag - grad(gfuXh[0]).imag, ref_sol_grad_d.imag - grad(gfuXh[0]).imag )     )   *  dx(definedon=mesh.Materials(err_d)) 
        h1err = sqrt( Integrate(err_form, mesh) ) 
        h1norm = sqrt( Integrate( (  (ref_sol.real)**2 + (ref_sol.imag)**2  
                                     + InnerProduct(ref_sol_grad_d.real, ref_sol_grad_d.real) 
                                     + InnerProduct(ref_sol_grad_d.imag, ref_sol_grad_d.imag ) )   *  dx(definedon=mesh.Materials(err_d)) , mesh) ) 
        relative_h1_errors.append( h1err/ h1norm )    
        print("Relative H1 error in {0} = {1}".format(err_d , h1err/ h1norm ))
    
    if export_vtk:
        VTKOutput(ma=mesh, coefs=[ IfPos(b-r,  gfuXh[0].real  , IfPos(c-r, gfuXh[1].real,   gfuXh[0].real   )  )  ],
                      names=["u"],
                      filename=vtk_str, subdivision=2).Do()
 
    
    #err =  (  (ref_sol.real - gfuXh[0].real)**2 + (ref_sol.imag - gfuXh[0].imag)**2)    *  dx(definedon=mesh.Materials("host-outer")) 
    #l2err = sqrt( Integrate(err, mesh) ) 
    #l2norm = sqrt( Integrate( (  (ref_sol.real)**2 + (ref_sol.imag)**2) *  dx(definedon=mesh.Materials("host-outer")) , mesh) ) 
    #print("relative l2err = ", l2err/ l2norm )
    #input("press enter to continue")
    return tuple(relative_h1_errors)

all_maxhs = np.linspace(0.8,0.2,6,endpoint=False).tolist() + np.linspace(0.2,0.1,6,endpoint=False).tolist() +  np.linspace(0.1,0.025,25,endpoint=True).tolist()

all_maxhs = np.linspace(0.8,0.2,6,endpoint=False).tolist() + np.linspace(0.2,0.1,6,endpoint=False).tolist() +  np.linspace(0.1,0.05,10,endpoint=True).tolist()
#all_maxhs = np.linspace(0.8,0.2,6,endpoint=False).tolist() + np.linspace(0.2,0.1,6,endpoint=False).tolist() +  np.linspace(0.1,0.05,10,endpoint=True).tolist()
#all_maxhs = np.linspace(0.08,0.07,20,endpoint=True).tolist()
#all_maxhs = np.linspace(0.8,0.2,6,endpoint=False).tolist() + np.linspace(0.2,0.1,6,endpoint=False).tolist() +  np.linspace(0.1,0.025,25,endpoint=True).tolist()


#for order in [1,2,3]:
for order in [2]:
#for order in [1]:

    stabs = stabs_order[order-1] 
    orders = {"primal-bulk": order,
          "primal-IF": order,
          "dual-bulk": order,
          "dual-IF": order}

    # Generate Plot of MetaMaterial 
    if order == 3:
        mesh = CreateMetamaterialMesh(maxh=0.08,order_geom=5)
        #mesh = CreateMetamaterialMesh(maxh=0.1,order_geom=5)
        SolveHybridStabilized(mesh,orders,stabs_order[order-1],plot=True, export_vtk=True,vtk_str="MetaMaterial-order{0}".format(order)) 

    if order == 1:
        maxhs = all_maxhs
    elif order == 2:
        maxhs = all_maxhs[:-5]
    elif order == 3:
        #maxhs = all_maxhs[:-6]
        maxhs = all_maxhs[:-11]
    elif order == 4:
        maxhs = np.linspace(0.8,0.15,16,endpoint=False).tolist() + [0.15] 
    else:
        maxhs = np.linspace(0.8,0.2,12,endpoint=False).tolist() + np.linspace(0.2,0.09,15,endpoint=False).tolist() 

    err_Galerkin_outer = [ ]
    err_Galerkin_inner = []
    err_Stabilized_outer = [ ]
    err_Stabilized_inner = [ ]
    err_Stabilized_object = [ ]


    for maxh in maxhs:
        print("maxh = ", maxh)
        mesh = CreateMetamaterialMesh(maxh=maxh,order_geom=5)
        print("Computing for Galerkin method")
        err_inner, err_outer =  SolveMetaMaterialStandardFEM(mesh, orders["primal-bulk"] )
        err_Galerkin_inner.append(err_inner)
        err_Galerkin_outer.append(err_outer)
        print("Computing for stabilized method")
        #err_inner, err_outer, err_object =  SolveHybridStabilized(mesh,orders,stabs,plot=True) 
        err_inner, err_outer =  SolveHybridStabilized(mesh,orders,stabs,plot=True) 
        err_Stabilized_inner.append(err_inner)
        err_Stabilized_outer.append(err_outer)
        #err_Stabilized_object.append(err_object)

    maxhnp = np.array(maxhs)
    plt.loglog(maxhnp, err_Galerkin_outer ,label="Galerkin-outer",marker='o')
    plt.loglog(maxhnp, err_Galerkin_inner ,label="Galerkin-inner",marker='+')
    plt.loglog(maxhnp,  err_Stabilized_outer ,label="Stabilized-outer",marker='x')
    plt.loglog(maxhnp,  err_Stabilized_inner ,label="Stabilized-inner",marker='x')
    #plt.loglog(maxhnp,  err_Stabilized_object ,label="Stabilized-object",marker='x')
    plt.loglog(maxhs, err_Stabilized_inner[4] * maxhnp**order/maxhnp[4]**order ,color="gray",label="$\mathcal{O}(h^k)$",marker='+')
    plt.title("Relative H1-error for k={0}".format(order))
    plt.xlabel("h")
    plt.legend()
    plt.show()
    #plt.savefig("Meta-k{0}".format(order))
    plt.clf()

    name_str = "MetaMaterial-k{0}-mPML.dat".format(order)
    results = [maxhnp, np.array(err_Galerkin_inner,dtype=float), np.array(err_Galerkin_outer,dtype=float), np.array( err_Stabilized_outer ,dtype=float), np.array( err_Stabilized_inner ,dtype=float) ]
    header_str = "h Galerkin-inner Galerkin-outer Hybridstab-outer Hybridstab-inner" 

    if False:
        np.savetxt(fname ="../data/{0}".format(name_str),
                                       X = np.transpose(results),
                                       header = header_str,
                                       comments = '')

    # Genrate plot without cloak (no MetaMaterial layer)
    if order == 3:
        mesh = CreateMetamaterialMesh(maxh=0.08,order_geom=5)
        #mesh = CreateMetamaterialMesh(maxh=0.1,order_geom=5)
        orders = {"primal-bulk": order,
                      "primal-IF": order,
                      "dual-bulk": order,
                      "dual-IF": order}
    #    SolveHybridStabilized(mesh,orders,stabs_order[order-1],plot=True, export_vtk=True,vtk_str="NoCloak-order{0}".format(order),remove_cloak=True) 

