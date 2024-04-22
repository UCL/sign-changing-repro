from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/janosch/projects//sign_changing_coeff/numexp")
from meshes import MakeStructuredCavityMesh,CreateUnstructuredMesh,CreateMetamaterialMesh
solver = "pardiso"
#solver = "umfpack"
from math import pi 
from ngs_refsol import FundamentalSolution,FundamentalSolution_grad


#('Gamma-inner', 'Gamma-inner', 'Gamma-inner', 'Gamma-inner', 'Gamma-internal', 'Gamma-internal', 'Gamma-internal', 'Gamma-internal', 'Gamma-outer', 'Gamma-outer', 'Gamma-outer', 'Gamma-outer', 'outer', 'outer', 'outer', 'outer')
#('object', 'cloak-inner', 'cloak-outer', 'host')

RPML = 3.75
RSource = 3.5
#RPML = 1.0
a = 1.0
b = 1.2
c = 1.44

kappa0 = 2.19*1e9
kappa1 = 0.48*kappa0

#kappa1 = 0.9*kappa0
#kappa1 = (4/9)*kappa0
#kappa1 = kappa0
rho0 = 998
rho1 = rho0

omega0 = 2*pi/0.5
#omega = 2*pi/0.5
omega = 2*pi*2963
#omega = 2*pi*1000
omega *= 0.5

r = sqrt(x**2+y**2)

##omega = 8
#kappa0 = 1
#kappa1 = kappa0
#rho0 = 1
#rho1 = 1 


# Parameters 


#sigma_dic = {"object": 1/rho0,
#         "cloak-inner": 1/rho1,
#         "cloak-outer": -1/rho1,
#         "host-inner": 1/rho0,
#         "host-outer": 1/rho0,
#         "source-buffer-inner": 1/rho0,
#         "source-buffer-outer": 1/rho0,
#         "PML": 1/rho0
#        }

sigma_dic = {"object": 1/rho0,
         "cloak-inner": 1/rho1,
         "cloak-outer": 1/rho1,
         "host-inner": 1/rho0,
         "host-outer": 1/rho0,
         "source-buffer-inner": 1/rho0,
         "source-buffer-outer": 1/rho0,
         "PML": 1/rho0
        }
#sigma_dic = {"object":  1/rho0,
#         "cloak-inner": 1/rho0,
#          "cloak-outer": 1/rho0,
#         "host": 1/rho0,
#        "PML": 1/rho0
#         }

sigma = [sigma_dic["object"], sigma_dic["cloak-inner"]]
#print("sigma = ", sigma)

#k2 =    {"object": omega**2*(b/a)**4,
#         "cloak-inner": omega**2*(b/r)**4,
#         "cloak-outer": omega**2,
#         "host": omega**2,
#         "PML": omega**2,
#        }


#k2 =    {"object": omega**2*(b/a)**4/kappa0,
#         "cloak-inner": -omega**2*(b/r)**4/kappa1,
#         "cloak-outer": omega**2/kappa1,
#         "host-inner": omega**2/kappa0,
#         "host-outer": omega**2/kappa0,
#         "source-buffer-inner": omega**2/kappa0,
#         "source-buffer-outer": omega**2/kappa0,
#         "PML": omega**2/kappa0,
#        }

k2 =    {"object": omega**2/kappa0,
         "cloak-inner": omega**2/kappa0,
         "cloak-outer": omega**2/kappa0,
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

error_inner = "host-inner"
error_outer = "host-outer"

#omega1 = 8
#k2 =    {"object": omega**2*(b/a)/kappa0 ,
#         "cloak-inner": omega**2/kappa0  ,
#         "cloak-outer":  omega**2/kappa0 ,
#         "host":  omega**2/kappa0 ,
#         "PML":  omega**2/kappa0  ,
#        }

#k2 =    {"object": omega**2*(b/a)**4,
#         "cloak-inner": omega**2,
#         "cloak-outer": omega**2,
#         "host": omega**2,
#         "PML": omega**2,
#        }



#stabs = # works good with full dual order
stabs_order = [{"CIP": 1e-6,"GLS": 1e-6,"Nitsche": 20,"Dual": 1e-6,"IF": 1e-2}, 
               {"CIP": 1e-6,"GLS": 1e-6,"Nitsche": 20,"Dual": 1e-6,"IF": 1e-2},
               #{"CIP": 1e-5,"GLS": 1e-5,"Nitsche": 20,"Dual": 1e-4,"IF": 1e-2},
               {"CIP": 1e-5,"GLS": 1e-5,"Nitsche": 20,"Dual": 1e-4,"IF": 1e-1},
               ]


# PML 
C_PML = 50
ROuter = RPML + 1.0 
eta = ROuter-RPML
omega_PML = 0.5*omega_eff
print("omega_eff = ", omega_eff)
sigma_r = IfPos(r - RPML, (C_PML/eta)*((r-RPML)/(eta))**2,0)
s_r = IfPos( r - RPML , 1.0/(1.0 + 1j*sigma_r/omega_PML ), 1.0 )
stilde_r = 1 + ((1j*C_PML)/(3*r*omega_PML))*((r-RPML)/eta)**3

#stabs = {"CIP": 1e-6,"GLS": 1e-6,"Nitsche": 20,"Dual": 1e-4,"IF": 1e-0}

#stabs = {"CIP": 1e-10,"GLS": 1e-10,"Nitsche": 20,"Dual": 1e-12,"IF": 1e-12}

#stabs = {"CIP": 1e-7,"GLS": 1e-4,"Nitsche": 20,"Dual": 1e-4,"IF": 1e-4} # good stabs

#mesh = CreateMetamaterialMesh(maxh=0.1,order_geom=5)
#mesh = CreateMetamaterialMesh(maxh=0.06,order_geom=5)
#Draw(mesh)
#input("")

#sigma_dic = {"object": (kappa0/rho0)*(a/b)**4,
#         "cloak-inner": (-kappa1/rho1)*(r/b)**4,
#         "cloak-outer": (-kappa1/rho1),
#         "host": kappa0/rho0,
#         "PML": kappa0/rho0
#         }

#f_pw_source = IfPos(-x,exp(-1j*x*omega0), exp(+1j*x*omega0))

#f_pw_dic = {"object": f_pw_source ,
#         "cloak-inner": f_pw_source,
#         "cloak-outer": f_pw_source,
#         "host": f_pw_source,
#         "PML": 0,
#        }


def SolveMetaMaterialStandardFEM(mesh,order=5):

    #omega = 8 
    #mesh.SetPML(pml.Radial(rad=RPML,alpha=1j,origin=(0,0)), "PML")
    #sigma_mat = { "object": kappa0/rho0,
    #              "cloak-inner":-kappa1/rho1,
    #              "cloak-outer":-kappa1/rho1,
    #              "host": kappa0/rho0,
    #              "PML":  kappa0/rho0
    #           }

    #sigma_mat = { "object": kappa0/rho0,
    #              "cloak-inner":kappa0/rho0,
    #              "cloak-outer":kappa0/rho0,
    #              "host": kappa0/rho0,
    #              "PML":  kappa0/rho0
    #           }
    #sigma_mat = { "object": kappa0/rho0,
    #              "cloak-inner": kappa0/rho1,
    #              "cloak-outer": kappa0/rho1,
    #              "host": kappa0/rho0,
    #              "PML":  kappa0/rho0
    #             }


    #k2 =    {"object": omega**2,
    #         "cloak-inner": omega**2,
    #         "cloak-outer": omega**2,
    #         "host": omega**2,
    #         "PML": omega**2,
    #        }
    
    #sigma_mat = { ,  "box" : 100 }
    sigma_coeff = CoefficientFunction([sigma_dic[mat] for mat in mesh.GetMaterials()])
    Draw(sigma_coeff,mesh,"sigma")
    k2_coeff = CoefficientFunction([k2[mat] for mat in mesh.GetMaterials()])

    #f_coeff = CoefficientFunction([ f_pw_dic[mat] for mat in mesh.GetMaterials()])

    #dX = tuple( [dx(definedon=mesh.Materials("")), dx(definedon=mesh.Materials("minus"))] ) 
    
    V = H1(mesh, order=order, dirichlet="outer",complex=True) 
    u,v = V.TnT()
    a = BilinearForm(V, symmetric=False)
    #sigma_coeff = CoefficientFunction( [sigma[0] if mat=="plus" else sigma[1] for mat in mesh.GetMaterials() ] )
    

    #a  += sigma_coeff*grad(u)*grad(v)*dx
    #a  += (-1)*k2_coeff*u*v*dx
    
    a  += sigma_coeff*grad(u)*grad(v)*dx( definedon=mesh.Materials("object|cloak-inner|cloak-outer|host-inner|host-outer|source-buffer-inner|source-buffer-outer")  ) 
    a  += (-1)*k2_coeff*u*v*dx( definedon=mesh.Materials("object|cloak-inner|cloak-outer|host-inner|host-outer|source-buffer-inner|source-buffer-outer") ) 

    # PML layer 
    u_r = (x/r)*grad(u)[0] + (y/r)*grad(u)[1]
    v_r = (x/r)*grad(v)[0] + (y/r)*grad(v)[1]
    u_phi = -y*grad(u)[0] + x*grad(u)[1] 
    v_phi = -y*grad(v)[0] + x*grad(v)[1] 

    a  += sigma_coeff * s_r *  stilde_r * u_r * v_r *  dx( definedon=mesh.Materials("PML")  ) 
    a  += sigma_coeff * (1/(r**2 * s_r *  stilde_r)) * u_phi * v_phi  * dx( definedon=mesh.Materials("PML")  ) 
    a  += (-1) * k2_coeff * (stilde_r / s_r)  * u * v  * dx( definedon=mesh.Materials("PML")  ) 

    #a  += sigma_coeff*grad(u)*grad(v)*dx
    #a  += (-1)*omega**2*u*v*dx
    #rtilde = CoefficientFunction((x,y))
    #a += (-4*kappa1*r**2/( rho1*b**4)) * InnerProduct(grad(u),rtilde) *  v * dx(definedon=mesh.Materials("cloak-inner"))

    #a  += grad(u)*grad(v)*dx
    #a  += (-1)*8**2*u*v*dx

    f = LinearForm(V)
    #f += 1e9 * exp( -100*( (x+2.5)**2 + y**2) ) * v *dx
    #f += f_coeff * v *dx

    with TaskManager():
        a.Assemble()
        f.Assemble()
    

    #dirac_source_pos = (-3.5,0.0)
    #dirac_source_pos = (-0.5,0.0)
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
    Draw(sigma_r,mesh,"PML")
    #err =  IfPos(r-c-0.25, (  (ref_sol.real - gfu.real)**2 + (ref_sol.imag - gfu.imag)**2),0.0)    *  dx(definedon=mesh.Materials("host")) 
    
    #err =  (  (ref_sol.real - gfu.real)**2 + (ref_sol.imag - gfu.imag)**2)    *  dx(definedon=mesh.Materials("host-outer")) 
    #l2err = sqrt( Integrate(err, mesh) ) 
    #l2norm = sqrt( Integrate( (  (ref_sol.real)**2 + (ref_sol.imag)**2) *  dx(definedon=mesh.Materials("host-outer")) , mesh) ) 
    
    #err_grad = ( InnerProduct(ref_sol_grad.real - grad(gfu).real, ref_sol_grad.real - grad(gfu).real ) 
    #             + InnerProduct(ref_sol_grad.imag - grad(gfu).imag, ref_sol_grad.imag - grad(gfu).imag )     )   *  dx(definedon=mesh.Materials("host-outer")) 
    #l2err_grad = sqrt( Integrate(err_grad, mesh) ) 
    #l2norm_grad =  sqrt( Integrate( (  InnerProduct(ref_sol_grad.real, ref_sol_grad.real  ) 
    #             + InnerProduct(ref_sol_grad.imag, ref_sol_grad.imag )     )   *  dx(definedon=mesh.Materials("host-outer")) , mesh) ) 
    
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

    
    #input("")
    #print("relative l2err = ", l2err/ l2norm )
    #print("relative l2err grad = ", l2err_grad/ l2norm_grad )
    #input("")
    return tuple(relative_h1_errors)


#print("Computing for Galerkin method")
#SolveMetaMaterialStandardFEM(mesh, orders["primal-bulk"] )



def SolveHybridStabilized(mesh,orders,stabs,plot=False):
    
    #mesh.SetPML(pml.Radial(rad=RPML,alpha=1j,origin=(0,0)), "PML")
    
    #mesh.SetPML(pml.Radial(rad=RPML,alpha=1j,origin=(0,0)), "PML")

    #g = exp(1j*omega0*x)

    # indicator function for facets on interface
    VG0 = FacetFESpace( mesh, order=0)
    #facets_Gamma_inner_indicator = GridFunction(VG0)
    #facets_Gamma_inner_indicator.vec[ VG0.GetDofs(mesh.Boundaries("Gamma-inner")) ]  = 1.0
    #facets_Gamma_outer_indicator = GridFunction(VG0)
    #facets_Gamma_outer_indicator.vec[ VG0.GetDofs(mesh.Boundaries("Gamma-outer")) ]  = 1.0
    
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


    #for el in mesh.Elements():
    #    Vh.SetOrder(NodeId(CELL,el.nr), orders["primal-bulk"])
    #    for ed in el.edges:
    #        Vh.SetOrder(ed, orders["primal-bulk"])
    #Vh.UpdateDofTables()


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
    #ddT = tuple( [dx( element_boundary=True, definedon=mesh.Materials("object|host")) , dx( element_boundary=True, definedon=mesh.Materials("cloak-inner|cloak-outer") )] )
    
    #dF = tuple( [dx(skeleton=True,  definedon=mesh.Materials("plus") ), dx(skeleton=True, definedon=mesh.Materials("minus") ) ] ) 
    #ddT = tuple( [dx( element_boundary=True, definedon=mesh.Materials("plus")) , dx( element_boundary=True, definedon=mesh.Materials("minus") )] )

    
    #dX = tuple( [dx(definedon=mesh.Materials("plus")), dx(definedon=mesh.Materials("minus"))] ) 

    #def calL(fun):
    #    hesse = [fun[i].Operator("hesse") for i in [0,1]]
    #    if  mesh.dim == 2:
    #        return (-sigma[0]*hesse[0][0,0]-sigma[0]*hesse[0][1,1], -sigma[1]*hesse[1][0,0]-sigma[1]*hesse[1][1,1]  )

    def calL(fun,reg):
        hesse = fun.Operator("hesse")
        if  mesh.dim == 2:
            return -sigma_dic[reg]*hesse[0,0] - sigma_dic[reg]*hesse[1,1] - k2[reg]*fun  
    

    region_list = [ "object", "cloak-inner", "cloak-outer", "host-inner", "host-outer", "source-buffer-inner", "source-buffer-outer", "PML"]
    idx_map = [0,0,1,0,0,0,0,0]

    aX = BilinearForm(Vh, symmetric=False)

    for reg,i in zip(region_list,idx_map):
        if reg != "PML":
            aX  += (sigma_dic[reg] * gradu[i] * gradw[i] - k2[reg]*u[i]*w[i]  ) * dX[reg]
        else:
            u_r = (x/r)*gradu[i][0] + (y/r)*gradu[i][1]
            w_r = (x/r)*gradw[i][0] + (y/r)*gradw[i][1]
            u_phi = -y*gradu[i][0] + x*gradu[i][1] 
            w_phi = -y*gradw[i][0] + x*gradw[i][1]
            aX += sigma_dic[reg]  * s_r *  stilde_r * u_r * w_r * dX[reg]
            aX += sigma_dic[reg]  * (1/(r**2 * s_r *  stilde_r)) * u_phi * w_phi * dX[reg]
            aX += (-1) * k2[reg] * (stilde_r / s_r)  * u[i] * w[i] * dX[reg]


          
    #a  += sigma_coeff * s_r *  stilde_r * u_r * v_r * dx( definedon=mesh.Materials("PML")  ) 
    #a  += sigma_coeff * (1/(r**2 * s_r *  stilde_r)) * u_phi * v_phi * dx( definedon=mesh.Materials("PML")  ) 
    #a  += (-1) * k2_coeff * (stilde_r / s_r)  * u * v * dx( definedon=mesh.Materials("PML")  ) 

    # a(u_pm,uG;w_pm,wG)
    #aX  += sum(  sigma[i] * gradu[i] * gradw[i] * dX[i] for i in [0, 1])
    #sub_idx_map = 
    #for reg_
    for reg,i in zip( jumping_materials, jumping_materials_idx ):
        aX += facets_G_indicator * ( (-1)*sigma_dic[reg]*gradu[i]*nF*jumpw[i] - sigma_dic[reg]*gradw[i]*nF*jumpu[i] 
               + stabs["Nitsche"] * np.sign(sigma_dic[reg]) * sigma_dic[reg] *order*order/h*jumpw[i]*jumpu[i]) * ddT[reg]   
    
    #for i in [0,1]:
    #    aX += facets_G_indicator * ( (-1)*sigma[i]*gradu[i]*nF*jumpw[i] - sigma[i]*gradw[i]*nF*jumpu[i] 
    #           + stabs["Nitsche"] * np.sign(sigma[i]) * sigma[i] *order*order/h*jumpw[i]*jumpu[i]) * ddT[i]   

    # a(v_pm,vG;z_pm,zG)
    #aX  += sum(  sigma[i] * gradv[i] * gradz[i] * dX[i] for i in [0, 1])
    for reg,i in zip(region_list,idx_map):
        if reg != "PML":
            aX  += (sigma_dic[reg] * gradv[i] * gradz[i]  -k2[reg]*v[i]*z[i] ) * dX[reg]
        else:
            v_r = (x/r)*gradv[i][0] + (y/r)*gradv[i][1]
            z_r = (x/r)*gradz[i][0] + (y/r)*gradz[i][1]
            v_phi = -y*gradv[i][0] + x*gradv[i][1] 
            z_phi = -y*gradz[i][0] + x*gradz[i][1]
            aX += sigma_dic[reg] * s_r *  stilde_r * v_r * z_r * dX[reg]
            aX += sigma_dic[reg]  * (1/(r**2 * s_r *  stilde_r)) * v_phi * z_phi * dX[reg]
            aX += (-1) * k2[reg] * (stilde_r / s_r)  * v[i] * z[i] * dX[reg]

    for reg,i in zip( jumping_materials, jumping_materials_idx ):
        aX += facets_G_indicator * ( (-1)*sigma_dic[reg]*gradv[i]*nF*jumpz[i] - sigma_dic[reg]*gradz[i]*nF*jumpv[i] 
               + stabs["Nitsche"] * np.sign(sigma_dic[reg]) * sigma_dic[reg]*order*order/h*jumpz[i]*jumpv[i]) * ddT[reg]   

    # primal stabilization s(u_pm,v_pm)
    for reg,i in zip(region_list,idx_map):
        if reg != "PML":
            aX +=  stabs["CIP"] * h * InnerProduct( (gradu[i] - gradu[i].Other()) * nF , (gradv[i] - gradv[i].Other()) * nF ) * dF[reg] 
            #if reg != "host":
            aX +=  stabs["GLS"] * h**2 * calL(u[i],reg) * calL(v[i],reg) * dX[reg] 


    #aX += sum( [ stabs["CIP"] * h * abs(sigma[i]) * InnerProduct( (gradu[i] - gradu[i].Other()) * nF , (gradv[i] - gradv[i].Other()) * nF ) * dF[i] for i in [0,1] ]  )
    
    #aX += sum( [ stabs["GLS"] * h**2 * calL(u)[i] * calL(v)[i] * dX[i] for i in [0, 1] ] )

    for reg,i in zip( jumping_materials, jumping_materials_idx ):
        #aX += facets_G_indicator * 2*10*order*order/h*jumpv[i]*jumpu[i] * ddT[i]   
        aX += facets_G_indicator * stabs["IF"]/h*jumpv[i]*jumpu[i] * ddT[reg]   

    # dual stabilization 
    for reg,i in zip(region_list,idx_map):
        if reg != "PML":
        #if True:
            aX += stabs["Dual"] * (-1)*gradz[i]*gradw[i]*dX[reg]
        else:
            w_r = (x/r)*gradw[i][0] + (y/r)*gradw[i][1]
            w_phi = -y*gradw[i][0] + x*gradw[i][1]
            z_r = (x/r)*gradz[i][0] + (y/r)*gradz[i][1]
            z_phi = -y*gradz[i][0] + x*gradz[i][1]
            aX += stabs["Dual"] * (-1)* (  s_r * stilde_r * z_r * w_r +  (1/(r**2 * s_r *  stilde_r)) * z_phi * w_phi + z[i]*w[i]  )  * dX[reg]


    # right hand side 
    fX = LinearForm(Vh)
    
    for reg,i in zip(region_list,idx_map):
        if reg not in ["PML"]:
            fX +=  stabs["GLS"] * h**2 * 0.0 * calL(v[i],reg) * dX[reg]
            

    
    #fX +=  1e5*w[0] * exp( -100*( (x+2.5)**2 + y**2) ) * dX["host"]  

    #fX += sum( coef_f[i] * w[i] * dX[i] for i in [0, 1])
    #fX += sum( [ stabs["GLS"] * h**2 * coef_f[i] * calL(v)[i] * dX[i] for i in [0, 1] ] )

    # Setting up matrix and vector
    print("assembling linear system")
    with TaskManager():
        aX.Assemble()
        fX.Assemble()


    gfuX = GridFunction(Vh)
    gfuXh = gfuX.components
    #gfuXh[0].Set(g,definedon=mesh.Boundaries("outer"))
    
    #dirac_source_pos = (-RSource,0.0)
    #dirac_source_pos = (-3.5,0.0)
    #dirac_source_pos = (-0.5,0.0)
    mesh_pt = mesh( dirac_source_pos[0] , dirac_source_pos[1] )
    #ref_sol = FundamentalSolution(omega, dirac_source_pos[0]  ,  dirac_source_pos[1]  ,False)
    
    offset = Vp_primal.ndof + Vm_primal.ndof +  VGamma_primal.ndof 
    #fX.vec[:] = 0.0 
    #print(" Vm_primal.ndof = ",Vm_primal.ndof  )
    #print(" len(gfuXh[0].vec) = ", len(gfuXh[0].vec)  )
    for i in range(Vp_dual.ndof):
        
        gfuXh[3].vec[:] = 0.0
        gfuXh[3].vec[i] = 1.0
        if abs(gfuXh[3](mesh_pt)) > 1e-10:
            print("i = {0}, gfuX(mesh_pt)  = {1} ".format(i, gfuXh[3](mesh_pt) ))
        fX.vec.FV().NumPy()[i+offset] += gfuXh[3](mesh_pt)
     
    #for i in range(Vp_primal.ndof):
    #    
    #    gfuXh[0].vec[:] = 0.0
    #    gfuXh[0].vec[i] = 1.0
    #    val = stabs["GLS"] * h(mesh_pt)**2 * calL(gfuXh[0],"source-buffer-inner")(mesh_pt)
    #    if abs(val) > 1e-15:
    #        print("i = {0}, val  = {1} ".format(i, val ))
    #    fX.vec.FV().NumPy()[i] += val    
    #print("fX.vec.FV().NumPy()[:] =", fX.vec.FV().NumPy()[:] )
    gfuX.vec[:] = 0.0
    #fX.vec.data -= aX.mat * gfuX.vec
    print("Solving linear system")
    gfuX.vec.data += aX.mat.Inverse(Vh.FreeDofs() ,inverse=solver  )* fX.vec
    
    #mesh.UnSetPML("PML")
    if True:
        #ref_sol = (1/sigma_dic["host-outer"]) * FundamentalSolution(sqrt(k2["host-outer"])/sqrt(sigma_dic["host-outer"]), dirac_source_pos[0]  ,  dirac_source_pos[1]  ,False)
        Draw(gfuXh[0]  ,mesh,"u0")
        Draw(gfuXh[1]  ,mesh,"u1")
        #Draw(IfPos(r-b,  gfuXh[0]  , IfPos(c-r, gfuXh[1], gfuXh[0]   )  ) ,mesh,"u")
        Draw(IfPos(b-r,  gfuXh[0]  , IfPos(c-r, gfuXh[1],   gfuXh[0]   )  ) ,mesh,"u")
        Draw(  IfPos(r-c,  IfPos( RPML -r,  sqrt( (ref_sol.real - gfuXh[0].real)**2 + (ref_sol.imag - gfuXh[0].imag)**2)  , 0.0 )   , 0.0 ) , mesh,"err")
        #input("")
    

    relative_h1_errors = [] 
    for err_d in [error_inner,error_outer]:
        err_form =  (  (ref_sol.real - gfuXh[0].real)**2 + (ref_sol.imag - gfuXh[0].imag)**2)    *  dx(definedon=mesh.Materials(err_d)) 
        err_form += ( InnerProduct(ref_sol_grad.real - grad(gfuXh[0]).real, ref_sol_grad.real - grad(gfuXh[0]).real ) 
                     + InnerProduct(ref_sol_grad.imag - grad(gfuXh[0]).imag, ref_sol_grad.imag - grad(gfuXh[0]).imag )     )   *  dx(definedon=mesh.Materials(err_d)) 
        h1err = sqrt( Integrate(err_form, mesh) ) 
        h1norm = sqrt( Integrate( (  (ref_sol.real)**2 + (ref_sol.imag)**2  
                                     + InnerProduct(ref_sol_grad.real, ref_sol_grad.real) 
                                     + InnerProduct(ref_sol_grad.imag, ref_sol_grad.imag ) )   *  dx(definedon=mesh.Materials(err_d)) , mesh) ) 
        relative_h1_errors.append( h1err/ h1norm )    
        print("Relative H1 error in {0} = {1}".format(err_d , h1err/ h1norm ))
    
    #err =  (  (ref_sol.real - gfuXh[0].real)**2 + (ref_sol.imag - gfuXh[0].imag)**2)    *  dx(definedon=mesh.Materials("host-outer")) 
    #l2err = sqrt( Integrate(err, mesh) ) 
    #l2norm = sqrt( Integrate( (  (ref_sol.real)**2 + (ref_sol.imag)**2) *  dx(definedon=mesh.Materials("host-outer")) , mesh) ) 
    #print("relative l2err = ", l2err/ l2norm )
    #input("press enter to continue")
    return tuple(relative_h1_errors)

#print("Computing for stabilized method")
#SolveHybridStabilized(mesh,orders,stabs,plot=True)

#all_maxhs = np.linspace(0.8,0.1,8,endpoint=False).tolist() +  np.linspace(0.1,0.025,25,endpoint=True).tolist()
#all_maxhs = [0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.095,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]
all_maxhs = [0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.095,0.09,0.08,0.07,0.06,0.05]
#all_maxhs = [0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.095,0.09,0.085,0.08]
#all_maxhs = [0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.095,0.09,0.085,0.08,0.07,0.06]
#all_maxhs = [0.09,0.085,0.08,0.07,0.06]

#for order,stabs in zip([1,2,3],stabs_order):
for order,stabs in zip([2,3],stabs_order[1:]):
#for order,stabs in zip([3],stabs_order[2:]):
#for order in [2,3]:

    orders = {"primal-bulk": order,
          "primal-IF": order,
          "dual-bulk": order,
          "dual-IF": order}
    #orders = {"primal-bulk": order, 
    #      "primal-IF": order,
    #      "dual-bulk": 1, # does not work
    #      "dual-IF": order-1}
    
    if order == 1:
        maxhs = all_maxhs
    elif order == 2:
        maxhs = all_maxhs[:-3]
    else:
        maxhs = all_maxhs[:-6]

    err_Galerkin_outer = [ ]
    err_Galerkin_inner = []
    err_Stabilized_outer = [ ]
    err_Stabilized_inner = [ ]

    for maxh in maxhs:
        mesh = CreateMetamaterialMesh(maxh=maxh,order_geom=5)
        print("Computing for Galerkin method")
        err_inner, err_outer =  SolveMetaMaterialStandardFEM(mesh, orders["primal-bulk"] )
        err_Galerkin_inner.append(err_inner)
        err_Galerkin_outer.append(err_outer)
        print("Computing for stabilized method")
        err_inner, err_outer =  SolveHybridStabilized(mesh,orders,stabs,plot=True) 
        err_Stabilized_inner.append(err_inner)
        err_Stabilized_outer.append(err_outer)

    maxhnp = np.array(maxhs)
    plt.loglog(maxhnp, err_Galerkin_outer ,label="Galerkin-outer",marker='o')
    plt.loglog(maxhnp, err_Galerkin_inner ,label="Galerkin-inner",marker='+')
    plt.loglog(maxhnp,  err_Stabilized_outer ,label="Stabilized-outer",marker='x')
    plt.loglog(maxhnp,  err_Stabilized_inner ,label="Stabilized-inner",marker='x')
    plt.loglog(maxhs, err_Stabilized_inner[4] * maxhnp**order/maxhnp[4]**order ,color="gray",label="$\mathcal{O}(h^k)$",marker='+')
    plt.title("Relative H1-error for k={0}".format(order))
    plt.xlabel("h")
    plt.legend()
    plt.show()
    #plt.savefig("Meta-k{0}".format(order))
    #plt.clf()

    #name_str = "MetaMaterial-k{0}.dat".format(order)
    #results = [maxhnp, np.array(err_Galerkin_inner,dtype=float), np.array(err_Galerkin_outer,dtype=float), np.array( err_Stabilized_outer ,dtype=float), np.array( err_Stabilized_inner ,dtype=float) ]
    #header_str = "h Galerkin-inner Galerkin-outer Hybridstab-outer Hybridstab-inner" 

    #np.savetxt(fname ="../data/{0}".format(name_str),
    #                               X = np.transpose(results),
    #                               header = header_str,
    #                               comments = '')
