from netgen.meshing import *
from netgen.csg import *
import ngsolve


def MakeStructuredCavityMesh(nstars=1,mapping=None):
    
    mesh = Mesh()
    mesh.dim=2

    pids = []

    pos_left = -1 
    pos_right = 0 
    pos_bottom = 0 
    pos_top = 1 

    idx_dom_pos = mesh.AddRegion("plus", dim=2)
    #print("idx_dom_pos = ", idx_dom_pos)  
    # First the pos subdomain    
    x_star_pts = [pos_left + (pos_right-pos_left)*i/nstars for i in range(nstars+1) ]
    y_star_pts = [pos_bottom + (pos_top-pos_bottom)*i/nstars for i in range(nstars+1) ]

    left_pos_bnd_pts = [] 
    top_pos_bnd_pts = [] 

    Gamma_pts = []
    all_pts = []

    x_top = [] 
    for i in range(nstars):    
        if i == 0:
            p1 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] , pos_bottom, 0)))
            all_pts.append(p1)
            x_top.append(p1)
        p2 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] + 0.5*(x_star_pts[i+1] - x_star_pts[i]) , pos_bottom, 0)))
        x_top.append(p2)
        all_pts.append(p2)
        p3 = mesh.Add (MeshPoint(Pnt( x_star_pts[i+1] , pos_bottom, 0)))
        x_top.append(p3)
        all_pts.append(p3)

    bottom_pos_bnd_pts = x_top.copy()
    #print("bottom_pos_bnd_pts =", bottom_pos_bnd_pts)  

    for j in range(nstars):
        y_bottom = y_star_pts[j]
        y_top = y_star_pts[j+1]
        y_middle = y_bottom + 0.5*(y_top - y_bottom) 
       
        x_bottom = x_top
        x_top = []

        p6_prev = None
        p9_prev = None
        
        for i in range(nstars):    
            
            p1 = x_bottom[2*i]
            p2 = x_bottom[2*i+1]
            p3 = x_bottom[2*i+2]

            if i == 0:
                p4 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] , y_middle, 0)))
                all_pts.append(p4)
            else:
                p4 = p6_prev 

            p5 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] + 0.5*(x_star_pts[i+1] - x_star_pts[i]) , y_middle, 0)))

            all_pts.append(p5)
            p6 = mesh.Add (MeshPoint(Pnt( x_star_pts[i+1] , y_middle , 0)))
            all_pts.append(p6)
            p6_prev = p6

            if i == 0:
                p7 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] , y_top, 0)))
                all_pts.append(p7)
            else:
                p7 = p9_prev
            p8 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] + 0.5*(x_star_pts[i+1] - x_star_pts[i]) , y_top, 0))) 
            all_pts.append(p8)
            p9 = mesh.Add (MeshPoint(Pnt( x_star_pts[i+1] , y_top , 0)))
            all_pts.append(p9)
            p9_prev = p9

            if i == 0:
                x_top.append(p7) 
            x_top.append(p8) 
            x_top.append(p9) 

            if i == 0:
                if j == 0:
                    left_pos_bnd_pts.append(p1) 
                left_pos_bnd_pts.append(p4)
                left_pos_bnd_pts.append(p7)
            if i == nstars -1:
                if j == 0:
                    Gamma_pts.append(p3)  
                Gamma_pts.append(p6)  
                Gamma_pts.append(p9)  
            
            if j == nstars -1:
                if i == 0:
                    top_pos_bnd_pts.append(p7) 
                top_pos_bnd_pts.append(p8) 
                top_pos_bnd_pts.append(p9) 

            # bottom of star
            elspidds = []
            elspidds.append([p1,p2,p4])
            elspidds.append([p2,p5,p4])
            elspidds.append([p2,p6,p5])
            elspidds.append([p2,p3,p6])
            # top of star
            elspidds.append([p4,p8,p7])
            elspidds.append([p4,p5,p8])
            elspidds.append([p5,p6,p8])
            elspidds.append([p6,p9,p8])

            for elpids in elspidds:
                mesh.Add(Element2D( idx_dom_pos  , elpids)) 

            #print("all_pts =", all_pts)
    

    # Now the neg subdomain

    neg_left = 0 
    neg_right = 1 
    neg_bottom = 0 
    neg_top = 1 

    idx_dom_neg = mesh.AddRegion("minus", dim=2)

    x_star_pts = [neg_left + (neg_right-neg_left)*i/nstars for i in range(nstars+1) ]
    y_star_pts = [neg_bottom + (neg_top-neg_bottom)*i/nstars for i in range(nstars+1) ]

    right_neg_bnd_pts = [] 
    top_neg_bnd_pts = [] 

    x_top = []
    
    all_pts_neg = [] 
    #print("Gamma_pts = ", Gamma_pts)
    
    for i in range(nstars):    
        if i == 0:
            p1 = Gamma_pts[0] 
            x_top.append(p1)
        #else:
        #    p1 = mesh.Add (MeshPoint(Pnt( x_star_pts[i], neg_bottom, 0)))

        p2 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] + 0.5*(x_star_pts[i+1] - x_star_pts[i]) , neg_bottom, 0)))
        all_pts_neg.append(p2)
        p3 = mesh.Add (MeshPoint(Pnt( x_star_pts[i+1] , neg_bottom, 0)))
        all_pts_neg.append(p3)
        x_top.append(p2)
        x_top.append(p3)

    bottom_neg_bnd_pts = x_top.copy()

    for j in range(nstars):
        y_bottom = y_star_pts[j]
        y_top = y_star_pts[j+1]
        y_middle = y_bottom + 0.5*(y_top - y_bottom) 
       
        x_bottom = x_top
        x_top = [] 
        
        p6_prev = Gamma_pts[2*j+1]
        p9_prev = Gamma_pts[2*j+2]

        
        for i in range(nstars):    

            p1 = x_bottom[2*i]
            p2 = x_bottom[2*i+1]
            p3 = x_bottom[2*i+2]

            p4 = p6_prev 
            p5 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] + 0.5*(x_star_pts[i+1] - x_star_pts[i]) , y_middle, 0)))
            all_pts_neg.append(p5)
            p6 = mesh.Add (MeshPoint(Pnt( x_star_pts[i+1] , y_middle , 0)))
            all_pts_neg.append(p6)
            p6_prev = p6

            p7 = p9_prev
            p8 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] + 0.5*(x_star_pts[i+1] - x_star_pts[i]) , y_top, 0))) 
            
            all_pts_neg.append(p8)
            p9 = mesh.Add (MeshPoint(Pnt( x_star_pts[i+1] , y_top , 0)))
            all_pts_neg.append(p9)
            p9_prev = p9

            if i == 0:
                x_top.append(p7) 
            x_top.append(p8) 
            x_top.append(p9) 

            if i == nstars-1:
                if j == 0:
                    right_neg_bnd_pts.append(p3) 
                right_neg_bnd_pts.append(p6)
                right_neg_bnd_pts.append(p9)
             
            if j == nstars -1:
                if i == 0:
                    top_neg_bnd_pts.append(p7) 
                top_neg_bnd_pts.append(p8) 
                top_neg_bnd_pts.append(p9) 

            # bottom of star
            elspidds = []
            elspidds.append([p1,p2,p4])
            elspidds.append([p2,p5,p4])
            elspidds.append([p2,p6,p5])
            elspidds.append([p2,p3,p6])
            # top of star
            elspidds.append([p4,p8,p7])
            elspidds.append([p4,p5,p8])
            elspidds.append([p5,p6,p8])
            elspidds.append([p6,p9,p8])

            for elpids in elspidds:
                mesh.Add(Element2D( idx_dom_neg , elpids)) 


    # mesh.Add(FaceDescriptor(surfnr=1,domin=1,bc=1))
    idx_outer = mesh.AddRegion("outer", dim=1)
    idx_gamma   = mesh.AddRegion("IF", dim=1)
    
    #print("all_pts_neg = ", all_pts_neg )
    
    left_bnd = left_pos_bnd_pts
    #print("left_bnd = ", left_bnd)
    for i in range(len(left_bnd)-1):
        mesh.Add(Element1D([left_bnd[i], left_bnd[i+1]], index=idx_outer))


    bottom_bnd = bottom_pos_bnd_pts + bottom_neg_bnd_pts[1:] 
    #print("bottom_bnd =", bottom_bnd) 
    for i in range(len(bottom_bnd)-1):
        mesh.Add(Element1D([bottom_bnd[i], bottom_bnd[i+1]], index=idx_outer))
    
    right_bnd = right_neg_bnd_pts 
    #print("right_bnd = ", right_bnd)
    for i in range(len(right_bnd)-1):
        mesh.Add(Element1D([right_bnd[i], right_bnd[i+1]], index=idx_outer))

    top_bnd = top_pos_bnd_pts + top_neg_bnd_pts[1:] 
    #print("top_bnd = ", top_bnd)
    for i in range(len(top_bnd)-1):
        mesh.Add(Element1D([top_bnd[i], top_bnd[i+1]], index=idx_outer))
    
    #print("Gamma_pts = ", Gamma_pts)
    for i in range(len(Gamma_pts)-1):
        mesh.Add(Element1D([ Gamma_pts[i], Gamma_pts[i+1]], index=idx_gamma))

    mesh.Compress()       
    
    if mapping:
        for p in mesh.Points():
            x,y,z = p.p
            x,y = mapping(x,y)
            p[0] = x
            p[1] = y
            
    ngsmesh = ngsolve.Mesh(mesh)
    return ngsmesh


def MakeUnsymmetricStructuredCavityMesh(nstars=1,mapping=None):
    
    mesh = Mesh()
    mesh.dim=2

    pids = []

    pos_left = -1 
    pos_right = 0 
    pos_bottom = 0 
    pos_top = 1 

    idx_dom_pos = mesh.AddRegion("plus", dim=2)
    #print("idx_dom_pos = ", idx_dom_pos)  
    # First the pos subdomain    
    x_star_pts = [pos_left + (pos_right-pos_left)*i/nstars for i in range(nstars+1) ]
    y_star_pts = [pos_bottom + (pos_top-pos_bottom)*i/nstars for i in range(nstars+1) ]

    left_pos_bnd_pts = [] 
    top_pos_bnd_pts = [] 

    Gamma_pts = []
    all_pts = []

    x_top = [] 
    for i in range(nstars):    
        if i == 0:
            p1 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] , pos_bottom, 0)))
            all_pts.append(p1)
            x_top.append(p1)
        p2 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] + 0.5*(x_star_pts[i+1] - x_star_pts[i]) , pos_bottom, 0)))
        x_top.append(p2)
        all_pts.append(p2)
        p3 = mesh.Add (MeshPoint(Pnt( x_star_pts[i+1] , pos_bottom, 0)))
        x_top.append(p3)
        all_pts.append(p3)

    bottom_pos_bnd_pts = x_top.copy()
    #print("bottom_pos_bnd_pts =", bottom_pos_bnd_pts)  

    for j in range(nstars):
        y_bottom = y_star_pts[j]
        y_top = y_star_pts[j+1]
        y_middle = y_bottom + 0.5*(y_top - y_bottom) 
       
        x_bottom = x_top
        x_top = []

        p6_prev = None
        p9_prev = None
        
        for i in range(nstars):    
            
            p1 = x_bottom[2*i]
            p2 = x_bottom[2*i+1]
            p3 = x_bottom[2*i+2]

            if i == 0:
                p4 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] , y_middle, 0)))
                all_pts.append(p4)
            else:
                p4 = p6_prev 

            p5 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] + 0.5*(x_star_pts[i+1] - x_star_pts[i]) , y_middle, 0)))

            all_pts.append(p5)
            p6 = mesh.Add (MeshPoint(Pnt( x_star_pts[i+1] , y_middle , 0)))
            all_pts.append(p6)
            p6_prev = p6

            if i == 0:
                p7 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] , y_top, 0)))
                all_pts.append(p7)
            else:
                p7 = p9_prev
            p8 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] + 0.5*(x_star_pts[i+1] - x_star_pts[i]) , y_top, 0))) 
            all_pts.append(p8)
            p9 = mesh.Add (MeshPoint(Pnt( x_star_pts[i+1] , y_top , 0)))
            all_pts.append(p9)
            p9_prev = p9

            if i == 0:
                x_top.append(p7) 
            x_top.append(p8) 
            x_top.append(p9) 

            if i == 0:
                if j == 0:
                    left_pos_bnd_pts.append(p1) 
                left_pos_bnd_pts.append(p4)
                left_pos_bnd_pts.append(p7)
            if i == nstars -1:
                if j == 0:
                    Gamma_pts.append(p3)  
                Gamma_pts.append(p6)  
                Gamma_pts.append(p9)  
            
            if j == nstars -1:
                if i == 0:
                    top_pos_bnd_pts.append(p7) 
                top_pos_bnd_pts.append(p8) 
                top_pos_bnd_pts.append(p9) 

            # bottom of star
            elspidds = []
            elspidds.append([p1,p2,p4])
            elspidds.append([p2,p5,p4])
            elspidds.append([p2,p6,p5])
            elspidds.append([p2,p3,p6])
            # top of star
            elspidds.append([p4,p8,p7])
            elspidds.append([p4,p5,p8])
            elspidds.append([p5,p9,p8])
            elspidds.append([p5,p6,p9])

            for elpids in elspidds:
                mesh.Add(Element2D( idx_dom_pos  , elpids)) 

            #print("all_pts =", all_pts)
    

    # Now the neg subdomain

    neg_left = 0 
    neg_right = 1 
    neg_bottom = 0 
    neg_top = 1 

    idx_dom_neg = mesh.AddRegion("minus", dim=2)

    x_star_pts = [neg_left + (neg_right-neg_left)*i/nstars for i in range(nstars+1) ]
    y_star_pts = [neg_bottom + (neg_top-neg_bottom)*i/nstars for i in range(nstars+1) ]

    right_neg_bnd_pts = [] 
    top_neg_bnd_pts = [] 

    x_top = []
    
    all_pts_neg = [] 
    #print("Gamma_pts = ", Gamma_pts)
    
    for i in range(nstars):    
        if i == 0:
            p1 = Gamma_pts[0] 
            x_top.append(p1)
        #else:
        #    p1 = mesh.Add (MeshPoint(Pnt( x_star_pts[i], neg_bottom, 0)))

        p2 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] + 0.5*(x_star_pts[i+1] - x_star_pts[i]) , neg_bottom, 0)))
        all_pts_neg.append(p2)
        p3 = mesh.Add (MeshPoint(Pnt( x_star_pts[i+1] , neg_bottom, 0)))
        all_pts_neg.append(p3)
        x_top.append(p2)
        x_top.append(p3)

    bottom_neg_bnd_pts = x_top.copy()

    for j in range(nstars):
        y_bottom = y_star_pts[j]
        y_top = y_star_pts[j+1]
        y_middle = y_bottom + 0.5*(y_top - y_bottom) 
       
        x_bottom = x_top
        x_top = [] 
        
        p6_prev = Gamma_pts[2*j+1]
        p9_prev = Gamma_pts[2*j+2]

        
        for i in range(nstars):    

            p1 = x_bottom[2*i]
            p2 = x_bottom[2*i+1]
            p3 = x_bottom[2*i+2]

            p4 = p6_prev 
            p5 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] + 0.5*(x_star_pts[i+1] - x_star_pts[i]) , y_middle, 0)))
            all_pts_neg.append(p5)
            p6 = mesh.Add (MeshPoint(Pnt( x_star_pts[i+1] , y_middle , 0)))
            all_pts_neg.append(p6)
            p6_prev = p6

            p7 = p9_prev
            p8 = mesh.Add (MeshPoint(Pnt( x_star_pts[i] + 0.5*(x_star_pts[i+1] - x_star_pts[i]) , y_top, 0))) 
            
            all_pts_neg.append(p8)
            p9 = mesh.Add (MeshPoint(Pnt( x_star_pts[i+1] , y_top , 0)))
            all_pts_neg.append(p9)
            p9_prev = p9

            if i == 0:
                x_top.append(p7) 
            x_top.append(p8) 
            x_top.append(p9) 

            if i == nstars-1:
                if j == 0:
                    right_neg_bnd_pts.append(p3) 
                right_neg_bnd_pts.append(p6)
                right_neg_bnd_pts.append(p9)
             
            if j == nstars -1:
                if i == 0:
                    top_neg_bnd_pts.append(p7) 
                top_neg_bnd_pts.append(p8) 
                top_neg_bnd_pts.append(p9) 

            # bottom of star
            elspidds = []
            elspidds.append([p1,p2,p4])
            elspidds.append([p2,p5,p4])
            elspidds.append([p2,p6,p5])
            elspidds.append([p2,p3,p6])
            # top of star
            elspidds.append([p4,p8,p7])
            elspidds.append([p4,p5,p8])
            elspidds.append([p5,p9,p8])
            elspidds.append([p5,p6,p9])

            for elpids in elspidds:
                mesh.Add(Element2D( idx_dom_neg , elpids)) 


    # mesh.Add(FaceDescriptor(surfnr=1,domin=1,bc=1))
    idx_outer = mesh.AddRegion("outer", dim=1)
    idx_gamma   = mesh.AddRegion("IF", dim=1)
    
    #print("all_pts_neg = ", all_pts_neg )
    
    left_bnd = left_pos_bnd_pts
    #print("left_bnd = ", left_bnd)
    for i in range(len(left_bnd)-1):
        mesh.Add(Element1D([left_bnd[i], left_bnd[i+1]], index=idx_outer))


    bottom_bnd = bottom_pos_bnd_pts + bottom_neg_bnd_pts[1:] 
    #print("bottom_bnd =", bottom_bnd) 
    for i in range(len(bottom_bnd)-1):
        mesh.Add(Element1D([bottom_bnd[i], bottom_bnd[i+1]], index=idx_outer))
    
    right_bnd = right_neg_bnd_pts 
    #print("right_bnd = ", right_bnd)
    for i in range(len(right_bnd)-1):
        mesh.Add(Element1D([right_bnd[i], right_bnd[i+1]], index=idx_outer))

    top_bnd = top_pos_bnd_pts + top_neg_bnd_pts[1:] 
    #print("top_bnd = ", top_bnd)
    for i in range(len(top_bnd)-1):
        mesh.Add(Element1D([top_bnd[i], top_bnd[i+1]], index=idx_outer))
    
    #print("Gamma_pts = ", Gamma_pts)
    for i in range(len(Gamma_pts)-1):
        mesh.Add(Element1D([ Gamma_pts[i], Gamma_pts[i+1]], index=idx_gamma))

    mesh.Compress()       
    
    if mapping:
        for p in mesh.Points():
            x,y,z = p.p
            x,y = mapping(x,y)
            p[0] = x
            p[1] = y
            
    ngsmesh = ngsolve.Mesh(mesh)
    return ngsmesh



from netgen.geom2d import SplineGeometry
from ngsolve.TensorProductTools import MeshingParameters

def CreateUnstructuredMesh(maxh=0.4,ref_lvl=0,b=1):

    geo = SplineGeometry()
    p1,p2,p3,p4 = [ geo.AppendPoint(x,y) for x,y in [(-1,0), (0,0), (0,1), (-1,1)] ]
    p5,p6 =  [ geo.AppendPoint(x,y) for x,y in [(b,0), (b,1)] ]
    geo.Append (["line", p1, p2], leftdomain=1, rightdomain=0,bc="outer")
    geo.Append (["line", p2, p3], leftdomain=1, rightdomain=2,bc="IF")
    geo.Append (["line", p3, p4], leftdomain=1, rightdomain=0,bc="outer")
    geo.Append (["line", p4, p1], leftdomain=1, rightdomain=0,bc="outer")
    geo.Append (["line", p2, p5], leftdomain=2, rightdomain=0,bc="outer")
    geo.Append (["line", p5, p6], leftdomain=2, rightdomain=0,bc="outer")
    geo.Append (["line", p6, p3], leftdomain=2, rightdomain=0,bc="outer")
    geo.SetMaterial(1, "plus")
    geo.SetMaterial(2, "minus")
    #geo.SetDomainMaxH(2, 0.4)
    #geo.SetDomainMaxH(1, 0.1)
    #maxh = 0.002
    mp = MeshingParameters (maxh = maxh)
    #print(mp)
    #for i in range(5):
    #    mp.RestrictH (x=0.1, y=0.5-0.1*i, z=0, h=4.0*maxh )
    #    mp.RestrictH (x=-0.1, y=0.5-0.1*i, z=0, h=0.25*maxh )
    mesh = ngsolve.Mesh(geo.GenerateMesh (mp=mp))

    for i in range(ref_lvl):
        mesh.Refine()
    
    return mesh


def CreateUnstructuredMeshSubdom(maxh=0.4,ref_lvl=0,b=1,s=0.15):

    geo = SplineGeometry()
    p1,p2,p3,p4 = [ geo.AppendPoint(x,y) for x,y in [(-1,0), (0,0), (0,1), (-1,1)] ]
    p5,p6 =  [ geo.AppendPoint(x,y) for x,y in [(b,0), (b,1)] ]
    p7,p8,p9,p10 = [ geo.AppendPoint(x,y) for x,y in [(-s,0), (s,0), (s,1), (-s,1)] ]

    geo.Append (["line", p1, p7], leftdomain=1, rightdomain=0,bc="outer")
    geo.Append (["line", p7, p10], leftdomain=1, rightdomain=2,bc="none")
    geo.Append (["line", p10, p4], leftdomain=1, rightdomain=0,bc="outer")
    geo.Append (["line", p4, p1], leftdomain=1, rightdomain=0,bc="outer")

    geo.Append (["line", p7, p2], leftdomain=2, rightdomain=0,bc="outer")
    geo.Append (["line", p2, p3], leftdomain=2, rightdomain=3,bc="IF")
    geo.Append (["line", p3, p10], leftdomain=2, rightdomain=0,bc="outer")

    geo.Append (["line", p2, p8], leftdomain=3, rightdomain=0,bc="outer")
    geo.Append (["line", p8, p9], leftdomain=3, rightdomain=4,bc="none")
    geo.Append (["line", p9, p3], leftdomain=3, rightdomain=0,bc="outer")

    geo.Append (["line", p8, p5], leftdomain=4, rightdomain=0,bc="outer")
    geo.Append (["line", p5, p6], leftdomain=4, rightdomain=0,bc="outer")
    geo.Append (["line", p6, p9], leftdomain=4, rightdomain=0,bc="outer")

    geo.SetMaterial(1, "plus-outer")
    geo.SetMaterial(2, "plus-inner")
    geo.SetMaterial(3, "minus-inner")
    geo.SetMaterial(4, "minus-outer")

    #geo.SetDomainMaxH(2, 0.4)
    #geo.SetDomainMaxH(1, 0.1)
    #maxh = 0.002
    mp = MeshingParameters (maxh = maxh)
    #print(mp)
    #for i in range(5):
    #    mp.RestrictH (x=0.1, y=0.5-0.1*i, z=0, h=4.0*maxh )
    #    mp.RestrictH (x=-0.1, y=0.5-0.1*i, z=0, h=0.25*maxh )
    mesh = ngsolve.Mesh(geo.GenerateMesh (mp=mp))

    for i in range(ref_lvl):
        mesh.Refine()
    
    return mesh


#from netgen.occ import *

def CreateMetamaterialMesh(maxh=0.4,order_geom=5,domain_maxh=0.03):
    #R = 3
    a = 1.0
    b = 1.2
    c = 1.44
    RHostinner = 1.7
    RHost = 3.25
    RSource = 3.5
    RPML = 3.75


    geo = SplineGeometry()
    geo.AddCircle( (0,0), a, leftdomain=1, rightdomain=2,bc="Gamma-inner")
    #geo.AddCircle( (0,0), b, leftdomain=2, rightdomain=3,bc="Gamma-internal",maxh= domain_maxh)
    #geo.AddCircle( (0,0), c, leftdomain=3, rightdomain=4,bc="Gamma-outer", maxh= domain_maxh )
    
    geo.AddCircle( (0,0), b, leftdomain=2, rightdomain=3,bc="Gamma-internal")
    geo.AddCircle( (0,0), c, leftdomain=3, rightdomain=4,bc="Gamma-outer")


    geo.AddCircle( (0,0), RHostinner, leftdomain=4, rightdomain=5,bc="host-inner")
    geo.AddCircle( (0,0), RHost, leftdomain=5, rightdomain=6,bc="host-outer")
    geo.AddCircle( (0,0), RSource, leftdomain=6, rightdomain=7,bc="source")

    geo.AddCircle( (0,0), RPML, leftdomain=7, rightdomain=8,bc="PML_start")
    geo.AddCircle( (0,0), RPML+1.0, leftdomain=8, rightdomain=0,bc="outer")

    geo.SetMaterial(1, "object")
    geo.SetMaterial(2, "cloak-inner")
    geo.SetMaterial(3, "cloak-outer")
    geo.SetMaterial(4, "host-inner")
    geo.SetMaterial(5, "host-outer")
    geo.SetMaterial(6, "source-buffer-inner")
    geo.SetMaterial(7, "source-buffer-outer")
    geo.SetMaterial(8, "PML")

    
    #geo.SetDomainMaxH(2, domain_maxh)
    #geo.SetDomainMaxH(3, domain_maxh)
    geo.SetDomainMaxH(2, maxh/3)
    geo.SetDomainMaxH(3, maxh/3)
    
    #geo.SetDomainMaxH(4, 0.2)
    
    #geo = SplineGeometry()
    #geo.AddCircle( (0,0), RPML, leftdomain=1, rightdomain=2,bc="inner")
    #geo.AddCircle( (0,0), RPML+1, leftdomain=2, rightdomain=0,bc="outer")

    #geo.SetMaterial(1, "object")
    #geo.SetMaterial(2, "PML")


    #outer = Circle((0,0), 1.4).Face()
    #outer.edges.name = 'outerbnd'
    #inner = Circle((0,0), 1).Face()
    #inner.edges.name = 'innerbnd'
    #inner.faces.name ='inner'
    #pmlregion = outer - inner
    #pmlregion.faces.name = 'PML'
    #geo = OCCGeometry(Glue([inner, pmlregion]), dim=2)

    mesh = ngsolve.Mesh(geo.GenerateMesh (maxh=maxh,quad_dominated=False))
    mesh.Curve(order_geom)
    return mesh


#mesh = CreateMetamaterialMesh(maxh=0.4,order_geom=5)
#Draw(mesh)
#print(mesh.GetBoundaries())
#print(mesh.GetMaterials())
#input("")

#fromrix Equations Package - MEPACK 1.1.0 released ngsolve import H1
import numpy as np
from ngsolve import H1
from math import sqrt

def draw_mesh_simple(mesh,name="dummy"):
    ll, ur = (-1.0, -1.0), (1.0, 1.0)
    square = SplineGeometry()
    square.AddRectangle(ll, ur, bc=1)
    fes = H1(mesh, order=1, dirichlet=[])

    ddx = 10
    rainbow = ["cyan","white"]
    file = open("../plots/{0}.tex".format(name),"w+")
    file.write("\\documentclass{standalone} \n")
    file.write("\\usepackage{xr} \n")
    file.write("\\usepackage{tikz} \n")
    file.write("\\usepackage{pgfplots} \n")
    file.write("\\usepackage{xcolor} \n")
    file.write("\\usepackage{} \n")
    file.write("\\usetikzlibrary{shapes,arrows,shadows,snakes,calendar,matrix,spy,backgrounds,folding,calc,positioning,patterns,hobby} \n")
    file.write("\\selectcolormodel{cmyk}  \n")
    file.write("\\definecolor{orange}{cmyk}{0,0.45,0.91,0} \n")
    file.write("\\definecolor{brightblue}{cmyk}{0.92,0,0.15,0.05} \n")
    file.write("\\definecolor{richred}{cmyk}{0,1,0.62,0} \n")
    file.write("\\definecolor{yellow}{cmyk}{0,0.25,0.95,0} \n")
    file.write("\\begin{document} \n")
    file.write("\\begin{tikzpicture}[scale = 1.0,use Hobby shortcut] \n")
    file.write("\\pgfresetboundingbox \n")
    #file.write("\\path[use as bounding box,draw,black,ultra thick] (-{0},-{0}) rectangle ({0},{0}); \n".format(ddx*1.5) )

    for el in fes.Elements():
        x_coords = []
        y_coords = []
        coords = []
        for vert in el.vertices:
            vx,vy,vz = mesh.ngmesh.Points()[vert.nr+1].p
            coords.append((ddx*vx,ddx*vy))
            x_coords.append(vx)
            y_coords.append(vy)
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        if len(coords) == 3:
            file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=1] {1} -- {2} -- {3} -- cycle; \n".format("white",coords[0],coords[1],coords[2] ))
        if len(coords) == 4:
            file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=1] {1} -- {2} -- {3} -- {4} -- cycle; \n".format("white",coords[0],coords[1],coords[2],coords[3]))
    
    
    file.write("\\draw[line width=1.0mm,draw =orange] {0} -- {1}; \n".format( (0*ddx,0*ddx) ,  (0*ddx,1*ddx)  ))
    file.write("\\draw[] ( -5 , 4.5  )  node[fill=white,above]{{ \\resizebox{ .25\\linewidth}{!}{ \\textcolor{black}{$\Omega_+ $} } }}; \n")
    file.write("\\draw[] ( 5 , 4.5  )  node[fill=white,above]{{ \\resizebox{ .25\\linewidth}{!}{ \\textcolor{black}{$\Omega_- $} } }}; \n")
    file.write("\\draw[] ( 1.0 , 0.25  )  node[above]{{ \\resizebox{ .175\\linewidth}{!}{ \\textcolor{orange}{$\Gamma $} } }}; \n")
    file.write("\\end{tikzpicture} \n")
    file.write("\\end{document} \n")
    file.close()


def draw_mesh_Meta(mesh,name="dummy"):
    
    a = 1.0
    b = 1.2
    c = 1.44
    RHostinner = 1.7
    RHost = 3.25
    RSource = 3.5
    RPML = 3.75

    fes = H1(mesh, order=1, dirichlet=[])

    ddx = 10
    rainbow = ["cyan","white"]
    file = open("../plots/{0}.tex".format(name),"w+")
    file.write("\\documentclass{standalone} \n")
    file.write("\\usepackage{xr} \n")
    file.write("\\usepackage{tikz} \n")
    file.write("\\usepackage{pgfplots} \n")
    file.write("\\usepackage{xcolor} \n")
    file.write("\\usetikzlibrary{shapes,arrows,shadows,snakes,calendar,matrix,spy,backgrounds,folding,calc,positioning,patterns,hobby} \n")
    file.write("\\selectcolormodel{cmyk}  \n")
    file.write("\\definecolor{orange}{cmyk}{0,0.45,0.91,0} \n")
    file.write("\\definecolor{brightblue}{cmyk}{0.92,0,0.15,0.05} \n")
    file.write("\\definecolor{richred}{cmyk}{0,1,0.62,0} \n")
    file.write("\\definecolor{yellow}{cmyk}{0,0.25,0.95,0} \n") 
    file.write("\\definecolor{aquamarine}{rgb}{0.5, 1.0, 0.83}\n")
    file.write("\\begin{document} \n")
    file.write("\\begin{tikzpicture}[scale = 1.0,use Hobby shortcut] \n")
    file.write("\\pgfresetboundingbox \n")
    file.write("\\tikzset{cross/.style={cross out, draw, line width=4.5mm,minimum size=2*(#1-\pgflinewidth), inner sep=0pt, outer sep=0pt}} \n")
        #file.write("\\path[use as bounding box,draw,black,ultra thick] (-{0},-{0}) rectangle ({0},{0}); \n".format(ddx*1.5) )

    for el in fes.Elements():
        x_coords = []
        y_coords = []
        coords = []
        c_x = 0
        c_y = 0
        L = len(el.vertices)
        for vert in el.vertices:
            vx,vy,vz = mesh.ngmesh.Points()[vert.nr+1].p
            coords.append((ddx*vx,ddx*vy))
            c_x = c_x + vx
            c_y = c_y + vy
            x_coords.append(vx)
            y_coords.append(vy)
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        c_x = c_x / L
        c_y = c_y / L
        r_c = sqrt(c_x**2+c_y**2)

        if r_c <= a:
            el_col = "yellow"
        elif a < r_c and r_c <= b:
            el_col = "lightgray"
        elif b < r_c and r_c <= c:
            el_col = "richred"
        elif c < r_c and r_c <= RHostinner:
            el_col = "cyan"
            #el_col = "magenta"
        elif RHostinner < r_c and r_c <= RHost:
            #el_col = "cyan"
            el_col = "magenta"
        elif RHost < r_c and r_c <= RPML:
            el_col = "green!90!black"
        else:
            #el_col = "magenta"
            el_col = "aquamarine"

        if len(coords) == 3:
            file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=1] {1} -- {2} -- {3} -- cycle; \n".format(el_col,coords[0],coords[1],coords[2] ))
        if len(coords) == 4:
            file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=1] {1} -- {2} -- {3} -- {4} -- cycle; \n".format("white",coords[0],coords[1],coords[2],coords[3]))
    
    
    #file.write("\\draw[line width=1.0mm,draw =orange] {0} -- {1}; \n".format( (0*ddx,0*ddx) ,  (0*ddx,1*ddx)  ))
    file.write("\\filldraw [fill=yellow, draw=black] (50,27) rectangle (54,31); ; \n")
    file.write("\\draw[] ( 55 , 29  )  node[right]{{ \\resizebox{ 2.5\\linewidth}{!}{ \\textcolor{black}{$\!: \; \\{ r \leq a  \\} $} } }}; \n")
    
    file.write("\\filldraw [fill=lightgray, draw=black] (50,19) rectangle (54,23); ; \n")
    file.write("\\draw[] ( 55 , 21  )  node[right]{{ \\resizebox{ 3.25\\linewidth}{!}{ \\textcolor{black}{$\!: \; \\{ a \\leq r \leq b  \\} $} } }}; \n")

    file.write("\\filldraw [fill=richred, draw=black] (50,11) rectangle (54,15); ; \n")
    file.write("\\draw[] ( 55 , 13  )  node[right]{{ \\resizebox{ 3.25\\linewidth}{!}{ \\textcolor{black}{$\!: \; \\{ b \\leq r \leq c  \\} $} } }}; \n")

    file.write("\\filldraw [fill=cyan, draw=black] (50,3) rectangle (54,7); ; \n")
    file.write("\\draw[] ( 55 , 5  )  node[right]{{ \\resizebox{ 1.4\\linewidth}{!}{ \\textcolor{black}{$\!: \; \\Omega_{ \\mathrm{i} } $} } }}; \n")

    #file.write("\\filldraw [fill=cyan, draw=black] (50,9) rectangle (54,13); ; \n")
    file.write("\\filldraw [fill=magenta, draw=black] (50,-5) rectangle (54,-1); ; \n")
    file.write("\\draw[] ( 55 , -3  )  node[right]{{ \\resizebox{ 1.4\\linewidth}{!}{ \\textcolor{black}{$\!: \; \\Omega_{ \\mathrm{e} } $} } }}; \n")
    
    file.write("\\filldraw [fill=green!90!black, draw=black] (50,-13) rectangle (54,-9); ; \n")
    file.write("\\draw (52,-11) node[cross=50pt,white]{}; \n")
    #file.write("\\draw [ line width=2.5mm, draw=white] (50,-13) -- (54,-9);  \n")
    #file.write("\\draw [ line width=2.5mm, draw=white] (50,-9) -- (54,-13);  \n")
    file.write("\\draw[] ( 55 , -11  )  node[right]{{ \\resizebox{ 2.0\\linewidth}{!}{ \\textcolor{black}{ Source } } }}; \n")

    #file.write("\\filldraw [fill=magenta, draw=black] (50,-7) rectangle (54,-3); ; \n")
    file.write("\\filldraw [fill=aquamarine, draw=black] (50,-21) rectangle (54,-17); ; \n")
    file.write("\\draw[] ( 55 , -19  )  node[right]{{ \\resizebox{ 1.5\\linewidth}{!}{ \\textcolor{black}{ PML } } }}; \n")
    
    file.write("\\draw (-35,0) node[cross=50pt,white]{}; \n")
        #file.write("\\draw[] ( 5 , 4.5  )  node[fill=white,above]{{ \\resizebox{ .25\\linewidth}{!}{ \\textcolor{black}{$\Omega_- $} } }}; \n")
    #file.write("\\draw[] ( 1.0 , 0.25  )  node[above]{{ \\resizebox{ .175\\linewidth}{!}{ \\textcolor{orange}{$\Gamma $} } }}; \n")
    
    file.write("\\end{tikzpicture} \n")
    file.write("\\end{document} \n")
    file.close()


#mesh_unstructured = CreateUnstructuredMesh(maxh=0.2,ref_lvl=0,b=1)
#draw_mesh_simple(mesh_unstructured ,name="mesh-unstructured-cavity")


#mesh_unstructured = MakeStructuredCavityMesh(nstars=5)
#draw_mesh_simple(mesh_unstructured ,name="mesh-structured-symmetric-cavity")

mesh = CreateMetamaterialMesh(maxh=0.6,order_geom=5,domain_maxh=0.03)
draw_mesh_Meta(mesh,name="mesh-MetaMaterial")


def CreateMetamaterialMeshOld(maxh=0.4,order_geom=5,domain_maxh=0.05):
    #R = 3
    a = 1.0
    b = 1.2
    c = 1.44
    RPML = 3.75

    geo = SplineGeometry()
    geo.AddCircle( (0,0), a, leftdomain=1, rightdomain=2,bc="Gamma-inner")
    geo.AddCircle( (0,0), b, leftdomain=2, rightdomain=3,bc="Gamma-internal")
    geo.AddCircle( (0,0), c, leftdomain=3, rightdomain=4,bc="Gamma-outer")
    
    geo.AddCircle( (0,0), RPML, leftdomain=4, rightdomain=5,bc="PML_start")
    geo.AddCircle( (0,0), RPML+1.0, leftdomain=5, rightdomain=0,bc="outer")

    geo.SetMaterial(1, "object")
    geo.SetMaterial(2, "cloak-inner")
    geo.SetMaterial(3, "cloak-outer")
    geo.SetMaterial(4, "host")
    geo.SetMaterial(5, "PML")

    geo.SetDomainMaxH(2, domain_maxh)
    geo.SetDomainMaxH(3, domain_maxh)
    geo.SetDomainMaxH(4, 0.1)
    
    #geo = SplineGeometry()
    #geo.AddCircle( (0,0), RPML, leftdomain=1, rightdomain=2,bc="inner")
    #geo.AddCircle( (0,0), RPML+1, leftdomain=2, rightdomain=0,bc="outer")

    #geo.SetMaterial(1, "object")
    #geo.SetMaterial(2, "PML")


    #outer = Circle((0,0), 1.4).Face()
    #outer.edges.name = 'outerbnd'
    #inner = Circle((0,0), 1).Face()
    #inner.edges.name = 'innerbnd'
    #inner.faces.name ='inner'
    #pmlregion = outer - inner
    #pmlregion.faces.name = 'PML'
    #geo = OCCGeometry(Glue([inner, pmlregion]), dim=2)

    mesh = ngsolve.Mesh(geo.GenerateMesh (maxh=maxh,quad_dominated=False))
    mesh.Curve(order_geom)
    return mesh
