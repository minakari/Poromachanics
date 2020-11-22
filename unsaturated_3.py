
# coding: utf-8

# In[72]:


from __future__ import print_function
from fenics import *
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import pylab  
import random


# In[73]:


parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True,                "eliminate_zeros": True,                "precompute_basis_const": True,                "precompute_ip_const": True}


# In[74]:


N = 50 # mesh density
domain = Rectangle(Point(-70.0,-100.0), Point(70.0, 40.0)) - Rectangle(Point(-1.0,0.0), Point(1.0, 40.95))
mesh = generate_mesh(domain, N)
plot ( mesh, title = 'mesh' )
mesh_points=mesh.coordinates()
print(mesh_points.shape)


# In[75]:


Dphi = Constant(1.0)
E = 100000000
nu = 0.3
lmbda = Constant(E*nu/((1+nu)*(1-2*nu))) 
mu = Constant(E/2/(1+nu))


# In[76]:


d = 1 # interpolation degree
Vue = VectorElement('CG', mesh.ufl_cell(), d) # displacement finite element
Vp1e = FiniteElement('CG', mesh.ufl_cell(), d) # concentration finite element
Vp2e = FiniteElement('CG', mesh.ufl_cell(), d) # concentration finite element
V = FunctionSpace(mesh, MixedElement([Vue, Vp1e, Vp2e]))

# Boundary conditions
def inner_b(x, on_boundary):
    return near(x[1], -0.0) and on_boundary
def inner_l(x, on_boundary):
    return near(x[0], -1.0) and on_boundary
def inner_r(x, on_boundary):
    return near(x[0], 1.0) and on_boundary
def bottom(x, on_boundary):
    return near(x[1], -100.0) and on_boundary
def left(x, on_boundary):
    return near(x[0], -70.0) and on_boundary
def right(x, on_boundary):
    return near(x[0], 70.0) and on_boundary
def top(x, on_boundary):
    return near(x[1], 40.0) and on_boundary
bc1 = DirichletBC(V.sub(1), Constant(1.), left)
bc2 = DirichletBC(V.sub(1), Constant(1.), right)
bc3 = DirichletBC(V.sub(1), Constant(1.), bottom)
bc4 = DirichletBC(V.sub(2), Constant(0.), left)
bc5 = DirichletBC(V.sub(2), Constant(0.), right)
bc6 = DirichletBC(V.sub(2), Constant(0.), bottom)
bc7 = DirichletBC(V.sub(0).sub(0), Constant(0.), left)
bc8 = DirichletBC(V.sub(0).sub(0), Constant(0.), right)
bc9 = DirichletBC(V.sub(0).sub(0), Constant(0.), bottom)
bc10 = DirichletBC(V.sub(0).sub(1), Constant(0.), left)
bc11 = DirichletBC(V.sub(0).sub(1), Constant(0.), right)
bc12 = DirichletBC(V.sub(0).sub(1), Constant(0.), bottom)

bc13 = DirichletBC(V.sub(1), Constant(1.), inner_b)


bcs = [bc1,bc2,bc3,bc7,bc8,bc9,bc10,bc11,bc12,bc13]


# In[77]:


# Defining multiple Neumann boundary conditions 
mf = MeshFunction("size_t", mesh, 1)
mf.set_all(0) # initialize the function to zero
class inner_b(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0) and on_boundary
inner_b = inner_b() # instantiate it
inner_b.mark(mf, 1)
ds = ds(subdomain_data = mf) 


# In[78]:


U_ = TestFunction(V)
(u_, P1_, P2_) = split(U_)  
dU = TrialFunction(V)
(du, dP1, dP2) = split(dU) 
U = Function(V)
(u, P1, P2) = split(U)  # P1 = \rho_aR and P2 = \rho_wR


# In[79]:


d = u.geometric_dimension()
I = Identity(d)             # Identity tensor

F = I + grad(u)             # Deformation gradient
C = F.T * F                # Elastic right Cauchy-Green tensor
Finv = inv(F)
k = 0.1            # Permeability of soil
phi = 0.9  # \phi_s
R = 8.32
T = 298
gamma_w = 1000

# Invariants of deformation tensors
Ic = tr(C)
J1 = det(F)

phi_a = 1-phi-P2/(J1*gamma_w)
phi_w = P2/(J1*gamma_w)
gamma_a = P1/(J1*phi_a)

F1 = phi* (mu * F - mu * Finv.T + lmbda*ln(J1) * Finv.T) - (1-phi)/phi_a *(P1*R*T* Finv.T) # S

F1 = (1/J1)*F.T * F1

J_a = k*R*T*phi_a*grad(gamma_a) # P_p1
J_w = k*R*T*phi_w*grad(gamma_a) # P_p2

#  Define time things.
dt = Constant(0.01)

P_init = Expression ( "1.0", degree = 0 )
P_old = project ( P_init, V.sub(1).collapse())

P_init2 = Expression ( "0.0", degree = 0 )
P_old2 = project ( P_init2, V.sub(1).collapse())

y_BC = Expression(("0.0", "0.0"),degree=0)
u = project(y_BC,V.sub(0).collapse())

f = Expression(("0.0", "0.0"),degree=0) 
g_a = Expression(("0.0"),degree=0)

g_w = Expression(("200.0"),degree=0)

mech_form = inner(F1, grad(u_))*dx + inner(140*f,u_)*dx
p_form1 = 1*J1*inner(((J_a) + 6.5*f), grad(P1_))*dx + ( P1 - P_old )/dt * P1_ * dx - J1*g_a*P1_*ds(1)
p_form2 = 1*J1*inner(((J_w) + 6.5*f), grad(P2_))*dx + ( P2 - P_old2 )/dt * P2_ * dx - J1*g_w*P2_*ds(1)

F = mech_form + p_form1 + p_form2 
J = derivative(F, U, dU)


# In[80]:


problem = NonlinearVariationalProblem(F, U, bcs, J)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-5
prm['newton_solver']['maximum_iterations'] = 15
prm['newton_solver']['relaxation_parameter'] = 1.0


# In[81]:


m = []
Nincr = 40
t = np.linspace(0.5 ,200, Nincr+1)
U = Function(V)
for (i, dti) in enumerate(np.array(t)):
    print("Increment " + str(i+1))
    dt.assign(dti)
    solver.solve()
    (u1, P11,P22) = U.split()
    P_old = P11.copy()


# In[82]:


M=plot(P2, title= 'water concentration')
plt.colorbar(M)  
filename = 'Phidistribution.png'
plt.savefig ( filename )
plt.show()


# In[84]:


W = TensorFunctionSpace(mesh, "Lagrange", 2)

por_p = project(1/det(inv(Finv))*P1*R*T*((1-phi)/phi_a)*Identity(2) , W)

M5 = plot(por_p[1,1])
plt.colorbar(M5)
plt.show()

