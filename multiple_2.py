
# coding: utf-8

# In[43]:


from __future__ import print_function
from fenics import *
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import pylab  
import random


# In[44]:


parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True,                "eliminate_zeros": True,                "precompute_basis_const": True,                "precompute_ip_const": True}


# In[45]:


N = 50 # mesh density
domain = Rectangle(Point(-70.0,-100.0), Point(70.0, 40.0)) - Rectangle(Point(-1.0,0.0), Point(1.0, 40.95))
mesh = generate_mesh(domain, N)
plot ( mesh, title = 'mesh' )
mesh_points=mesh.coordinates()
print(mesh_points.shape)


# In[46]:


E = 100000000
nu = 0.3
lmbda = Constant(E*nu/((1+nu)*(1-2*nu))) 
mu = Constant(E/2/(1+nu))


# In[47]:


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
bc1 = DirichletBC(V.sub(1), Constant(0.), left)
bc2 = DirichletBC(V.sub(1), Constant(0.), right)
bc3 = DirichletBC(V.sub(1), Constant(0.), bottom)
bc4 = DirichletBC(V.sub(2), Constant(0.), left)
bc5 = DirichletBC(V.sub(2), Constant(0.), right)
bc6 = DirichletBC(V.sub(2), Constant(0.), bottom)
bc7 = DirichletBC(V.sub(0).sub(0), Constant(0.), left)
bc8 = DirichletBC(V.sub(0).sub(0), Constant(0.), right)
bc9 = DirichletBC(V.sub(0).sub(0), Constant(0.), bottom)
bc10 = DirichletBC(V.sub(0).sub(1), Constant(0.), left)
bc11 = DirichletBC(V.sub(0).sub(1), Constant(0.), right)
bc12 = DirichletBC(V.sub(0).sub(1), Constant(0.), bottom)

bcs = [bc1,bc2,bc3,bc4,bc5,bc6,bc7,bc8,bc9,bc10,bc11,bc12]


# In[48]:


# Defining multiple Neumann boundary conditions 
mf = MeshFunction("size_t", mesh, 1)
mf.set_all(0) # initialize the function to zero
class inner_b(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0) and on_boundary
inner_b = inner_b() # instantiate it
inner_b.mark(mf, 1)
ds = ds(subdomain_data = mf) 


# In[49]:


U_ = TestFunction(V)
(u_, P1_, P2_) = split(U_)
dU = TrialFunction(V)
(du, dP1, dP2) = split(dU) 
U = Function(V)
(u, P1, P2) = split(U)

Un = Function(V)
(un, Pn1, Pn2) = split(Un)


# In[50]:


d = u.geometric_dimension()
I = Identity(d)             # Identity tensor

F = I + grad(u)             # Deformation gradient
C = F.T * F                # Elastic right Cauchy-Green tensor
Finv = inv(F)
k = 0.01            # Permeability of soil
k2 = 0.02            # Permeability of soil
phi = 0.8
R = 8.32
T = 298

# Invariants of deformation tensors
Ic = tr(C)
J1 = det(F)

ph2 = (1-phi)/(1+(P2+0.001)/(P1+0.001))
ph3 = ph2*(P2+0.001)/(P1+0.001)

F1 = phi* (mu * F - mu * Finv.T + lmbda*ln(J1) * Finv.T) - (P1*R*T* Finv.T) - (P2*R*T* Finv.T)# S

F1 = (1/J1)*F.T * F1

par_psi_phi1 = k*R*T*ph2*grad(P1/(ph2*J1)) # P_p1
par_psi_phi2 = k2*R*T*ph3*grad(P2/(ph3*J1)) # P_p2

#  Define time things.
Tt, num_steps = 550.0 , 30                   
dt = Tt / (1.0*num_steps)

P_init = Expression ( "0.1", degree = 0 )
Pn1 = project ( P_init, V.sub(1).collapse())

P_init1 = Expression ( "0.0", degree = 0 )
Pn2 = project ( P_init1, V.sub(1).collapse())

y_BC = Expression(("0.0", "0.0"),degree=0)
u = project(y_BC,V.sub(0).collapse())

f = Expression(("0.0", "0.0"),degree=0)
g1 = Expression(("0.0"),degree=0)
g2 = Expression(("10.0"),degree=0)

mech_form = inner(F1, grad(u_))*dx + inner(140*f,u_)*dx
p_form1 = 1*J1*inner(((par_psi_phi1) + 6.5*f), grad(P1_))*dx + ( P1 - Pn1 )*J1/dt * P1_ * dx - J1*g1*P1_*ds(1)
p_form2 = 1*J1*inner(((par_psi_phi2) + 6.5*f), grad(P2_))*dx + ( P2 - Pn2 )*J1/dt * P2_ * dx - J1*g2*P2_*ds(1)

F = mech_form + p_form1 + p_form2 
J = derivative(F, U, dU)


# In[51]:


problem = NonlinearVariationalProblem(F, U, bcs, J)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-5
prm['newton_solver']['relative_tolerance'] = 1E-3
prm['newton_solver']['maximum_iterations'] = 6
prm['newton_solver']['relaxation_parameter'] = 1.0 


# In[52]:


t = 0
for n in range(num_steps):
	print('n=',n)
	t += dt
	solver.solve()      
	(u1, P12, P22) = U.split()
	assign(Un,U)


# In[57]:


W = TensorFunctionSpace(mesh, "Lagrange", 2)

por_p = project((1/det(inv(Finv)))*P1*R*T*Identity(2) , W)

M5 = plot(por_p[1,1])
plt.colorbar(M5)
plt.show()


# In[58]:


W = TensorFunctionSpace(mesh, "Lagrange", 2)

por_p = project((1/det(inv(Finv)))*P2*R*T*Identity(2) , W)

M5 = plot(por_p[1,1])
plt.colorbar(M5)
plt.show()

