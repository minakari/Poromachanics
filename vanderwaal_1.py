
# coding: utf-8

# In[69]:


from fenics import *
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import pylab  
import random


# In[70]:


parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True,                "eliminate_zeros": True,                "precompute_basis_const": True,                "precompute_ip_const": True}


# In[71]:


N = 50 # mesh density
domain = Rectangle(Point(-70.0,-100.0), Point(70.0, 40.0)) - Rectangle(Point(-2.0,0.0), Point(2.0, 40.95))#-Circle(Point(0., 0.), R,100)
mesh = generate_mesh(domain, N)
plot ( mesh, title = 'mesh' )
mesh_points=mesh.coordinates()
print(mesh_points.shape)


# In[72]:


Dphi = Constant(1.0)
E = 100000000
nu = 0.3
lmbda = Constant(E*nu/((1+nu)*(1-2*nu)))
mu = Constant(E/2/(1+nu))


# In[73]:


d = 1 # interpolation degree
Vue = VectorElement('CG', mesh.ufl_cell(), d) # displacement finite element
Vpe = FiniteElement('CG', mesh.ufl_cell(), d) # temperature finite element
V = FunctionSpace(mesh, MixedElement([Vue, Vpe]))

def inner_b(x, on_boundary):
    return near(x[1], -0.0) and on_boundary
def inner_l(x, on_boundary):
    return near(x[0], -2.0) and on_boundary
def inner_r(x, on_boundary):
    return near(x[0], 2.0) and on_boundary
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
bc4 = DirichletBC(V.sub(1), Constant(0.), top)
bc5 = DirichletBC(V.sub(1), Constant(0.), inner_l)
bc6 = DirichletBC(V.sub(1), Constant(0.), inner_r)
bc7 = DirichletBC(V.sub(0).sub(0), Constant(0.), left)
bc8 = DirichletBC(V.sub(0).sub(0), Constant(0.), right)
bc9 = DirichletBC(V.sub(0).sub(0), Constant(0.), bottom)
bc10 = DirichletBC(V.sub(0).sub(1), Constant(0.), left)
bc11 = DirichletBC(V.sub(0).sub(1), Constant(0.), right)
bc12 = DirichletBC(V.sub(0).sub(1), Constant(0.), bottom)

bc13 = DirichletBC(V.sub(1), Dphi, inner_b)
bcs = [bc1,bc2,bc3,bc7,bc8,bc9,bc10,bc11,bc12]


# In[74]:


# Defining multiple Neumann boundary conditions 
mf = MeshFunction("size_t", mesh, 1)
mf.set_all(0) # initialize the function to zero
class inner_b(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0) and on_boundary
inner_b = inner_b() # instantiate it
inner_b.mark(mf, 1)
ds = ds(subdomain_data = mf)


# In[75]:


U_ = TestFunction(V)
(u_, P_) = split(U_)
dU = TrialFunction(V)
(du, dP) = split(dU) 
U = Function(V)
(u, P) = split(U) 

Un = Function(V)
(un, Pn) = split(Un)


# In[76]:


d = u.geometric_dimension()
I = Identity(d)             # Identity tensor

F = I + grad(u)             # Deformation gradient
C = F.T * F                # Elastic right Cauchy-Green tensor
Finv = inv(F)
k = 0.01
phi = .2
R = 8.32 
#####################
T = 285
a = 0.364
b = 0.00004267

# Invariants of deformation tensors
Ic = tr(C)
J = det(F)

gamma = P/(J*phi) 

F1 = (1-phi)* (mu * F - mu * Finv.T + lmbda*ln(J) * Finv.T) - 1*P*(R*T*Finv.T/(1-(gamma)*b) - a*gamma*Finv.T) # S
par_psi_phi = phi*k*(R*T*grad(gamma)/(1-b*gamma) + b*R*T*gamma*grad(gamma)/(1-b*gamma)**2 - 2*a*gamma*grad(gamma)) # J

F1 = (1/J)*F.T *F1

#  Define time things.
Tt, num_steps = 20.0 , 300                   
dt = Tt / (1.0*num_steps)

P_init = Expression ( "0.0", degree = 0 )
P_old = project ( P_init, V.sub(1).collapse())

y_BC = Expression(("0.0", "0.0"),degree=0)
u = project(y_BC,V.sub(0).collapse())

f = Expression(("0.0", "0.0"),degree=0) 
g = Expression(("10000.0"),degree=0)

mech_form = inner(F1, grad(u_))*dx + inner(140*f,u_)*dx
phi_form = J*inner(((par_psi_phi) + 6.5*f), grad(P_))*dx + ( P - Pn )/dt * P_ * dx - J*g*P_*ds(1)
F = mech_form + phi_form
J = derivative(F, U, dU)


# In[77]:


problem = NonlinearVariationalProblem(F, U, bcs, J)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-1
prm['newton_solver']['relative_tolerance'] = 1E-1
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['relaxation_parameter'] = 1.0


# In[78]:



t = 0
for n in range(num_steps):
	print('n=',n)
	t += dt
	solver.solve()   
	(u1, P1) = U.split()
	assign(Un,U)


# In[80]:


W = TensorFunctionSpace(mesh, "Lagrange", 2)
gamma = P/(det(inv(Finv))*phi)
por_p = project((gamma)*(R*T/(1-(gamma)*b) - a*gamma)*Identity(2) , W)

M5 = plot(por_p[1,1], title='$P_p$')
plt.colorbar(M5)
plt.show()

P0 = por_p[1,1]

tol = 0.00001  # avoid hitting points outside the domain
yy = np.linspace(0.0 + tol, 100.0 - tol, 101)
points = [(0. , -yy_ ) for yy_ in yy]  # 2D points
u_line = np.array([P0(point) for point in points])
plt.plot(yy, u_line, 'k', linewidth=2)
plt.ylabel('$P_p$ value')
plt.xlabel('distance')
plt.show()


# In[81]:


W = TensorFunctionSpace(mesh, "Lagrange", 2)
gamma = P/(det(inv(Finv))*phi)
por_p2 = project(gamma*Identity(2) , W)

M5 = plot(por_p2[1,1], title='$gamma$')
plt.colorbar(M5)
plt.show()

P1 = por_p2[1,1]

tol = 0.00001  # avoid hitting points outside the domain
yy = np.linspace(0.0 + tol, 60.0 - tol, 101)
points = [(0. , -yy_ ) for yy_ in yy]  # 2D points
u_line1 = np.array([P1(point) for point in points])
u_line2 = 1/u_line1
print(np.min(u_line2))
u_line = np.array([P0(point) for point in points])
plt.plot(u_line2, u_line , 'k', linewidth=2)
plt.ylabel('pressure')
plt.xlabel('gamma')
plt.show()

