import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
nx, ny = 50, 50  # Number of grid points in x and y
lx, ly = 2.0, 2.0  # Physical dimensions of the domain
dx, dy = lx / (nx - 1), ly / (ny - 1)  # Grid spacing

# Simulation parameters
dt = 0.01  # Time step
nt = 100  # Number of time steps
rho = 1.0  # Density
nu = 0.1  # Kinematic viscosity

# Boundary conditions (example: no-slip boundaries)
def apply_boundary_conditions(u, v):
    u[0, :] = u[-1, :] = 0
    u[:, 0] = u[:, -1] = 0
    v[0, :] = v[-1, :] = 0
    v[:, 0] = v[:, -1] = 0
    return u, v

def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt *
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                             (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, b, dx, dy):
    pn = np.empty_like(p)
    pn[:] = p[:]
    for q in range(nt):
        pn[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                           (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        # Boundary conditions for pressure
        pn[:, -1] = pn[:, -2]  # dp/dy = 0 at y = 2
        pn[0, :] = pn[1, :]  # dp/dx = 0 at x = 0
        pn[:, 0] = pn[:, 1]  # dp/dy = 0 at y = 0
        pn[-1, :] = 0  # p = 0 at x = 2

        p = pn
    return p

def velocity_update(u, v, p, rho, nu, dt, dx, dy):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    un[:] = u[:]
    vn[:] = v[:]

    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                     (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                     (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt / (2 * rho * dx) *
                     (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (dt / dx**2 *
                           (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                           dt / dy**2 *
                           (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                     (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                     (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                     dt / (2 * rho * dy) *
                     (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * (dt / dx**2 *
                           (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                           dt / dy**2 *
                           (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

    u, v = apply_boundary_conditions(u, v)
    return u, v

def main_simulation():
    # Initialize fields
    u = np.zeros((nx, ny))  # Velocity field in x-direction
    v = np.zeros((nx, ny))  # Velocity field in y-direction
    p = np.zeros((nx, ny))  # Pressure field
    b = np.zeros((nx, ny))  # Source term in the pressure Poisson equation

    # Initial conditions
    u[:, :] = 0
    v[:, :] = 0
    p[:, :] = 0

    u, v = apply_boundary_conditions(u, v)

    for n in range(nt):
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, b, dx, dy)
        u, v = velocity_update(u, v, p, rho, nu, dt, dx, dy)
        
        if n % 10 == 0:  # Plot every 10 time steps
            plot_fields(u, v, p)

def plot_fields(u, v, p):
    plt.figure(figsize=(10, 8))

    plt.subplot(1, 2, 1)
    plt.contourf(u, cmap='jet')
    plt.colorbar()
    plt.title('Velocity Field (u)')

    plt.subplot(1, 2, 2)
    plt.contourf(v, cmap='jet')
    plt.colorbar()
    plt.title('Velocity Field (v)')

    plt.figure()
    plt.contourf(p, cmap='jet')
    plt.colorbar()
    plt.title('Pressure Field')
    
    plt.show()

# Run the simulation
main_simulation()


