"""Core physics engine for 2D molecular simulation."""

import numpy as np


# Particle initialization
def init_state(seed=42, N=12, L=10.0, v_max=4.4, margin=0.8, dtype=np.float64):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(margin, L - margin, size=(N, 2)).astype(dtype)
    vel = rng.uniform(-v_max, v_max, size=(N, 2)).astype(dtype)
    mass = np.ones(N, dtype=dtype)
    return pos, vel, mass


# Pairwise displacements, distances, and neighbor mask
def pairwise_distances(pos, rc=1.5):
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1, dtype=pos.dtype))
    neighbors = (dist > 1e-14) & (dist <= rc)
    return d, dist, neighbors


# Hookean spring pair forces
def pair_forces(pos, k=4000.0, r0=0.70, rc=0.70):
    scalar = pos.dtype.type
    d, dist, neighbors = pairwise_distances(pos, rc=rc)
    denom = dist[:, :, None]
    u = np.divide(
        d,
        denom,
        out=np.zeros_like(d),
        where=denom > 1e-14,
    )
    fmag = -scalar(k) * (dist - scalar(r0)) * neighbors
    return np.sum(fmag[:, :, None] * u, axis=1, dtype=pos.dtype)


# Total force combining spring forces and optional viscous damping
def total_forces(pos, vel=None, mass=None, gamma=0.0, k=4000.0, r0=0.70, rc=0.70, pair_force_fn=None):
    if pair_force_fn is not None:
        F_spring = pair_force_fn(pos, k=k, r0=r0, rc=rc)
    else:
        F_spring = pair_forces(pos, k=k, r0=r0, rc=rc)
    if vel is None or gamma == 0.0:
        return F_spring
    return F_spring - vel.dtype.type(gamma) * vel


# Kinetic energy
def kinetic_energy(vel, mass=None):
    scalar = vel.dtype.type
    if mass is None:
        mass = np.ones(len(vel), dtype=vel.dtype)
    return scalar(0.5) * np.sum(mass[:, None] * (vel * vel), dtype=vel.dtype)


# Potential energy
def potential_energy(pos, k=4000.0, r0=0.70, rc=0.70):
    scalar = pos.dtype.type
    _, dist, neighbors = pairwise_distances(pos, rc=rc)
    pair_u = scalar(0.5) * scalar(k) * ((dist - scalar(r0)) ** 2) * neighbors
    return scalar(0.5) * np.sum(pair_u, dtype=pos.dtype)


# Specular wall reflections
def walls_reflect(pos, vel, L=10.0, radius=0.0):
    scalar = pos.dtype.type
    r_val = scalar(radius)
    L_val = scalar(L) - r_val
    for dim in range(pos.shape[1]):
        low = pos[:, dim] < r_val
        vel[low, dim] = np.abs(vel[low, dim])
        pos[low, dim] = r_val

        high = pos[:, dim] > L_val
        vel[high, dim] = -np.abs(vel[high, dim])
        pos[high, dim] = L_val
    return pos, vel


# Particle-particle elastic collision handling
def particle_collisions(pos, vel, radius=0.0):
    if radius <= 0.0:
        return pos, vel
    N_p = len(pos)
    min_dist = 2.0 * radius
    for i in range(N_p):
        for j in range(i + 1, N_p):
            delta = pos[i] - pos[j]
            dist = np.hypot(delta[0], delta[1])
            if 0.0 < dist < min_dist:
                n = delta / dist
                overlap = min_dist - dist
                pos[i] += 0.5 * overlap * n
                pos[j] -= 0.5 * overlap * n
                v_rel = vel[i] - vel[j]
                vn = np.dot(v_rel, n)
                if vn < 0.0:
                    vel[i] -= vn * n
                    vel[j] += vn * n
    return pos, vel


# Forward Euler integrator step
def step_euler(pos, vel, mass, dt, L=10.0, gamma=0.0, k=4000.0, r0=0.70, rc=0.70, radius=0.0):
    scalar = pos.dtype.type
    dt_val = scalar(dt)
    F = total_forces(pos, vel=vel, mass=mass, gamma=gamma, k=k, r0=r0, rc=rc)
    acc = F / mass[:, None]
    pos_new = pos + vel * dt_val
    vel_new = vel + acc * dt_val
    pos_new, vel_new = walls_reflect(pos_new, vel_new, L, radius=radius)
    return pos_new, vel_new


# Velocity Verlet integrator step
def step_verlet(
    pos, vel, mass, dt, L=10.0, gamma=0.0, k=4000.0, r0=0.70, rc=0.70,
    radius=0.0, F_prev=None, force_previous=None, pair_force_fn=None
):
    if isinstance(L, np.ndarray):
        if force_previous is None and F_prev is None:
            F_prev = L
        L = 10.0
    if F_prev is None and force_previous is not None:
        F_prev = force_previous

    scalar = pos.dtype.type
    dt_val = scalar(dt)
    half = scalar(0.5)

    if F_prev is None:
        F_prev = total_forces(
            pos, vel=vel, mass=mass, gamma=gamma, k=k, r0=r0, rc=rc, pair_force_fn=pair_force_fn
        )
    a_prev = F_prev / mass[:, None]
    vel_half = vel + half * a_prev * dt_val
    pos_new = pos + vel_half * dt_val
    pos_new, vel_half = walls_reflect(pos_new, vel_half, L, radius=radius)

    F_new = total_forces(
        pos_new, vel=vel_half, mass=mass, gamma=gamma, k=k, r0=r0, rc=rc, pair_force_fn=pair_force_fn
    )
    a_new = F_new / mass[:, None]
    vel_new = vel_half + half * a_new * dt_val
    return pos_new, vel_new, F_new


# Full trajectory simulation runner
def run_sim(
    dt, T_max, method="verlet", seed=42, gamma=0.0, k=15.0, r0=1.5, rc=1.5,
    L=10.0, N=12, radius=0.0, save_pos=False, dtype=np.float64, v_max=4.4
):
    pos, vel, mass = init_state(seed=seed, N=N, L=L, v_max=v_max, margin=0.8, dtype=dtype)
    n_steps = int(round(T_max / dt))

    t_arr = np.linspace(0.0, n_steps * dt, n_steps + 1, dtype=np.float64)
    K_arr = np.zeros(n_steps + 1, dtype=dtype)
    U_arr = np.zeros(n_steps + 1, dtype=dtype)
    E_arr = np.zeros(n_steps + 1, dtype=dtype)
    v_mean = np.zeros(n_steps + 1, dtype=dtype)
    pos_hist = np.zeros((n_steps + 1, N, 2), dtype=dtype) if save_pos else None

    K_arr[0] = kinetic_energy(vel, mass)
    U_arr[0] = potential_energy(pos, k=k, r0=r0, rc=rc)
    E_arr[0] = K_arr[0] + U_arr[0]
    v_mean[0] = np.mean(np.linalg.norm(vel, axis=1))
    if save_pos:
        pos_hist[0] = pos.copy()

    F_prev = None
    for s in range(1, n_steps + 1):
        if method == "euler":
            pos, vel = step_euler(
                pos, vel, mass, dt, L=L, gamma=gamma, k=k, r0=r0, rc=rc, radius=radius
            )
        else:
            pos, vel, F_prev = step_verlet(
                pos, vel, mass, dt, L=L, gamma=gamma, k=k, r0=r0, rc=rc, radius=radius, F_prev=F_prev
            )

        K_arr[s] = kinetic_energy(vel, mass)
        U_arr[s] = potential_energy(pos, k=k, r0=r0, rc=rc)
        E_arr[s] = K_arr[s] + U_arr[s]
        v_mean[s] = np.mean(np.linalg.norm(vel, axis=1))
        if save_pos:
            pos_hist[s] = pos.copy()

    rel_err = np.abs(E_arr - E_arr[0]) / E_arr[0]
    eps_E_max = float(np.max(rel_err))
    eps_E_rms = float(np.sqrt(np.mean(rel_err ** 2)))

    return {
        "t": t_arr, "K": K_arr, "U": U_arr, "E": E_arr,
        "rel_err": rel_err, "v_mean": v_mean, "pos_hist": pos_hist,
        "eps_E_max": eps_E_max, "eps_E_rms": eps_E_rms
    }
