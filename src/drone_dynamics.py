import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment


def assign_targets(initial_positions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    dists = np.linalg.norm(initial_positions[:, None] - targets[None], axis=-1) ** 2
    row_ind, col_ind = linear_sum_assignment(dists)
    return targets[col_ind]


def repulsion_force(positions: np.ndarray, k_rep: float, r_safe: float) -> np.ndarray:
    N = positions.shape[0]
    forces = np.zeros_like(positions)
    for i in range(N):
        for j in range(i + 1, N):
            diff = positions[i] - positions[j]
            dist = np.linalg.norm(diff)
            if 1e-6 < dist < r_safe:
                force = k_rep * diff / dist ** 3
                forces[i] += force
                forces[j] -= force
    return forces


def rhs(t: float, y: np.ndarray, targets: np.ndarray, params: dict) -> np.ndarray:
    N = len(targets)
    positions = y[:3 * N].reshape(N, 3)
    velocities = y[3 * N:].reshape(N, 3)
    dxdt = np.zeros_like(positions)
    dvdt = np.zeros_like(velocities)
    for i in range(N):
        v_norm = np.linalg.norm(velocities[i])
        dxdt[i] = velocities[i] * (min(1, params['v_max'] / v_norm) if v_norm > 0 else 0)
    rep_forces = repulsion_force(positions, params['k_rep'], params['r_safe'])
    for i in range(N):
        attract = params['k_p'] * (targets[i] - positions[i])
        damp = -params['k_d'] * velocities[i]
        dvdt[i] = (1 / params['m']) * (attract + rep_forces[i] + damp)
    return np.concatenate([dxdt.ravel(), dvdt.ravel()])


def compute_trajectories(initial_positions: np.ndarray, targets: np.ndarray, T_final: float, dt: float) -> np.ndarray:
    N = len(initial_positions)
    assigned_targets = assign_targets(initial_positions, targets)
    params = {'m': 1.0, 'k_p': 2.0, 'k_d': 1.5, 'v_max': 5.0, 'k_rep': 20.0, 'r_safe': 0.5}
    y0 = np.concatenate([initial_positions.ravel(), np.zeros(3 * N)])
    times = np.arange(0, T_final + dt, dt)
    sol = solve_ivp(lambda t, y: rhs(t, y, assigned_targets, params), [0, T_final], y0, t_eval=times, method='RK45')
    if not sol.success:
        raise ValueError("IVP solver failed")
    return sol.y[:3*N].reshape(len(times), N, 3)