import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment


np.random.seed(95)  # ⚡
params = {'m': 1.0, 'k_p': 1.0, 'k_v': 2.5, 'k_d': 3.0, 'v_max': 5.0, 'k_rep': 5.0, 'r_safe': 0.1}


def assign_targets(initial_positions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    dists = np.linalg.norm(initial_positions[:, None] - targets[None], axis=-1) ** 2
    row_ind, col_ind = linear_sum_assignment(dists)
    return targets[col_ind]


def repulsion_force(positions: np.ndarray, k_rep: float, r_safe: float) -> np.ndarray:
    diffs = positions[:, None] - positions
    dists = np.linalg.norm(diffs, axis=-1)
    mask = (dists > 1e-6) & (dists < r_safe)
    forces = np.zeros_like(diffs)
    forces[mask] = k_rep * diffs[mask] / dists[mask, None] ** 3
    return np.sum(forces, axis=1)


def velocity_saturation(velocity: np.ndarray, v_max: float) -> np.ndarray:
    v_norm = np.linalg.norm(velocity)
    if v_norm > 0:
        return velocity * min(1.0, v_max / v_norm)
    return velocity


def rhs_target_tracking(t: float, y: np.ndarray, targets: np.ndarray, params: dict) -> np.ndarray:
    N = len(targets)
    positions = y[:3 * N].reshape(N, 3)
    velocities = y[3 * N:].reshape(N, 3)
    dxdt = np.zeros_like(positions)
    dvdt = np.zeros_like(velocities)
    for i in range(N):
        dxdt[i] = velocity_saturation(velocities[i], params['v_max'])
    rep_forces = repulsion_force(positions, params['k_rep'], params['r_safe'])
    for i in range(N):
        attract = params['k_p'] * (targets[i] - positions[i])
        damp = -params['k_d'] * velocities[i]
        dvdt[i] = (1 / params['m']) * (attract + rep_forces[i] + damp)
    return np.concatenate([dxdt.ravel(), dvdt.ravel()])


def rhs_velocity_field(t: float, y: np.ndarray, velocity_field_func, params: dict) -> np.ndarray:
    N = int(len(y) // 6)
    positions = y[:3 * N].reshape(N, 3)
    velocities = y[3 * N:].reshape(N, 3)
    dxdt = np.zeros_like(positions)
    dvdt = np.zeros_like(velocities)
    for i in range(N):
        dxdt[i] = velocity_saturation(velocities[i], params['v_max'])
    rep_forces = repulsion_force(positions, params['k_rep'], params['r_safe'])
    for i in range(N):
        v_field = velocity_field_func(positions[i], t)
        v_field_sat = velocity_saturation(v_field, params['v_max'])
        vel_track = params['k_v'] * (v_field_sat - velocities[i])
        damp = -params['k_d'] * velocities[i]
        dvdt[i] = (1 / params['m']) * (vel_track + rep_forces[i] + damp)
    return np.concatenate([dxdt.ravel(), dvdt.ravel()])


def compute_trajectories_static(initial_positions: np.ndarray, targets: np.ndarray, T_final: float, dt: float) -> np.ndarray:
    N = len(initial_positions)
    assigned_targets = assign_targets(initial_positions, targets)
    global params
    y0 = np.concatenate([initial_positions.ravel(), np.zeros(3 * N)])
    times = np.arange(0, T_final + dt, dt)
    sol = solve_ivp(lambda t, y: rhs_target_tracking(t, y, assigned_targets, params), [0, T_final], y0, t_eval=times, method='LSODA')
    if not sol.success:
        raise ValueError("IVP solver failed")
    return sol.y[:3 * N].reshape(len(times), N, 3)


def compute_trajectories_transition(initial_positions: np.ndarray, initial_velocities: np.ndarray, targets: np.ndarray, T_final: float, dt: float) -> np.ndarray:
    N = len(initial_positions)
    assigned_targets = assign_targets(initial_positions, targets)
    global params
    y0 = np.concatenate([initial_positions.ravel(), initial_velocities.ravel()])
    times = np.arange(0, T_final + dt, dt)
    sol = solve_ivp(lambda t, y: rhs_target_tracking(t, y, assigned_targets, params), [0, T_final], y0, t_eval=times, method='LSODA')
    if not sol.success:
        raise ValueError("IVP solver failed")
    return sol.y[:3 * N].reshape(len(times), N, 3)


def compute_trajectories_dynamic(initial_positions: np.ndarray, initial_velocities: np.ndarray, velocity_field_func, T_final: float, dt: float) -> np.ndarray:
    N = len(initial_positions)
    global params
    y0 = np.concatenate([initial_positions.ravel(), initial_velocities.ravel()])
    times = np.arange(0, T_final + dt, dt)
    sol = solve_ivp(lambda t, y: rhs_velocity_field(t, y, velocity_field_func, params), [0, T_final], y0, t_eval=times, method='LSODA', max_step=dt)
    if not sol.success:
        raise ValueError("IVP solver failed")
    return sol.y[:3 * N].reshape(len(times), N, 3)