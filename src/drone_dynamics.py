import logging
import numpy as np
import time

from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


np.random.seed(95)  # ⚡
params = {'m': 1.0, 'k_p': 5.0, 'k_v': 2.0, 'k_d': 6.0, 'v_max': 10.0, 'k_rep': 5.0, 'r_safe': 2.0}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def assign_targets(initial_positions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    dists = np.linalg.norm(initial_positions[:, None] - targets[None], axis=-1) ** 2
    row_ind, col_ind = linear_sum_assignment(dists)
    return targets[col_ind]


def repulsion_force(positions: np.ndarray, k_rep: float, r_safe: float) -> np.ndarray:
    N = len(positions)
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    mask = (dists < r_safe) & (dists > 1e-6)
    forces = np.zeros((N, positions.shape[1]))
    dists_safe = np.where(mask, dists, 1.0)
    rep_magnitude = np.where(mask, k_rep / (dists_safe ** 3), 0.0)
    forces = np.sum(diffs * rep_magnitude[:, :, np.newaxis], axis=1)
    return forces


def clip_velocity(velocity: np.ndarray, v_max: float) -> np.ndarray:
    v_norms = np.linalg.norm(velocity, axis=-1, keepdims=True)
    v_norms_safe = np.where(v_norms > v_max, v_norms, v_max)
    return velocity * (v_max / v_norms_safe)


def rhs_target_tracking(y: np.ndarray, targets: np.ndarray, params: dict) -> np.ndarray:
    N = len(targets)
    dim = targets.shape[1]
    positions = y[:dim * N].reshape(N, dim)
    velocities = y[dim * N:].reshape(N, dim)

    velocities_clipped = clip_velocity(velocities, params['v_max'])
    dxdt = velocities_clipped

    rep_forces = repulsion_force(positions, params['k_rep'], params['r_safe'])
    attract = params['k_p'] * (targets - positions)
    damp = -params['k_d'] * velocities
    dvdt = (1 / params['m']) * (attract + rep_forces + damp)

    return np.concatenate([dxdt.ravel(), dvdt.ravel()])


def rhs_velocity_field(t: float, y: np.ndarray, velocity_field_func, params: dict) -> np.ndarray:
    N = int(len(y) // 6)
    positions = y[:3 * N].reshape(N, 3)
    velocities = y[3 * N:].reshape(N, 3)

    velocities_clipped = clip_velocity(velocities, params['v_max'])
    dxdt = velocities_clipped

    rep_forces = repulsion_force(positions, params['k_rep'], params['r_safe'])

    v_fields = np.array([velocity_field_func(positions[i], t) for i in range(N)])
    v_fields_clipped = clip_velocity(v_fields, params['v_max'])

    vel_track = params['k_v'] * (v_fields_clipped - velocities)
    damp = -params['k_d'] * velocities
    dvdt = (1 / params['m']) * (vel_track + rep_forces + damp)

    return np.concatenate([dxdt.ravel(), dvdt.ravel()])


def compute_trajectories_static(initial_positions: np.ndarray, targets: np.ndarray, T_final: float, dt: float) -> np.ndarray:
    N = len(initial_positions)
    dim = initial_positions.shape[1]
    assigned_targets = assign_targets(initial_positions, targets)

    logger.info(f"<<TwinkleSwarmLogger>>:  Target assignment complete:")
    logger.info(f"<<TwinkleSwarmLogger>>:  Initial spread: {np.std(initial_positions, axis=0)}")
    logger.info(f"<<TwinkleSwarmLogger>>:  Target spread: {np.std(assigned_targets, axis=0)}")
    logger.info(f"<<TwinkleSwarmLogger>>:  Mean distance to target: {np.mean(np.linalg.norm(initial_positions - assigned_targets, axis=1)):.2f}")

    global params
    y0 = np.concatenate([initial_positions.ravel(), np.zeros(dim * N)])
    times = np.arange(0, T_final + dt, dt)

    logger.info(f"<<TwinkleSwarmLogger>>:  Solving ODE: {len(times)} timesteps, T_final={T_final}, dt={dt}")

    start_time = time.time()
    last_print = [start_time]
    pbar = tqdm(total=100, desc="Solving ODE", bar_format='{l_bar}{bar}| {n:.1f}% [{elapsed}<{remaining}]')

    def rhs_with_progress(t, y):
        current_time = time.time()
        if current_time - last_print[0] > 2.0:
            pct = (t / T_final) * 100 if T_final > 0 else 0
            pbar.update(pct - pbar.n)
            last_print[0] = current_time
        return rhs_target_tracking(y, assigned_targets, params)

    sol = solve_ivp(rhs_with_progress, [0, T_final], y0, t_eval=times, method='RK45', max_step=dt, rtol=1e-6, atol=1e-9)

    elapsed = time.time() - start_time
    pbar.update(100 - pbar.n)
    pbar.close()

    if not sol.success:
        logger.warning(f"<<TwinkleSwarmLogger>>:  WARNING: Solver failed with message: {sol.message}")
        raise ValueError("IVP solver failed")

    logger.info(f"<<TwinkleSwarmLogger>>:  Solver success in {elapsed:.2f} seconds! ({sol.nfev} function evaluations)")
    traj = sol.y[:dim * N, :].T.reshape(len(times), N, dim)

    final_error = np.mean(np.linalg.norm(traj[-1] - assigned_targets, axis=1))
    logger.info(f"<<TwinkleSwarmLogger>>:  Final mean error: {final_error:.3f}")
    logger.info(f"<<TwinkleSwarmLogger>>:  Final position spread: {np.std(traj[-1], axis=0)}")

    return traj


def compute_trajectories_transition(initial_positions: np.ndarray, initial_velocities: np.ndarray, targets: np.ndarray, T_final: float, dt: float) -> np.ndarray:
    N = len(initial_positions)
    dim = initial_positions.shape[1]
    assigned_targets = assign_targets(initial_positions, targets)

    logger.info(f"<<TwinkleSwarmLogger>>:  Solving transition ODE...")

    global params
    y0 = np.concatenate([initial_positions.ravel(), initial_velocities.ravel()])
    times = np.arange(0, T_final + dt, dt)

    start_time = time.time()
    last_print = [start_time]
    pbar = tqdm(total=100, desc="Transition ODE", bar_format='{l_bar}{bar}| {n:.1f}% [{elapsed}<{remaining}]')

    def rhs_with_progress(t, y):
        current_time = time.time()
        if current_time - last_print[0] > 2.0:
            pct = (t / T_final) * 100 if T_final > 0 else 0
            pbar.update(pct - pbar.n)
            last_print[0] = current_time
        return rhs_target_tracking(y, assigned_targets, params)

    sol = solve_ivp(rhs_with_progress, [0, T_final], y0, t_eval=times, method='RK45', max_step=dt, rtol=1e-6, atol=1e-9)

    elapsed = time.time() - start_time
    pbar.update(100 - pbar.n)
    pbar.close()

    if not sol.success:
        raise ValueError("IVP solver failed")

    return sol.y[:dim * N, :].T.reshape(len(times), N, dim)


def compute_trajectories_dynamic(initial_positions: np.ndarray, initial_velocities: np.ndarray, velocity_field_func, T_final: float, dt: float) -> np.ndarray:
    N = len(initial_positions)
    global params
    y0 = np.concatenate([initial_positions.ravel(), initial_velocities.ravel()])
    times = np.arange(0, T_final + dt, dt)

    sol = solve_ivp(lambda t, y: rhs_velocity_field(t, y, velocity_field_func, params),[0, T_final], y0, t_eval=times, method='RK45', max_step=dt, rtol=1e-6, atol=1e-9)

    if not sol.success:
        raise ValueError("IVP solver failed")

    return sol.y[:3 * N, :].T.reshape(len(times), N, 3)