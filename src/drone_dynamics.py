import logging
import numpy as np
import time

from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from numba import njit
from tqdm import tqdm


np.random.seed(95)  # ⚡
params = {'m': 1.0, 'k_p': 5.0, 'k_v': 2.0, 'k_d': 6.0, 'v_max': 10.0, 'k_rep': 1.0, 'r_safe': 0.2}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def assign_targets(initial_positions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    dists = np.linalg.norm(initial_positions[:, None] - targets[None], axis=-1) ** 2
    row_ind, col_ind = linear_sum_assignment(dists)
    return targets[col_ind]


@njit
def compute_repulsion_numba_core(positions: np.ndarray, forces: np.ndarray, k_rep: float, r_safe: float) -> None:
    N = len(positions)
    r_safe_sq = r_safe ** 2
    epsilon = 0.1

    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq < r_safe_sq:
                dist_sq_safe = dist_sq + epsilon
                inv_dist_sq = 1.0 / dist_sq_safe
                inv_dist = np.sqrt(inv_dist_sq)
                rep_mag = k_rep * inv_dist_sq * inv_dist

                force_x = dx * rep_mag
                force_y = dy * rep_mag
                force_z = dz * rep_mag

                forces[i, 0] += force_x
                forces[i, 1] += force_y
                forces[i, 2] += force_z

                forces[j, 0] -= force_x
                forces[j, 1] -= force_y
                forces[j, 2] -= force_z


def repulsion_force_numba(positions: np.ndarray, k_rep: float, r_safe: float) -> np.ndarray:
    forces = np.zeros_like(positions)
    if len(positions) > 0:
        compute_repulsion_numba_core(positions, forces, k_rep, r_safe)
    return forces


def repulsion_force_spatial(positions: np.ndarray, k_rep: float, r_safe: float) -> np.ndarray:
    N = len(positions)
    if N == 0:
        return np.zeros_like(positions)

    tree = KDTree(positions)
    pairs = tree.query_pairs(r=r_safe, output_type='ndarray')

    if len(pairs) == 0:
        return np.zeros_like(positions)

    forces = np.zeros_like(positions)
    epsilon = 0.1

    i_indices = pairs[:, 0]
    j_indices = pairs[:, 1]
    diffs = positions[i_indices] - positions[j_indices]
    dists_sq = np.sum(diffs ** 2, axis=1) + epsilon
    inv_dists_sq = 1.0 / dists_sq
    inv_dists = np.sqrt(inv_dists_sq)
    rep_magnitude = k_rep * inv_dists_sq * inv_dists
    force_vectors = diffs * rep_magnitude[:, np.newaxis]

    np.add.at(forces, i_indices, force_vectors)
    np.add.at(forces, j_indices, -force_vectors)

    return forces


def repulsion_force_hybrid(positions: np.ndarray, k_rep: float, r_safe: float) -> np.ndarray:
    N = len(positions)
    if N < 50:
        return repulsion_force_numba(positions, k_rep, r_safe)
    else:
        return repulsion_force_spatial(positions, k_rep, r_safe)


def clip_velocity(velocity: np.ndarray, v_max: float) -> np.ndarray:
    v_norms = np.linalg.norm(velocity, axis=-1, keepdims=True)
    scale = np.minimum(v_max / np.maximum(v_norms, 1e-10), 1.0)
    return velocity * scale


def rhs_target_tracking(t: float, y: np.ndarray, targets: np.ndarray, params: dict) -> np.ndarray:
    N = len(targets)
    dim = targets.shape[1]
    positions = y[:dim * N].reshape(N, dim)
    velocities = y[dim * N:].reshape(N, dim)

    velocities_clipped = clip_velocity(velocities, params['v_max'])
    dxdt = velocities_clipped

    rep_forces = repulsion_force_hybrid(positions, params['k_rep'], params['r_safe'])
    attract = params['k_p'] * (targets - positions)
    damp = -params['k_d'] * velocities
    dvdt = (1 / params['m']) * (attract + rep_forces + damp)

    return np.concatenate([dxdt.ravel(), dvdt.ravel()])


def rhs_velocity_field(t: float, y: np.ndarray, velocity_field_func, params: dict) -> np.ndarray:
    N = len(y) // 6
    dim = 3
    positions = y[:dim * N].reshape(N, dim)
    velocities = y[dim * N:].reshape(N, dim)
    v_field = np.zeros((N, dim))
    for i in range(N):
        v_field[i] = velocity_field_func(positions[i], t)
    v_norms = np.linalg.norm(v_field, axis=1, keepdims=True)
    v_sat = np.where(v_norms > 0, v_field * np.minimum(1.0, params['v_max'] / v_norms), 0.0)
    v_sat = np.full_like(v_sat, np.mean(v_sat, axis=0))
    repulsion = repulsion_force_hybrid(positions, params['k_rep'], params['r_safe'])
    accelerations = (params['k_v'] * v_sat - params['k_d'] * velocities + repulsion) / params['m']
    return np.concatenate([clip_velocity(velocities, params['v_max']).ravel(), accelerations.ravel()])


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
    times = np.linspace(0, T_final, int(T_final / dt) + 1)

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
        return rhs_target_tracking(t, y, assigned_targets, params)

    sol = solve_ivp(rhs_with_progress, [0, T_final], y0, t_eval=times, method='RK45', rtol=5e-3, atol=1e-4)

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
    times = np.linspace(0, T_final, int(T_final / dt) + 1)

    start_time = time.time()
    last_print = [start_time]
    pbar = tqdm(total=100, desc="Transition ODE", bar_format='{l_bar}{bar}| {n:.1f}% [{elapsed}<{remaining}]')

    def rhs_with_progress(t, y):
        current_time = time.time()
        if current_time - last_print[0] > 2.0:
            pct = (t / T_final) * 100 if T_final > 0 else 0
            pbar.update(pct - pbar.n)
            last_print[0] = current_time
        return rhs_target_tracking(t, y, assigned_targets, params)

    sol = solve_ivp(rhs_with_progress, [0, T_final], y0, t_eval=times, method='RK45', rtol=5e-3, atol=1e-4)

    pbar.update(100 - pbar.n)
    pbar.close()

    if not sol.success:
        raise ValueError("IVP solver failed")

    return sol.y[:dim * N, :].T.reshape(len(times), N, dim)


def compute_trajectories_dynamic(initial_positions: np.ndarray, initial_velocities: np.ndarray, velocity_field_func, T_final: float, dt: float) -> np.ndarray:
    N = len(initial_positions)
    global params
    y0 = np.concatenate([initial_positions.ravel(), initial_velocities.ravel()])
    times = np.linspace(0, T_final, int(T_final / dt) + 1)

    sol = solve_ivp(lambda t, y: rhs_velocity_field(t, y, velocity_field_func, params), [0, T_final], y0, t_eval=times, method='RK45', rtol=5e-3, atol=1e-4)

    if not sol.success:
        raise ValueError("IVP solver failed")

    return sol.y[:3 * N, :].T.reshape(len(times), N, 3)