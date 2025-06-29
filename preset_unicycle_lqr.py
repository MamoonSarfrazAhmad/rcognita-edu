import numpy as np
import matplotlib.pyplot as plt

from systems import SysUnicycle
from controllers import LQRController
from simulator import Simulator

# Time settings
dt = 0.1
T = 15  # Slightly reduced time for stabilization
N = int(T / dt)

# Initial and goal states
x0 = np.array([0.0, 0.0, 0.1])  # Small theta to avoid being exactly at unstable point
goal = np.array([2.0, 2.0, 0.0])  # Optional: use (5.0, 5.0, 0.0) if system works

# Improved A and B matrices (discrete-time approximation around theta ≈ 0.1)
A = np.array([
    [1, 0, -dt * 1.0],
    [0, 1,  dt * 1.0],
    [0, 0, 1]
])

B = np.array([
    [dt, 0],
    [0, 0],
    [0, dt]
])

# Cost matrices (can tweak to make controller more aggressive)
Q = np.diag([20, 20, 2])
R = np.diag([0.5, 0.1])

# Instantiate controller
controller = LQRController(A, B, Q, R)

# Instantiate system
system = SysUnicycle(
    sys_type='diff_eqn',
    dim_state=3,
    dim_input=2,
    dim_output=3,
    dim_disturb=0,
    ctrl_bnds=np.array([[-1.0, 1.0], [-2.0, 2.0]]),
    is_dyn_ctrl=0,
    is_disturb=0
)

# Closed-loop system
def closed_loop_rhs(t, state):
    u = controller.compute_control(state, goal)
    u = np.clip(u, system.ctrl_bnds[:, 0], system.ctrl_bnds[:, 1])
    system.receive_action(u)
    return system.closed_loop_rhs(t, state)

# Simulator
sim = Simulator(
    sys_type='diff_eqn',
    closed_loop_rhs=closed_loop_rhs,
    sys_out=system.out,
    state_init=x0,
    disturb_init=[],
    action_init=[],
    t0=0,
    t1=T,
    dt=dt,
    is_disturb=0,
    is_dyn_ctrl=0
)

# Simulation loop
trajectory = []
controls = []
errors = []
cost_lqr = 0.0
alpha = 1.0
beta = 0.1

for _ in range(N):
    t, state, obs, _ = sim.get_sim_step_data()
    u = controller.compute_control(state, goal)
    u = np.clip(u, system.ctrl_bnds[:, 0], system.ctrl_bnds[:, 1])
    system.receive_action(u)

    trajectory.append(state.copy())
    controls.append(u.copy())
    error = np.linalg.norm(state[:2] - goal[:2])
    errors.append(error)
    cost_lqr += (error**2 + alpha * u[0]**2 + beta * u[1]**2) * dt

    sim.sim_step()

trajectory = np.array(trajectory)
controls = np.array(controls)
errors = np.array(errors)

# Plots
plt.figure()
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory')
plt.plot(goal[0], goal[1], 'ro', label='Goal')
plt.title("Trajectory (LQR Controller)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.savefig("trajectory_lqr.png")
plt.close()

plt.figure()
plt.plot(np.arange(N) * dt, errors)
plt.title("Tracking Error Over Time (LQR)")
plt.xlabel("Time [s]")
plt.ylabel("Distance to Goal")
plt.grid(True)
plt.savefig("error_lqr.png")
plt.close()

plt.figure()
plt.plot(np.arange(N) * dt, controls[:, 0], label='v')
plt.plot(np.arange(N) * dt, controls[:, 1], label='w')
plt.title("Control Inputs Over Time (LQR)")
plt.xlabel("Time [s]")
plt.ylabel("Input")
plt.grid(True)
plt.legend()
plt.savefig("inputs_lqr.png")
plt.close()

with open("cost_lqr.txt", "w") as f:
    f.write(str(cost_lqr))

print(f"✅ LQR Simulation Done! Total Cost: {cost_lqr:.2f}")

