import numpy as np
import matplotlib.pyplot as plt

from systems import SysUnicycle
from controllers import MPCController
from simulator import Simulator

# Time settings
dt = 0.1
T = 20
N = int(T / dt)

# Initial and goal states
x0 = np.array([0.0, 0.0, 0.0])
goal = np.array([5.0, 5.0, 0.0])

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

# Wrap RHS to match signature
def rhs_wrapper(t, state, action):
    return system._state_dyn(t, state, action)

# Cost matrices
Q = np.diag([10, 10, 1])
R = np.diag([1, 0.1])

# Instantiate controller
controller = MPCController(
    sys_rhs=rhs_wrapper,
    sys_out=system.out,
    state_dim=3,
    input_dim=2,
    ctrl_bnds=system.ctrl_bnds,
    horizon=10,
    dt=dt,
    Q=Q,
    R=R,
    goal=goal
)

# Define closed-loop RHS
def closed_loop_rhs(t, state):
    u = controller.compute_control(state)
    u = np.clip(u, system.ctrl_bnds[:, 0], system.ctrl_bnds[:, 1])
    system.receive_action(u)
    return system.closed_loop_rhs(t, state)

# Instantiate simulator
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

# Run simulation
trajectory = []
controls = []
errors = []
cost_mpc = 0.0

for _ in range(N):
    t, state, obs, _ = sim.get_sim_step_data()
    u = controller.compute_control(state)
    u = np.clip(u, system.ctrl_bnds[:, 0], system.ctrl_bnds[:, 1])
    system.receive_action(u)

    trajectory.append(state.copy())
    controls.append(u.copy())
    error = np.linalg.norm(state[:2] - goal[:2])
    errors.append(error)
    cost_mpc += error**2 * dt

    sim.sim_step()

trajectory = np.array(trajectory)
controls = np.array(controls)
errors = np.array(errors)

# Plots
plt.figure()
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory')
plt.plot(goal[0], goal[1], 'ro', label='Goal')
plt.title("Trajectory (MPC Controller)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.savefig("trajectory_mpc.png")
plt.close()

plt.figure()
plt.plot(np.arange(N) * dt, errors)
plt.title("Tracking Error Over Time (MPC)")
plt.xlabel("Time [s]")
plt.ylabel("Distance to Goal")
plt.grid(True)
plt.savefig("error_mpc.png")
plt.close()

plt.figure()
plt.plot(np.arange(N) * dt, controls[:, 0], label='v')
plt.plot(np.arange(N) * dt, controls[:, 1], label='w')
plt.title("Control Inputs Over Time (MPC)")
plt.xlabel("Time [s]")
plt.ylabel("Input")
plt.grid(True)
plt.legend()
plt.savefig("inputs_mpc.png")
plt.close()

with open("cost_mpc.txt", "w") as f:
    f.write(str(cost_mpc))

print(f"\n✅ MPC Simulation Done! Total Cost: {cost_mpc:.2f}")
