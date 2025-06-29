import numpy as np
import matplotlib.pyplot as plt

from systems import SysUnicycle
from controllers import N_CTRL
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

# Instantiate controller
controller = N_CTRL()

# Define closed-loop right-hand side
def closed_loop_rhs(t, state):
    x, y, theta = state
    state_vec = np.array([x, y, theta])
    u = controller.pure_loop([state_vec, goal])
    u = np.clip(u, system.ctrl_bnds[:, 0], system.ctrl_bnds[:, 1])
    system.receive_action(u)
    return system.closed_loop_rhs(t, state)

# Simulator
sim = Simulator(
    sys_type='diff_eqn',
    closed_loop_rhs=closed_loop_rhs,
    sys_out=system.out,
    state_init=x0,
    disturb_init=[],  # No disturbances
    action_init=[],   # Not used
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
cost_nominal = 0.0
alpha = 1.0  # cost weight for v^2
beta = 0.1   # cost weight for w^2

for _ in range(N):
    t, state, obs, _ = sim.get_sim_step_data()
    u = controller.pure_loop([state, goal])
    u = np.clip(u, system.ctrl_bnds[:, 0], system.ctrl_bnds[:, 1])
    system.receive_action(u)

    # Logging
    trajectory.append(state.copy())
    controls.append(u.copy())
    error = np.linalg.norm(state[:2] - goal[:2])
    errors.append(error)

    # Accumulated cost
    cost_nominal += (error**2 + alpha * u[0]**2 + beta * u[1]**2) * dt

    sim.sim_step()

# Convert to numpy arrays
trajectory = np.array(trajectory)
controls = np.array(controls)
errors = np.array(errors)

# ----------------------------------
# 📊 Plotting and Saving
# ----------------------------------

# 1. Trajectory plot
plt.figure(figsize=(8, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Robot Trajectory')
plt.plot(goal[0], goal[1], 'ro', label='Goal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectory (Nominal Controller)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig("trajectory_nominal.png")
plt.close()

# 2. Tracking error over time
plt.figure()
plt.plot(np.arange(len(errors)) * dt, errors)
plt.title("Tracking Error Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Distance to Goal")
plt.grid(True)
plt.savefig("error_nominal.png")
plt.close()

# 3. Control inputs over time
plt.figure()
plt.plot(np.arange(len(controls)) * dt, controls[:, 0], label='v (linear)')
plt.plot(np.arange(len(controls)) * dt, controls[:, 1], label='w (angular)')
plt.title("Control Inputs Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Input Value")
plt.legend()
plt.grid(True)
plt.savefig("inputs_nominal.png")
plt.close()

# 4. Save cost to file
with open("cost_nominal.txt", "w") as f:
    f.write(str(cost_nominal))

print(f"✅ All outputs saved! Total accumulated cost: {cost_nominal:.2f}")

