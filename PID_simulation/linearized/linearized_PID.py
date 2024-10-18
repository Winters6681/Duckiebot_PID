import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, process_variable, dt):
        error = setpoint - process_variable
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class LinearizedDuckiebotModel:
    def __init__(self, wheel_distance, wheel_radius):
        self.L = wheel_distance
        self.R = wheel_radius

    def dynamics(self, t, state, v_l, v_r):
        x, y, theta = state
        v = self.R * (v_r + v_l) / 2
        omega = self.R * (v_r - v_l) / self.L
        dx = v * (1 - theta**2 / 2)  # Taylor expansion of cos(theta)
        dy = v * theta  # Taylor expansion of sin(theta)
        dtheta = omega
        return [dx, dy, dtheta]

class NonlinearDuckiebotModel:
    def __init__(self, wheel_distance, wheel_radius):
        self.L = wheel_distance
        self.R = wheel_radius

    def dynamics(self, t, state, v_l, v_r):
        x, y, theta = state
        v = self.R * (v_r + v_l) / 2
        omega = self.R * (v_r - v_l) / self.L
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega
        return [dx, dy, dtheta]

def simulate_duckiebot(linear_model, nonlinear_model, pid, setpoint, dt, total_time, v0):
    time_span = (0, total_time)
    t_eval = np.arange(0, total_time, dt)
    initial_state = [0, 0, 0.1]  # [x, y, theta]

    def ode_function_linear(t, state):
        _, _, theta = state
        control_output = pid.compute(setpoint, theta, dt)
        base_speed = v0
        v_l = base_speed - control_output / 2
        v_r = base_speed + control_output / 2
        return linear_model.dynamics(t, state, v_l, v_r)

    def ode_function_nonlinear(t, state):
        _, _, theta = state
        control_output = pid.compute(setpoint, theta, dt)
        base_speed = v0
        v_l = base_speed - control_output / 2
        v_r = base_speed + control_output / 2
        return nonlinear_model.dynamics(t, state, v_l, v_r)

    sol_linear = solve_ivp(ode_function_linear, time_span, initial_state, t_eval=t_eval, method='RK45')
    sol_nonlinear = solve_ivp(ode_function_nonlinear, time_span, initial_state, t_eval=t_eval, method='RK45')

    return sol_linear, sol_nonlinear

def main():
    # Parameters
    wheel_distance = 0.1
    wheel_radius = 0.025
    v0 = 0.5  # Reduced nominal forward velocity for more visible differences
    setpoint = 0  # Desired angle (radians)
    total_time = 20
    dt = 0.05

    # Create models and controller
    linear_model = LinearizedDuckiebotModel(wheel_distance, wheel_radius)
    nonlinear_model = NonlinearDuckiebotModel(wheel_distance, wheel_radius)
    pid = PIDController(kp=5.0, ki=0.1, kd=0.5)  # Adjusted PID parameters

    # Run simulation
    sol_linear, sol_nonlinear = simulate_duckiebot(linear_model, nonlinear_model, pid, setpoint, dt, total_time, v0)

    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Trajectory plot
    axs[0, 0].plot(sol_linear.y[0], sol_linear.y[1], label='Linearized')
    axs[0, 0].plot(sol_nonlinear.y[0], sol_nonlinear.y[1], label='Nonlinear')
    axs[0, 0].set_title("Duckiebot Trajectory")
    axs[0, 0].set_xlabel("X position")
    axs[0, 0].set_ylabel("Y position")
    axs[0, 0].axis('equal')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Heading plot
    axs[0, 1].plot(sol_linear.t, sol_linear.y[2], label='Linearized')
    axs[0, 1].plot(sol_nonlinear.t, sol_nonlinear.y[2], label='Nonlinear')
    axs[0, 1].set_title("Duckiebot Heading Over Time")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Heading (radians)")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # X position over time
    axs[1, 0].plot(sol_linear.t, sol_linear.y[0], label='Linearized')
    axs[1, 0].plot(sol_nonlinear.t, sol_nonlinear.y[0], label='Nonlinear')
    axs[1, 0].set_title("X Position Over Time")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("X Position")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Y position over time
    axs[1, 1].plot(sol_linear.t, sol_linear.y[1], label='Linearized')
    axs[1, 1].plot(sol_nonlinear.t, sol_nonlinear.y[1], label='Nonlinear')
    axs[1, 1].set_title("Y Position Over Time")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Y Position")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
