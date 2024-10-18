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

class DuckiebotODEModel:
    def __init__(self, wheel_distance, wheel_radius, omega_max):
        self.L = wheel_distance
        self.R = wheel_radius
        self.omega_max = omega_max

    def dynamics(self, t, state, v_l, v_r):
        x, y, theta = state
        v = self.R * (v_r + v_l) / 2
        omega = self.R * (v_r - v_l) / self.L
        
        # Apply saturation to omega
        omega = np.clip(omega, -self.omega_max, self.omega_max)
        
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega
        
        return [dx, dy, dtheta]

def simulate_duckiebot_with_pid(model, pid, setpoint, dt, total_time):
    time_span = (0, total_time)
    t_eval = np.linspace(0, total_time, int(total_time / dt) + 1)
    initial_state = [0, 0, 0.2]  # [x, y, theta] - np.pi/2 :facing upwards
    
    def ode_function(t, state):
        _, _, theta = state
        control_output = pid.compute(setpoint, theta, dt)
        
        base_speed = 0.5  # Reduced base speed for smoother motion
        v_l = base_speed - control_output
        v_r = base_speed + control_output
        
        return model.dynamics(t, state, v_l, v_r)

    solution = solve_ivp(ode_function, time_span, initial_state, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-8)
    
    # Calculate omega_history after the simulation
    omega_history = np.diff(solution.y[2]) / np.diff(solution.t)
    # Pad omega_history with the last value to match the length of solution.t
    omega_history = np.append(omega_history, omega_history[-1])
    
    return solution.t, solution.y[0], solution.y[1], solution.y[2], omega_history

def main():
    # Create Duckiebot model and PID controller
    omega_max = 2.0  # Set maximum angular velocity (rad/s)
    duckiebot = DuckiebotODEModel(wheel_distance=0.1, wheel_radius=0.025, omega_max=omega_max)
    pid = PIDController(kp=5.0, ki=0.1, kd=0.5)  # Adjusted PID parameters

    # Simulation parameters
    setpoint = 0  # Desired angle (radians) 
    total_time = 20
    dt = 0.01  # Smaller time step for smoother simulation

    # Run simulation
    time, x_history, y_history, theta_history, omega_history = simulate_duckiebot_with_pid(duckiebot, pid, setpoint, dt, total_time)

    # Plot results
    plt.figure(figsize=(12, 12))
    
    plt.subplot(3, 1, 1)
    plt.plot(x_history, y_history)
    plt.title("Duckiebot Trajectory")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.axis('equal')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, theta_history)
    plt.title("Duckiebot Heading Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (radians)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, omega_history)
    plt.title("Duckiebot Angular Velocity (Omega) Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.axhline(y=omega_max, color='r', linestyle='--', label='Max Omega')
    plt.axhline(y=-omega_max, color='r', linestyle='--', label='Min Omega')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
