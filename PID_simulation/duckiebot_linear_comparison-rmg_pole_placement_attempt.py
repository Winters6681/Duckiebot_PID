import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import signal

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

class PolePlacementController:
    def __init__(self, A, B, desired_poles):
        self.A = A
        self.B = B
        self.K = self._compute_gain_matrix(desired_poles)
        
    def _compute_gain_matrix(self, desired_poles):
        n = self.A.shape[0]
        C = np.hstack([np.linalg.matrix_power(self.A, i) @ self.B for i in range(n)])
        rank = np.linalg.matrix_rank(C)
        
        if rank < n:
            raise ValueError(f"System is not controllable. Rank of controllability matrix is {rank} < {n}")
            
        K = signal.place_poles(self.A, self.B, desired_poles)
        return K.gain_matrix
    
    def compute_control_input(self, state, reference_state):
        state_error = state - reference_state
        u = -self.K @ state_error
        return u

class DuckiebotODEModel:
    def __init__(self, wheel_distance, wheel_radius, omega_max):
        self.L = wheel_distance
        self.R = wheel_radius
        self.omega_max = omega_max
    
    def dynamics(self, t, state, v, omega):
        x, y, theta = state
        omega = np.clip(omega, -self.omega_max, self.omega_max)
        
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega
        
        return [dx, dy, dtheta]

class LinearizedDuckiebotModel:
    def __init__(self, wheel_distance, wheel_radius, omega_max):
        self.L = wheel_distance
        self.R = wheel_radius
        self.omega_max = omega_max
        self.base_speed = 0.5
    
    def get_linearized_matrices(self, theta_operating=0.0):
        A = np.array([
            [0, 0, 0],
            [0, 0, 0.5],
            [0, 0, 0]
        ])

        B = np.array([
            [1, 0],
            [0, 0],
            [0, 1]
        ])

        return A, B

    def linear_dynamics(self, t, state, v, omega, theta_operating=0.0):
        A, B = self.get_linearized_matrices(theta_operating)
        state_deviation = state 
        u = np.array([v, omega])
        u_operating = np.array([0, 0])
        input_deviation = u - u_operating
        state_dot = A @ state_deviation + B @ input_deviation
        return state_dot

def simulate_duckiebot_with_pid(model, pid, setpoint, dt, total_time):
    time_span = (0, total_time)
    t_eval = np.linspace(0, total_time, int(total_time / dt) + 1)
    initial_state = [0, 0, 0.1]
    
    def ode_function(t, state):
        _, _, theta = state
        control_output = pid.compute(setpoint, theta, dt)
        base_speed = 0.5
        v = base_speed
        omega = control_output
        return model.dynamics(t, state, v, omega)

    solution = solve_ivp(ode_function, time_span, initial_state, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-8)
    
    omega_history = np.diff(solution.y[2]) / np.diff(solution.t)
    omega_history = np.append(omega_history, omega_history[-1])
    
    return solution.t, solution.y[0], solution.y[1], solution.y[2], omega_history

def simulate_duckiebot_pole_placement(model, controller, reference_state, dt, total_time):
    time_span = (0, total_time)
    t_eval = np.linspace(0, total_time, int(total_time/dt) + 1)
    initial_state = np.array([0, 0, 0.1])
    
    def ode_function(t, state):
        u = controller.compute_control_input(state, reference_state)
        v, omega = u
        omega = np.clip(omega, -model.omega_max, model.omega_max)  # Add saturation
        return model.linear_dynamics(t, state, v, omega)
    
    solution = solve_ivp(
        ode_function,
        time_span,
        initial_state,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-8
    )
    
    omega_history = np.diff(solution.y[2]) / np.diff(solution.t)
    omega_history = np.append(omega_history, omega_history[-1])
    
    return solution.t, solution.y[0], solution.y[1], solution.y[2], omega_history

def compare_controllers(nonlinear_model, linear_model, pid_controller, pole_controller, 
                      setpoint, reference_state, dt, total_time):
    """Compare PID and Pole Placement controllers"""
    
    # Simulate PID control
    t_pid, x_pid, y_pid, theta_pid, omega_pid = simulate_duckiebot_with_pid(
        nonlinear_model, pid_controller, setpoint, dt, total_time
    )
    
    # Simulate Pole Placement control
    t_pole, x_pole, y_pole, theta_pole, omega_pole = simulate_duckiebot_pole_placement(
        linear_model, pole_controller, reference_state, dt, total_time
    )
    
    # Plot comparison
    plt.figure(figsize=(15, 12))
    
    # Trajectory comparison
    plt.subplot(3, 1, 1)
    plt.plot(x_pid, y_pid, 'b-', label='PID Control')
    plt.plot(x_pole, y_pole, 'r--', label='Pole Placement')
    plt.plot(reference_state[0], reference_state[1], 'k*', label='Reference Point')
    plt.title("Trajectory Comparison")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

    # Heading comparison
    plt.subplot(3, 1, 2)
    plt.plot(t_pid, theta_pid, 'b-', label='PID Control')
    plt.plot(t_pole, theta_pole, 'r--', label='Pole Placement')
    plt.axhline(y=reference_state[2], color='k', linestyle=':', label='Reference')
    plt.title("Heading Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (radians)")
    plt.grid(True)
    plt.legend()

    # Angular velocity comparison
    plt.subplot(3, 1, 3)
    plt.plot(t_pid, omega_pid, 'b-', label='PID Control')
    plt.plot(t_pole, omega_pole, 'r--', label='Pole Placement')
    plt.axhline(y=nonlinear_model.omega_max, color='k', linestyle=':', label='Max/Min Omega')
    plt.axhline(y=-nonlinear_model.omega_max, color='k', linestyle=':')
    plt.title("Angular Velocity Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    # System parameters
    omega_max = 2.0
    wheel_distance = 0.1
    wheel_radius = 0.025
    
    # Create models
    duckiebot_nonlinear = DuckiebotODEModel(
        wheel_distance=wheel_distance,
        wheel_radius=wheel_radius,
        omega_max=omega_max
    )
    
    duckiebot_linear = LinearizedDuckiebotModel(
        wheel_distance=wheel_distance,
        wheel_radius=wheel_radius,
        omega_max=omega_max
    )
    
    # Get linearized system matrices for pole placement
    A, B = duckiebot_linear.get_linearized_matrices()
    
    # Create controllers
    pid_controller = PIDController(kp=5, ki=0, kd=0)
    desired_poles = [-2, -2 + 1j, -2 - 1j]
    pole_controller = PolePlacementController(A, B, desired_poles)
    
    # Simulation parameters
    setpoint = 0  # For PID
    reference_state = np.array([1.0, 1.0, 0.0])  # For pole placement
    total_time = 20
    dt = 0.01
    
    # Compare controllers
    compare_controllers(
        duckiebot_nonlinear,
        duckiebot_linear,
        pid_controller,
        pole_controller,
        setpoint,
        reference_state,
        dt,
        total_time
    )

if __name__ == "__main__":
    main()