import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class WheelVelocityCalculator:
    def __init__(self, wheel_distance, wheel_radius):
        """
        Initialize the wheel velocity calculator
        
        Args:
            wheel_distance (float): Distance between wheels
            wheel_radius (float): Radius of the wheels
        """
        self.wheel_distance = wheel_distance
        self.wheel_radius = wheel_radius
    
    def v_omega_to_wheel_velocities(self, v, omega):
        """
        Convert linear velocity and angular velocity to left and right wheel velocities
        Using the inverse of the relationships:
        v = wheel_radius * (v_r + v_l) / 2
        omega = wheel_radius * (v_r - v_l) / wheel_distance
        
        Args:
            v (float): Linear velocity
            omega (float): Angular velocity
            
        Returns:
            tuple: (v_l, v_r) Left and right wheel velocities
        """
        # Solve the system of equations:
        # v = wheel_radius * (v_r + v_l) / 2
        # omega = wheel_radius * (v_r - v_l) / wheel_distance
        
        # First equation: 2v/wheel_radius = v_r + v_l
        # Second equation: omega*wheel_distance/wheel_radius = v_r - v_l
        
        # Add equations to get v_r:
        # 2v/wheel_radius + omega*wheel_distance/wheel_radius = 2v_r
        v_r = (2 * v + omega * self.wheel_distance) / (2 * self.wheel_radius)
        
        # Subtract equations to get v_l:
        # 2v/wheel_radius - omega*wheel_distance/wheel_radius = 2v_l
        v_l = (2 * v - omega * self.wheel_distance) / (2 * self.wheel_radius)
        
        return v_l, v_r
    
    def wheel_velocities_to_v_omega(self, v_l, v_r):
        """
        Convert left and right wheel velocities to linear and angular velocity
        Using the relationships:
        v = wheel_radius * (v_r + v_l) / 2
        omega = wheel_radius * (v_r - v_l) / wheel_distance
        
        Args:
            v_l (float): Left wheel velocity
            v_r (float): Right wheel velocity
            
        Returns:
            tuple: (v, omega) Linear and angular velocity
        """
        v = self.wheel_radius * (v_r + v_l) / 2
        omega = self.wheel_radius * (v_r - v_l) / self.wheel_distance
        
        return v, omega

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
        self.wheel_distance = wheel_distance
        self.wheel_radius = wheel_radius
        self.omega_max = omega_max
        self.vel_calc = WheelVelocityCalculator(wheel_distance, wheel_radius)
    
    def dynamics(self, state, v, omega):
        x, y, theta = state
        
        # Apply saturation to omega
        omega = np.clip(omega, -self.omega_max, self.omega_max)
        
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega
        
        return [dx, dy, dtheta]

class LinearizedDuckiebotModel:
    def __init__(self, wheel_distance, wheel_radius, omega_max):
        self.wheel_distance = wheel_distance
        self.wheel_radius = wheel_radius
        self.omega_max = omega_max
        self.base_speed = 0.5  # Operating point for linearization
        self.vel_calc = WheelVelocityCalculator(wheel_distance, wheel_radius)

    def get_linearized_matrices(self, theta_operating=0.0):
        """
        Returns the A and B matrices for the linearized system around an operating point
        State vector: [x, y, theta]
        Input vector: [v, omega]
        """
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

    def linear_dynamics(self, state, v, omega, theta_operating=0.0):
        """
        Implements the linearized dynamics around the operating point
        """
        A, B = self.get_linearized_matrices(theta_operating)
        
        # Calculate state deviation from operating point
        state_deviation = state 
        
        # Calculate input
        u = np.array([v, omega])
        u_operating = np.array([0, 0])
        input_deviation = u - u_operating

        # Linearized dynamics
        state_dot = A @ state_deviation + B @ input_deviation
        
        return state_dot

def simulate_duckiebot_with_pid(model, pid, setpoint, dt, total_time):
    time_span = (0, total_time)
    t_eval = np.linspace(0, total_time, int(total_time / dt) + 1)
    initial_state = [0, 0, 0.1]  # [x, y, theta]
    
    # Pre-allocate arrays for wheel velocities with the same length as t_eval
    v_l_history = np.zeros_like(t_eval)
    v_r_history = np.zeros_like(t_eval)
    current_index = [0]  # Use list to allow modification in nested function
    
    def ode_function(t, state):
        _, _, theta = state
        control_output = pid.compute(setpoint, theta, dt)
        
        base_speed = 0.5  # Reduced base speed for smoother motion
        # Convert to wheel velocities
        v_l, v_r = model.vel_calc.v_omega_to_wheel_velocities(base_speed, control_output)
        
        # Store wheel velocities at the current index
        if current_index[0] < len(t_eval):
            v_l_history[current_index[0]] = v_l
            v_r_history[current_index[0]] = v_r
            current_index[0] += 1
        
        # Convert back to v, omega for the dynamics function
        v, omega = model.vel_calc.wheel_velocities_to_v_omega(v_l, v_r)
        
        return model.dynamics(state, v, omega)

    solution = solve_ivp(ode_function, time_span, initial_state, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-8)
    
    omega_history = np.diff(solution.y[2]) / np.diff(solution.t)
    omega_history = np.append(omega_history, omega_history[-1])
    
    return solution.t, solution.y[0], solution.y[1], solution.y[2], omega_history, v_l_history, v_r_history

def simulate_linearized_duckiebot(model, pid, setpoint, dt, total_time):
    time_span = (0, total_time)
    t_eval = np.linspace(0, total_time, int(total_time/dt) + 1)
    initial_state = np.array([0, 0, 0.1])
    
    # Pre-allocate arrays for wheel velocities with the same length as t_eval
    v_l_history = np.zeros_like(t_eval)
    v_r_history = np.zeros_like(t_eval)
    current_index = [0]  # Use list to allow modification in nested function
    
    def ode_function(t, state):
        theta = state[2]
        control_output = pid.compute(setpoint, theta, dt)
        
        base_speed = 0.5
        # Convert to wheel velocities
        v_l, v_r = model.vel_calc.v_omega_to_wheel_velocities(base_speed, control_output)
        
        # Store wheel velocities at the current index
        if current_index[0] < len(t_eval):
            v_l_history[current_index[0]] = v_l
            v_r_history[current_index[0]] = v_r
            current_index[0] += 1
        
        # Convert back to v, omega for the dynamics function
        v, omega = model.vel_calc.wheel_velocities_to_v_omega(v_l, v_r)
        
        return model.linear_dynamics(state, v, omega)

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
    
    return solution.t, solution.y[0], solution.y[1], solution.y[2], omega_history, v_l_history, v_r_history

def compare_models(nonlinear_model, linear_model, pid, setpoint, dt, total_time):
    """
    Simulates both models and plots comparison including wheel velocities
    """
    # Simulate both models
    t_nl, x_nl, y_nl, theta_nl, omega_nl, vl_nl, vr_nl = simulate_duckiebot_with_pid(
        nonlinear_model, pid, setpoint, dt, total_time
    )
    t_l, x_l, y_l, theta_l, omega_l, vl_l, vr_l = simulate_linearized_duckiebot(
        linear_model, pid, setpoint, dt, total_time
    )
    
    # Plot comparison
    plt.figure(figsize=(15, 15))  # Made figure taller to accommodate new plots
    
    # Trajectory comparison
    plt.subplot(4, 1, 1)  # Changed to 4 rows
    plt.plot(x_nl, y_nl, 'b-', label='Nonlinear')
    plt.plot(x_l, y_l, 'r--', label='Linearized')
    plt.title("Trajectory Comparison")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

    # Heading comparison
    plt.subplot(4, 1, 2)
    plt.plot(t_nl, theta_nl, 'b-', label='Nonlinear')
    plt.plot(t_l, theta_l, 'r--', label='Linearized')
    plt.title("Heading Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (radians)")
    plt.grid(True)
    plt.legend()

    # Angular velocity comparison
    plt.subplot(4, 1, 3)
    plt.plot(t_nl, omega_nl, 'b-', label='Nonlinear')
    plt.plot(t_l, omega_l, 'r--', label='Linearized')
    plt.title("Angular Velocity Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.axhline(y=nonlinear_model.omega_max, color='k', linestyle=':', label='Max/Min Omega')
    plt.axhline(y=-nonlinear_model.omega_max, color='k', linestyle=':')
    plt.grid(True)
    plt.legend()

    # Wheel velocities comparison
    plt.subplot(4, 1, 4)
    plt.plot(t_nl, vl_nl, 'b-', label='Nonlinear Left Wheel')
    plt.plot(t_nl, vr_nl, 'b--', label='Nonlinear Right Wheel')
    plt.plot(t_l, vl_l, 'r-', label='Linear Left Wheel')
    plt.plot(t_l, vr_l, 'r--', label='Linear Right Wheel')
    plt.title("Wheel Velocities Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Wheel Velocity (m/s)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    # Create both nonlinear and linearized models
    omega_max = 2.0
    wheel_distance = 0.1
    wheel_radius = 0.025
    
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
    
    # Create PID controller
    pid = PIDController(kp=5, ki=0, kd=0)
    
    # Simulation parameters
    setpoint = 0
    total_time = 20
    dt = 0.01
    
    # Compare models
    compare_models(
        duckiebot_nonlinear,
        duckiebot_linear,
        pid,
        setpoint,
        dt,
        total_time
    )

if __name__ == "__main__":
    main()