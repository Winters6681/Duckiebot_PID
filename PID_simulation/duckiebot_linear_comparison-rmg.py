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

    # def dynamics(self, t, state, v_l, v_r):
    #     x, y, theta = state
    #     v = self.R * (v_r + v_l) / 2
    #     omega = self.R * (v_r - v_l) / self.L
        
    #     # Apply saturation to omega
    #     omega = np.clip(omega, -self.omega_max, self.omega_max)
        
    #     dx = v * np.cos(theta)
    #     dy = v * np.sin(theta)
    #     dtheta = omega
        
    #     return [dx, dy, dtheta]
    
    def dynamics(self, t, state, v, omega):
        x, y, theta = state
        # v = self.R * (v_r + v_l) / 2
        # omega = self.R * (v_r - v_l) / self.L
        
        # Apply saturation to omega
        omega = np.clip(omega, -self.omega_max, self.omega_max)
        
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega
        
        return [dx, dy, dtheta]

def simulate_duckiebot_with_pid(model, pid, setpoint, dt, total_time):
    time_span = (0, total_time)
    t_eval = np.linspace(0, total_time, int(total_time / dt) + 1)
    initial_state = [0, 0, 0.1]  # [x, y, theta] - np.pi/2 :facing upwards
    
    # def ode_function(t, state):
    #     _, _, theta = state
    #     control_output = pid.compute(setpoint, theta, dt)
        
    #     base_speed = 0.5  # Reduced base speed for smoother motion
    #     v = base_speed
    #     omega = control_output
    #     # v_l = base_speed - control_output
    #     # v_r = base_speed + control_output
        
    #     return model.dynamics(t, state, v_l, v_r)
    def ode_function(t, state):
        _, _, theta = state
        control_output = pid.compute(setpoint, theta, dt)
        
        base_speed = 0.5  # Reduced base speed for smoother motion
        v = base_speed
        omega = control_output
        # v_l = base_speed - control_output
        # v_r = base_speed + control_output
        
        return model.dynamics(t, state, v, omega)

    solution = solve_ivp(ode_function, time_span, initial_state, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-8)
    
    # Calculate omega_history after the simulation
    omega_history = np.diff(solution.y[2]) / np.diff(solution.t)
    # Pad omega_history with the last value to match the length of solution.t
    omega_history = np.append(omega_history, omega_history[-1])
    
    return solution.t, solution.y[0], solution.y[1], solution.y[2], omega_history

class LinearizedDuckiebotModel:
    def __init__(self, wheel_distance, wheel_radius, omega_max):
        self.L = wheel_distance
        self.R = wheel_radius
        self.omega_max = omega_max
        self.base_speed = 0.5  # Operating point for linearization

    # def get_linearized_matrices(self, theta_operating=0.0):
    #     """
    #     Returns the A and B matrices for the linearized system around an operating point
    #     State vector: [x, y, theta]
    #     Input vector: [v_l, v_r]
    #     """
    #     # System matrices at operating point
    #     # A = np.array([
    #     #     [0, 0, self.base_speed * np.cos(theta_operating)],
    #     #     [0, 0, self.base_speed * theta_operating],
    #     #     [0, 0, 0]
    #     # ])
    #     A = np.array([
    #         [0, 0, 0],
    #         [0, 0, self.R/2],
    #         [0, 0, 0]
    #     ])

    #     # Input matrix
    #     B = np.array([
    #         [self.R/2 , self.R/2],
    #         [0, 0],
    #         [-self.R/self.L, self.R/self.L]
    #     ])
    #     # B = np.array([
    #     #     [self.R/2 * np.cos(theta_operating), self.R/2 * np.cos(theta_operating)],
    #     #     [self.R/2 * theta_operating, self.R/2 * theta_operating],
    #     #     [-self.R/self.L, self.R/self.L]
    #     # ])

    #     return A, B
    
    def get_linearized_matrices(self, theta_operating=0.0):
        """
        Returns the A and B matrices for the linearized system around an operating point
        State vector: [x, y, theta]
        Input vector: [v_l, v_r]
        """
        # System matrices at operating point
        # A = np.array([
        #     [0, 0, self.base_speed * np.cos(theta_operating)],
        #     [0, 0, self.base_speed * theta_operating],
        #     [0, 0, 0]
        # ])
        A = np.array([
            [0, 0, 0],
            [0, 0, 0.5],
            [0, 0, 0]
        ])

        # Input matrix
        B = np.array([
            [1, 0],
            [0, 0],
            [0, 1]
        ])
        # B = np.array([
        #     [self.R/2 * np.cos(theta_operating), self.R/2 * np.cos(theta_operating)],
        #     [self.R/2 * theta_operating, self.R/2 * theta_operating],
        #     [-self.R/self.L, self.R/self.L]
        # ])

        return A, B

    # def linear_dynamics(self, t, state, v_l, v_r, theta_operating=0.0):
    #     """
    #     Implements the linearized dynamics around the operating point
    #     """
    #     A, B = self.get_linearized_matrices(theta_operating)
        
    #     # Calculate state deviation from operating point
    #     state_deviation = state - np.array([0, 0, theta_operating])
        
    #     # Calculate input
    #     u = np.array([v_l, v_r])
    #     u_operating = np.array([self.base_speed, self.base_speed])
    #     input_deviation = u - u_operating

    #     # Linearized dynamics
    #     state_dot = A @ state_deviation + B @ input_deviation
        
    #     return state_dot
    def linear_dynamics(self, t, state, v, omega, theta_operating=0.0):
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

def simulate_linearized_duckiebot(model, pid, setpoint, dt, total_time):
    time_span = (0, total_time)
    t_eval = np.linspace(0, total_time, int(total_time/dt) + 1)
    initial_state = np.array([0, 0, 0.1])  # Same initial conditions as nonlinear system
    
    def ode_function(t, state):
        theta = state[2]
        control_output = pid.compute(setpoint, theta, dt)
        
        # Calculate wheel velocities
        # v_l = model.base_speed - control_output
        # v_r = model.base_speed + control_output
        
        return model.linear_dynamics(t, state, 0.5, control_output)

    solution = solve_ivp(
        ode_function, 
        time_span, 
        initial_state, 
        t_eval=t_eval, 
        method='RK45',
        rtol=1e-8,
        atol=1e-8
    ) 
    
    # Calculate omega history
    omega_history = np.diff(solution.y[2]) / np.diff(solution.t)
    omega_history = np.append(omega_history, omega_history[-1])
    
    return solution.t, solution.y[0], solution.y[1], solution.y[2], omega_history

def compare_models(nonlinear_model, linear_model, pid, setpoint, dt, total_time):
    """
    Simulates both models and plots comparison
    """
    # Simulate both models
    t_nl, x_nl, y_nl, theta_nl, omega_nl = simulate_duckiebot_with_pid(
        nonlinear_model, pid, setpoint, dt, total_time
    )
    t_l, x_l, y_l, theta_l, omega_l = simulate_linearized_duckiebot(
        linear_model, pid, setpoint, dt, total_time
    )
    print(x_l)
    # Plot comparison
    plt.figure(figsize=(15, 12))
    
    # Trajectory comparison
    plt.subplot(3, 1, 1)
    plt.plot(x_nl, y_nl, 'b-', label='Nonlinear')
    plt.plot(x_l, y_l, 'r--', label='Linearized')
    plt.title("Trajectory Comparison")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

    # Heading comparison
    plt.subplot(3, 1, 2)
    plt.plot(t_nl, theta_nl, 'b-', label='Nonlinear')
    plt.plot(t_l, theta_l, 'r--', label='Linearized')
    plt.title("Heading Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (radians)")
    plt.grid(True)
    plt.legend()

    # Angular velocity comparison
    plt.subplot(3, 1, 3)
    plt.plot(t_nl, omega_nl, 'b-', label='Nonlinear')
    plt.plot(t_l, omega_l, 'r--', label='Linearized')
    plt.title("Angular Velocity Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.axhline(y=nonlinear_model.omega_max, color='k', linestyle=':', label='Max/Min Omega')
    plt.axhline(y=-nonlinear_model.omega_max, color='k', linestyle=':')
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
