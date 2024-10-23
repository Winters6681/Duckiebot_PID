import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import List, Tuple
import itertools

@dataclass
class PerformanceMetrics:
    kp: float
    ki: float
    kd: float
    rise_time: float
    settling_time: float
    overshoot: float
    steady_state_error: float
    
class PIDTuningAnalysis:
    def __init__(self, model_params: dict):
        self.wheel_distance = model_params['wheel_distance']
        self.wheel_radius = model_params['wheel_radius']
        self.omega_max = model_params['omega_max']
        self.base_speed = model_params['base_speed']
        self.dt = model_params['dt']
        self.total_time = model_params['total_time']
        self.setpoint = model_params['setpoint']
        self.initial_theta = model_params['initial_theta']
        
    def simulate_system(self, kp: float, ki: float, kd: float) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the system with given PID parameters"""
        model = DuckiebotODEModel(self.wheel_distance, self.wheel_radius, self.omega_max)
        pid = PIDController(kp, ki, kd)
        
        t, _, _, theta, _ = simulate_duckiebot_with_pid(
            model, pid, self.setpoint, self.dt, self.total_time
        )
        return t, theta
    
    def calculate_metrics(self, t: np.ndarray, theta: np.ndarray, kp: float, ki: float, kd: float) -> PerformanceMetrics:
        """Calculate performance metrics from simulation results"""
        # Find steady state value (average of last 10% of simulation)
        steady_idx = int(0.9 * len(theta))
        steady_state = np.mean(theta[steady_idx:])
        
        # Calculate steady state error
        steady_state_error = abs(self.setpoint - steady_state)
        
        # Calculate overshoot
        if self.initial_theta > self.setpoint:
            min_value = np.min(theta)
            overshoot = abs((min_value - self.setpoint) / (self.initial_theta - self.setpoint) * 100)
        else:
            max_value = np.max(theta)
            overshoot = abs((max_value - self.setpoint) / (self.initial_theta - self.setpoint) * 100)
        
        # Calculate rise time (time to reach 90% of setpoint)
        target = self.setpoint + 0.1 * (self.initial_theta - self.setpoint)
        rise_indices = np.where(np.abs(theta - target) <= 0.01)[0]
        rise_time = t[rise_indices[0]] if len(rise_indices) > 0 else np.inf
        
        # Calculate settling time (time to stay within ±2% of setpoint)
        settling_band = 0.02 * abs(self.initial_theta - self.setpoint)
        settled = np.where(np.abs(theta - self.setpoint) <= settling_band)[0]
        settling_time = t[settled[0]] if len(settled) > 0 else np.inf
        
        return PerformanceMetrics(
            kp=kp,
            ki=ki,
            kd=kd,
            rise_time=rise_time,
            settling_time=settling_time,
            overshoot=overshoot,
            steady_state_error=steady_state_error
        )
    
    def parameter_sweep(self, kp_range: List[float], ki_range: List[float], 
                       kd_range: List[float]) -> List[PerformanceMetrics]:
        """Perform parameter sweep and return metrics for all combinations"""
        results = []
        total_combinations = len(kp_range) * len(ki_range) * len(kd_range)
        
        print(f"Starting parameter sweep with {total_combinations} combinations...")
        
        for i, (kp, ki, kd) in enumerate(itertools.product(kp_range, ki_range, kd_range)):
            print(f"Progress: {i+1}/{total_combinations}", end='\r')
            
            t, theta = self.simulate_system(kp, ki, kd)
            metrics = self.calculate_metrics(t, theta, kp, ki, kd)  # Pass PID parameters here
            results.append(metrics)
            
        print("\nParameter sweep complete!")
        return results
    
    def plot_best_results(self, results: List[PerformanceMetrics], num_best: int = 3):
        """Plot the best results based on a weighted combination of metrics"""
        # Calculate weighted score for each result
        weighted_scores = []
        for r in results:
            score = (
                -0.3 * r.rise_time +  # Faster rise time is better
                -0.3 * r.settling_time +  # Faster settling time is better
                -0.2 * r.overshoot +  # Less overshoot is better
                -0.2 * r.steady_state_error  # Less steady state error is better
            )
            weighted_scores.append((score, r))
        
        # Sort by score and get best results
        weighted_scores.sort(key=lambda x: x[0], reverse=True)
        best_results = [r for _, r in weighted_scores[:num_best]]
        
        # Create figure with extra space for text
        plt.figure(figsize=(15, 10))
        
        for r in best_results:
            t, theta = self.simulate_system(r.kp, r.ki, r.kd)
            label = f'Kp={r.kp}, Ki={r.ki}, Kd={r.kd}'
            plt.plot(t, theta, label=label)
        
        plt.axhline(y=self.setpoint, color='r', linestyle='--', label='Setpoint')
        plt.title('Best PID Controller Responses')
        plt.xlabel('Time (s)')
        plt.ylabel('Heading (radians)')
        plt.legend()
        plt.grid(True)
        
        # Add performance metrics as text
        text_str = "Performance Metrics:\n\n"
        for i, r in enumerate(best_results, 1):
            text_str += f"Parameter Set {i}:\n"
            text_str += f"Kp={r.kp}, Ki={r.ki}, Kd={r.kd}\n"
            text_str += f"Rise Time: {r.rise_time:.3f}s\n"
            text_str += f"Settling Time: {r.settling_time:.3f}s\n"
            text_str += f"Overshoot: {r.overshoot:.1f}%\n"
            text_str += f"Steady State Error: {r.steady_state_error:.4f}\n\n"
        
        plt.figtext(1.02, 0.5, text_str, fontsize=10, va='center')
        plt.subplots_adjust(right=0.85)  # Make room for text
        plt.show()

def main():
    # Model parameters
    model_params = {
        'wheel_distance': 0.1,
        'wheel_radius': 0.025,
        'omega_max': 2.0,
        'base_speed': 0.5,
        'dt': 0.01,
        'total_time': 20,
        'setpoint': 0,
        'initial_theta': 0.2
    }
    
    # Create analyzer
    analyzer = PIDTuningAnalysis(model_params)
    
    # Define parameter ranges to test
    kp_range = [1.0, 2.0, 3.0, 4.0, 5.0]
    ki_range = [0.0, 0.1, 0.2, 0.3]
    kd_range = [0.1, 0.3, 0.5, 0.7]
    
    # Run parameter sweep
    results = analyzer.parameter_sweep(kp_range, ki_range, kd_range)
    
    # Plot best results
    analyzer.plot_best_results(results, num_best=3)
    
    # Print detailed results
    print("\nDetailed Results for Best Parameters:")
    results.sort(key=lambda x: x.settling_time + x.rise_time + x.overshoot + x.steady_state_error)
    for i, r in enumerate(results[:5], 1):
        print(f"\nParameter Set {i}:")
        print(f"Kp={r.kp}, Ki={r.ki}, Kd={r.kd}")
        print(f"Rise Time: {r.rise_time:.3f}s")
        print(f"Settling Time: {r.settling_time:.3f}s")
        print(f"Overshoot: {r.overshoot:.1f}%")
        print(f"Steady State Error: {r.steady_state_error:.4f}")

if __name__ == "__main__":
    # Import the required classes from the previous file
    from duckiebot_linear_comparison import PIDController, DuckiebotODEModel, simulate_duckiebot_with_pid
    main()