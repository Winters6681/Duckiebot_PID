import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import control

@dataclass
class PerformanceSpecs:
    rise_time: Optional[float] = None          # Time to go from 10% to 90% of setpoint
    settling_time: Optional[float] = None      # Time to stay within ±2% of setpoint
    overshoot_percentage: Optional[float] = None  # Maximum overshoot percentage
    steady_state_error: Optional[float] = None # Steady-state error
    peak_time: Optional[float] = None          # Time to reach first peak
    delay_time: Optional[float] = None         # Time to reach 50% of setpoint

class PIDParameterCalculator:
    def __init__(self, system_params: Dict):
        """
        Initialize calculator with system parameters
        system_params should contain:
        - wheel_distance
        - wheel_radius
        - base_speed
        """
        self.L = system_params['wheel_distance']
        self.R = system_params['wheel_radius']
        self.v = system_params['base_speed']
        
        # Create linearized system transfer function
        self.sys = self._create_linear_system()

    def _create_linear_system(self):
        """Create linearized system transfer function"""
        # For Duckiebot, linearized around θ=0:
        # The transfer function is approximately:
        # G(s) = (R/L) / s
        num = [self.R/self.L]
        den = [1, 0]
        return control.TransferFunction(num, den)

    def _simulate_response(self, kp: float, ki: float, kd: float) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate closed-loop system response with given PID parameters"""
        # Create PID controller transfer function
        pid = control.TransferFunction([kd, kp, ki], [1, 0])
        
        # Create closed-loop system
        closed_loop = control.feedback(self.sys * pid, 1)
        
        # Simulate step response
        t = np.linspace(0, 10, 1000)
        y, t = control.step_response(closed_loop, t)
        
        return t, y

    def _calculate_performance_metrics(self, t: np.ndarray, y: np.ndarray) -> Dict:
        """Calculate performance metrics from time response"""
        steady_state = y[-1]
        
        # Rise time (10% to 90%)
        y_10 = 0.1 * steady_state
        y_90 = 0.9 * steady_state
        t_10 = t[np.where(y >= y_10)[0][0]]
        t_90 = t[np.where(y >= y_90)[0][0]]
        rise_time = t_90 - t_10
        
        # Peak value and time
        peak_idx = np.argmax(y)
        peak_value = y[peak_idx]
        peak_time = t[peak_idx]
        
        # Overshoot
        overshoot = (peak_value - steady_state) / steady_state * 100 if peak_value > steady_state else 0
        
        # Settling time (±2% band)
        settling_band = 0.02 * steady_state
        settling_idx = np.where(np.abs(y - steady_state) <= settling_band)[0][0]
        settling_time = t[settling_idx]
        
        # Delay time (50% of final value)
        delay_idx = np.where(y >= 0.5 * steady_state)[0][0]
        delay_time = t[delay_idx]
        
        return {
            'rise_time': rise_time,
            'settling_time': settling_time,
            'overshoot': overshoot,
            'steady_state_error': 1 - steady_state,
            'peak_time': peak_time,
            'delay_time': delay_time
        }

    def _objective_function(self, x: np.ndarray, specs: PerformanceSpecs) -> float:
        """Objective function for optimization"""
        kp, ki, kd = x
        
        # Simulate system
        t, y = self._simulate_response(kp, ki, kd)
        metrics = self._calculate_performance_metrics(t, y)
        
        # Calculate error terms
        error = 0
        weights = {
            'rise_time': 1.0,
            'settling_time': 1.0,
            'overshoot': 2.0,
            'steady_state_error': 2.0,
            'peak_time': 1.0,
            'delay_time': 1.0
        }
        
        if specs.rise_time is not None:
            error += weights['rise_time'] * (metrics['rise_time'] - specs.rise_time)**2
        
        if specs.settling_time is not None:
            error += weights['settling_time'] * (metrics['settling_time'] - specs.settling_time)**2
            
        if specs.overshoot_percentage is not None:
            error += weights['overshoot'] * (metrics['overshoot'] - specs.overshoot_percentage)**2
            
        if specs.steady_state_error is not None:
            error += weights['steady_state_error'] * (metrics['steady_state_error'] - specs.steady_state_error)**2
            
        if specs.peak_time is not None:
            error += weights['peak_time'] * (metrics['peak_time'] - specs.peak_time)**2
            
        if specs.delay_time is not None:
            error += weights['delay_time'] * (metrics['delay_time'] - specs.delay_time)**2
            
        return error

    def find_pid_parameters(self, specs: PerformanceSpecs, num_attempts: int = 5) -> Tuple[Dict, Dict]:
        """Find PID parameters that meet the specifications"""
        best_error = float('inf')
        best_result = None
        best_metrics = None
        
        for _ in range(num_attempts):
            # Random initial guess
            x0 = np.random.uniform([0.1, 0.01, 0.01], [10.0, 1.0, 1.0])
            
            # Optimize
            result = minimize(
                self._objective_function,
                x0,
                args=(specs,),
                bounds=[(0.1, 20.0), (0.0, 2.0), (0.0, 2.0)],
                method='SLSQP'
            )
            
            if result.fun < best_error:
                best_error = result.fun
                best_result = result
                # Calculate metrics for best result
                t, y = self._simulate_response(*result.x)
                best_metrics = self._calculate_performance_metrics(t, y)
        
        # Extract best parameters
        kp, ki, kd = best_result.x
        params = {'Kp': kp, 'Ki': ki, 'Kd': kd}
        
        return params, best_metrics

    def plot_response(self, kp: float, ki: float, kd: float):
        """Plot step response with given PID parameters"""
        t, y = self._simulate_response(kp, ki, kd)
        metrics = self._calculate_performance_metrics(t, y)
        
        plt.figure(figsize=(12, 8))
        plt.plot(t, y, 'b-', label='System Response')
        plt.axhline(y=1, color='r', linestyle='--', label='Setpoint')
        
        # Add annotations for metrics
        plt.plot(metrics['peak_time'], max(y), 'go', label='Peak')
        plt.plot(metrics['settling_time'], y[np.where(t >= metrics['settling_time'])[0][0]], 
                'mo', label='Settling Point')
        
        plt.title(f'Step Response (Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f})')
        plt.xlabel('Time [s]')
        plt.ylabel('Response')
        plt.grid(True)
        plt.legend()
        
        # Add metrics text
        text = f"Rise Time: {metrics['rise_time']:.3f}s\n"
        text += f"Settling Time: {metrics['settling_time']:.3f}s\n"
        text += f"Overshoot: {metrics['overshoot']:.1f}%\n"
        text += f"Steady State Error: {metrics['steady_state_error']:.3f}\n"
        text += f"Peak Time: {metrics['peak_time']:.3f}s\n"
        text += f"Delay Time: {metrics['delay_time']:.3f}s"
        
        plt.figtext(0.02, 0.02, text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.show()

def main():
    # System parameters
    system_params = {
        'wheel_distance': 0.1,
        'wheel_radius': 0.025,
        'base_speed': 0.5
    }
    
    # Create calculator
    calculator = PIDParameterCalculator(system_params)
    
    # Define desired specifications
    specs = PerformanceSpecs(
        rise_time=1,          # 0.5 seconds rise time
        overshoot_percentage=20, # 10% maximum overshoot
        steady_state_error=0  # 1% steady-state error
    )
    
    # Find PID parameters
    print("Finding PID parameters...")
    params, metrics = calculator.find_pid_parameters(specs)
    
    print("\nCalculated PID Parameters:")
    print(f"Kp: {params['Kp']:.3f}")
    print(f"Ki: {params['Ki']:.3f}")
    print(f"Kd: {params['Kd']:.3f}")
    
    print("\nAchieved Performance Metrics:")
    print(f"Rise Time: {metrics['rise_time']:.3f} s")
    print(f"Settling Time: {metrics['settling_time']:.3f} s")
    print(f"Overshoot: {metrics['overshoot']:.1f} %")
    print(f"Steady State Error: {metrics['steady_state_error']:.3f}")
    print(f"Peak Time: {metrics['peak_time']:.3f} s")
    print(f"Delay Time: {metrics['delay_time']:.3f} s")
    
    # Plot response
    calculator.plot_response(params['Kp'], params['Ki'], params['Kd'])

if __name__ == "__main__":
    main()