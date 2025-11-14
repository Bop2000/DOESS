from typing import Dict, List, Callable, Tuple, Optional, Any
import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate

# Type aliases
ObjectiveFunction = Callable[[List[float], Dict[str, Any]], float]
FittingResult = Tuple[np.ndarray, np.ndarray, np.ndarray]

def exp_func(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    t = t.astype(np.float64)
    return np.where(-b*t > -500, a*np.exp(-b*t), 0) + c


def fit_exponential(
    results: List[float], 
    training_params: Dict[str, Any],
) -> FittingResult:
    
    max_time = training_params['max_time']
    min_time = training_params['min_time']
    num_points = training_params['num_points']

    x_data = np.linspace(min_time, max_time, num_points)
    y_data = np.array(results)

    p0 = [0.9, 0.0003, 0.05]  # Initial guess for parameters a, b, and c

    try:
        popt, _ = curve_fit(exp_func, x_data, y_data, p0=p0)
    except RuntimeError:
        popt = np.array([0.0, 0.0, 0.0])

    fit_points = training_params['fit_points']
    x_fit = np.linspace(min_time, max_time, fit_points)
    y_fit = exp_func(x_fit, *popt)
    
    return popt, x_fit, y_fit

def find_x_for_y(y: float, popt: np.ndarray) -> float:
    """
    Calculate the x-value for a given y-value on the fitted exponential curve.

    Parameters:
    y (float): The y-value to find the corresponding x-value for.
    popt (np.ndarray): Optimal values for the parameters (a, b, c).

    Returns:
    float: The corresponding x-value.

    Raises:
    ValueError: If the given y-value is outside the function range.
    """
    a, b, c = popt
    
    if y > a + c:
        return 0
    if y < c:
        return np.inf
    
    return -np.log((y - c) / a) / b

# Objective functions
def arith_mean_to_criticals(
    results: List[float],
    training_params: Dict[str, Any]
) -> float:
    """Calculate mean time to reach critical points."""
    popt, _, _ = fit_exponential(results, training_params)
    critical_points = training_params['critical_points']
    times_to_critical = [
        find_x_for_y(point, popt) for point in critical_points
    ]
    return np.mean(times_to_critical)

def geo_mean_to_criticals(
    results: List[float],
    training_params: Dict[str, Any]
) -> float:
    """Calculate geometric mean time to reach critical points."""
    popt, _, _ = fit_exponential(results, training_params)
    critical_points = training_params['critical_points']
    times_to_critical = [
        find_x_for_y(point, popt) for point in critical_points
    ]
    return np.prod(times_to_critical) ** (1 / len(times_to_critical))


def decay_time(
    results: List[float],
    training_params: Dict[str, Any]
) -> float:
    """Extract decay rate from fitted exponential."""
    popt, _, _ = fit_exponential(results, training_params)
    return 1 / popt[1]

def enhanced_decay_time(
    results: List[float],
    training_params: Dict[str, Any]
) -> float:
    mean_to_criticals = arith_mean_to_criticals(results, training_params)
    popt, _, _ = fit_exponential(results, training_params)

    return mean_to_criticals / popt[1]

def area_under_curve(
    results: List[float],
    training_params: Dict[str, Any]
) -> float:
    """Calculate area under the fitted curve."""
    _, x_fit, y_fit = fit_exponential(results, training_params)
    return integrate.trapz(y_fit, x_fit)


def sensitivity(
    results: List[float],
    training_params: Dict[str, Any]
) -> float:
    """Calculate the sensitivity of the results."""
    popt, _, _ = fit_exponential(results, training_params)
    return popt[0] / popt[1]


def arithmetic_mean(
    results: List[float],
    training_params: Dict[str, Any]
) -> float:
    """Calculate the arithmetic mean of the results."""
    return np.mean(results)

def geometric_mean(
    results: List[float],
    training_params: Dict[str, Any]
) -> float:
    """Calculate the geometric mean of the results."""
    return np.prod(results) ** (1 / len(results))

# Dictionary to store all objective functions
OBJECTIVE_FUNCTIONS: Dict[str, ObjectiveFunction] = {
    'arithmetic_mean': arithmetic_mean,
    'geometric_mean': geometric_mean,
}

def register_objective_function(name: str, func: ObjectiveFunction) -> None:
    """Register a new objective function."""
    OBJECTIVE_FUNCTIONS[name] = func

def get_objective_function(name: str) -> ObjectiveFunction:
    """
    Get an objective function by name.

    Raises:
    KeyError: If the objective function is not found.
    """
    return OBJECTIVE_FUNCTIONS[name]

def get_all_objective_functions() -> Dict[str, ObjectiveFunction]:
    """Get all registered objective functions."""
    return OBJECTIVE_FUNCTIONS.copy()