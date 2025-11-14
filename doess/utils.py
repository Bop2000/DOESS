import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import json
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

def load_data(
        path1,
        file_name='seq',
        num_initial=2000,
        num_per_round=20,
        ):
    data = pd.read_csv(path1+'/data.csv',index_col=0)
    input_x = np.array(data)[:,:24]
    input_y = np.array(data)[:,-1]
    indicators = pd.read_csv(path1+'/indicators.csv',index_col=0)
    indicators = np.array(indicators)
    
    pulse_matrices = np.load(path1+'/pulse_matrices.npy')

    print(input_x.shape,input_y.shape,indicators.shape)

    """Visulization"""
    print('fano factor of Full score:',np.var(input_y)/np.mean(input_y))
    plt.figure()
    _, bins = np.histogram(input_y, bins=50)
    plt.hist(input_y[:num_initial],
             bins=bins,
             label='Initial data')
    for i in range(1,1 + round((len(input_y)-num_initial)/num_per_round)):
        plt.hist(
            input_y[num_initial+(i-1)*num_per_round:num_initial+i*num_per_round],
            bins=bins,
            label=f'Round{i}')
    plt.xlabel('Simplified sim. score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(path1+'/data_distribution.png')
    plt.close()

    return input_x, pulse_matrices, input_y.flatten(), indicators


def get_initial_points(
        x: np.ndarray, 
        y: np.ndarray,
        y2: np.ndarray,
        threshold_array: np.ndarray,
        n_points: int = 1,
        ) -> np.ndarray:

    mask = np.all(y2 < threshold_array, axis=1)
    valid_indices = np.where(mask)[0]
    print(f"找到 {len(valid_indices)} 行满足所有阈值条件")
    print("前10个满足条件的行索引:", valid_indices[:10])
    x_valid = x[valid_indices]
    y_valid = y[valid_indices]
    
    ind = np.argsort(y_valid)[-n_points:]
    x_current_top = x_valid[ind]
    y_top = y_valid[ind]
    print('selected root nodes:',x_current_top)    
    print('y score of selected root nodes:',y_top)    

    return x_current_top.flatten(), float(y_top)


def BoxFeature(input_list):
    """
    get the feature of box figure.
    
    > @param[in] input_list:    the series
    return: 
    < @param[out] out_list:     the feature value
    < @param[out_note]:         [ave,min,Q1,Q2,Q3,max,error_number]
    """
    percentile = np.percentile(input_list, (25, 50, 75), interpolation='linear')
    Q1 = percentile[0]  # upper quartile
    Q2 = percentile[1]
    Q3 = percentile[2]  # lower quartile
    IQR = Q3 - Q1       # Interquartile range
    ulim = Q3 + 1.5*IQR # upper limit
    llim = Q1 - 1.5*IQR # lower limit
    # llim = 0 if llim < 0 else llim
    # out_list = [llim,Q1,Q2,Q3,ulim]
    # ------- count the number of anomalies ----------
    right_list = []     # normal data
    Error_Point_num = 0
    value_total = 0
    average_num = 0
    for item in input_list:
        if item < llim or item > ulim:
            Error_Point_num += 1
        else:
            right_list.append(item)
            value_total += item
            average_num += 1
    average_value =  value_total/average_num
    out_list = [average_value,min(right_list), Q1, Q2, Q3, max(right_list), Error_Point_num]
    return out_list




@dataclass
class Tracker:
    """A class for tracking optimization results and saving them periodically."""

    folder_name: str
    _counter: int = field(init=False, default=0)
    _results: list[float] = field(init=False, default_factory=list)
    _x_values: list[Optional[np.ndarray]] = field(init=False, default_factory=list)
    _current_best: float = field(init=False, default=float("-inf"))
    _current_best_x: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self):
        """Initialize the tracker and create the folder after instance creation."""

        self._create_folder()

    def _create_folder(self) -> None:
        """Create a folder to store results."""
        try:
            os.mkdir(self.folder_name)
            print(f"Successfully created the directory {self.folder_name}")
        except OSError:
            print(f"Creation of the directory {self.folder_name} failed")

    def dump_trace(self) -> None:
        """Save the current results to a file."""
        np.save(
            f"{self.folder_name}/result.npy", np.array(self._results), allow_pickle=True
        )

    def track(
        self, result: float, x: Optional[np.ndarray] = None, save: bool = False
    ) -> None:
        """Track a new result and update the best if necessary.

        Args:
            result: The current optimization result.
            x: The current x value.
            save: Whether to save results immediately.
        """
        self._counter += 1
        if result > self._current_best:
            self._current_best = result
            self._current_best_x = x

        self._print_status()
        self._results.append(result)
        self._x_values.append(x)

        # if save or self._counter % 20 == 0 or round(self._current_best, 5) == 0:
        #     self.dump_trace()

    def _print_status(self) -> None:
        """Print the current status of the optimization."""
        print("\n" + "=" * 10)
        print(f"# totalsamples: {self._counter}")
        print("=" * 10)
        print(f"current best f(x): {self._current_best}")
        print(f"current best x: {np.around(self._current_best_x, decimals=4)}")
        

    
    