from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import numpy as np
from typing import Any, Set, Optional, Dict, List


import ray
ray.init(num_gpus=0,ignore_reinit_error=True)

from .utils import Tracker
from sim.simulation import simulator as Simulator


@dataclass
class ObjectiveFunction(ABC):
    
    dims: int = 10
    lb: np.ndarray = field(init=False)
    ub: np.ndarray = field(init=False)
    
    tracker: Tracker = field(init=False)

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        pass

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x).round(0)
        assert len(x) == self.dims
        assert x.ndim == 1
        return x


@dataclass
class HamEngineering(ObjectiveFunction):
    name: str       = "simplified_sim_score"
    
    dims: int       = 24
    lb: int         = 0
    ub: int         = 12
    interval: int   = 1

    file_name: str  = 'configurations'
    file_path: str  = 'configN2'
    objective: str  = 'arithmetic_mean'
    threshold: Dict[Any, int] = field(default_factory=lambda: {
        '#1': 0.2, # 0.2, # disorder during pulse intervals
        '#2': 0.4, # 0.4, # disorder during finite pulse
        '#3': 1.0, # interactions during pulse intervals
        '#4': 0.8, # interactions during finite pulse
        '#5': 0.5, # pulse errors
        })
    
    "parameters for simulation"
    repetition: int = 500


    def __post_init__(self):
        self.tracker = Tracker(self.name + str(self.dims))

        self.SIM = Simulator(self.file_name, self.file_path)
        self.SIM.print_info()

        self.allowed_values = np.arange(self.lb, self.ub + self.interval*0.1, self.interval).round(5)

    
    def __call__(self, x: np.ndarray) -> float:
        """Return simplified simulation score of this sequence"""
        x = self._preprocess(x)
        self.SIM.set_sequence(input=list([int(i) for i in list(x)]))
        sim_score, _ = self.SIM.get_performance_metrics(
            repetitions=self.repetition, objective=self.objective)
        sim_score = (sim_score-0.5)*2
        return sim_score
    
    def get_score_ray(self, all_tuples0, num_workers=None):
        
        # filtered by predicted indicators #1-#3
        inds = self.aht_filter(all_tuples0) 
        all_tuples = [all_tuples0[ind] for ind in inds]
        
        # initialize ray remote
        if num_workers is None:
            num_workers = min(16, len(all_tuples))
        
        workers = [
            MetricComputer.remote(self.repetition,
                                  self.file_name, self.file_path,
                                  self.threshold,
                                  self.objective) 
            for _ in range(num_workers)
        ]
        
        batch_size = max(1, len(all_tuples) // num_workers)
        batches = [
            all_tuples[i:i + batch_size] 
            for i in range(0, len(all_tuples), batch_size)
        ]
        
        futures = []
        for i, batch in enumerate(batches):
            worker = workers[i % len(workers)]
            future = worker.compute_batch.remote(batch)
            futures.append(future)
        
        # parallelized computing by ray
        all_results = []
        for future in futures:
            batch_results = ray.get(future)
            all_results.extend(batch_results)
        
        all_scores = np.zeros(len(all_tuples0))
        all_scores[inds] = np.array(all_results)[:,-1]
        for sim_score,x in zip(all_scores,all_tuples0):
            # print('sim. score:',sim_score)
            self.tracker.track(sim_score, x)
        return all_scores
    
    def aht_filter(self, all_tuples):
        """Return the valid indices filtered by predicted indicators #1-#3"""
        pulse_matrices = [self.get_pulse_matrix(x) for x in all_tuples]
        s1_pred,s2_pred,s3_pred = self.get_aht_s1_s2_s3_pred(np.array(pulse_matrices))
        thre = self.threshold
        inds = []
        for i,(s1,s2,s3) in enumerate(zip(s1_pred,s2_pred,s3_pred)):
            if s1 < thre['#1'] and s2 < thre['#2'] and s3 < thre['#3']:
                inds.append(i)
        return inds
    
    def get_score_with_constraints(self, x: np.ndarray, track: bool = True) -> float:
        x = self._preprocess(x)
        self.SIM.set_sequence(input=list([int(i) for i in list(x)]))
        pulse_matrices = self.SIM.seq.pulse_matrices()

        s4, s5 = self.SIM.aht_s4_s5

        s1 = float(self.model1.pred(pulse_matrices[np.newaxis]))
        s2 = float(self.model2.pred(pulse_matrices[np.newaxis]))
        s3 = float(self.model3.pred(pulse_matrices[np.newaxis]))
        print('s1-s5:',s1,s2,s3,s4,s5)
        
        # computing simplified simulated score only if the indicator scores are 
        # below the acceptance threshold
        thre = self.threshold
        if s1 < thre['#1'] and s2 < thre['#2'] and s3 < thre['#3'] and s4 < thre['#4'] and s5 < thre['#5']:
            t1=time.time()
            sim_score, results = self.SIM.get_performance_metrics(
                repetitions=self.repetition, objective=self.objective)
            sim_score = (sim_score-0.5)*2
            t2=time.time()
            print('computing time: ',t2-t1,'s')
        else:
            sim_score = 0
        if track:
            self.tracker.track(sim_score, x)
        return sim_score
    
    def get_pulse_matrix_indicators_ray(self, all_tuples, num_workers=None):
        
        # initialize ray remote
        if num_workers is None:
            num_workers = min(16, len(all_tuples))
        
        workers = [
            MetricComputer.remote(self.repetition,
                                  self.file_name, self.file_path,
                                  self.threshold,
                                  self.objective) 
            for _ in range(num_workers)
        ]
        
        batch_size = max(1, len(all_tuples) // num_workers)
        batches = [
            all_tuples[i:i + batch_size] 
            for i in range(0, len(all_tuples), batch_size)
        ]
        
        futures = []
        for i, batch in enumerate(batches):
            worker = workers[i % len(workers)]
            future = worker.compute_batch2.remote(batch)
            futures.append(future)
        
        # parallelized computing by ray
        all_results = []
        for future in futures:
            batch_results = ray.get(future)
            all_results.extend(batch_results)
        
        new_x2 = [results[0] for results in all_results]
        new_y2 = [results[1] for results in all_results]
        return new_x2, new_y2
    
    def get_pulse_matrix(self, x):
        self.SIM.set_sequence(input=list([int(i) for i in list(x)]))
        return self.SIM.seq.pulse_matrices()
    
    def get_aht_score(self, x):
        self.SIM.set_sequence(input=list([int(i) for i in list(x)]))
        s1, s2, s3, s4, s5 = self.SIM.aht_score
        return s1, s2, s3, s4, s5
    
    def get_aht_s1_s2_s3(self, x):
        self.SIM.set_sequence(input=list([int(i) for i in list(x)]))
        s1, s2, s3 = self.SIM.aht_s1_s2_s3
        return s1, s2, s3
    
    def get_aht_s1_s2_s3_pred(self, pulse_matrices):
        s1s = self.model1.pred(pulse_matrices)
        s2s = self.model2.pred(pulse_matrices)
        s3s = self.model3.pred(pulse_matrices)
        return s1s, s2s, s3s
    
    def get_aht_s4_s5(self, x):
        self.SIM.set_sequence(input=list([int(i) for i in list(x)]))
        s4, s5 = self.SIM.aht_s4_s5
        return s4, s5

    


@ray.remote
class MetricComputer:
    
    def __init__(self, repetition,file_name,file_path,threshold,objective):
        self.repetition = repetition
        self.SIM = Simulator(file_name,file_path)
        self.threshold = threshold
        self.objective = objective

    def compute_batch(self, batch_tuples):
        """Computing simplified simulation score"""
        results = []
        for x in batch_tuples:
            self.SIM.set_sequence(input=list([int(i) for i in list(x)]))
            s4, s5 = self.SIM.aht_s4_s5

            # computing simplified simulated score only if the 
            # indicator scores #4 and #5 are below the acceptance threshold
            thre = self.threshold
            if s4 < thre['#4'] and s5 < thre['#5']:
                
                sim_score, _ = self.SIM.get_performance_metrics(
                    repetitions=self.repetition, objective=self.objective)
                sim_score = (sim_score-0.5)*2
            else:
                sim_score = 0
                print('indicator scores #4 and #5 are not satisfied')
            results.append(list(x)+[sim_score])
        return results
    
    def compute_batch2(self, batch_tuples):
        """Computing pulse_matrices and indicators"""
        results = []
        for x in batch_tuples:
            self.SIM.set_sequence(input=list([int(i) for i in list(x)]))
            pulse_matrix = self.SIM.seq.pulse_matrices()
            s1, s2, s3, s4, s5 = self.SIM.aht_score
            results.append([pulse_matrix,[s1, s2, s3, s4, s5]])
        return results
    
