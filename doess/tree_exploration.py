import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from typing import Any, Set, Optional, Dict, List

import time

import keras
import numpy as np

from .objective_func import ObjectiveFunction


@dataclass
class TreeExploration:
    func: ObjectiveFunction = None
    N: Dict[Any, int] = field(default_factory=lambda: defaultdict(int))
    children: Dict[Any, Set] = field(default_factory=dict)
    rollout_round: int = 100
    ratio: float = 0.01 # exploration weight ratio

    def choose(self, node):
        """Choose the best successor of node."""
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            """Upper confidence bound for trees"""
            return n.value + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n] + 1)
            )

        media_node = max(self.children[node], key=uct)

        node_all = [
            list(list(self.children[node])[i].tup) + [list(self.children[node])[i].value]
            for i in range(len(self.children[node]))
        ]
        print('uct of root:',uct(node),'value of root:',node.value)
        print('uct of best leaf:',uct(media_node),'value of best leaf:',media_node.value)

        return (
            (media_node, node_all)
            if uct(media_node) > uct(node)
            else (node, node_all)
        )

    def do_rollout(self, node):
        """Make the tree one layer better. (Train for one iteration.)"""
        self._expand(node)
        self._backpropagate(path=node)

    @staticmethod
    def data_process(x: np.ndarray, boards: List[list]) -> np.ndarray:
        new_x = []
        boards = np.unique(np.array(boards), axis=0)
        new_x = [board for board in boards if not np.any(np.all(board == x, axis=1))]
        # print(f"Unique number of boards: {len(new_x)}")
        return np.array(new_x)

    def rollout(
        self,
        initial_X: np.ndarray,
        initial_y: np.ndarray,
    ) -> np.ndarray:
        """Perform rollout."""
        initial_X = initial_X.flatten()
        self.exploration_weight = self.ratio * abs(initial_y)
        board_uct = OptTask(tup=tuple(initial_X), value=abs(initial_y), terminal=False)

        nodes_all = []
        for _ in range(0, self.rollout_round):
            self.do_rollout(board_uct)
            board_uct, node_all = self.choose(board_uct)
            nodes_all.extend(list(node_all))
        
        # new_x, new_x2, new_y, new_y2
        new_x = [nodes[:-1] for nodes in nodes_all]
        new_y = [nodes[-1] for nodes in nodes_all]
        inds = np.argsort(new_y)[-len(new_y)//10:]
        new_x2, new_y2 = self.func.get_pulse_matrix_indicators_ray(
            [new_x[i] for i in inds]
            )
        return new_x, new_x2, new_y, new_y2 


    def _expand(self, node):
        """Update the `children` dict with the children of `node`"""
        action = list(range(len(node.tup)))
        self.children[node] = node.find_children(
            node, action, self.func
        )

    def _backpropagate(self, path):
        """Send the reward back up to the ancestors of the leaf"""
        self.N[path] += 1


class Node(ABC):
    """
    A representation of a single board state.
    DOTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        """All possible successors of this board state"""
        return set()

    @abstractmethod
    def is_terminal(self):
        """Returns True if the node has no children"""
        return True

    @abstractmethod
    def __hash__(self):
        """Nodes must be hashable"""
        return 123456789

    @abstractmethod
    def __eq__(self, node2):
        """Nodes must be comparable"""
        return True


_OT = namedtuple("OptTask", "tup value terminal")


class OptTask(_OT, Node):
    """Represents an optimization task node in the search tree."""

    @staticmethod
    def find_children(board, action, func):
        """Find all possible child nodes for the current board state."""
        if board.terminal:
            return set()

        all_tuples = OptTask._generate_child_tuples(board, action, func)
        
        t1=time.time()
        """sequential: low efficiency"""
        # all_values = [func.get_score_with_constraints(tup) for tup in all_tuples] 
        
        """parallel: use ray"""
        all_values = func.get_score_ray(all_tuples)
        print(f'Computing time: {time.time()-t1}s')
        
        return {OptTask(tuple(t), v, False) for t, v in zip(all_tuples, all_values)}

    @staticmethod
    def _generate_child_tuples(board, action, func):
        """Generate child tuples based on the current board state and function parameters."""
        
        all_tuples = []
        for index in action:
            tup = list(board.tup)
            tup = OptTask._apply_random_modification(
                tup, index, func
            )
            all_tuples.append(tup)

        return all_tuples

    @staticmethod
    def _apply_random_modification(tup0, index, func):
        """Apply a random modification to the tuple."""
        
        tup = list(tup0)
        possible_values = func.allowed_values
        
        flip = random.randint(0,7)
        if flip in (0,1,2):
            ind = np.where(possible_values==tup[index])[0][0]
            if flip == 0:
                try:
                    tup[index] = possible_values[ind+1]
                except:
                    tup[index] = possible_values[ind-1]
                
            elif flip == 1:
                try:
                    tup[index] = possible_values[ind-1]
                except:
                    tup[index] = possible_values[ind+1]
            else:
                tup[index] = np.random.choice(possible_values)

        elif flip in (3,4,5,6,7):
            d = func.dims
            num_flip = np.random.choice((int(d/2),int(d/3),int(d/4),int(d/5),int(d/10)))
            ind_flip = np.random.choice(np.arange(len(tup)),num_flip,replace=False)
            for ind in ind_flip:
                tup[ind] = np.random.choice(possible_values)
        tup[index] = round(tup[index])
        return np.array(tup)

    def is_terminal(self):
        """Check if the current board state is terminal."""
        return self.terminal