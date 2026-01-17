from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .engine import DotsAndBoxes, GameState


class PolicyValueNet:
    def predict(self, state: GameState) -> Tuple[np.ndarray, float]:
        raise NotImplementedError


class RandomPolicyValueNet(PolicyValueNet):
    def __init__(self, action_size: int) -> None:
        self.action_size = action_size

    def predict(self, state: GameState) -> Tuple[np.ndarray, float]:
        policy = np.ones(self.action_size, dtype=np.float32)
        policy /= policy.sum()
        value = 0.0
        return policy, value


@dataclass
class Node:
    prior: float
    player: int
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, game: DotsAndBoxes, network: PolicyValueNet, simulations: int = 100, c_puct: float = 1.5) -> None:
        self.game = game
        self.network = network
        self.simulations = simulations
        self.c_puct = c_puct

    def run(self, state: GameState) -> Node:
        root = Node(prior=1.0, player=state.current_player)
        self._expand(root, state)
        for _ in range(self.simulations):
            state_copy = state.copy()
            path = [root]
            node = root

            while node.expanded() and not self.game.is_terminal(state_copy):
                action, node = self._select_child(node)
                self.game.apply_action(state_copy, action)
                path.append(node)

            if self.game.is_terminal(state_copy):
                value = self._terminal_value(state_copy, node.player)
            else:
                value = self._expand(node, state_copy)

            self._backpropagate(path, value)
        return root

    def _select_child(self, node: Node) -> Tuple[int, Node]:
        best_score = -float("inf")
        best_action = -1
        best_child = None
        for action, child in node.children.items():
            score = self._ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        if best_child is None:
            raise RuntimeError("No child selected")
        return best_action, best_child

    def _ucb_score(self, parent: Node, child: Node) -> float:
        pb_c = math.log((parent.visit_count + 1) / 1.0) + self.c_puct
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        value_score = child.value()
        return prior_score + value_score

    def _expand(self, node: Node, state: GameState) -> float:
        policy, value = self.network.predict(state)
        legal_actions = self.game.legal_actions(state)
        if not legal_actions:
            return 0.0
        policy = policy.copy()
        mask = np.zeros_like(policy)
        mask[legal_actions] = 1.0
        policy *= mask
        policy_sum = policy.sum()
        if policy_sum <= 0:
            policy = mask / mask.sum()
        else:
            policy /= policy_sum
        for action in legal_actions:
            next_state = state.copy()
            self.game.apply_action(next_state, action)
            node.children[action] = Node(prior=float(policy[action]), player=next_state.current_player)
        return float(value)

    def _terminal_value(self, state: GameState, player: int) -> float:
        winner = state.winner()
        if winner == -1:
            return 0.0
        return 1.0 if winner == player else -1.0

    def _backpropagate(self, path: List[Node], value: float) -> None:
        for index in range(len(path) - 1, -1, -1):
            node = path[index]
            node.visit_count += 1
            node.value_sum += value
            if index > 0:
                parent = path[index - 1]
                if parent.player != node.player:
                    value = -value


def select_action_from_root(root: Node, action_size: int, temperature: float = 1.0) -> Tuple[int, np.ndarray]:
    visit_counts = np.array([child.visit_count for child in root.children.values()], dtype=np.float32)
    actions = list(root.children.keys())
    if temperature <= 0:
        best_index = int(np.argmax(visit_counts))
        action = actions[best_index]
        probs = np.zeros_like(visit_counts)
        probs[best_index] = 1.0
    else:
        adjusted = np.power(visit_counts, 1.0 / temperature)
        adjusted /= adjusted.sum()
        action = int(np.random.choice(actions, p=adjusted))
        probs = adjusted
    policy = np.zeros(action_size, dtype=np.float32)
    policy[actions] = probs
    return action, policy


def self_play(
    game: DotsAndBoxes,
    network: PolicyValueNet,
    simulations: int = 50,
    temperature: float = 1.0,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    state = game.new_game()
    mcts = MCTS(game, network, simulations=simulations)
    trajectory: List[Tuple[np.ndarray, np.ndarray, int]] = []

    while not game.is_terminal(state):
        root = mcts.run(state)
        action, policy = select_action_from_root(root, state.action_size, temperature)
        trajectory.append((state.to_tensor(), policy, state.current_player))
        game.apply_action(state, action)

    winner = state.winner()
    results: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for obs, policy, player in trajectory:
        if winner == -1:
            value = 0.0
        else:
            value = 1.0 if player == winner else -1.0
        results.append((obs, policy, value))
    return results
