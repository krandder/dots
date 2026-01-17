from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .engine import GameState


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


@dataclass
class LinearPolicyValueNet:
    input_dim: int
    action_size: int
    learning_rate: float = 0.01

    def __post_init__(self) -> None:
        rng = np.random.default_rng(0)
        self.policy_weights = rng.normal(scale=0.01, size=(self.action_size, self.input_dim)).astype(np.float32)
        self.policy_bias = np.zeros(self.action_size, dtype=np.float32)
        self.value_weights = rng.normal(scale=0.01, size=(self.input_dim,)).astype(np.float32)
        self.value_bias = np.zeros(1, dtype=np.float32)

    def predict(self, state: GameState) -> Tuple[np.ndarray, float]:
        x = state.to_tensor().astype(np.float32).ravel()
        logits = self.policy_weights @ x + self.policy_bias
        policy = _softmax(logits)
        value = float(np.tanh(self.value_weights @ x + self.value_bias))
        return policy, value

    def train_batch(
        self,
        observations: Sequence[np.ndarray],
        target_policies: Sequence[np.ndarray],
        target_values: Sequence[float],
    ) -> float:
        batch_size = len(observations)
        if batch_size == 0:
            return 0.0
        policy_loss = 0.0
        value_loss = 0.0
        grad_policy_w = np.zeros_like(self.policy_weights)
        grad_policy_b = np.zeros_like(self.policy_bias)
        grad_value_w = np.zeros_like(self.value_weights)
        grad_value_b = np.zeros_like(self.value_bias)

        for obs, target_policy, target_value in zip(observations, target_policies, target_values):
            x = obs.astype(np.float32).ravel()
            logits = self.policy_weights @ x + self.policy_bias
            policy = _softmax(logits)
            policy_loss += -float(np.sum(target_policy * np.log(policy + 1e-8)))
            grad_logits = policy - target_policy
            grad_policy_w += np.outer(grad_logits, x)
            grad_policy_b += grad_logits

            value = float(np.tanh(self.value_weights @ x + self.value_bias))
            value_error = value - target_value
            value_loss += 0.5 * value_error**2
            grad_value = value_error * (1 - value**2)
            grad_value_w += grad_value * x
            grad_value_b += grad_value

        scale = 1.0 / batch_size
        self.policy_weights -= self.learning_rate * grad_policy_w * scale
        self.policy_bias -= self.learning_rate * grad_policy_b * scale
        self.value_weights -= self.learning_rate * grad_value_w * scale
        self.value_bias -= self.learning_rate * grad_value_b * scale

        total_loss = (policy_loss + value_loss) * scale
        return total_loss

    def serialize(self) -> dict:
        return {
            "policy_weights": self.policy_weights,
            "policy_bias": self.policy_bias,
            "value_weights": self.value_weights,
            "value_bias": self.value_bias,
        }

    def load_state(self, state: dict) -> None:
        self.policy_weights = state["policy_weights"]
        self.policy_bias = state["policy_bias"]
        self.value_weights = state["value_weights"]
        self.value_bias = state["value_bias"]


def apply_symmetry_observation(observation: np.ndarray, symmetry: str) -> np.ndarray:
    if symmetry == "identity":
        return observation
    if symmetry == "rot90":
        return np.rot90(observation, k=1, axes=(1, 2))
    if symmetry == "rot180":
        return np.rot90(observation, k=2, axes=(1, 2))
    if symmetry == "rot270":
        return np.rot90(observation, k=3, axes=(1, 2))
    if symmetry == "flip_h":
        return np.flip(observation, axis=2)
    if symmetry == "flip_v":
        return np.flip(observation, axis=1)
    raise ValueError(f"Unknown symmetry: {symmetry}")


def apply_symmetry_policy(
    policy: np.ndarray,
    rows: int,
    cols: int,
    symmetry: str,
) -> np.ndarray:
    h_count = (rows + 1) * cols
    h_edges = policy[:h_count].reshape(rows + 1, cols)
    v_edges = policy[h_count:].reshape(rows, cols + 1)

    if symmetry in {"rot90", "rot270"} and rows != cols:
        raise ValueError("90-degree rotations require square boards")

    def transform(point: Tuple[int, int]) -> Tuple[int, int]:
        r, c = point
        if symmetry == "identity":
            return r, c
        if symmetry == "rot90":
            return cols - c, r
        if symmetry == "rot180":
            return rows - r, cols - c
        if symmetry == "rot270":
            return c, rows - r
        if symmetry == "flip_h":
            return r, cols - c
        if symmetry == "flip_v":
            return rows - r, c
        raise ValueError(f"Unknown symmetry: {symmetry}")

    new_h = np.zeros_like(h_edges)
    new_v = np.zeros_like(v_edges)

    for r in range(rows + 1):
        for c in range(cols):
            value = h_edges[r, c]
            p1 = transform((r, c))
            p2 = transform((r, c + 1))
            if p1[0] == p2[0]:
                nr = p1[0]
                nc = min(p1[1], p2[1])
                new_h[nr, nc] = value
            else:
                nr = min(p1[0], p2[0])
                nc = p1[1]
                new_v[nr, nc] = value

    for r in range(rows):
        for c in range(cols + 1):
            value = v_edges[r, c]
            p1 = transform((r, c))
            p2 = transform((r + 1, c))
            if p1[0] == p2[0]:
                nr = p1[0]
                nc = min(p1[1], p2[1])
                new_h[nr, nc] = value
            else:
                nr = min(p1[0], p2[0])
                nc = p1[1]
                new_v[nr, nc] = value

    return np.concatenate([new_h.ravel(), new_v.ravel()])


def available_symmetries(rows: int, cols: int) -> List[str]:
    symmetries = ["identity", "rot180", "flip_h", "flip_v"]
    if rows == cols:
        symmetries.extend(["rot90", "rot270"])
    return symmetries
