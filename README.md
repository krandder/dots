# Dots and Boxes Engine

This repository contains a lightweight Dots-and-Boxes engine along with an AlphaZero-style
self-play loop. The implementation supports arbitrary `(rows, cols)` board sizes and is
intended as a foundation for reinforcement learning experiments.

## Package Overview

- `dots/engine.py`: Core game engine, including game state, legal moves, scoring, and
  observation tensor encoding.
- `dots/alphazero.py`: MCTS implementation, a random policy/value network stub, and
  a self-play loop that returns `(observation, policy, value)` training data.
- `dots/__init__.py`: Public API exports.

## Quick Start

```python
from dots import DotsAndBoxes, RandomPolicyValueNet, self_play

game = DotsAndBoxes(rows=2, cols=3)
network = RandomPolicyValueNet(game.new_game().action_size)
trajectory = self_play(game, network, simulations=25, temperature=1.0)
```

## Command Line Usage

Run a small self-play training loop (defaults: 4x4 board, 50 simulations per move) and
save loss plot and best models into `runs/`:

```bash
python -m dots.cli train
```

Print a single self-play game to the terminal:

```bash
python -m dots.cli print-game
```

## Notes

- The `RandomPolicyValueNet` is a stub meant to be replaced with a learned model.
- The engine uses NumPy for state storage and tensor creation.
