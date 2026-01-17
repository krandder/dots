from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .alphazero import MCTS, RandomPolicyValueNet, select_action_from_root, self_play
from .engine import DotsAndBoxes
from .model import (
    LinearPolicyValueNet,
    apply_symmetry_observation,
    apply_symmetry_policy,
    available_symmetries,
)


def run_self_play(args: argparse.Namespace) -> None:
    game = DotsAndBoxes(args.rows, args.cols)
    state = game.new_game()
    network = RandomPolicyValueNet(state.action_size)
    mcts = MCTS(game, network, simulations=args.num_sims)

    move_index = 1
    print(state.render())
    while not game.is_terminal(state):
        root = mcts.run(state)
        action, _ = select_action_from_root(root, state.action_size, temperature=0.0)
        game.apply_action(state, action)
        print(f"\nMove {move_index}: player {state.current_player}")
        print(state.render())
        move_index += 1

    winner = state.winner()
    if winner == -1:
        print("\nGame finished in a draw.")
    else:
        print(f"\nWinner: player {winner}")


def train(args: argparse.Namespace) -> None:
    game = DotsAndBoxes(args.rows, args.cols)
    state = game.new_game()
    input_dim = int(np.prod(state.to_tensor().shape))
    network = LinearPolicyValueNet(input_dim, state.action_size, learning_rate=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    losses: List[float] = []
    best_loss = float("inf")
    symmetries = available_symmetries(args.rows, args.cols)

    for game_index in range(1, args.games + 1):
        trajectory = self_play(game, network, simulations=args.num_sims, temperature=args.temperature)
        observations: List[np.ndarray] = []
        policies: List[np.ndarray] = []
        values: List[float] = []
        for obs, policy, value in trajectory:
            for symmetry in symmetries:
                observations.append(apply_symmetry_observation(obs, symmetry))
                policies.append(apply_symmetry_policy(policy, args.rows, args.cols, symmetry))
                values.append(value)

        loss = network.train_batch(observations, policies, values)
        losses.append(loss)
        print(f"Game {game_index}: loss={loss:.4f}")

        if game_index % args.save_every == 0:
            avg_loss = float(np.mean(losses[-args.save_every:]))
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = out_dir / f"best_model_game_{game_index}.npz"
                np.savez(save_path, **network.serialize())
                print(f"Saved best model to {save_path}")

    fig, ax = plt.subplots()
    ax.plot(range(1, len(losses) + 1), losses, marker="o")
    ax.set_xlabel("Game")
    ax.set_ylabel("Loss")
    ax.set_title("Self-play Training Loss")
    fig.tight_layout()
    plot_path = out_dir / "loss_curve.png"
    fig.savefig(plot_path)
    print(f"Saved loss plot to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dots-and-Boxes self-play tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run self-play training loop")
    train_parser.add_argument("--games", type=int, default=20, help="Number of self-play games")
    train_parser.add_argument("--rows", type=int, default=4, help="Board rows")
    train_parser.add_argument("--cols", type=int, default=4, help="Board columns")
    train_parser.add_argument("--num-sims", type=int, default=50, help="MCTS simulations per move")
    train_parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    train_parser.add_argument("--save-every", type=int, default=5, help="Save best model every N games")
    train_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    train_parser.add_argument("--out-dir", type=str, default="runs", help="Output directory")
    train_parser.set_defaults(func=train)

    print_parser = subparsers.add_parser("print-game", help="Print a self-play game to the terminal")
    print_parser.add_argument("--rows", type=int, default=4, help="Board rows")
    print_parser.add_argument("--cols", type=int, default=4, help="Board columns")
    print_parser.add_argument("--num-sims", type=int, default=50, help="MCTS simulations per move")
    print_parser.set_defaults(func=run_self_play)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
