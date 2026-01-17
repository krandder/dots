import copy
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Move = Tuple[str, int, int]


def _empty_edges(rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray]:
    h_edges = np.zeros((rows + 1, cols), dtype=bool)
    v_edges = np.zeros((rows, cols + 1), dtype=bool)
    return h_edges, v_edges


def _empty_boxes(rows: int, cols: int) -> np.ndarray:
    return -np.ones((rows, cols), dtype=int)


@dataclass
class GameState:
    rows: int
    cols: int
    h_edges: np.ndarray
    v_edges: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    current_player: int

    @classmethod
    def new(cls, rows: int, cols: int) -> "GameState":
        h_edges, v_edges = _empty_edges(rows, cols)
        boxes = _empty_boxes(rows, cols)
        scores = np.zeros(2, dtype=int)
        return cls(rows, cols, h_edges, v_edges, boxes, scores, 0)

    def copy(self) -> "GameState":
        return GameState(
            self.rows,
            self.cols,
            self.h_edges.copy(),
            self.v_edges.copy(),
            self.boxes.copy(),
            self.scores.copy(),
            self.current_player,
        )

    @property
    def action_size(self) -> int:
        return (self.rows + 1) * self.cols + self.rows * (self.cols + 1)

    def is_terminal(self) -> bool:
        return self.h_edges.all() and self.v_edges.all()

    def winner(self) -> int:
        if self.scores[0] > self.scores[1]:
            return 0
        if self.scores[1] > self.scores[0]:
            return 1
        return -1

    def legal_moves(self) -> List[Move]:
        moves: List[Move] = []
        for r in range(self.rows + 1):
            for c in range(self.cols):
                if not self.h_edges[r, c]:
                    moves.append(("h", r, c))
        for r in range(self.rows):
            for c in range(self.cols + 1):
                if not self.v_edges[r, c]:
                    moves.append(("v", r, c))
        return moves

    def move_to_index(self, move: Move) -> int:
        kind, r, c = move
        if kind == "h":
            return r * self.cols + c
        if kind == "v":
            return (self.rows + 1) * self.cols + r * (self.cols + 1) + c
        raise ValueError(f"Unknown move kind: {kind}")

    def index_to_move(self, index: int) -> Move:
        h_count = (self.rows + 1) * self.cols
        if index < h_count:
            r, c = divmod(index, self.cols)
            return ("h", r, c)
        index -= h_count
        r, c = divmod(index, self.cols + 1)
        return ("v", r, c)

    def apply_move(self, move: Move) -> int:
        kind, r, c = move
        if kind == "h":
            if self.h_edges[r, c]:
                raise ValueError("Horizontal edge already filled")
            self.h_edges[r, c] = True
        elif kind == "v":
            if self.v_edges[r, c]:
                raise ValueError("Vertical edge already filled")
            self.v_edges[r, c] = True
        else:
            raise ValueError(f"Unknown move kind: {kind}")

        completed = self._claim_completed_boxes(kind, r, c)
        if completed == 0:
            self.current_player = 1 - self.current_player
        return completed

    def _claim_completed_boxes(self, kind: str, r: int, c: int) -> int:
        completed = 0
        for box_r, box_c in self._adjacent_boxes(kind, r, c):
            if self.boxes[box_r, box_c] != -1:
                continue
            if self._is_box_complete(box_r, box_c):
                self.boxes[box_r, box_c] = self.current_player
                self.scores[self.current_player] += 1
                completed += 1
        return completed

    def _adjacent_boxes(self, kind: str, r: int, c: int) -> Iterable[Tuple[int, int]]:
        if kind == "h":
            if r > 0:
                yield (r - 1, c)
            if r < self.rows:
                yield (r, c)
        else:
            if c > 0:
                yield (r, c - 1)
            if c < self.cols:
                yield (r, c)

    def _is_box_complete(self, r: int, c: int) -> bool:
        return (
            self.h_edges[r, c]
            and self.h_edges[r + 1, c]
            and self.v_edges[r, c]
            and self.v_edges[r, c + 1]
        )

    def to_tensor(self) -> np.ndarray:
        h = self.h_edges.astype(np.float32)
        v = self.v_edges.astype(np.float32)
        player_plane = np.full((1, self.rows + 1, self.cols + 1), self.current_player, dtype=np.float32)
        h_pad = np.pad(h, ((0, 0), (0, 1)))
        v_pad = np.pad(v, ((0, 1), (0, 0)))
        stacked = np.stack([h_pad, v_pad, player_plane.squeeze(0)], axis=0)
        return stacked

    def render(self) -> str:
        lines: List[str] = []
        for r in range(self.rows):
            top = []
            for c in range(self.cols):
                top.append("+")
                top.append("--" if self.h_edges[r, c] else "  ")
            top.append("+")
            lines.append("".join(top))
            middle = []
            for c in range(self.cols):
                middle.append("|" if self.v_edges[r, c] else " ")
                owner = self.boxes[r, c]
                middle.append("P" if owner == 0 else "Q" if owner == 1 else " ")
            middle.append("|" if self.v_edges[r, self.cols] else " ")
            lines.append("".join(middle))
        bottom = []
        for c in range(self.cols):
            bottom.append("+")
            bottom.append("--" if self.h_edges[self.rows, c] else "  ")
        bottom.append("+")
        lines.append("".join(bottom))
        return "\n".join(lines)

class DotsAndBoxes:
    def __init__(self, rows: int, cols: int) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        self.rows = rows
        self.cols = cols

    def new_game(self) -> GameState:
        return GameState.new(self.rows, self.cols)

    def clone(self, state: GameState) -> GameState:
        return state.copy()

    def legal_actions(self, state: GameState) -> List[int]:
        return [state.move_to_index(move) for move in state.legal_moves()]

    def apply_action(self, state: GameState, action: int) -> int:
        return state.apply_move(state.index_to_move(action))

    def outcome(self, state: GameState) -> int:
        winner = state.winner()
        if winner == -1:
            return 0
        return 1 if winner == state.current_player else -1

    def is_terminal(self, state: GameState) -> bool:
        return state.is_terminal()

    def action_size(self, state: GameState) -> int:
        return state.action_size
