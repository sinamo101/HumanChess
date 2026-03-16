import numpy as np
import chess
import math

# this code is largely very similar to the conversion strategy used by DeepMind and other papers!

NUM_BUCKETS = 128
FULL = np.linspace(0.0, 1.0, NUM_BUCKETS + 1, dtype=np.float32)
BUCKET_EDGES = FULL[1:-1]
BUCKET_MIDPOINTS = (FULL[:-1] + FULL[1:]) * 0.5

def win_to_bucket(win_pct):
    return np.searchsorted(BUCKET_EDGES, win_pct, side='left')

def hl_gauss(win_pct):
    w = np.atleast_1d(win_pct).astype(np.float32)
    N = w.shape[0]
    sigma = 1.0 / NUM_BUCKETS
    diffs = w[:, None] - BUCKET_MIDPOINTS[None, :]
    logits = -0.5 * (diffs / sigma) ** 2
    exps   = np.exp(logits)
    probs  = exps / exps.sum(axis=1, keepdims=True)
    return probs[0] if np.ndim(win_pct) == 0 else probs


chars = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h',
    'p','n','r','k','q','P','B','N','R','Q','K','w','.',]

idx = {}
for i, c in enumerate(chars):
    idx[c] = i

empty_set = set()
for i in range(1, 9):
    empty_set.add(chr(i + 48))

def convert_to_token(fen, seq_len=77):
    board, side, castling, en_passant, halfmoves_last, fullmoves = fen.split(' ')
    board = board.replace('/', '')
    board = side + board

    token = []

    for char in board:
        if char in empty_set:
            token.extend(int(char) * [idx['.']])
        else:
            token.append(idx[char])

    if castling == '-':
        token.extend(4 * [idx['.']])
    else:
        for char in castling:
            token.append(idx[char])
        if len(castling) < 4:
            token.extend((4 - len(castling)) * [idx['.']])

    if en_passant == '-':
        token.extend(2 * [idx['.']])
    else:
        for char in en_passant:
            token.append(idx[char])

    halfmoves_last += '.' * (3 - len(halfmoves_last))
    token.extend([idx[x] for x in halfmoves_last])

    fullmoves += '.' * (3 - len(fullmoves))
    token.extend([idx[x] for x in fullmoves])

    assert len(token) == seq_len

    return np.asarray(token, dtype=np.uint8)


ROWS = ['a','b','c','d','e','f','g','h']

def compute_all_possible_actions():
    all_moves = []
    board = chess.BaseBoard.empty()

    for sq in range(64):
        next_loc = []
        board.set_piece_at(sq, chess.Piece.from_symbol('Q'))
        next_loc += board.attacks(sq)

        board.set_piece_at(sq, chess.Piece.from_symbol('N'))
        next_loc += board.attacks(sq)
        board.remove_piece_at(sq)

        for tgt in next_loc:
            all_moves.append(chess.square_name(sq) + chess.square_name(tgt))

    for (rank, next_rank) in [('2','1'), ('7','8')]:
        for i, cur_file in enumerate(ROWS):
            for promo in ['q','r','b','n']:
                all_moves.append(f"{cur_file}{rank}{cur_file}{next_rank}{promo}")

            if cur_file > 'a':
                left = ROWS[i-1]
                for promo in ['q','r','b','n']:
                    all_moves.append(f"{cur_file}{rank}{left}{next_rank}{promo}")

            if cur_file < 'h':
                right = ROWS[i+1]
                for promo in ['q','r','b','n']:
                    all_moves.append(f"{cur_file}{rank}{right}{next_rank}{promo}")

    unique = sorted(set(all_moves))
    assert len(unique) == 1968, f"Expected 1968 moves, got {len(unique)}"

    move_to_id = {m:i for i,m in enumerate(unique)}
    id_to_move = {i:m for i,m in enumerate(unique)}
    return move_to_id, id_to_move

MOVE_TO_ID, ID_TO_MOVE = compute_all_possible_actions()


def move_to_id(move: str) -> int:
    return MOVE_TO_ID[move]

def id_to_move(idx: int) -> str:
    return ID_TO_MOVE[idx]

def centipawns_to_win_probability(centipawns: int) -> float:
  return 0.5 + 0.5 * (2 / (1 + math.exp(-0.00368208 * centipawns)) - 1)

def get_uniform_buckets_edges_values(num_buckets: int):
    full = np.linspace(0.0, 1.0, num_buckets + 1)
    edges  = full[1:-1]
    values = (full[:-1] + full[1:]) * 0.5
    return edges, values

def bucketize(win_percents: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, win_percents, side='left')
