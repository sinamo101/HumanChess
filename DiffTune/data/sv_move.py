import os
import torch
# from infra import TransformerDecoder
from data.infra_2d import TransformerDecoder2D
from data.fen_conv_diff import NUM_BUCKETS, BUCKET_MIDPOINTS, convert_to_token
from data.config import ACTION_SIZE, SEQ_LEN, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, DROPOUT, OUTPUT_SIZE, MAX_DISTANCE
import chess
import random

action_size = ACTION_SIZE
seq_len     = SEQ_LEN
d_model     = D_MODEL
num_layers  = NUM_LAYERS
num_heads   = NUM_HEADS
d_ff        = D_FF
dropout     = DROPOUT
output_size = OUTPUT_SIZE

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

model = TransformerDecoder2D(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    dropout=dropout,
    action_size=action_size,            
    seq_len=seq_len,
    output_size=output_size,
    max_distance=MAX_DISTANCE,         # how far apart you want to model relative bias
    use_causal_mask=False   
).to(device)

# load cur params
_dir = os.path.dirname(os.path.abspath(__file__))
state = torch.load(os.path.join(_dir, "..", "trainer", "model_epoch_7.pth"), map_location=device)
model.eval()
model.load_state_dict(state)      # strict=True by default


buckets = NUM_BUCKETS
midpoints = BUCKET_MIDPOINTS

def return_next_move(fen):
    board = chess.Board(fen) # set up current FEN

    results = []
    for move in board.legal_moves:
        copy_board = board.copy()
        copy_board.push(move)
        new_fen = copy_board.fen()
        tokens = convert_to_token(new_fen) #this new fen will be the state of the opponent, so we want to choose the lowest score here
        tokens = torch.from_numpy(tokens).long().unsqueeze(0).to(device)
    
        with torch.no_grad():
            logits = model(tokens)
            probs = torch.softmax(logits, dim = -1)
            win_p = float((probs * torch.from_numpy(BUCKET_MIDPOINTS).to(device)).sum())
        
        results.append((move.uci(), win_p))
    
    results.sort(key=lambda x: x[1]) 
    return results

def return_next_move_subsample(fen, num_moves = 5):
    board = chess.Board(fen) # set up current FEN
    
    # Get all legal moves
    legal_moves = list(board.legal_moves)
    
    # Take up to 25 random moves
    
    if len(legal_moves) > num_moves:
        legal_moves = random.sample(legal_moves, num_moves)
    
    results = []
    for move in legal_moves:
        copy_board = board.copy()
        copy_board.push(move)
        new_fen = copy_board.fen()
        tokens = convert_to_token(new_fen) #this new fen will be the state of the opponent, so we want to choose the lowest score here
        tokens = torch.from_numpy(tokens).long().unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(tokens)
            probs = torch.softmax(logits, dim = -1)
            win_p = float((probs * torch.from_numpy(BUCKET_MIDPOINTS).to(device)).sum())
            
        results.append((move.uci(), win_p))
    
    results.sort(key=lambda x: x[1])
    
    return results
    

#next_move = return_next_move("6k1/8/5PK1/8/8/8/8/8 w - - 0 1")
#print(next_move)


 
