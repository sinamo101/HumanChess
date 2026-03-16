import os
import numpy as np
import torch
import chess
import time
import csv
from fen_conv import NUM_BUCKETS, BUCKET_MIDPOINTS, convert_to_token, id_to_move, move_to_id
from model_v2 import load_base_model
from sv_move import return_next_move, return_next_move_subsample

SCALE = 0.6
EPSILON = 0.00001
MAX_LINES = 10_000_000  # Process only first 5 million lines

model, device = load_base_model()

def get_win_percentage(fen):
    tokens = convert_to_token(fen)
    tokens = torch.from_numpy(tokens).long().unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tokens)
        probs = torch.softmax(logits, dim=-1)
        win_p = float((probs * torch.from_numpy(BUCKET_MIDPOINTS).to(device)).sum())

    return win_p

def calc_encoding(cur_id, top_6_winps, top_6_ids, target_winp):
    encoding = np.zeros(1968, dtype=float)

    deltas = [abs(target_winp - winp) for winp in top_6_winps]
    similarities = [1.0 / (delta + 0.001) for delta in deltas]
    total_similarity = sum(similarities)

    for i in range(len(top_6_ids)):
        if deltas[i] <= 0.05:
            sim = SCALE * similarities[i] / total_similarity
            encoding[top_6_ids[i]] = sim
    
    encoding[encoding == 0] = EPSILON  # add epsilon
    encoding[cur_id] = 0.0  # set current move to 0
    encoding = encoding / np.sum(encoding)  # normalize to sum to 1
    
    encoding = encoding * (SCALE / (np.sum(encoding)))
    encoding[cur_id] = 1.0 - SCALE
    
    return encoding

def main():
    # Create headers for output CSV
    indexes = np.arange(0, 1968)
    str_indexes = [str(i) for i in indexes]
    headers = ["FEN"] + str_indexes
    
    _dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_FILE = os.path.join(_dir, "chess_moves_with_elo.csv")
    OUTPUT_FILE = os.path.join(_dir, "relabeled_chess_moves.csv")

    # Open files for reading and writing
    with open(INPUT_FILE, 'r') as infile, open(OUTPUT_FILE, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Skip header in input file and write header to output file
        header = next(reader)
        writer.writerow(headers)
        
        # Process each row up to MAX_LINES
        for i, row in enumerate(reader):
            if i >= MAX_LINES:
                print(f"Reached maximum limit of {MAX_LINES:,} lines. Stopping processing.")
                break
                
            fen, move = row[0], row[1]
            # print(f"Processing {i+1:,}: FEN: {fen}, Move: {move}")
            print(f"Processed {i+1}, Percent: {i/MAX_LINES * 100}%")
            
            # Print progress every 100,000 lines
            # if (i + 1) % 100000 == 0:
            #     # print(f"Progress: {i+1:,} / {MAX_LINES:,} lines processed ({((i+1)/MAX_LINES)*100:.1f}%)")
            #     print(f"Percent: {i }")
            
            # Get next fen string win percentage
            board = chess.Board(fen)
            copy_board = board.copy()
            copy_board.push(chess.Move.from_uci(move))
            new_fen = copy_board.fen()
            
            new_fen_winp = get_win_percentage(new_fen)
            
            # Get next top moves with closest win percentages
            results = return_next_move_subsample(fen, num_moves=10)
            sorted_results = sorted(results, key=lambda x: abs(x[1] - new_fen_winp))
            num_moves = 6
            top_6_moves = sorted_results[1:num_moves+1]
            target_move = sorted_results[0]
            
            top_6_winps = [top_6_moves[i][1] for i in range(len(top_6_moves))]
            top_6_ids = [move_to_id(top_6_moves[i][0]) for i in range(len(top_6_moves))]
            
            # Calculate encoding
            encoding = calc_encoding(move_to_id(move), top_6_winps, top_6_ids, target_move[1])
            
            # Prepare row data and write to output file
            output_row = [fen] + encoding.tolist()
            writer.writerow(output_row)
        
        print(f"Processing complete! Total lines processed: {min(i+1, MAX_LINES):,}")

if __name__ == "__main__":
    main()