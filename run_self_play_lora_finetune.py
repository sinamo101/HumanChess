import os
import sys
sys.path.append('..')
from model.lora_action_model import LoraActionModel  # Updated import
from data.fen_conv_lora_action import convert_to_token, move_to_id, id_to_move  # Updated import
import chess
import torch
import torch.nn.functional as F
from data.sv_move import return_next_move as baseline_next_move
import random
import numpy as np
from data.fen_conv_lora_action import MOVE_TO_ID, ID_TO_MOVE   # Updated import

# For basic lora finetune, not for diffusion model

action_size = 1968  # Updated to match LoRA action model      
seq_len     = 77
d_model     = 256
num_layers  = 8
num_heads   = 8
d_ff        = d_model * 4
dropout     = 0.1

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

def evaluate_position(board):
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }
    
    white_material = 0
    black_material = 0
    white_position_bonus = 0
    black_position_bonus = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values.get(piece.piece_type, 0)
            
            if piece.color == chess.WHITE:
                white_material += value
                
                if piece.piece_type in [chess.PAWN, chess.KNIGHT]:
                    if square in [chess.E4, chess.D4, chess.E5, chess.D5]:
                        white_position_bonus += 10
                    elif square in [chess.C3, chess.C4, chess.C5, chess.C6,
                                   chess.F3, chess.F4, chess.F5, chess.F6]:
                        white_position_bonus += 5
                        
                if piece.piece_type == chess.KING:
                    if board.has_castling_rights(chess.WHITE):
                        white_position_bonus += 20
                        
            else:
                black_material += value
                
                if piece.piece_type in [chess.PAWN, chess.KNIGHT]:
                    if square in [chess.E4, chess.D4, chess.E5, chess.D5]:
                        black_position_bonus += 10
                    elif square in [chess.C3, chess.C4, chess.C5, chess.C6,
                                   chess.F3, chess.F4, chess.F5, chess.F6]:
                        black_position_bonus += 5
                        
                if piece.piece_type == chess.KING:
                    if board.has_castling_rights(chess.BLACK):
                        black_position_bonus += 20
    
    mobility_bonus = len(list(board.legal_moves)) * 2
    
    if board.turn == chess.WHITE:
        total = (white_material + white_position_bonus) - (black_material + black_position_bonus) + mobility_bonus
    else:
        total = (black_material + black_position_bonus) - (white_material + white_position_bonus) + mobility_bonus
    
    return total / 100.0 

def legal_action_generate(model, s_tokens, elo_float, board, temperature=1.0, device=None):
    with torch.no_grad():
        logits = model(s_tokens, elo_float)  # [batch_size, 1968]
 
        legal_moves_indices = []
        legal_moves_uci = []
        for m in board.legal_moves:
            uci = m.uci()
            if uci in MOVE_TO_ID:
                move_idx = MOVE_TO_ID[uci]
                legal_moves_indices.append(move_idx)
                legal_moves_uci.append(uci)
        
        if len(legal_moves_indices) == 0:
            return None, None

        masked_logits = logits.clone()
        legal_mask = torch.zeros(1968, device=device).bool()
        for idx in legal_moves_indices:
            legal_mask[idx] = True
        
        masked_logits[0, ~legal_mask] = -1e9
        
        scaled_logits = masked_logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        move_idx = torch.multinomial(probs[0], 1).item()
        log_probs = F.log_softmax(masked_logits, dim=-1)
        log_prob = log_probs[0, move_idx]
        return move_idx, log_prob

def get_winner(board):
    if not board.is_game_over():
        return None
    result = board.result()
    if result == "1-0":
        return 0
    elif result == "0-1":
        return 1
    else:
        return -1

def train_against_baseline_online(
        action_model,
        num_episodes=1000,
        learning_rate=1e-5,
        elo_range=(1200, 1800),
        device=None):    
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, action_model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    wins = 0
    losses = 0
    draws = 0
    update_steps = 0
    total_loss = 0
    game_lengths = []
    reset_interval = 200 # lip_DhEUr58YaiaoBAiFZY3Y

    for episode in range(num_episodes):
        action_plays_white = (episode % 2 == 0)
        action_elo = random.choice([0, 1, 2, 3, 4, 5, 6])
        
        board = chess.Board()
        action_player = 0 if action_plays_white else 1
        current_player = 0
        
        last_action_move_info = None
        position_history = []
        move_count = 0
        
        while not board.is_game_over():
            move_count += 1
            if current_player == action_player:
                fen_before = board.fen()
                position_value_before = evaluate_position(board)

                state_tokens = convert_to_token(fen_before)
                state_tokens = torch.from_numpy(state_tokens).long().unsqueeze(0).to(device)
                move_idx, log_prob = legal_action_generate(
                    model=action_model,
                    s_tokens=state_tokens,
                    elo_float=torch.tensor([action_elo]).to(device),
                    board=board,
                    temperature=1.2,  # Add some exploration
                    device=device
                )
                
                if move_idx is None:
                    break
                
                # Convert move index to UCI and make move
                uci_move = id_to_move(move_idx)
                move = chess.Move.from_uci(uci_move)

                persp = chess.WHITE if action_player == 0 else chess.BLACK
                val_before = evaluate_position(board)   
                board.push(move)
                val_after = evaluate_position(board)

                if persp == chess.BLACK:
                    val_before *= -1
                    val_after *= -1

                material_change = val_after - val_before           
                delayed_reward = 0
                if len(position_history) >= 4:
                    old_val = position_history[-4]['value']        
                    delayed_reward = (val_after - old_val) * 0.05

                immediate_bonus = 0
                if board.is_check():
                    immediate_bonus += 0.05
                if board.is_checkmate():            
                    immediate_bonus += 1.0
                if move.uci() in ('e1g1','e1c1','e8g8','e8c8'):  # Castling
                    immediate_bonus += 0.1

                immediate_reward = 0.1 * material_change + delayed_reward + immediate_bonus
                
                # UPDATE MODEL WITH IMMEDIATE REWARD
                if abs(immediate_reward) > 0.001: 
                    action_model.train()
                    logits = action_model(state_tokens, torch.tensor([action_elo]).to(device))
                    if immediate_reward > 0:
                        # Positive reward: encourage this move with cross-entropy
                        target = torch.zeros(1968, device=device)
                        target[move_idx] = 1.0
                        log_probs = F.log_softmax(logits, dim=-1)
                        loss = -torch.sum(target * log_probs[0]) * immediate_reward
                    else:
                        # Negative reward: discourage this move, encourage exploration
                        loss_weight = abs(immediate_reward)
                        legal_moves_indices = []
                        for m in chess.Board(fen_before).legal_moves:
                            uci = m.uci()
                            if uci in MOVE_TO_ID:
                                legal_moves_indices.append(MOVE_TO_ID[uci])
                        
                        uniform_target = torch.zeros(1968, device=device)
                        for idx in legal_moves_indices:
                            uniform_target[idx] = 1.0 / len(legal_moves_indices)
                        
                        log_probs = F.log_softmax(logits, dim=-1)
                        loss = F.kl_div(log_probs, uniform_target.unsqueeze(0), reduction='batchmean') * loss_weight
                        
                    if loss.requires_grad:
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(action_model.parameters(), 1.0)
                        optimizer.step()
                        
                        update_steps += 1
                        total_loss += loss.item()

                
                last_action_move_info = {
                    'state': state_tokens.clone(),
                    'action': move_idx,
                    'elo': action_elo,
                    'log_prob': log_prob.item(),
                    'delivered_checkmate': board.is_checkmate(),
                    'fen_before': fen_before
                }
                
                position_history.append({
                    'value': val_after,
                    'fen': board.fen()
                })
                if len(position_history) > 4:
                    position_history.pop(0)
                
            else:
                fen_str = board.fen()
                move_list = baseline_next_move(fen_str)
                if not move_list:
                    break
                    
                best_move_uci = move_list[0][0]
                move = chess.Move.from_uci(best_move_uci)
                board.push(move)
                
                val_opp = evaluate_position(board)
                if persp == chess.BLACK:
                    val_opp *= -1
                position_history.append({'value': val_opp, 'fen': board.fen()})

                if len(position_history) > 4:
                    position_history.pop(0)
            
            current_player = 1 - current_player
        
        game_lengths.append(move_count)
        winner = get_winner(board)
        
        if winner == 0:
            if action_plays_white:
                wins += 1
            else:
                losses += 1
        elif winner == 1:
            if action_plays_white:
                losses += 1
            else:
                wins += 1
        else:
            draws += 1
        
        if last_action_move_info is not None:
            terminal_reward = 0
            
            if last_action_move_info['delivered_checkmate']:
                terminal_reward = 5.0
            elif (winner == 0 and action_plays_white) or (winner == 1 and not action_plays_white):
                terminal_reward = 2.0
            elif (winner == 1 and action_plays_white) or (winner == 0 and not action_plays_white):
                terminal_reward = -2.0
            elif winner == -1 and len(position_history) > 0 and position_history[-1]['value'] > 5:
                terminal_reward = -0.5
            
            if abs(terminal_reward) > 0.001:
                action_model.train()
                
                last_board = chess.Board(last_action_move_info['fen_before'])
                
                logits = action_model(last_action_move_info['state'], 
                                    torch.tensor([last_action_move_info['elo']]).to(device))
                
                if terminal_reward > 0:
                    target = torch.zeros(1968, device=device)
                    target[last_action_move_info['action']] = 1.0
                    log_probs = F.log_softmax(logits, dim=-1)
                    terminal_loss = -torch.sum(target * log_probs[0]) * terminal_reward
                else:
                    legal_moves_indices = []
                    for m in last_board.legal_moves:
                        uci = m.uci()
                        if uci in MOVE_TO_ID:
                            legal_moves_indices.append(MOVE_TO_ID[uci])
                    
                    uniform_target = torch.zeros(1968, device=device)
                    for idx in legal_moves_indices:
                        uniform_target[idx] = 1.0 / len(legal_moves_indices)
                    
                    log_probs = F.log_softmax(logits, dim=-1)
                    terminal_loss = F.kl_div(log_probs, uniform_target.unsqueeze(0), 
                                           reduction='batchmean') * abs(terminal_reward)
                
                if terminal_loss.requires_grad:
                    optimizer.zero_grad()
                    terminal_loss.backward()
                    torch.nn.utils.clip_grad_norm_(action_model.parameters(), 1.0)
                    optimizer.step()

        if episode % reset_interval == 0:
            win_rate = wins / max(wins + losses + draws, 1)
            avg_loss = total_loss / max(update_steps, 1)
            print(f"\nEpisode {episode}")
            print(f"Win rate: {win_rate:.2%} (W:{wins}/L:{losses}/D:{draws})")
            if game_lengths:
                avg_game_length = sum(game_lengths) / len(game_lengths)
                min_game_length = min(game_lengths)
                max_game_length = max(game_lengths)
                print(f"Game lengths: avg={avg_game_length}, min={min_game_length}, max={max_game_length}")
        
        if episode % reset_interval == 0 and episode > 0:
            game_lengths = []
            update_steps = 0
            total_loss = 0
            
        if episode % 2000 == 0 and episode > 0:
            torch.save(action_model.state_dict(), 
                      f"action_model_selfplay_{episode}.pth")
    
    torch.save(action_model.state_dict(), "action_model_selfplay_final.pth")

action_model = LoraActionModel(
    base_model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "DiffTune", "trainer", "model_epoch_7.pth"),
    elo_min=1200, 
    elo_max=1800, 
    bucket_size=100,
    betas=None,  
    num_moves=1968
).to(device)

checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DiffTune", "trainer", "human_full_model_epoch_20.pth")
action_model.load_state_dict(torch.load(checkpoint_path, map_location=device))


action_model.train()

for name, param in action_model.named_parameters():
    if '.lora_' in name or 'input_emb' in name or 'norm' in name or 'move_head' in name or 'elo_buckets' in name:
        param.requires_grad_(True)
    else:
        param.requires_grad_(False)

trainable_params = sum(p.numel() for p in action_model.parameters() if p.requires_grad)

# Start self-play training
train_against_baseline_online(
    action_model=action_model,
    num_episodes=200000,
    learning_rate=5e-6,  # Lower learning rate for fine-tuning
    elo_range=(1200, 2200),
    device=device
)
