import os
import sys
sys.path.append('..')
from model.lora_model import LoraDiffusionModel
from data.fen_conv_diff import convert_to_token, move_to_id, id_to_move
import chess
import torch
import torch.nn.functional as F
from data.sv_move import return_next_move as baseline_next_move
import random
import numpy as np
from data.fen_conv_diff import MOVE_TO_ID, ID_TO_MOVE
from data.infra_2d import TransformerDecoder2D
from data.config import ACTION_SIZE, SEQ_LEN, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, DROPOUT, OUTPUT_SIZE, MAX_DISTANCE

# For diffusion model

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
    max_distance=MAX_DISTANCE,
    use_causal_mask=False
).to(device)

_dir = os.path.dirname(os.path.abspath(__file__))
_trainer_dir = os.path.join(_dir, "DiffTune", "trainer")
state = torch.load(os.path.join(_trainer_dir, "model_epoch_7.pth"), map_location=device)
model.eval()
model.load_state_dict(state)     



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


def calculate_windowed_rewards(game_history, window_size=4):
    enhanced_history = []
    
    for i, transition in enumerate(game_history):
        window_positions = []
        
        for j in range(min(window_size, len(game_history) - i)):
            if i + j < len(game_history):
                window_positions.append(game_history[i + j])
        
        if len(window_positions) >= 2:
            initial_value = window_positions[0].get("position_value_before", 0)
            final_value = window_positions[-1].get("position_value_after", 0)
            window_value_change = final_value - initial_value
        else:
            window_value_change = transition.get("position_value_after", 0) - transition.get("position_value_before", 0)
        
        transition["window_reward"] = window_value_change
        transition["window_size_actual"] = len(window_positions)
        
        enhanced_history.append(transition)
    
    return enhanced_history


def legal_diffuse_generate(model, s_tokens, elo_float, board, T=20, vocab_size=2000, move_offset=31, device=None):
    B = s_tokens.size(0)
    x_t = torch.randint(0, vocab_size, (B, 312), device=device)
    frozen = torch.zeros_like(x_t, dtype=torch.bool)
    final_logits = None

    legal_moves_indices = []
    for m in board.legal_moves:
        uci = m.uci()
        if uci in MOVE_TO_ID:
            move_idx = MOVE_TO_ID[uci] + move_offset
            legal_moves_indices.append(move_idx)
    
    if len(legal_moves_indices) == 0:
        return None, None 

    for t in range(T, 0, -1):                          
        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
        logits = model(s_tokens, x_t, elo_float, t_tensor)  
        if t == 1:
            final_logits = logits[:, 0, :].clone()
        
        if not frozen[0, 0]:
            legal_mask = torch.zeros(vocab_size, device=device).bool()
            for idx in legal_moves_indices:
                legal_mask[idx] = True
            
            illegal_mask = ~legal_mask
            logits[0, 0, illegal_mask] -= 1e9

        probs = torch.softmax(logits[:, 0, :], dim=-1)
        
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum().item()
        max_entropy = 7.6
        
        if not np.isfinite(entropy):
            entropy = 0.0
            
        entropy_norm = min(max(entropy / max_entropy, 0.0), 1.0)
        
        probs = F.softmax(logits, dim=-1)
        
        x_hat = torch.multinomial(
            probs.view(-1, vocab_size), 1
        ).view(B, 312)
        
        conf = torch.max(probs, dim=-1).values
        conf = torch.clamp(conf, min=0.0, max=1.0)
        
        base_rate = t / T
        min_rate = 0.2 * t / T
        keep_ratio = (1 - entropy_norm) * base_rate + entropy_norm * min_rate
        
        if not np.isfinite(keep_ratio):
            keep_ratio = base_rate
        
        for b in range(B):
            num_unfrozen = (~frozen[b]).sum().item()
            num_to_freeze = int(num_unfrozen * keep_ratio)

            if num_to_freeze == 0:
                continue

            idx_unfrozen = torch.where(~frozen[b])[0]
            conf_sorted = conf[b, idx_unfrozen].argsort(descending=True)
            top_k = idx_unfrozen[conf_sorted[:num_to_freeze]]
            x_t[b, top_k] = x_hat[b, top_k]
            frozen[b, top_k] = True
        
        if t > 1:  
            for b in range(B):
                other_unfrozen = torch.where(~frozen[b, 1:])[0] + 1
                if len(other_unfrozen) > 0:
                    x_t[b, other_unfrozen] = torch.randint(
                        0, vocab_size, (len(other_unfrozen),), device=device
                    )

    move_token = x_t[0, 0]
    
    if move_token.item() not in legal_moves_indices:
        import random
        move_token = torch.tensor(random.choice(legal_moves_indices), device=device)
    
    log_prob = None
    if final_logits is not None:
        masked_logits = final_logits.clone()
        legal_mask = torch.zeros(vocab_size, device=device).bool()
        for idx in legal_moves_indices:
            legal_mask[idx] = True
            
        masked_logits[0, ~legal_mask] = -1e9
        log_probs = F.log_softmax(masked_logits, dim=-1)
        log_prob = log_probs[0, move_token]

    return move_token, log_prob


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
        diffusion_model,
        num_episodes=1000,
        learning_rate=1e-5,
        elo_range=(1200, 1800),
        device=None):    
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=learning_rate)
    
    wins = 0
    losses = 0
    draws = 0
    update_steps = 0
    total_loss = 0
    game_lengths = []
    reset_interval = 200 


    
    for episode in range(num_episodes):
        diffusion_plays_white = (episode % 2 == 0)
        diffusion_elo = random.choice([0, 1, 2, 3, 4, 5, 6])
        
        board = chess.Board()
        diffusion_player = 0 if diffusion_plays_white else 1
        current_player = 0
        
        last_diffusion_move_info = None
        position_history = []
        move_count = 0

        
        while not board.is_game_over():
            move_count += 1
            if current_player == diffusion_player:
                fen_before = board.fen()
                position_value_before = evaluate_position(board)
                
                first_state = convert_to_token(fen_before)
                first_state = torch.from_numpy(first_state).long().unsqueeze(0).to(device)
                
                output, log_prob = legal_diffuse_generate(
                    model=diffusion_model,
                    s_tokens=first_state,
                    elo_float=torch.tensor([diffusion_elo]).to(device),
                    board=board,
                    T=20,
                    vocab_size=2000,
                    move_offset=31,
                    device=device
                )
                
                if output is None:
                    break
                
                uci_move  = id_to_move(output.item() - 31)
                move = chess.Move.from_uci(uci_move)

                persp = chess.WHITE if diffusion_player == 0 else chess.BLACK

                val_before = evaluate_position(board)   

                board.push(move)

                val_after  = evaluate_position(board)

                if persp == chess.BLACK:
                    val_before *= -1
                    val_after  *= -1

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
                if move.uci() in ('e1g1','e1c1','e8g8','e8c8'):
                    immediate_bonus += 0.1

                immediate_reward = 0.1 * material_change + delayed_reward + immediate_bonus

                
                # UPDATE MODEL WITH IMMEDIATE REWARD
                if abs(immediate_reward) > 0.001: 
                    immediate_loss = -log_prob * immediate_reward
                    
                    optimizer.zero_grad()
                    immediate_loss.backward()
                    torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
                    optimizer.step()
                    
                    update_steps += 1
                    total_loss += immediate_loss.item()
                
                last_diffusion_move_info = {
                    'state': first_state.clone(),
                    'action': output.item(),
                    'elo': diffusion_elo,
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
                
                #if episode % 50 == 0 and (abs(immediate_reward) > 0.5 or board.is_check()):
                    #print(f"Move: {uci_move}, Immediate reward: {immediate_reward:.3f}")
                
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
        
        # adding end of gme rews
        winner = get_winner(board)
        
        if winner == 0:
            if diffusion_plays_white:
                wins += 1
            else:
                losses += 1
        elif winner == 1:
            if diffusion_plays_white:
                losses += 1
            else:
                wins += 1
        else:
            draws += 1
        
        if last_diffusion_move_info is not None:
            terminal_reward = 0
            
            if last_diffusion_move_info['delivered_checkmate']:
                terminal_reward = 5.0
            elif (winner == 0 and diffusion_plays_white) or (winner == 1 and not diffusion_plays_white):
                terminal_reward = 2.0
            elif (winner == 1 and diffusion_plays_white) or (winner == 0 and not diffusion_plays_white):
                terminal_reward = -2.0
            elif winner == -1 and len(position_history) > 0 and position_history[-1]['value'] > 5:
                terminal_reward = -0.5
            
            if abs(terminal_reward) > 0.001:
                last_board = chess.Board(last_diffusion_move_info['fen_before'])
                
                output_check, log_prob_current = legal_diffuse_generate(
                    model=diffusion_model,
                    s_tokens=last_diffusion_move_info['state'],
                    elo_float=torch.tensor([last_diffusion_move_info['elo']]).to(device),
                    board=last_board,
                    T=20,
                    vocab_size=2000,
                    move_offset=31,
                    device=device
                )
                
                if log_prob_current is not None:
                    terminal_loss = -log_prob_current * terminal_reward
                    
                    optimizer.zero_grad()
                    terminal_loss.backward()
                    torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
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
                print(f"Game lengths: avg={avg_game_length:.1f}, min={min_game_length}, max={max_game_length}")
        if episode % reset_interval == 0 and episode > 0:
            game_lengths = []
            update_steps = 0
            total_loss = 0
            
        if episode % 2000 == 0 and episode > 0:
            torch.save(diffusion_model.state_dict(), 
                      f"diffusion_online_{episode}.pth")
    
    torch.save(diffusion_model.state_dict(), "diffusion_online_final.pth")

diffusion_model = LoraDiffusionModel(
    base_model_path=os.path.join(_trainer_dir, "model_epoch_7.pth"),
    elo_min=1200,
    elo_max=1800,
    bucket_size=100,
    betas=[(i + 1) / (20 * 10) for i in range(20)],
    num_moves=2000
).to(device)

ckpt = torch.load(os.path.join(_trainer_dir, "full_model_epoch_10.pth"), map_location=device)
for k in list(ckpt.keys()):              
    if any(tag in k for tag in ["pos_enc.pe", "token_coords", "rel_idx"]):
        del ckpt[k]                       
diffusion_model.load_state_dict(ckpt, strict=False) 
diffusion_model.eval()

train_against_baseline_online(
    diffusion_model=diffusion_model,
    num_episodes=200000,
    learning_rate=1e-5,
    elo_range=(1200, 2200),
    device=device
)
