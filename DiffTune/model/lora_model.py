import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel
import math
from model.infra_2d_diff import TransformerDecoder2D
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.config import ACTION_SIZE, SEQ_LEN, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, DROPOUT, MAX_DISTANCE

class LoraDiffusionModel(nn.Module):
    def __init__(self, base_model_path, elo_min, elo_max, bucket_size, betas, num_moves):
        super().__init__()

        self.piece_vocab = ACTION_SIZE
        self.transformer = TransformerDecoder2D(
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            dropout=DROPOUT,
            action_size=num_moves,
            seq_len=SEQ_LEN,
            max_distance=MAX_DISTANCE,
            use_causal_mask=False,
            output_size=num_moves,
        )

        self.d_model = D_MODEL
        # self.embed_peices = nn.Embedding(self.piece_vocab, self.d_model)
        # self.embed_moves = nn.Embedding(num_moves, self.d_model)
        self.t_embed = nn.Embedding(21, self.d_model)
        nn.init.zeros_(self.t_embed.weight)
        self.final_ln = nn.LayerNorm(self.d_model)

        pretrained_model = torch.load(base_model_path, map_location="cpu")

        filtered = {
            k: v for k, v in pretrained_model.items()
            if not k.startswith("out_proj")      
            and not k.startswith("input_emb")     
        }
        
        self.transformer.load_state_dict(filtered, strict=False)

        old_embed = pretrained_model["input_emb.weight"]
        old_action, dim = old_embed.shape
        new_embed = torch.empty(num_moves, dim)
        new_embed[:old_action] = old_embed
        nn.init.normal_(new_embed[old_action:], std=0.01)
        self.transformer.input_emb = nn.Embedding.from_pretrained(new_embed, freeze=False)

        for block in self.transformer.layers:
            block.attn.rel_bias.data.zero_()
            block.attn.rel_bias.requires_grad_(False)

        for n, p in self.transformer.named_parameters():
            if n == "input_emb.weight":
                p.requires_grad_(True)
            elif 'norm' in n:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM", r=24, lora_alpha=48, lora_dropout=0.1, use_rslora=True,
            target_modules=["q_proj","k_proj","v_proj","o_proj"]
        )
        self.lora_transformer = get_peft_model(self.transformer, lora_cfg)

        self.elo_min = elo_min
        self.bucket_size = bucket_size
        self.num_buckets = (elo_max-elo_min)//bucket_size + 1
        self.elo_buckets = nn.Embedding(self.num_buckets, self.d_model)
        self.move_head  = nn.Linear(self.d_model, num_moves)
        nn.init.normal_(self.transformer.input_emb.weight, std=0.02)
        nn.init.zeros_(self.move_head.bias)

    def forward(self, s_tokens, x_t, elo_idx_float, t):
        batch_size, state_size = s_tokens.shape
        max_length = x_t.shape[1]
        d_model = self.d_model 
        embeds = self.transformer.input_emb
        s_emb = embeds(s_tokens) 
        pool  = self.lora_transformer.pool.expand(batch_size, -1, -1)                                                                                    

        idxf = elo_idx_float.clamp(0, self.num_buckets-1)
        lo, hi = idxf.floor().long(), idxf.ceil().long()                        
        w_hi = (idxf - lo.float()).unsqueeze(-1)                                 
        w_lo = 1.0 - w_hi                                                       
        emb_lo = self.elo_buckets(lo)                                            
        emb_hi = self.elo_buckets(hi)                                            
        elo_emb = emb_lo * w_lo + emb_hi * w_hi                                        
        
        f_emb = embeds(x_t)
        x = torch.cat([pool, s_emb, f_emb], dim=1) 
        x = self.lora_transformer.pos_enc(x)
        elo_exp = elo_emb.unsqueeze(1).expand(-1, x.size(1), -1)   
        x = x + elo_exp                                                
        x = x + self.t_embed(t).unsqueeze(1)
 
        #NO MASKING TO BASICALLY LET IT SEE EVERYTHING AND REVERSE THE NOISE
        for layer in self.lora_transformer.layers:
            x = layer(x)                                                  

        x = self.final_ln(x)
        future_hidden = x[:, 1+state_size :, :]                                           
        logits = self.move_head(future_hidden)                          
        return logits
