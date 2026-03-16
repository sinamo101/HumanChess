"""
Parameter count script for all HumanChess models.
No training, no weight loading - just architecture instantiation.
"""
import sys
import os

sys.path.insert(0, "/Users/simohammadi-mbp/Desktop/HC/HumanChess")
sys.path.insert(0, "/Users/simohammadi-mbp/Desktop/HC/HumanChess/DiffTune")
sys.path.insert(0, "/Users/simohammadi-mbp/Desktop/HC/HumanChess/DiffTune/model")

import torch
import torch.nn as nn

def count_params(model, label=""):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable parameters: {trainable:>12,}")
    if frozen > 0:
        print(f"  Frozen parameters:    {frozen:>12,}")
    print(f"  Trainable %:          {100*trainable/total:>11.2f}%")
    print()
    return total, trainable

def separator(title):
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)

# ──────────────────────────────────────────────────────────────
# 1. Base TransformerDecoder (infra.py)
# ──────────────────────────────────────────────────────────────
separator("1. TransformerDecoder (infra.py)")
from infra import TransformerDecoder

model1 = TransformerDecoder(
    action_size=31, seq_len=77, d_model=256, num_layers=8,
    num_heads=8, d_ff=1024, dropout=0.1, output_size=128,
    use_causal_mask=False, apply_qk_layernorm=False
)
count_params(model1)

# ──────────────────────────────────────────────────────────────
# 2. Base TransformerDecoder2D (infra_2d.py)
# ──────────────────────────────────────────────────────────────
separator("2. TransformerDecoder2D (infra_2d.py)")
from infra_2d import TransformerDecoder2D

model2 = TransformerDecoder2D(
    action_size=31, seq_len=77, d_model=256, num_layers=8,
    num_heads=8, d_ff=1024, dropout=0.1, output_size=128,
    max_distance=8, use_causal_mask=False
)
count_params(model2)

# ──────────────────────────────────────────────────────────────
# 3. LoraDiffusionModel (manual reconstruction)
#    We can't call the real constructor (needs base_model_path),
#    so we build the same architecture manually.
# ──────────────────────────────────────────────────────────────
separator("3. LoraDiffusionModel components (DiffTune)")

# 3a. The TransformerDecoder2D backbone (diffusion variant)
from infra_2d_diff import TransformerDecoder2D as TransformerDecoder2DDiff

backbone = TransformerDecoder2DDiff(
    num_layers=8, d_model=256, num_heads=8, d_ff=1024,
    dropout=0.1, action_size=2000, seq_len=77,
    max_distance=8, use_causal_mask=False, output_size=2000
)

print("  [Backbone] TransformerDecoder2D (action_size=2000, output_size=2000)")
backbone_total = sum(p.numel() for p in backbone.parameters())
print(f"    Parameters: {backbone_total:>12,}")

# 3b. Extra components
elo_buckets = nn.Embedding(7, 256)
t_embed = nn.Embedding(21, 256)
final_ln = nn.LayerNorm(256)
move_head = nn.Linear(256, 2000)

extras = {
    "elo_buckets (7x256)": elo_buckets,
    "t_embed (21x256)": t_embed,
    "final_ln (256)": final_ln,
    "move_head (256->2000)": move_head,
}

extra_total = 0
print("\n  [Extra components]")
for name, mod in extras.items():
    n = sum(p.numel() for p in mod.parameters())
    extra_total += n
    print(f"    {name}: {n:>10,}")

full_total = backbone_total + extra_total
print(f"\n  [Full LoraDiffusionModel total]")
print(f"    Backbone:   {backbone_total:>12,}")
print(f"    Extras:     {extra_total:>12,}")
print(f"    TOTAL:      {full_total:>12,}")

# 3c. LoRA trainable parameter estimate
# LoRA on q_proj, k_proj, v_proj, o_proj in each of 8 layers
# Each proj is Linear(256, 256, bias=False)
# LoRA adds: A (256 x r) + B (r x 256) per target per layer
# r = 24, 4 targets, 8 layers
r = 24
d = 256
n_targets = 4  # q_proj, k_proj, v_proj, o_proj (sometimes called out_proj)
n_layers = 8
lora_per_target = d * r + r * d  # A + B
lora_total = lora_per_target * n_targets * n_layers

# Trainable in lora_model.py: LoRA params + input_emb + all norm layers + extra components
# From the code: input_emb.weight is trainable, all 'norm' params are trainable

# input_emb
input_emb_params = 2000 * 256

# norm layers inside backbone: each block has norm1 and norm2, each LayerNorm(256) = 256 weight + 256 bias = 512
norm_per_block = 2 * (256 + 256)  # norm1 + norm2
norm_backbone = norm_per_block * n_layers

# Extra components are all trainable (they're separate nn.Modules on LoraDiffusionModel)
extra_trainable = extra_total

# Pool token (it's a Parameter but set requires_grad=False by the loop since it's not 'norm' and not 'input_emb.weight')
# Actually let's check: the loop does `for n, p in self.transformer.named_parameters()` and only keeps input_emb.weight and 'norm' params trainable
# pool is a Parameter of transformer named "pool" - no 'norm' in name, not input_emb.weight -> frozen

trainable_from_backbone = input_emb_params + norm_backbone
trainable_lora = lora_total
trainable_extras = extra_trainable
total_trainable = trainable_from_backbone + trainable_lora + trainable_extras
total_frozen = full_total - trainable_from_backbone - trainable_extras + lora_total  # lora params are added on top

print(f"\n  [LoRA Analysis] (r=24, alpha=48, targets: q/k/v/o_proj)")
print(f"    LoRA params per target:  {lora_per_target:>10,}  (A: {d}x{r} + B: {r}x{d})")
print(f"    LoRA targets per layer:  {n_targets}")
print(f"    LoRA layers:             {n_layers}")
print(f"    Total LoRA params:       {lora_total:>10,}")
print()
print(f"    Trainable breakdown:")
print(f"      LoRA adapters:         {trainable_lora:>10,}")
print(f"      input_emb (2000x256):  {input_emb_params:>10,}")
print(f"      Backbone norms:        {norm_backbone:>10,}")
print(f"      Extra components:      {trainable_extras:>10,}")
print(f"      ---------------------------------")
print(f"      Total trainable:       {total_trainable:>10,}")
print()

# Total with LoRA = original full_total + lora_total (LoRA adds new params)
grand_total_with_lora = full_total + lora_total
frozen_with_lora = grand_total_with_lora - total_trainable
print(f"    Grand total (with LoRA): {grand_total_with_lora:>10,}")
print(f"    Frozen:                  {frozen_with_lora:>10,}")
print(f"    Trainable:               {total_trainable:>10,}")
print(f"    Trainable %:             {100*total_trainable/grand_total_with_lora:>9.2f}%")
