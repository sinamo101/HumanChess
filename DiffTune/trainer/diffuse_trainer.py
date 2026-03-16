import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from peft import get_peft_model_state_dict
from transformers import get_linear_schedule_with_warmup
import sys
sys.path.append('..')
from model.lora_model import LoraDiffusionModel

from diffusion.diffusion import Diffusion
from datasets import load_from_disk
from tqdm import tqdm

def train(max_samples: int | None = 1000):
    batch_size = 10
    num_epochs = 10
    lr = 1e-4
    total_t = 20
    betas = [(i + 1) / (total_t * 10) for i in range(total_t)]
    num_moves = 2000
    elo_min, elo_max, bucket_size = 1200,1800,100
    d_elo = 256

    _dir = os.path.dirname(os.path.abspath(__file__))
    ds = load_from_disk(os.path.join(_dir, "..", "data", "data", "sasa_1200_1800"))
    if max_samples is not None:
        ds = ds.select(range(max_samples))
    ds.set_format(type="torch", columns=["s_tokens","f_tokens","elo_idx_float"])
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = LoraDiffusionModel(
        base_model_path=os.path.join(_dir, "model_epoch_7.pth"),
        elo_min=elo_min, elo_max=elo_max, bucket_size=bucket_size,
        betas=betas, num_moves=num_moves
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    total_steps = (len(ds)//batch_size + 1) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # this is kind of a general discrete loss weighting schedule
    betas_t = torch.tensor(betas)
    alphas_bar = torch.cumprod(1 - betas_t, dim=0)
    lambdas = (betas_t / (1 - alphas_bar)).to(device)
    
    diffuser = Diffusion(num_moves, elo_min, elo_max, bucket_size, 256, betas)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=diffuser.diffuse)

    model.train()
    for epoch in range(num_epochs):
    # for epoch in [1]:
        epoch_pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in epoch_pbar:
            batch_size = batch["x_t"].size(0)
            for k in batch:
                batch[k] = batch[k].to(device)

            logits = model(
                batch["s_tokens"],
                batch["x_t"],
                batch["elo_idx_float"],
                batch["t"]
            )                                      
            x0 = batch["x_0"]
            xt = batch["x_t"]
            t = batch["t"]

            B, L, V = logits.shape
            flat_logits = logits.view(-1, V)
            flat_x0 = x0.view(-1)
            ce = nn.functional.cross_entropy(flat_logits, flat_x0, reduction="none").view(B, L) 
            pad_mask = (x0 != 1999).float()     
            mask = (xt != x0).float()   
            # Weight by lambdas
            weight = lambdas[t]
            loss = ((ce * mask).sum(dim=1) * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_pbar.set_postfix(loss=f"{loss.item():.4f}")

            if (epoch + 1) % 5 == 0:
                state = get_peft_model_state_dict(model.lora_transformer)
                # torch.save(state, "/home/ankush/repos/chess_train/HumanChess/DiffTune/trainer/lora_adapters.pt")
                save_path = os.path.join(_dir, f"model_lora_epoch_{epoch+1}.pt")
                torch.save(state, save_path)
                torch.save(model.state_dict(), f"full_model_epoch_{epoch+1}.pth")
                print(f"Model weights saved at {save_path}")

        print(f"Epoch {epoch+1}/{num_epochs} — loss {loss.item():.4f}")

    state = get_peft_model_state_dict(model.lora_transformer)
    torch.save(state, os.path.join(_dir, "lora_adapters.pt"))
    print("Training complete.")

if __name__ == "__main__":
    train(max_samples=1000000)