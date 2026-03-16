import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, layer, rank, alpha):
        super().__init__()
        self.layer = layer
        self.rank = rank
        self.alpha = alpha
        self.input_d, self.output_d = layer.weight.shape

        self.lora_a = nn.Parameter(torch.zeros((rank, self.output_d)))
        self.lora_b = nn.Parameter(torch.zeros(self.input_d, rank))

        for param in self.layer.parameters():
            param.requires_grad = False
        
        nn.init.normal_(self.lora_a, std=0.02)
    
    def forward(self, x):
        output = self.layer(x) 
        new_output = torch.matmul(x, torch.matmul(self.lora_b, self.lora_a)) * self.alpha / self.rank
        return output + new_output
