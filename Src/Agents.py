import torch
import math


class Lander_Policy():
    def __init__(
        self,
        archi,
        device,
        Parallel_models,
        Train_Config
        ):
        
        # Reproductibilité
        self.seed = int(self.Train_Config["seed"]) 
        torch.manual_seed(self.seed)
        
        self.device = device
        self.n_layer = len(archi) -1
        self.Parallel_models = Parallel_models
        self.Train_Config = Train_Config
        
        self.model = {}
        for i in range(self.n_layer):
            n_in = archi[i]
            n_out = archi[i + 1]
            bound = 1.0 / math.sqrt(n_in)
            
            self.model[f"w{i}"] = torch.empty(self.Parallel_models , n_in, n_out, device=self.device).uniform_(-bound, bound)
            self.model[f"b{i}"] = torch.empty(self.Parallel_models , 1, n_out, device=self.device).uniform_(-bound, bound)

    def forward(self, x):
        act = x.unsqueeze(1)
        for i in range(self.n_layer):
            act = act @ self.model[f"w{i}"]
            act += self.model[f"b{i}"]
            if i != self.n_layer - 1:
                act = torch.clamp(act, min=0)
    
        return torch.sigmoid(act.squeeze(1))
    
    def mutation(self, mutate):
        for i in range(self.n_layer):
            torch.rand_like(W) < taux
            torch.normal(0.0, 0.1, (3, 4))