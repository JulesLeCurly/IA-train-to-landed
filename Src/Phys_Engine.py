import torch

class Phys_Engine:
    def __init__(
    self,
    Train_Config,
    Env_Config,
    Spawn_pos,
    mode,
    seed,
    device="cpu"
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.mode = mode
        self.seed = seed
        self.Train_Config = Train_Config
        self.Env_Config = Env_Config
        self.Spawn_pos = Spawn_pos

        # Reproductibilité
        self.seed = int(self.Train_Config["seed"]) if seed is None else int(seed)
        torch.manual_seed(self.seed)

        # Dimensions d'exécution (Validation stricte sans fallback)
        self.episode_length = int(self.Train_Config["episode_length"])
        if self.mode == "train":
            self.Parallel_Scenarios = int(self.Train_Config["Parallel_Scenarios_train"])
        elif self.mode == "val":
            self.Parallel_Scenarios = int(self.Train_Config["Parallel_Scenarios_Val"])
        else:
            raise ValueError(f"Mode invalide '{self.mode}'. Choisissez 'train' ou 'val'.")

        self.current_iteration = 0
        self._init_parameters()
        self._allocate_memory()

        self.scenarios_idx = torch.arange(self.Parallel_Scenarios, device=self.device, dtype=torch.int64)
        self.reset(self.scenarios_idx)
        
    def _init_parameters(self):
        self.GRAVITY = self.Env_Config["GRAVITY"]  # m/s^2 per frame

        # Victory conditions
        self.MAX_LANDING_SPEED = self.Env_Config["MAX_LANDING_SPEED"]  # m/s
        self.LANDING_ANGLE_TOLERANCE = self.Env_Config["LANDING_ANGLE_TOLERANCE"]  # degrees from vertical
    
    def _allocate_memory(self):
        """Alloue les tenseurs d'état vivants en 1D compact (Parallel_Scenarios,)."""
        n = self.Parallel_Scenarios
        dev = self.device

        # Progression temporelle
        self.current_step = torch.zeros(n, dtype=torch.int32, device=dev)
        self.starting_indices = torch.zeros(n, dtype=torch.int32, device=dev)
        
        self.pos_x = torch.zeros(n, dtype=torch.float32, device=dev)
        self.pos_y = torch.zeros(n, dtype=torch.float32, device=dev)
        
        self.velocity_x = torch.zeros(n, dtype=torch.float32, device=dev)
        self.velocity_y = torch.zeros(n, dtype=torch.float32, device=dev)
        
        self.rotation = torch.zeros(n, dtype=torch.float32, device=dev)
    
    def reset(self, scenarios_to_reset):
        if scenarios_to_reset is None:
            scenarios_to_reset = self.scenarios_idx
        
        idx = scenarios_to_reset
        n_reset = len(idx)
        if n_reset == 0:
                return
    
        self.current_step[idx] = 0
        self.pos_x[idx] = self.Spawn_pos["x"]
        self.pos_y[idx] = self.Spawn_pos["y"]
        
        set_vel_x = self.Env_Config["Max_vel_x"]
        set_vel_y = self.Env_Config["Max_vel_y"]
        
        self.velocity_x[idx] = torch.empty(n_reset, device=self.device).uniform_(-set_vel_x, set_vel_x)
        self.velocity_y[idx] = torch.empty(n_reset, device=self.device).uniform_(set_vel_y[0], set_vel_y[1])
        self.rotation[idx] = torch.empty(n_reset, device=self.device).uniform_(1, 360)
    
    def _apply_model_action(self, action):
        """Applique l'action discrète du modèle à l'Open de la bougie courante."""
        action = action.to(dtype=torch.float32, device=self.device)
        
        self.raw_rotation = action[0] * 10 - 5
        self.raw_trust = action[1] * 10
        
        self.rotation = (self.rotation + ) % 360
        self.trust

        thrust_angle_rad = math.radians(self.rotation)
        self.velocity_x += math.sin(thrust_angle_rad) * self.config.THRUST_POWER
        self.velocity_y += math.cos(thrust_angle_rad) * self.config.THRUST_POWER
        
        # Update position
        self.x += self.velocity_x
        self.y -= self.velocity_y
        
        self.x = self.x % 998698
        
        # Wrap around screen horizontally
        if self.x > self.config.SCREEN_WIDTH:
            self.x = 0
        elif self.x < 0:
            self.x = self.config.SCREEN_WIDTH
            
    
    
    def _advance_env_step(self, ):
        self.velocity_y -= self.config.GRAVITY
        
        # 5. Conditions de terminaison (Faillite et Truncation)
        newly_bankrupt = (self.wallet <= (self.initial_wallet * (1.0 + self.bankruptcy_threshold))) & ~self.is_bankrupt
        self.is_bankrupt = self.is_bankrupt | newly_bankrupt

        terminated = self.is_bankrupt.clone()
        truncated = (self.current_step >= self.episode_length)
        return terminated, truncated