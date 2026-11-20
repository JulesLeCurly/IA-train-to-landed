import torch

class Phys_Engine:
    def __init__(
    self,
    Train_Config,
    Env_Config,
    Spawn_pos,
    mode,
    device="cpu"
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.mode = mode
        self.Train_Config = Train_Config
        self.Env_Config = Env_Config
        self.Spawn_pos = Spawn_pos

        # Reproductibilité
        self.seed = int(self.Train_Config["seed"])
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

        self.GROUND_LEVEL = self.Env_Config["GROUND_LEVEL"]
        self.Max_rotation = self.Env_Config["Max_rotation"]
        self.Max_trust = self.Env_Config["Max_trust"]
        self.Max_width = self.Env_Config["Max_width"]
    
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

        self.alive = torch.ones(n, dtype=torch.bool, device=dev)
        self.win = torch.zeros(n, dtype=torch.bool, device=dev)
    
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
        
        self.alive[idx] = True
        self.win[idx] = False
        
        
    def _apply_model_action(self, action):
        """Applique l'action discrète du modèle à l'Open de la bougie courante."""
        action = action.to(dtype=torch.float32, device=self.device)
        alive = self.alive

        raw_rotation = action[:, 0] * self.Max_rotation - self.Max_rotation / 2
        raw_thrust = torch.where(action[:, 1] > 0.05, action[:, 1] * self.Max_trust, 0.0)
        
        raw_rotation = raw_rotation[alive]
        raw_thrust = raw_thrust[alive]

        self.rotation[alive] = (self.rotation[alive] + raw_rotation) % 360
        thrust_angle_rad = torch.deg2rad(self.rotation[alive])
        
        self.velocity_x[alive] += torch.sin(thrust_angle_rad) * raw_thrust
        self.velocity_y[alive] += torch.cos(thrust_angle_rad) * raw_thrust - self.GRAVITY

        # Update position
        self.pos_x[alive] += self.velocity_x[alive]
        self.pos_y[alive] += self.velocity_y[alive]
        
        self.pos_x[alive] = self.pos_x[alive] % self.Max_width

        has_touch = alive & (self.pos_y <= self.GROUND_LEVEL)

        speed = torch.hypot(self.velocity_x, self.velocity_y)
        tol = self.LANDING_ANGLE_TOLERANCE

        is_slow = speed < self.MAX_LANDING_SPEED
        is_upright = (self.rotation > 360 - tol) | (self.rotation < tol)

        self.win = torch.where(has_touch, is_slow & is_upright, self.win)

        self.alive = self.alive & ~has_touch