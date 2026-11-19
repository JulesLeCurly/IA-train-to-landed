import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import Src.Model_gestion.Scheduler as Scheduler


# ============================================================================
# Constantes Symboliques NEW_NERV / MERV
# ============================================================================
ACTION_WAIT = 0
ACTION_LONG = 1
ACTION_SHORT = 2
ACTION_CLOSE = 3

STATUS_SHORT = -1
STATUS_FLAT = 0
STATUS_LONG = 1

REASON_NONE = -1
REASON_MODEL_CLOSE = 0
REASON_TAKE_PROFIT = 1
REASON_STOP_LOSS = 2
REASON_LIQUIDATION = 3


class VectorizedTradingEngine:
    """
    VectorizedTradingEngine (Architecture NEW_NERV / MERV - 1D Pure)
    
    Moteur de simulation financière 100% vectorisé :
    - Espace d'action discret strict (4 actions : Wait, Long, Short, Close).
    - Dimensionnement déterministe invariant : L* = (R_max * C) / (c * d_SL).
    - Trailing Stop Chandelier 3 paliers : Invalidation -> Breakeven (1.5*d_SL) -> Chandelier Ratchet.
    - Facturation sur valeur notionnelle réelle (Taker Open, Close, Spread, Funding forfaitaire ou réel signé : FUNDING_MODE).
    - Clôtures d'urgence et gaps intrabar modélisés au Stop Loss : min(SL, R_open).
    - États vivants 1D compacts (Parallel_Scenarios,). Zéro rétrocompatibilité.
    """

    def __init__(
        self,
        Finance_data: Union[Dict[str, torch.Tensor], torch.Tensor],
        Finance_Timestamps: torch.Tensor,
        Symbols: List[str],
        Train_Config: Dict[str, Any],
        Dataset_Config: Dict[str, Any],
        NERV_Env_Config: Dict[str, Any],
        mode: str = "train",
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        Swarm_Timestamps: Optional[torch.Tensor] = None,
        NERV_Curriculum_Config: Optional[Dict[str, Any]] = None,
        swarm_seq_len: int = 1,
        **kwargs
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.mode = mode
        self.swarm_seq_len = max(1, int(swarm_seq_len))
        self.Symbols = Symbols
        self.Train_Config = Train_Config
        self.Dataset_Config = Dataset_Config
        self.NERV_Env_Config = NERV_Env_Config
        self.Swarm_Timestamps = Swarm_Timestamps
        self.NERV_Curriculum_Config = NERV_Curriculum_Config

        # Données de marché sur le device cible
        if isinstance(Finance_data, dict):
            self.Finance_data = {k: torch.as_tensor(v, device=self.device) for k, v in Finance_data.items()}
        else:
            self.Finance_data = torch.as_tensor(Finance_data, device=self.device)
        self.Finance_Timestamps = torch.as_tensor(Finance_Timestamps, device=self.device)

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
        if self.FUNDING_MODE == "real" and not (isinstance(self.Finance_data, dict) and "funding_rate" in self.Finance_data):
            raise ValueError("FUNDING_MODE 'real' : Finance_data['funding_rate'] manquant (voir Manage_Data.Get_Finance_Data).")
        self._init_time_bounds()
        self._allocate_memory()

        self.scenarios_idx = torch.arange(self.Parallel_Scenarios, device=self.device, dtype=torch.int64)
        self.reset(self.scenarios_idx)

    @property
    def min_holding_bars_before_close(self) -> int:
        """Retourne la durée minimale de portage courante (dynamique via Scheduler si dict)."""
        val = Scheduler.get_scheduled_value(self.min_holding_bars_cfg, self.current_iteration)
        return int(round(float(val)))

    def _init_parameters(self):
        """Initialise tous les paramètres financiers et de risque (Accès strict sans valeur par défaut silencieuse)."""
        cfg = self.NERV_Env_Config

        # Frais réels d'exchange
        self.FEE_TAKER_OPEN = float(cfg["FEE_TAKER_OPEN"])
        self.FEE_TAKER_CLOSE = float(cfg["FEE_TAKER_CLOSE"])
        self.FEE_SPREAD = float(cfg["FEE_SPREAD"])
        self.FEE_FUNDING_HOURLY = float(cfg["FEE_FUNDING_HOURLY"])
        # Financement : "real" = taux réels de chaque perpétuel (Finance_data["funding_rate"], réglés à l'ouverture des heures
        # de règlement pendant que la position est ouverte ; un taux > 0 est payé par les longs et REÇU par les shorts) ;
        # "flat" = ancien forfait FEE_FUNDING_HOURLY par heure, payé par les deux côtés. Clé absente -> "flat" (anciennes configs).
        self.FUNDING_MODE = str(cfg.get("FUNDING_MODE", "flat"))
        if self.FUNDING_MODE not in ("flat", "real"):
            raise ValueError(f"FUNDING_MODE invalide '{self.FUNDING_MODE}' (attendu 'flat' ou 'real').")
        self.LIQUIDATION_THRESHOLD = float(cfg["LIQUIDATION_THRESHOLD"])
        self.LIQUIDATION_PENALTY = float(cfg["LIQUIDATION_PENALTY"])

        # Paramètres d'exposition et risque
        self.Cooldown = int(cfg["Cooldown"])
        self.bankruptcy_threshold = float(cfg["Bankruptcy_Threshold"])
        self.invest_capital_fixed = float(cfg["Invest_capital"])

        raw_lev = cfg["fixed_leverage"]
        self.fixed_leverage = float(raw_lev) if (raw_lev is not None and raw_lev is not False) else None

        self.k_atr_stop = float(cfg["k_atr_stop"])
        self.min_sl_config = float(cfg["min_sl_config"])
        self.max_sl_config = float(cfg["max_sl_config"])

        self.k_atr_trail = float(cfg["k_atr_trail"])
        self.breakeven_ratio = float(cfg["breakeven_ratio"])
        self.risk_target_max = float(cfg["risk_target_max"])

        self.min_lever_config = float(cfg["min_lever_config"])
        self.max_lever_config = float(cfg["max_lever_config"])

        self.min_holding_bars_cfg = cfg["min_holding_bars_before_close"]
        self.early_close_profit_threshold = float(cfg["early_close_profit_threshold"])

        self._fees_roundtrip = self.FEE_TAKER_OPEN + self.FEE_TAKER_CLOSE + (2.0 * self.FEE_SPREAD)

    def _init_time_bounds(self):
        """Découpage des bornes temporelles et alignement garanti anti-OOB."""
        self.Train_Bloc = self.Train_Config["Train_Bloc"]
        self.Val_Bloc = self.Train_Config["Val_Bloc"]

        def _ts_idx(bloc_name: str, point: int) -> int:
            val = self.Dataset_Config[bloc_name][point]
            return torch.searchsorted(self.Finance_Timestamps, torch.tensor(val, device=self.device)).item()

        train_s = _ts_idx(self.Train_Bloc, 0)
        train_e = _ts_idx(self.Train_Bloc, 1)
        val_s = _ts_idx(self.Val_Bloc, 0)
        val_e = _ts_idx(self.Val_Bloc, 1)

        # Alignement avec l'historique Swarm EVA disponible
        seq_len = self.swarm_seq_len
        if self.Swarm_Timestamps is not None and len(self.Swarm_Timestamps) > 0:
            swarm_min_ts = self.Swarm_Timestamps[seq_len - 1]
            swarm_start = torch.searchsorted(self.Finance_Timestamps, swarm_min_ts).item()
            swarm_end = torch.searchsorted(self.Finance_Timestamps, self.Swarm_Timestamps[-1], right=True).item() - 1
        else:
            swarm_start = 0
            swarm_end = self.Finance_data["open"].shape[1] - 1

        max_t = self.Finance_data["open"].shape[1]
        max_allowed_start = max_t - self.episode_length - 1
        if max_allowed_start < 0:
            raise ValueError(f"Finance_data trop court ({max_t}) pour episode_length={self.episode_length}")

        def _clip_slice(s_raw: int, e_raw: int) -> Tuple[int, int]:
            s = max(s_raw, swarm_start)
            e = min(e_raw - self.episode_length, swarm_end - self.episode_length - 2, max_allowed_start + 1)
            if e <= s:
                s = max(0, min(s, max_allowed_start))
                e = s + 1
            return int(s), int(e)

        self.train_start_idx, self.train_end_idx = _clip_slice(train_s, train_e)
        # "b1" = the whole OOF training period (Train_Bloc: Bloc_1, or an alias such as Bloc_01 = Bloc_0 + Bloc_1),
        # "b2" = the Master period. Naming "Bloc_1" here would silently drop Bloc_0 from the curriculum.
        if self.Train_Bloc in self.Dataset_Config and "Bloc_2" in self.Dataset_Config:
            self.train_b1_start_idx, self.train_b1_end_idx = _clip_slice(_ts_idx(self.Train_Bloc, 0), _ts_idx(self.Train_Bloc, 1))
            self.train_b2_start_idx, self.train_b2_end_idx = _clip_slice(_ts_idx("Bloc_2", 0), _ts_idx("Bloc_2", 1))

        self.val_start_idx, self.val_end_idx = _clip_slice(val_s, val_e)

        # Curriculum scheduler optionnel
        self.curriculum_scheduler = None
        if self.NERV_Curriculum_Config is not None:
            try:
                from Src.Training.NERV.Curriculum_learning import CurriculumScheduler
                self.curriculum_scheduler = CurriculumScheduler(self.NERV_Curriculum_Config, device=self.device)
                if self.mode == "train" and self.curriculum_scheduler.enabled:
                    print(f">>> [Curriculum] CurriculumScheduler activé avec succès (Stratégie: {self.curriculum_scheduler.strategy}, min_b2: {self.curriculum_scheduler.min_bloc2_prob*100:.0f}%, max_b2: {self.curriculum_scheduler.max_bloc2_prob*100:.0f}%)")
            except Exception as e:
                print(f"[Attention] Échec de l'initialisation du CurriculumScheduler: {e}")
                self.curriculum_scheduler = None
        self.current_iteration = 0

        # Grille de validation déterministe
        if self.mode == "val":
            num_sym = len(self.Symbols)
            self.val_offsets_per_symbol = int(self.Train_Config["Val_Offsets_Per_Symbol"])
            self.Parallel_Scenarios = num_sym * self.val_offsets_per_symbol

            val_syms, val_starts = [], []
            stride = (self.val_end_idx - self.val_start_idx) / max(1, self.val_offsets_per_symbol - 1) if self.val_offsets_per_symbol > 1 else 0
            for s_idx in range(num_sym):
                for k in range(self.val_offsets_per_symbol):
                    st = int(round(self.val_start_idx + k * stride)) if self.val_offsets_per_symbol > 1 else int(self.val_start_idx)
                    val_syms.append(s_idx)
                    val_starts.append(st)

            self.val_grid_symbols = torch.tensor(val_syms, dtype=torch.int32, device=self.device)
            self.val_grid_starting_indices = torch.tensor(val_starts, dtype=torch.int32, device=self.device)

    def _allocate_memory(self):
        """Alloue les tenseurs d'état vivants en 1D compact (Parallel_Scenarios,)."""
        n = self.Parallel_Scenarios
        dev = self.device

        # Progression temporelle
        self.current_step = torch.zeros(n, dtype=torch.int32, device=dev)
        self.starting_indices = torch.zeros(n, dtype=torch.int32, device=dev)
        self.symbols_scenarios = torch.zeros(n, dtype=torch.int32, device=dev)

        # Portefeuille et banqueroute
        self.wallet = torch.ones(n, dtype=torch.float32, device=dev)
        self.initial_wallet = torch.ones(n, dtype=torch.float32, device=dev)
        self.is_bankrupt = torch.zeros(n, dtype=torch.bool, device=dev)

        # Paramètres du trade actif
        self.trade_status = torch.zeros(n, dtype=torch.int32, device=dev)
        self.entry_price = torch.zeros(n, dtype=torch.float32, device=dev)
        self.entry_atr_pct = torch.zeros(n, dtype=torch.float32, device=dev)
        self.entry_sl_dist = torch.zeros(n, dtype=torch.float32, device=dev)
        self.invest_capital = torch.zeros(n, dtype=torch.float32, device=dev)
        self.leverage = torch.zeros(n, dtype=torch.float32, device=dev)
        self.current_sl_pct = torch.zeros(n, dtype=torch.float32, device=dev)
        self.peak_gross_return = torch.zeros(n, dtype=torch.float32, device=dev)
        self.gross_return = torch.zeros(n, dtype=torch.float32, device=dev)
        self.funding_paid = torch.zeros(n, dtype=torch.float32, device=dev)   # financement réel cumulé du trade (fraction du notionnel)

        # Compteurs et horloges
        self.time_in_trade = torch.zeros(n, dtype=torch.int32, device=dev)
        self.time_since_last_trade = torch.zeros(n, dtype=torch.int32, device=dev)
        self.trades_count = torch.zeros(n, dtype=torch.int32, device=dev)

        # Événements et métriques de step
        self.trade_just_opened = torch.zeros(n, dtype=torch.bool, device=dev)
        self.trade_just_closed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_realized_pnl = torch.zeros(n, dtype=torch.float32, device=dev)
        self.last_close_reason = torch.full((n,), REASON_NONE, dtype=torch.int32, device=dev)
        self.last_trade_duration = torch.zeros(n, dtype=torch.int32, device=dev)
        self.last_trade_mfe = torch.zeros(n, dtype=torch.float32, device=dev)
        self.last_trade_leverage = torch.zeros(n, dtype=torch.float32, device=dev)
        self.last_trade_direction = torch.zeros(n, dtype=torch.int32, device=dev)

        # Mémoire Post-Trade persistante (survit aux étapes Flat)
        self.prev_trade_pnl = torch.zeros(n, dtype=torch.float32, device=dev)
        self.prev_trade_reason = torch.full((n,), REASON_NONE, dtype=torch.int32, device=dev)
        self.prev_trade_direction = torch.zeros(n, dtype=torch.int32, device=dev)

    def _clear_trade_state(self, mask: torch.Tensor):
        """Réinitialise les paramètres internes de trade pour les scénarios dont la position se clôture."""
        z = torch.tensor(0.0, device=self.device)
        self.entry_price = torch.where(mask, z, self.entry_price)
        self.entry_atr_pct = torch.where(mask, z, self.entry_atr_pct)
        self.entry_sl_dist = torch.where(mask, z, self.entry_sl_dist)
        self.invest_capital = torch.where(mask, z, self.invest_capital)
        self.leverage = torch.where(mask, z, self.leverage)
        self.current_sl_pct = torch.where(mask, z, self.current_sl_pct)
        self.peak_gross_return = torch.where(mask, z, self.peak_gross_return)
        self.funding_paid = torch.where(mask, z, self.funding_paid)

    def _funding_cost(self, held_bars: torch.Tensor) -> torch.Tensor:
        """Coût de financement du trade en cours (fraction du notionnel, négatif = gain) selon FUNDING_MODE."""
        if self.FUNDING_MODE == "real":
            return self.funding_paid
        return self.FEE_FUNDING_HOURLY * held_bars.float()

    def reset(self, scenarios_to_reset: Optional[torch.Tensor] = None):
        """Réinitialise les scénarios spécifiés de manière 100% Flat sans récursion."""
        if scenarios_to_reset is None:
            scenarios_to_reset = self.scenarios_idx

        idx = scenarios_to_reset
        n_reset = len(idx)
        if n_reset == 0:
            return

        self.current_step[idx] = 0
        self.trades_count[idx] = 0
        self.is_bankrupt[idx] = False

        self.wallet[idx] = 1.0
        self.initial_wallet[idx] = 1.0

        self.trade_status[idx] = STATUS_FLAT
        self._clear_trade_state(torch.isin(self.scenarios_idx, idx))
        self.gross_return[idx] = 0.0

        self.time_in_trade[idx] = 0
        self.time_since_last_trade[idx] = self.Cooldown

        self.trade_just_opened[idx] = False
        self.trade_just_closed[idx] = False
        self.last_realized_pnl[idx] = 0.0
        self.last_close_reason[idx] = REASON_NONE
        self.last_trade_duration[idx] = 0
        self.last_trade_mfe[idx] = 0.0
        self.prev_trade_pnl[idx] = 0.0
        self.prev_trade_reason[idx] = REASON_NONE
        self.prev_trade_direction[idx] = 0

        if self.mode == "train":
            if self.curriculum_scheduler is not None and self.curriculum_scheduler.enabled:
                self.starting_indices[idx] = self.curriculum_scheduler.sample_indices(
                    n_samples=n_reset,
                    current_iteration=self.current_iteration,
                    b1_bounds=(self.train_b1_start_idx, self.train_b1_end_idx),
                    b2_bounds=(self.train_b2_start_idx, self.train_b2_end_idx)
                )
            else:
                self.starting_indices[idx] = torch.randint(
                    self.train_start_idx, self.train_end_idx, (n_reset,), device=self.device, dtype=torch.int32
                )
            self.symbols_scenarios[idx] = torch.randint(0, len(self.Symbols), (n_reset,), device=self.device, dtype=torch.int32)
        elif self.mode == "val":
            self.starting_indices[idx] = self.val_grid_starting_indices[idx]
            self.symbols_scenarios[idx] = self.val_grid_symbols[idx]

    def get_action_mask(self) -> torch.Tensor:
        """
        Retourne le masque booléen (N, 4) des actions autorisées.
        [0: Wait, 1: Long, 2: Short, 3: Close]
        """
        n = self.Parallel_Scenarios
        mask = torch.zeros((n, 4), dtype=torch.bool, device=self.device)

        is_flat = (self.trade_status == STATUS_FLAT)
        in_cooldown = (self.time_since_last_trade < self.Cooldown)

        # Scénarios Flat : Wait toujours autorisé ; Long et Short autorisés hors cooldown
        mask[is_flat, ACTION_WAIT] = True
        mask[is_flat & ~in_cooldown, ACTION_LONG] = True
        mask[is_flat & ~in_cooldown, ACTION_SHORT] = True

        # Scénarios en position : Wait (Hold) toujours autorisé.
        # Close n'est autorisé que si la durée minimale de portage est atteinte
        # OU si le rendement brut a atteint le seuil de gain anticipé.
        can_close = (self.time_in_trade >= self.min_holding_bars_before_close) | (self.gross_return >= self.early_close_profit_threshold)
        mask[~is_flat, ACTION_WAIT] = True
        mask[~is_flat & can_close, ACTION_CLOSE] = True

        # Scénarios en Faillite : Seul Wait est autorisé
        mask[self.is_bankrupt, :] = False
        mask[self.is_bankrupt, ACTION_WAIT] = True
        return mask

    def step(self, action: torch.Tensor, conviction: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exécute la transition d'état complète (Décision modèle -> Avancement intrabar marché -> Stops)."""
        self._apply_model_action(action, conviction)
        terminated, truncated = self._advance_market_step()
        return terminated, truncated

    def _apply_model_action(self, action: torch.Tensor, conviction: Optional[torch.Tensor] = None):
        """Applique l'action discrète du modèle à l'Open de la bougie courante."""
        action = action.to(dtype=torch.int32, device=self.device)
        status_before = self.trade_status.clone()
        action = torch.where(self.is_bankrupt, torch.tensor(ACTION_WAIT, dtype=torch.int32, device=self.device), action)

        wants_open_long = (status_before == STATUS_FLAT) & (action == ACTION_LONG)
        wants_open_short = (status_before == STATUS_FLAT) & (action == ACTION_SHORT)
        can_close = (self.time_in_trade >= self.min_holding_bars_before_close) | (self.gross_return >= self.early_close_profit_threshold)
        wants_close = (status_before != STATUS_FLAT) & (action == ACTION_CLOSE) & can_close

        in_cooldown = (self.time_since_last_trade < self.Cooldown)
        can_open_long = wants_open_long & ~in_cooldown
        can_open_short = wants_open_short & ~in_cooldown
        newly_opened = can_open_long | can_open_short
        new_direction = torch.where(can_open_long, STATUS_LONG, torch.where(can_open_short, STATUS_SHORT, status_before))

        open_price_now = self.Finance_data["open"][self.symbols_scenarios, self.starting_indices + self.current_step]

        # 1. Clôture volontaire par l'IA (Action Close)
        safe_entry = torch.where(status_before != STATUS_FLAT, self.entry_price, open_price_now)
        gross_return_close = (open_price_now - safe_entry) / (safe_entry + 1e-8) * status_before
        funding_fees = self._funding_cost(self.time_in_trade)
        total_fees = self.FEE_TAKER_OPEN + self.FEE_TAKER_CLOSE + (2.0 * self.FEE_SPREAD) + funding_fees
        pnl_net_close = self.leverage * (gross_return_close - total_fees)

        self.wallet = torch.where(
            wants_close,
            torch.clamp(self.wallet + (self.wallet * self.invest_capital * pnl_net_close), min=0.0, max=1e30),
            self.wallet
        )
        self.last_realized_pnl = torch.where(wants_close, self.invest_capital * pnl_net_close, torch.tensor(0.0, device=self.device))
        self.last_close_reason = torch.where(
            wants_close, torch.tensor(REASON_MODEL_CLOSE, dtype=torch.int32, device=self.device), torch.tensor(REASON_NONE, dtype=torch.int32, device=self.device)
        )
        self.last_trade_duration = torch.where(wants_close, self.time_in_trade, torch.tensor(0, dtype=torch.int32, device=self.device))
        self.last_trade_mfe = torch.where(wants_close, torch.maximum(self.peak_gross_return, gross_return_close), torch.tensor(0.0, device=self.device))
        self.last_trade_leverage = torch.where(wants_close, self.leverage, torch.tensor(0.0, device=self.device))
        self.last_trade_direction = torch.where(wants_close, status_before, torch.tensor(0, dtype=torch.int32, device=self.device))

        # Persistance de la mémoire du dernier trade pour les futures étapes Flat
        self.prev_trade_pnl = torch.where(wants_close, self.invest_capital * pnl_net_close, self.prev_trade_pnl)
        self.prev_trade_reason = torch.where(
            wants_close, torch.tensor(REASON_MODEL_CLOSE, dtype=torch.int32, device=self.device), self.prev_trade_reason
        )
        self.prev_trade_direction = torch.where(wants_close, status_before, self.prev_trade_direction)

        self._clear_trade_state(wants_close)

        # 2. Ouverture d'un nouveau trade (Dimensionnement déterministe invariant)
        if torch.any(newly_opened):
            cur_t = self.starting_indices + self.current_step
            sym_t = self.symbols_scenarios

            if "atr_pct" in self.Finance_data:
                atr_raw = self.Finance_data["atr_pct"][sym_t, cur_t]
                atr_cur = torch.nan_to_num(atr_raw, nan=0.015)
            else:
                atr_cur = torch.full_like(self.wallet, 0.015)

            # Distance au Stop Loss d'invalidation initiale d_SL
            d_sl = torch.clamp(self.k_atr_stop * atr_cur, min=self.min_sl_config, max=self.max_sl_config)

            c_factor = conviction if conviction is not None else torch.ones_like(self.wallet)
            c_factor = torch.clamp(c_factor, min=0.05, max=1.0)
            e_optimal = (self.risk_target_max * c_factor) / (d_sl + 1e-8)

            c_val = self.invest_capital_fixed
            if self.fixed_leverage is not None:
                l_star = torch.full_like(self.wallet, self.fixed_leverage)
            else:
                l_star = torch.clamp(e_optimal / (c_val + 1e-8), min=self.min_lever_config, max=self.max_lever_config)

            self.entry_price = torch.where(newly_opened, open_price_now, self.entry_price)
            self.entry_atr_pct = torch.where(newly_opened, atr_cur, self.entry_atr_pct)
            self.entry_sl_dist = torch.where(newly_opened, d_sl, self.entry_sl_dist)
            self.invest_capital = torch.where(newly_opened, torch.tensor(c_val, device=self.device), self.invest_capital)
            self.leverage = torch.where(newly_opened, l_star, self.leverage)
            self.current_sl_pct = torch.where(newly_opened, -d_sl, self.current_sl_pct)
            self.peak_gross_return = torch.where(newly_opened, torch.tensor(0.0, device=self.device), self.peak_gross_return)
            self.funding_paid = torch.where(newly_opened, torch.tensor(0.0, device=self.device), self.funding_paid)
            self.trades_count += newly_opened.to(torch.int32)

        # 3. Actualisation des statuts et horloges
        self.trade_status = torch.where(wants_close, STATUS_FLAT, new_direction)
        self.trade_just_closed = wants_close.clone()
        self.trade_just_opened = newly_opened.clone()

        reset_clock = wants_close | newly_opened
        self.time_in_trade = torch.where(reset_clock, torch.tensor(0, dtype=torch.int32, device=self.device), self.time_in_trade)
        self.time_since_last_trade = torch.where(reset_clock, torch.tensor(0, dtype=torch.int32, device=self.device), self.time_since_last_trade)

        is_active = (self.trade_status != STATUS_FLAT)
        safe_entry_now = torch.where(is_active, self.entry_price, open_price_now)
        self.gross_return = torch.where(
            is_active, (open_price_now - safe_entry_now) / (safe_entry_now + 1e-8) * self.trade_status, torch.tensor(0.0, device=self.device)
        )

    def _advance_market_step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fait progresser le marché d'une bougie intrabar (t -> t+1) et surveille les stops/liquidations."""
        prev_step = self.current_step.clone()
        self.current_step += 1

        sym_idx = self.symbols_scenarios
        step_eval = self.starting_indices + prev_step

        open_bar = self.Finance_data["open"][sym_idx, step_eval]
        high_bar = self.Finance_data["high"][sym_idx, step_eval]
        low_bar = self.Finance_data["low"][sym_idx, step_eval]

        active_mask = (self.trade_status != STATUS_FLAT)
        safe_entry = torch.where(active_mask, self.entry_price, open_bar)

        gross_at_open = (open_bar - safe_entry) / (safe_entry + 1e-8) * self.trade_status
        gross_at_high = (high_bar - safe_entry) / (safe_entry + 1e-8) * self.trade_status
        gross_at_low = (low_bar - safe_entry) / (safe_entry + 1e-8) * self.trade_status

        max_gross_bar = torch.maximum(gross_at_high, gross_at_low)
        min_gross_bar = torch.minimum(gross_at_high, gross_at_low)

        # 1. Vérification Liquidation & Stop Loss
        time_held = self.time_in_trade + 1
        if self.FUNDING_MODE == "real":
            # Règlement à l'ouverture de cette barre si la position était déjà ouverte (pas sur la barre d'entrée :
            # la position s'ouvre à l'ouverture, après le règlement). Le short reçoit ce que le long paie.
            rate_bar = self.Finance_data["funding_rate"][sym_idx, step_eval]
            settles = active_mask & (self.time_in_trade > 0)
            self.funding_paid = torch.where(settles, self.funding_paid + self.trade_status.float() * rate_bar, self.funding_paid)
        funding_fees = self._funding_cost(time_held)
        worst_margin_ret = self.leverage * (min_gross_bar - (self.FEE_TAKER_OPEN + self.FEE_SPREAD + funding_fees))

        trade_liquidated = active_mask & (worst_margin_ret <= self.LIQUIDATION_THRESHOLD)
        trade_hit_sl = active_mask & (min_gross_bar <= self.current_sl_pct) & ~trade_liquidated
        trade_closed_auto = trade_liquidated | trade_hit_sl
        self.trade_just_closed = self.trade_just_closed | trade_closed_auto

        # 2. Règlement financier des sorties automatiques (avec slippage au Stop Loss : min(SL, R_open))
        fill_gross = torch.where(trade_hit_sl, torch.minimum(self.current_sl_pct, gross_at_open), torch.tensor(0.0, device=self.device))
        total_exit_fees = self.FEE_TAKER_OPEN + self.FEE_TAKER_CLOSE + (2.0 * self.FEE_SPREAD) + funding_fees
        auto_pnl = self.leverage * (fill_gross - total_exit_fees)

        liq_pnl = torch.clamp(torch.tensor(self.LIQUIDATION_THRESHOLD - self.LIQUIDATION_PENALTY, device=self.device), min=-1.0, max=0.0)
        auto_pnl = torch.where(trade_liquidated, liq_pnl, auto_pnl)

        self.wallet = torch.where(
            trade_closed_auto,
            torch.clamp(self.wallet + (self.wallet * self.invest_capital * auto_pnl), min=0.0, max=1e30),
            self.wallet
        )
        self.last_realized_pnl = torch.where(trade_closed_auto, self.invest_capital * auto_pnl, self.last_realized_pnl)

        is_tp = trade_hit_sl & (fill_gross > self._fees_roundtrip)
        auto_reasons = torch.where(
            trade_liquidated,
            torch.tensor(REASON_LIQUIDATION, dtype=torch.int32, device=self.device),
            torch.where(
                is_tp,
                torch.tensor(REASON_TAKE_PROFIT, dtype=torch.int32, device=self.device),
                torch.where(trade_hit_sl, torch.tensor(REASON_STOP_LOSS, dtype=torch.int32, device=self.device), self.last_close_reason)
            )
        )
        self.last_close_reason = torch.where(trade_closed_auto, auto_reasons, self.last_close_reason)
        self.last_trade_duration = torch.where(trade_closed_auto, time_held, self.last_trade_duration)
        self.last_trade_mfe = torch.where(trade_closed_auto, torch.maximum(self.peak_gross_return, max_gross_bar), self.last_trade_mfe)

        # 3. Trailing Stop Chandelier pour les positions maintenues
        trade_continues = active_mask & ~trade_closed_auto
        self.peak_gross_return = torch.where(trade_continues, torch.maximum(self.peak_gross_return, max_gross_bar), self.peak_gross_return)

        # Palier Breakeven (Exact : MFE >= breakeven_ratio * entry_sl_dist)
        is_breakeven = trade_continues & (self.peak_gross_return >= self.breakeven_ratio * self.entry_sl_dist)
        sl_be = torch.where(is_breakeven, torch.tensor(self._fees_roundtrip, device=self.device), self.current_sl_pct)

        # Palier Chandelier (Sécurisation des gains : actif UNIQUEMENT quand raw_chandelier >= Breakeven)
        # atr_pct[t+1] = ATR de la barre t, tout juste complétée (atr_pct est décalé d'une barre dans Manage_Data)
        cur_atr = self.Finance_data["atr_pct"][sym_idx, self.starting_indices + self.current_step] if "atr_pct" in self.Finance_data else self.entry_atr_pct
        cur_atr = torch.nan_to_num(cur_atr, nan=0.015)
        raw_chandelier = self.peak_gross_return - (self.k_atr_trail * cur_atr)
        sl_chandelier = torch.where(raw_chandelier >= self._fees_roundtrip, raw_chandelier, sl_be)

        # Cliquet irréversible : ne resserre JAMAIS dans la zone de perte négative
        can_trail = trade_continues & (self.time_in_trade > 0)
        sl_candidate = torch.where(can_trail, torch.maximum(sl_be, sl_chandelier), self.current_sl_pct)
        self.current_sl_pct = torch.where(trade_continues, torch.maximum(self.current_sl_pct, sl_candidate), self.current_sl_pct)

        # 4. Clôture des états et compteurs
        self.last_trade_leverage = torch.where(trade_closed_auto, self.leverage, self.last_trade_leverage)
        self.last_trade_direction = torch.where(trade_closed_auto, self.trade_status, self.last_trade_direction)

        # Persistance de la mémoire du dernier trade pour les futures étapes Flat
        self.prev_trade_pnl = torch.where(trade_closed_auto, self.invest_capital * auto_pnl, self.prev_trade_pnl)
        self.prev_trade_reason = torch.where(trade_closed_auto, auto_reasons, self.prev_trade_reason)
        self.prev_trade_direction = torch.where(trade_closed_auto, self.trade_status, self.prev_trade_direction)

        self.trade_status = torch.where(trade_closed_auto, STATUS_FLAT, self.trade_status)
        self.time_in_trade = torch.where(trade_continues, self.time_in_trade + 1, torch.tensor(0, dtype=torch.int32, device=self.device))
        self.time_since_last_trade = torch.where(
            ~active_mask | trade_closed_auto, self.time_since_last_trade + 1, torch.tensor(0, dtype=torch.int32, device=self.device)
        )

        self._clear_trade_state(trade_closed_auto)

        # Rendement brut pour le pas t+1
        price_now = self.Finance_data["open"][sym_idx, self.starting_indices + self.current_step]
        safe_entry_next = torch.where(self.trade_status != STATUS_FLAT, self.entry_price, price_now)
        self.gross_return = torch.where(
            self.trade_status != STATUS_FLAT,
            (price_now - safe_entry_next) / (safe_entry_next + 1e-8) * self.trade_status,
            torch.tensor(0.0, device=self.device)
        )

        # 5. Conditions de terminaison (Faillite et Truncation)
        newly_bankrupt = (self.wallet <= (self.initial_wallet * (1.0 + self.bankruptcy_threshold))) & ~self.is_bankrupt
        self.is_bankrupt = self.is_bankrupt | newly_bankrupt

        terminated = self.is_bankrupt.clone()
        truncated = (self.current_step >= self.episode_length)
        return terminated, truncated

    def get_equity(self) -> torch.Tensor:
        """Calcule la valeur liquidative instantanée Mark-to-Market nette du portefeuille."""
        active = (self.trade_status != STATUS_FLAT)
        funding = self._funding_cost(self.time_in_trade)
        total_fees = self.FEE_TAKER_OPEN + self.FEE_TAKER_CLOSE + (2.0 * self.FEE_SPREAD) + funding
        margin_pnl = self.leverage * (self.gross_return - total_fees)
        unrealized_delta = self.wallet * self.invest_capital * margin_pnl
        equity = torch.where(active, self.wallet + unrealized_delta, self.wallet)
        return torch.clamp(equity, min=0.0)

    def get_trade_net_return(self) -> torch.Tensor:
        """Calcule le rendement net de frais instantané non-levier des trades ouverts (0.0 si flat)."""
        active = (self.trade_status != STATUS_FLAT)
        funding = self._funding_cost(self.time_in_trade)
        total_fees = self.FEE_TAKER_OPEN + self.FEE_TAKER_CLOSE + (2.0 * self.FEE_SPREAD) + funding
        return torch.where(active, self.gross_return - total_fees, torch.tensor(0.0, device=self.device))