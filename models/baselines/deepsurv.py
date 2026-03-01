import torch
import torch.nn as nn
from typing import Optional
from models.interface import TorchSurvivalModel


class DeepSurv(TorchSurvivalModel):
    """
    DeepSurv 深度 Cox 比例风险模型
    
    使用父类的:
    - predict_survival_function (基于 Breslow 估计)
    - _cox_predict_time (中位生存时间)
    - _cox_compute_hazard_rate (对数风险函数)
    """
    
    def __init__(self, in_dim: int, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config or {}
        hidden_dims = self.config.get("hidden_dims", [128, 128, 64, 32])
        dropout = self.config.get("dropout", 0.1)
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.SELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.risk_net = nn.Sequential(*layers)
        self._baseline_cum_haz = None
        self._baseline_times = None
        self._train_times = None
        self._train_events = None
        self._train_log_haz = None

    def forward_loss(self, features: torch.Tensor, times: torch.Tensor, events: torch.Tensor, **kwargs):
        log_haz = self.risk_net(features).squeeze(-1)
        sort_idx = torch.argsort(times)
        log_haz_s = log_haz[sort_idx]
        events_s = events[sort_idx]
        exp_haz = torch.exp(log_haz_s)
        risk_set = torch.flip(torch.cumsum(torch.flip(exp_haz, [0]), 0), [0])
        event_mask = events_s.bool()
        if not event_mask.any():
            return torch.tensor(0.0, device=features.device, requires_grad=True), {"neg_partial_ll": 0.0}
        log_lik = log_haz_s[event_mask] - torch.log(risk_set[event_mask] + 1e-10)
        neg_ll = -log_lik.mean()
        return neg_ll, {"neg_partial_ll": neg_ll.item()}

    def _fit_baseline_hazard(self, times: torch.Tensor, events: torch.Tensor, log_haz: torch.Tensor):
        self._baseline_times, self._baseline_cum_haz = super()._fit_breslow_baseline_hazard(times, events, log_haz)
        self._train_times = times
        self._train_events = events
        self._train_log_haz = log_haz

    def predict_risk(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            risk = self.risk_net(features).squeeze(-1)
            return torch.nan_to_num(risk, nan=0.0, posinf=20.0, neginf=-20.0).float()

    def predict_time(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._cox_predict_time(features)

    def compute_hazard_rate(self, features: torch.Tensor, time_grid: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._cox_compute_hazard_rate(features, time_grid)
