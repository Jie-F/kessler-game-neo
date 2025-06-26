from src.kesslergame import KesslerController

import math
from typing import Tuple
import sys

def is_close_to_zero(x: float, eps: float = 1e-9) -> bool:
    return abs(x) < eps

def collision_prediction(ax: float, ay: float, vax: float, vay: float, ra: float, bx: float, by: float, vbx: float, vby: float, rb: float) -> tuple[float, float]:
    separation = ra + rb
    dx = ax - bx
    dy = ay - by
    dvx = vax - vbx
    dvy = vay - vby
    dist_sq = dx * dx + dy * dy
    speed_sq = dvx * dvx + dvy * dvy
    dot = dx * dvx + dy * dvy
    sep_sq = separation * separation
    if abs(speed_sq) < 1e-10:
        if dist_sq <= sep_sq:
            return (-math.inf, math.inf)  # Overlapping forever
        else:
            return (math.nan, math.nan)   # Never collide
    if dot >= 0.0 and dist_sq > sep_sq:
        return (math.nan, math.nan)       # Moving apart or tangent
    cos_theta_sq = (dot * dot) / (dist_sq * speed_sq)
    sin_theta_sq = 1.0 - cos_theta_sq
    min_sin_sq = sep_sq / dist_sq
    if sin_theta_sq > min_sin_sq:
        return (math.nan, math.nan)       # Will miss each other
    root_term = math.sqrt((sep_sq - dist_sq * sin_theta_sq) / speed_sq)
    t_mid = -dot / speed_sq
    t_enter = t_mid - root_term
    t_exit  = t_mid + root_term
    return (t_enter, t_exit)

class NovaController(KesslerController):
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_ts_fired = -100

    def actions(self, ship_state: dict, game_state: dict) -> tuple[float, float, bool, bool]:
        if game_state["sim_frame"] == 0:
            self.reset()
        fire = True
        drop_mine = False
        thrust = 0
        turn_rate = 1
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Nova"
