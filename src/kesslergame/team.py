# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from statistics import mean, median


class Team:
    def __init__(self, id: int, name: str) -> None:
        self.team_id = id
        self.team_name = name

        self.total_bullets: int = 0
        self.total_asteroids: int = 0

        self.bullet_asteroid_hits: int = 0
        self.ship_asteroid_hits: int = 0
        self.ship_ship_hits: int = 0
        self.mine_ship_hits: int = 0
        self.mine_asteroid_hits: int = 0

        self.shots_fired: int = 0
        self.bullets_remaining: int = 0
        self.mines_dropped: int = 0
        self.mines_remaining: int = 0
        self.deaths: int = 0
        self.eval_times: list[float] = []
        self.lives_remaining: int = 0

    @property
    def asteroids_hit(self) -> int:
        return self.bullet_asteroid_hits + self.ship_asteroid_hits + self.mine_asteroid_hits

    @property
    def ships_hit(self) -> int:
        return self.ship_ship_hits + self.mine_ship_hits

    @property
    def accuracy(self) -> float:
        return self.bullet_asteroid_hits / self.shots_fired if self.shots_fired != 0 else 0.0

    @property
    def fraction_total_asteroids_hit(self) -> float:
        return self.asteroids_hit / self.total_asteroids if self.total_asteroids != 0 else 0.0

    @property
    def fraction_bullets_used(self) -> float:
        return self.shots_fired / self.total_bullets if self.total_bullets != 0 else 0.0

    @property
    def ratio_bullets_needed(self) -> float:
        return self.shots_fired / self.total_asteroids if self.total_asteroids != 0 else 0.0

    @property
    def mean_eval_time(self) -> float:
        if self.eval_times:
            return float(mean(self.eval_times))
        else:
            return 0.0

    @property
    def median_eval_time(self) -> float:
        if self.eval_times:
            return float(median(self.eval_times))
        else:
            return 0.0

    @property
    def min_eval_time(self) -> float:
        if self.eval_times:
            return min(self.eval_times)
        else:
            return 0.0

    @property
    def max_eval_time(self) -> float:
        if self.eval_times:
            return max(self.eval_times)
        else:
            return 0.0

    def __repr__(self) -> str:
        return (
            f"Team {self.team_id} ({self.team_name}): "
            f"Asteroids Hit={self.asteroids_hit}, Bullet-Asteroid Hits={self.bullet_asteroid_hits}, "
            f"Shots Fired={self.shots_fired}, Bullets Remaining={self.bullets_remaining}, "
            f"Mine-Asteroid Hits={self.mine_asteroid_hits}, Mines Dropped={self.mines_dropped}, "
            f"Mines Remaining={self.mines_remaining}, "
            f"Ship-Asteroid Hits={self.ship_asteroid_hits}, Ship-Ship Hits={self.ship_ship_hits}, "
            f"Mine-Ship Hits={self.mine_ship_hits}, "
            f"Deaths={self.deaths}, "
            f"Lives Remaining={self.lives_remaining}, "
            f"Eval Times={self.eval_times}"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Team):
            return False

        return (
            self.team_id == other.team_id and
            self.team_name == other.team_name and
            self.total_bullets == other.total_bullets and
            self.total_asteroids == other.total_asteroids and
            self.bullet_asteroid_hits == other.bullet_asteroid_hits and
            self.ship_asteroid_hits == other.ship_asteroid_hits and
            self.ship_ship_hits == other.ship_ship_hits and
            self.mine_ship_hits == other.mine_ship_hits and
            self.mine_asteroid_hits == other.mine_asteroid_hits and
            self.shots_fired == other.shots_fired and
            self.bullets_remaining == other.bullets_remaining and
            self.mines_dropped == other.mines_dropped and
            self.mines_remaining == other.mines_remaining and
            self.deaths == other.deaths and
            self.lives_remaining == other.lives_remaining# and
            #self.eval_times == other.eval_times
        )
