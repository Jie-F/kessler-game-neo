# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from typing import TYPE_CHECKING

import numpy as np

from .ship import Ship
from .scenario import Scenario
from .team import Team
if TYPE_CHECKING:
    from .kessler_game import StopReason


class Score:
    def __init__(self, scenario: Scenario) -> None:
        self.sim_time: float = 0.0
        self.stop_reason: 'StopReason' | None = None


        # Initialize team classes to score team-specific scores
        team_ids = [ship.team for ship in scenario.ships()]
        team_names = [ship.team_name for ship in scenario.ships()]
        self.teams = [Team(int(team_id), str(team_name)) for team_id, team_name in zip(np.unique(team_ids), np.unique(team_names))]

        # Populate scenario initial conditions into score parameters
        for team in self.teams:
            team.total_asteroids = scenario.max_asteroids
            for ship in scenario.ships():
                if team.team_id == ship.team:
                    team.total_bullets += scenario.bullet_limit

    def update(self, ships: list[Ship], sim_time: float, controller_perf: list[float] | None = None) -> None:
        self.sim_time = sim_time
        for team in self.teams:
            ast_hit, bul_hit, shots, bullets, mines, deaths, lives = (0, 0, 0, 0, 0, 0, 0)
            for idx, ship in enumerate(ships):
                if team.team_id == ship.team:
                    ast_hit += ship.asteroids_hit
                    bul_hit += ship.bullets_hit
                    shots += ship.bullets_shot
                    bullets += ship.bullets_remaining
                    mine_hit += ship.mines_hit
                    mines_dropped += ship.mines_dropped
                    mines += ship.mines_remaining
                    deaths += ship.deaths
                    lives += ship.lives
                    if controller_perf is not None and controller_perf[idx] > 0:
                        team.eval_times.append(controller_perf[idx])
            team.asteroids_hit, team.bullets_hit, team.shots_fired, team.bullets_remaining, team.mines_hit, team.mines_dropped, team.mines_remaining, team.deaths, team.lives_remaining = (ast_hit, bul_hit, shots, bullets, mine_hit, mines_dropped, mines, deaths, lives)

    def finalize(self, sim_time: float, stop_reason: 'StopReason', ships: list[Ship]) -> None:
        self.sim_time = sim_time
        self.stop_reason = stop_reason
        self.final_controllers = [ship.controller for ship in ships]

    def __repr__(self) -> str:
        team_summaries = []
        for team in self.teams:
            summary = (
                f"Team {team.team_id} ({team.team_name}): "
                f"Asteroids Hit={team.asteroids_hit}, Bullets Hit={team.bullets_hit}, "
                f"Shots Fired={team.shots_fired}, Bullets Remaining={team.bullets_remaining}, "
                f"Mines Hit={team.mines_hit}, Mines Dropped={team.mines_dropped}, "
                f"Mines Remaining={team.mines_remaining}, Deaths={team.deaths}, "
                f"Lives Remaining={team.lives_remaining}"
            )
            team_summaries.append(summary)
        
        stop = self.stop_reason.name if self.stop_reason else "None"
        return (
            f"<Score(sim_time={self.sim_time:.2f}, stop_reason={stop},\n"
            f" Teams:\n  " + "\n  ".join(team_summaries) + ")>"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Score):
            return False

        if self.sim_time != other.sim_time or self.stop_reason != other.stop_reason:
            return False

        if len(self.teams) != len(other.teams):
            return False

        for team_self, team_other in zip(self.teams, other.teams):
            if team_self != team_other:
                return False

        return True
