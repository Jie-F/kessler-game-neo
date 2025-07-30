# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from __future__ import annotations

import time

from math import inf, nan, isfinite, isnan, ceil, sqrt
from typing import Any, TypedDict, cast
from enum import Enum, IntEnum

from .scenario import Scenario
from .score import Score
from .controller import KesslerController
from .collisions import circle_line_collision_continuous, collision_time_interval, ship_asteroid_continuous_collision_time, ship_ship_continuous_collision_time
from .graphics import GraphicsType, GraphicsHandler, KesslerGraphics
from .mines import Mine
from .asteroid import Asteroid
from .ship import Ship
from .bullet import Bullet
from .settings_dicts import SettingsDict, UISettingsDict
from .state_models import GameState, ShipState


class StopReason(Enum):
    not_stopped = 0
    no_ships = 1
    no_asteroids = 2
    time_expired = 3
    out_of_bullets = 4


class PerfDict(TypedDict, total=False):
    controller_times: list[float]
    total_controller_time: float
    physics_update: float
    collisions_check: float
    score_update: float
    graphics_draw: float
    total_frame_time: float


class CollisionType(IntEnum):
    BULLET_ASTEROID = 0
    MINE_ASTEROID = 1
    MINE_SHIP = 2
    SHIP_ASTEROID = 3
    SHIP_SHIP = 4


class CollisionEvent:
    __slots__ = ("time_offset", "distance", "object_a_idx", "object_b_idx", "collision_type", "index_sum")

    TOLERANCE = 1e-10

    def __init__(self, time_offset: float, distance: float, object_a_idx: int, object_b_idx: int, collision_type: CollisionType):
        """
        Represents a collision event between two game objects at a specific time.

        :param time_offset: Time (in seconds) relative to the end of the frame. Must be in [-dt, 0.0].
        :param distance: Squared wrapped distance between the colliding objects. Used to break ties
        :param object_a_idx: Index of first object involved in the collision.
        :param object_b_idx: Index of second object involved in the collision.
        :param collision_type: Type of collision as defined in CollisionType enum.
        """
        self.time_offset = time_offset  # Time offset in seconds relative to frame end, e.g., -0.001
        self.distance = distance
        # Time offset should be in range [-delta_time, 0.0]
        self.object_a_idx = object_a_idx
        self.object_b_idx = object_b_idx
        self.collision_type = collision_type
        self.index_sum = object_a_idx + object_b_idx

    # Hand-implement each of the comparison dunders for speed, instead of using one for the others
    @staticmethod
    def _less(a: float, b: float) -> bool:
        return a < b - CollisionEvent.TOLERANCE

    @staticmethod
    def _greater(a: float, b: float) -> bool:
        return a > b + CollisionEvent.TOLERANCE

    @staticmethod
    def _equal(a: float, b: float) -> bool:
        return abs(a - b) <= CollisionEvent.TOLERANCE

    def __lt__(self, other: CollisionEvent) -> bool:
        if self._less(self.time_offset, other.time_offset):
            return True
        if self._equal(self.time_offset, other.time_offset):
            if self._less(self.distance, other.distance):
                return True
            if self._equal(self.distance, other.distance):
                return self.index_sum < other.index_sum
        return False

    def __le__(self, other: CollisionEvent) -> bool:
        if self._less(self.time_offset, other.time_offset):
            return True
        if self._equal(self.time_offset, other.time_offset):
            if self._less(self.distance, other.distance):
                return True
            if self._equal(self.distance, other.distance):
                return self.index_sum <= other.index_sum
        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CollisionEvent):
            return NotImplemented
        return (
            self._equal(self.time_offset, other.time_offset) and
            self._equal(self.distance, other.distance) and
            self.index_sum == other.index_sum
        )

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, CollisionEvent):
            return NotImplemented
        return (
            not self._equal(self.time_offset, other.time_offset) or
            not self._equal(self.distance, other.distance) or
            self.index_sum != other.index_sum
        )

    def __gt__(self, other: CollisionEvent) -> bool:
        if self._greater(self.time_offset, other.time_offset):
            return True
        if self._equal(self.time_offset, other.time_offset):
            if self._greater(self.distance, other.distance):
                return True
            if self._equal(self.distance, other.distance):
                return self.index_sum > other.index_sum
        return False

    def __ge__(self, other: CollisionEvent) -> bool:
        if self._greater(self.time_offset, other.time_offset):
            return True
        if self._equal(self.time_offset, other.time_offset):
            if self._greater(self.distance, other.distance):
                return True
            if self._equal(self.distance, other.distance):
                return self.index_sum >= other.index_sum
        return False

    def __repr__(self) -> str:
        return f"<CollisionEvent time_offset={self.time_offset}s distance={self.distance} type={self.collision_type} obj_a_idx={self.object_a_idx} obj_b_idx={self.object_b_idx}>"


class KesslerGame:
    def __init__(self, settings: SettingsDict | None = None) -> None:
        if settings is None:
            settings = {}
        # Game settings
        self.frequency: float = settings.get("frequency", 30.0)
        self.delta_time: float = 1.0 / settings.get("frequency", 30.0)
        self.perf_tracker: bool = settings.get("perf_tracker", False)
        self.prints_on: bool = settings.get("prints_on", True)
        self.graphics_type: GraphicsType = settings.get("graphics_type", GraphicsType.Tkinter)
        self.graphics_obj: KesslerGraphics | None = settings.get("graphics_obj", None)
        self.realtime_multiplier: float = settings.get("realtime_multiplier", 0.0 if self.graphics_type==GraphicsType.NoGraphics else 1.0)
        self.frame_skip: int = max(1, int(settings.get("frame_skip", int(self.frequency) if self.realtime_multiplier == 0.0 else round(self.realtime_multiplier))))
        self.time_limit: float = settings.get("time_limit", inf)
        self.random_ast_splits: bool = settings.get("random_ast_splits", False)
        self.competition_safe_mode: bool = settings.get("competition_safe_mode", True)
        self.map_width: float = 0.0 # To be set later
        self.map_height: float = 0.0

        self.collision_queue: list[CollisionEvent] = []

        # UI settings
        default_ui: UISettingsDict = {'ships': True, 'lives_remaining': True, 'accuracy': True,
                      'asteroids_hit': True, 'bullets_remaining': True, 'controller_name': True, 'scale': 1.0}
        UI_settings: UISettingsDict | str = settings.get("UI_settings", default_ui)
        if UI_settings == 'all':
            UI_settings = {'ships': True, 'lives_remaining': True, 'accuracy': True,
                                'asteroids_hit': True, 'shots_fired': True, 'bullets_remaining': True,
                                'controller_name': True, 'scale': 1.0}
        self.UI_settings = cast(UISettingsDict, UI_settings)

    def enqueue_bullet_asteroid_collisions(self, bullets: list[Bullet], asteroids: list[Asteroid], asteroid_past_time_clamp: float, asteroid_list_idx_offset: int = 0) -> None:
        # Collect all potential bullet-asteroid collisions
        # Since bullets do not wrap, we treat the bullet hitbox as being clamped at the map edge.
        # The way to calculate the collision time interval is elegant. It considers two virtual bullets,
        # the intersection of the two being the true bullet's position:
        # 1. The bullet that passes right through without getting clamped
        # 2. The bullet that is stationary, with its head right on the border, and the tail sticking into the map
        # The intersection of these two form the clamped bullet hitbox we're interested in, during the time
        # interval from when the bullet head first hits the edge, until the tail also leaves. This construction
        # is invalid before this time interval!

        def time_until_exit(x: float, y: float, vx: float, vy: float) -> float:
            """Returns the time when a point moving at (vx, vy) will fully exit the map."""
            tx = inf
            ty = inf

            if vx > 0.0:
                tx = (self.map_width - x) / vx
            elif vx < 0.0:
                tx = -x / vx

            if vy > 0.0:
                ty = (self.map_height - y) / vy
            elif vy < 0.0:
                ty = -y / vy

            return min(tx, ty)

        # We might not be able to go back a full delta_time if the asteroid wasn't alive for that long yet!
        # So we clamp that time
        collision_past_time_clamp = min(asteroid_past_time_clamp, self.delta_time)

        for bul_idx, bullet in enumerate(bullets):
            for ast_idx, asteroid in enumerate(asteroids):
                # Center the asteroid position relative to the bullet, accounting for wrapping of asteroids.
                if asteroid.x - bullet.x > 0.5 * self.map_width:
                    ast_x_centered = asteroid.x - self.map_width
                elif asteroid.x - bullet.x < -0.5 * self.map_width:
                    ast_x_centered = asteroid.x + self.map_width
                else:
                    ast_x_centered = asteroid.x

                if asteroid.y - bullet.y > 0.5 * self.map_height:
                    ast_y_centered = asteroid.y - self.map_height
                elif asteroid.y - bullet.y < -0.5 * self.map_height:
                    ast_y_centered = asteroid.y + self.map_height
                else:
                    ast_y_centered = asteroid.y

                # Compute unclamped bullet path (as if bullet passes through the map edge)
                bullet_head_x = bullet.x
                bullet_head_y = bullet.y
                bullet_tail_x = bullet.x + bullet.tail_delta_x
                bullet_tail_y = bullet.y + bullet.tail_delta_y

                # Compute when head and tail leave the visible map
                t_head_exit = time_until_exit(bullet_head_x, bullet_head_y, bullet.vx, bullet.vy)
                t_tail_exit = time_until_exit(bullet_tail_x, bullet_tail_y, bullet.vx, bullet.vy)

                # Determine valid time window for clamped bullet
                t_clamp_start = t_head_exit
                t_clamp_end = t_tail_exit
                assert t_clamp_start <= t_clamp_end
                if t_clamp_start > 0.0:
                    # This means that the bullet hasn't begun clipping on the map border yet!
                    # This is the simplest case to handle. Do the normal collision check:
                    if circle_line_collision_continuous(
                        bullet_head_x, bullet_head_y,
                        bullet_tail_x, bullet_tail_y,
                        bullet.vx, bullet.vy,
                        ast_x_centered, ast_y_centered,
                        asteroid.vx, asteroid.vy,
                        asteroid.radius,
                        collision_past_time_clamp
                    ):
                        collision_start_time, _ = collision_time_interval(
                            bullet_head_x, bullet_head_y,
                            bullet_tail_x, bullet_tail_y,
                            bullet.vx, bullet.vy,
                            ast_x_centered, ast_y_centered,
                            asteroid.vx, asteroid.vy,
                            asteroid.radius
                        )
                        if isnan(collision_start_time):
                            # This case should NEVER happen, but maybe due to some pathological numeric instability,
                            # it might be that the first function detected a barely collision, and then the second function
                            # missed it and the discriminant was -0.0000000000000001, and returns nan for collision time ¯\_(ツ)_/¯
                            continue
                        collision_time = max(-collision_past_time_clamp, collision_start_time)
                        assert -collision_past_time_clamp <= collision_time <= 0.0

                        collision_event = CollisionEvent(collision_time, 0.0, bul_idx, ast_idx + asteroid_list_idx_offset, CollisionType.BULLET_ASTEROID)
                        i = len(self.collision_queue)
                        while i > 0 and self.collision_queue[i - 1] > collision_event:
                            i -= 1
                        self.collision_queue.insert(i, collision_event)
                else:
                    # During the time interval we're checking, either the whole time or a part of the time,
                    # the bullet is at least partially out of bounds and has to have its hitbox clipped at the edge
                    
                    # Virtual bullet 1: normal moving bullet that is unclamped as it goes beyond the border
                    hit1 = circle_line_collision_continuous(
                        bullet_head_x, bullet_head_y,
                        bullet_tail_x, bullet_tail_y,
                        bullet.vx, bullet.vy,
                        ast_x_centered, ast_y_centered,
                        asteroid.vx, asteroid.vy,
                        asteroid.radius,
                        collision_past_time_clamp
                    )
                    if not hit1:
                        continue
                    t1_start, t1_end = collision_time_interval(
                        bullet_head_x, bullet_head_y,
                        bullet_tail_x, bullet_tail_y,
                        bullet.vx, bullet.vy,
                        ast_x_centered, ast_y_centered,
                        asteroid.vx, asteroid.vy,
                        asteroid.radius
                    )
                    if isnan(t1_start) or isnan(t1_end):
                        # This should never happen, but is here in case of numeric instability in a barely collision
                        continue

                    # Virtual bullet 2: bullet head pinned at boundary, tail sticking FAR into map, stationary!
                    # Remember that the t_clamp_start is a negative number. The bullet head at the end of the frame is already past bound.
                    pinned_head_x = bullet_head_x + bullet.vx * t_clamp_start
                    pinned_head_y = bullet_head_y + bullet.vy * t_clamp_start
                    # Just stick the tail of the bullet waaaaaay into the map to make sure we don't clamp on that end at all!
                    pinned_tail_x = bullet_tail_x + bullet.vx * (t_clamp_start - collision_past_time_clamp)
                    pinned_tail_y = bullet_tail_y + bullet.vy * (t_clamp_start - collision_past_time_clamp)

                    hit2 = circle_line_collision_continuous(
                        pinned_head_x, pinned_head_y,
                        pinned_tail_x, pinned_tail_y,
                        0.0, 0.0, # Stationary bullet for clamping to bound!
                        ast_x_centered, ast_y_centered,
                        asteroid.vx, asteroid.vy,
                        asteroid.radius,
                        collision_past_time_clamp
                    )
                    if not hit2:
                        continue
                    t2_start, t2_end = collision_time_interval(
                        pinned_head_x, pinned_head_y,
                        pinned_tail_x, pinned_tail_y,
                        0.0, 0.0,
                        ast_x_centered, ast_y_centered,
                        asteroid.vx, asteroid.vy,
                        asteroid.radius
                    )
                    if isnan(t2_start) or isnan(t2_end):
                        # This should never happen, but is here in case of numeric instability in a barely collision
                        continue

                    # Take the intersection of the intervals
                    # Use nested min/max instead of 3-arg min/max, because this is much faster for MyPyC compilation
                    t_start = max(-collision_past_time_clamp, max(t1_start, t2_start))
                    t_end = min(0.0, min(t1_end, t2_end))

                    if t_start <= t_end:
                        # The interval actually exists
                        collision_time = t_start
                        assert -collision_past_time_clamp <= collision_time <= 0.0
                        collision_event = CollisionEvent(collision_time, 0.0, bul_idx, ast_idx + asteroid_list_idx_offset, CollisionType.BULLET_ASTEROID)
                        i = len(self.collision_queue)
                        while i > 0 and self.collision_queue[i - 1] > collision_event:
                            i -= 1
                        self.collision_queue.insert(i, collision_event)

    def enqueue_mine_asteroid_collisions(self, mines: list[Mine], asteroids: list[Asteroid], asteroid_list_idx_offset: int = 0) -> None:
        for mine_idx, mine in enumerate(mines):
            if mine.detonating:
                for ast_idx, asteroid in enumerate(asteroids):
                    dx = abs(asteroid.x - mine.x)
                    dy = abs(asteroid.y - mine.y)
                    if dx > 0.5 * self.map_width:
                        dx = self.map_width - dx
                    if dy > 0.5 * self.map_height:
                        dy = self.map_height - dy
                    
                    radius_sum = mine.blast_radius + asteroid.radius
                    sq_dist = dx * dx + dy * dy
                    if sq_dist <= radius_sum * radius_sum:
                        collision_time = 0.0
                        collision_event = CollisionEvent(collision_time, sq_dist, mine_idx, ast_idx + asteroid_list_idx_offset, CollisionType.MINE_ASTEROID)
                        i = len(self.collision_queue)
                        while i > 0 and self.collision_queue[i - 1] > collision_event:
                            i -= 1
                        self.collision_queue.insert(i, collision_event)

    def enqueue_mine_ship_collisions(self, mines: list[Mine], ships: list[Ship]) -> None:
        for mine_idx, mine in enumerate(mines):
            if mine.detonating:
                # For each live, non-respawning ship, apply damage only from the closest mine within range
                for ship_idx, ship in enumerate(ships):
                    if ship.is_respawning or not ship.alive:
                        continue
                    dx = abs(ship.x - mine.x)
                    dy = abs(ship.y - mine.y)
                    if dx > 0.5 * self.map_width:
                        dx = self.map_width - dx
                    if dy > 0.5 * self.map_height:
                        dy = self.map_height - dy
                    
                    radius_sum = mine.blast_radius + ship.radius
                    sq_dist = dx * dx + dy * dy
                    if sq_dist <= radius_sum * radius_sum:
                        collision_time = 0.0
                        collision_event = CollisionEvent(collision_time, sq_dist, mine_idx, ship_idx, CollisionType.MINE_SHIP)
                        i = len(self.collision_queue)
                        while i > 0 and self.collision_queue[i - 1] > collision_event:
                            i -= 1
                        self.collision_queue.insert(i, collision_event)

    def enqueue_ship_asteroid_collisions(self, ships: list[Ship], asteroids: list[Asteroid], asteroid_past_time_clamp: float, asteroid_list_idx_offset: int = 0) -> None:
        for ship_idx, ship in enumerate(ships):
            if ship.is_respawning or not ship.alive:
                continue
            for ast_idx, asteroid in enumerate(asteroids):
                # Check for collisions in time interval [t - delta_time, t]
                if asteroid.x - ship.x > 0.5 * self.map_width:
                    ast_x_centered_around_ship = asteroid.x - self.map_width
                elif asteroid.x - ship.x < -0.5 * self.map_width:
                    ast_x_centered_around_ship = asteroid.x + self.map_width
                else:
                    ast_x_centered_around_ship = asteroid.x
                
                if asteroid.y - ship.y > 0.5 * self.map_height:
                    ast_y_centered_around_ship = asteroid.y - self.map_height
                elif asteroid.y - ship.y < -0.5 * self.map_height:
                    ast_y_centered_around_ship = asteroid.y + self.map_height
                else:
                    ast_y_centered_around_ship = asteroid.y
                assert ship._respawning <= 1e-12, f"{ship._respawning=}"
                collision_start_time = ship_asteroid_continuous_collision_time(
                    ship.x, ship.y, ship.radius, ship.speed, ship.integration_initial_states,
                    ast_x_centered_around_ship, ast_y_centered_around_ship, asteroid.vx, asteroid.vy, asteroid.radius, asteroid.speed,
                    max(-min(asteroid_past_time_clamp, self.delta_time), min(0.0, ship._respawning)), 0.0 # Only check collisions starting from when the ship's respawn invincibility wore off
                )
                if not isnan(collision_start_time):
                    assert -self.delta_time <= collision_start_time <= 0.0 # Collision happened within past frame
                    # As a tiebreaker, we need to get the positions of the objects during the collision, which is at offset collision_start_time
                    # This is VERY IMPORTANT because if a ship was respawning and suddenly it wears off while the ship is inside multiple asteroids,
                    # the tiebreaker will be used, and the ship will collide with whatever is closer. Otherwise, framerate-dependent behavior will leak in,
                    # and asteroid order will then decide how it ends up in the queue and subsequently gets resolved.
                    ship_past_x, ship_past_y = ship.get_past_position(collision_start_time, (self.map_width, self.map_height))
                    dx = abs((asteroid.x + asteroid.vx * collision_start_time) - ship_past_x)
                    dy = abs((asteroid.y + asteroid.vy * collision_start_time) - ship_past_y)
                    if dx > 0.5 * self.map_width:
                        dx = self.map_width - dx
                    if dy > 0.5 * self.map_height:
                        dy = self.map_height - dy
                    sq_dist = dx * dx + dy * dy
                    collision_event = CollisionEvent(collision_start_time, sq_dist, ship_idx, ast_idx + asteroid_list_idx_offset, CollisionType.SHIP_ASTEROID)
                    i = len(self.collision_queue)
                    while i > 0 and self.collision_queue[i - 1] > collision_event:
                        i -= 1
                    self.collision_queue.insert(i, collision_event)

    def enqueue_ship_ship_collisions(self, ships: list[Ship]) -> None:
        num_ships = len(ships)
        for ship1_idx, ship1 in enumerate(ships):
            if ship1.alive and not ship1.is_respawning:
                for ship2_idx in range(ship1_idx + 1, num_ships):
                    ship2 = ships[ship2_idx]
                    if ship2.alive and not ship2.is_respawning:
                        # Check for collisions in time interval [t - delta_time, t]
                        # But clamp the start time to when both ships are out of respawn
                        if ship2.x - ship1.x > 0.5 * self.map_width:
                            ship2_x_centered_around_ship1 = ship2.x - self.map_width
                        elif ship2.x - ship1.x < -0.5 * self.map_width:
                            ship2_x_centered_around_ship1 = ship2.x + self.map_width
                        else:
                            ship2_x_centered_around_ship1 = ship2.x

                        if ship2.y - ship1.y > 0.5 * self.map_height:
                            ship2_y_centered_around_ship1 = ship2.y - self.map_height
                        elif ship2.y - ship1.y < -0.5 * self.map_height:
                            ship2_y_centered_around_ship1 = ship2.y + self.map_height
                        else:
                            ship2_y_centered_around_ship1 = ship2.y

                        collision_start_time = ship_ship_continuous_collision_time(
                            ship1.x, ship1.y, ship1.radius, ship1.speed, ship1.integration_initial_states,
                            ship2_x_centered_around_ship1, ship2_y_centered_around_ship1, ship2.radius, ship2.speed, ship2.integration_initial_states,
                            max(-self.delta_time, min(0.0, max(ship1._respawning, ship2._respawning))), 0.0 # Clamp to when ships are out of respawn. Double max/min calls is MUCH faster than calling max/min with 3-4 args, in MyPyC compiled code!
                        )
                        if not isnan(collision_start_time):
                            assert -self.delta_time <= collision_start_time <= 0.0 # Collision happened within past frame
                            # Insert chronologically
                            ship1_past_x, ship1_past_y = ship1.get_past_position(collision_start_time, (self.map_width, self.map_height))
                            ship2_past_x, ship2_past_y = ship2.get_past_position(collision_start_time, (self.map_width, self.map_height))
                            dx = abs(ship1_past_x - ship2_past_x)
                            dy = abs(ship1_past_y - ship2_past_y)
                            if dx > 0.5 * self.map_width:
                                dx = self.map_width - dx
                            if dy > 0.5 * self.map_height:
                                dy = self.map_height - dy
                            sq_dist = dx * dx + dy * dy
                            collision_event = CollisionEvent(collision_start_time, sq_dist, ship1_idx, ship2_idx, CollisionType.SHIP_SHIP)
                            i = len(self.collision_queue)
                            while i > 0 and self.collision_queue[i - 1] > collision_event:
                                i -= 1
                            self.collision_queue.insert(i, collision_event)

    def run(self, scenario: Scenario, controllers: list[KesslerController]) -> tuple[Score, PerfDict]:
        """
        Run an entire scenario from start to finish and return score and stop reason
        """
        ##################
        # INITIALIZATION #
        ##################
        # Initialize objects lists from scenario
        asteroids: list[Asteroid] = scenario.asteroids()
        ships: list[Ship] = scenario.ships() # Keep full list of ships (dead or alive) for score reporting
        liveships: list[Ship] = list(ships) # Maintain a parallel list of just live ships
        bullets: list[Bullet] = []
        mines: list[Mine] = []

        # Initialize Scoring class
        score = Score(scenario)

        # Initialize environment parameters
        stop_reason = StopReason.not_stopped
        sim_time: float = 0.0
        sim_frame: int = 0
        time_limit = scenario.time_limit if scenario.time_limit else self.time_limit
        self.map_width = float(scenario.map_size[0])
        self.map_height = float(scenario.map_size[1])

        # Assign controllers to each ship
        assert len(controllers) >= len(ships), f"There are not enough controllers ({len(controllers)}) to assign to the {len(ships)} ships!"
        for controller, ship in zip(controllers, ships):
            controller.ship_id = ship.id
            ship.controller = controller
            if hasattr(controller, "custom_sprite_path"):
                ship.custom_sprite_path = controller.custom_sprite_path

        # Initialize graphics display
        graphics = GraphicsHandler(type=self.graphics_type, scenario=scenario, UI_settings=self.UI_settings, graphics_obj=self.graphics_obj)

        # Initialize list of dictionary for performance tracking (will remain empty if perf_tracker is false
        perf_dict: PerfDict = {
            'controller_times': [0.0] * len(ships),
            'total_controller_time': 0.0,
            'physics_update': 0.0,
            'collisions_check': 0.0,
            'score_update': 0.0,
            'graphics_draw': 0.0,
            'total_frame_time': 0.0
        }

        ######################
        # MAIN SCENARIO LOOP #
        ######################

        ships_to_cull: list[int] = []
        asteroids_to_cull: list[int] = []
        bullets_to_cull: list[int] = []
        mines_to_cull: list[int] = []
        new_asteroids: list[Asteroid] = []

        # Maintain game_state dict to send to teams
        game_state: GameState | None = None
        if not self.competition_safe_mode:
            game_state = GameState(
                # Game entities
                ships=[ship.state for ship in liveships],
                asteroids=[asteroid.state for asteroid in asteroids],
                bullets=[bullet.state for bullet in bullets],
                mines=[mine.state for mine in mines],
                # Environment
                map_size=scenario.map_size,
                time_limit=time_limit,
                # Simulation timing
                time=sim_time,
                frame=sim_frame,
                delta_time=self.delta_time,
                frame_rate=self.frequency,
                # Game settings
                random_asteroid_splits=self.random_ast_splits,
                competition_safe_mode=self.competition_safe_mode
            )

        while stop_reason == StopReason.not_stopped:
            # Get perf time at the start of time step evaluation and initialize performance tracker
            step_start = time.perf_counter()

            # --- CALL CONTROLLER FOR EACH SHIP ------------------------------------------------------------------------

            # Initialize controller time recording in performance tracker
            if self.perf_tracker:
                t_start = time.perf_counter()

            # Loop through each controller/ship combo and apply their actions
            for ship_idx, ship in enumerate(ships):
                if ship.alive:
                    ship.update_state() # The ship's state might have changed between the last update call and now, if it got hit
                    if controllers[ship_idx].ship_id != ship.id:
                        raise RuntimeError("Controller and ship ID do not match")
                    
                    # Generate game_state info to send to controller
                    game_state_to_controller: GameState
                    if self.competition_safe_mode:
                        # Must recreate GameState object, so competitors do not accidentally or maliciously modify the true game state
                        game_state_to_controller = GameState(
                            # Game entities
                            ships=[ship.state.copy() for ship in liveships],
                            asteroids=[asteroid.state.copy() for asteroid in asteroids],
                            bullets=[bullet.state.copy() for bullet in bullets],
                            mines=[mine.state.copy() for mine in mines],
                            # Environment
                            map_size=scenario.map_size,
                            time_limit=time_limit,
                            # Simulation timing
                            time=sim_time,
                            frame=sim_frame,
                            delta_time=self.delta_time,
                            frame_rate=self.frequency,
                            # Game settings
                            random_asteroid_splits=self.random_ast_splits,
                            competition_safe_mode=self.competition_safe_mode
                        )
                    else:
                        assert game_state is not None
                        game_state_to_controller = game_state
                    
                    # Evaluate each controller letting control be applied
                    thrust, turn_rate, fire, drop_mine = controllers[ship_idx].actions(ShipState(ship.ownstate), game_state_to_controller)

                    assert isinstance(thrust, (int, float)),    f"Controller {ship_idx} thrust is not a number: {thrust!r}"
                    assert isfinite(float(thrust)),             f"Controller {ship_idx} thrust is not finite: {thrust!r}"
                    assert isinstance(turn_rate, (int, float)), f"Controller {ship_idx} turn_rate is not a number: {turn_rate!r}"
                    assert isfinite(float(turn_rate)),          f"Controller {ship_idx} turn_rate is not finite: {turn_rate!r}"
                    assert isinstance(fire, bool),              f"Controller {ship_idx} fire is not bool: {fire!r}"
                    assert isinstance(drop_mine, bool),         f"Controller {ship_idx} drop_mine is not bool: {drop_mine!r}"

                    ship.thrust = float(thrust) # Upcast potential ints to float
                    ship.turn_rate = float(turn_rate)
                    ship.fire = fire
                    ship.drop_mine = drop_mine

                    # Update controller evaluation time if performance tracking
                    if self.perf_tracker:
                        controller_time = time.perf_counter() - t_start if ship.alive else 0.00
                        perf_dict['controller_times'][ship_idx] += controller_time
                        t_start = time.perf_counter()

            if self.perf_tracker:
                perf_dict['total_controller_time'] += time.perf_counter() - step_start
                prev = time.perf_counter()

            # --- UPDATE TIME TO THE TIME AT THE END OF THIS FRAME
            sim_frame += 1
            sim_time = sim_frame / self.frequency # Derive time from integer frames, to avoid accumulated floating point errors
            if not self.competition_safe_mode:
                assert game_state is not None
                game_state.time = sim_time
                game_state.frame = sim_frame
            
            # --- UPDATE STATE INFORMATION OF EACH OBJECT --------------------------------------------------------------
            
            # Update each Asteroid, Bullet, and Ship
            # Because the game_state stores a mutable reference to the internal states of the ship/asteroid/bullet/mine,
            # these updates automatically reflect in the game_state
            for ship in liveships:
                # The ships shoot at the start of the frame
                new_bullet, new_mine = ship.update(self.delta_time, scenario.map_size, True)
                if new_bullet is not None:
                    bullets.append(new_bullet)
                    if not self.competition_safe_mode:
                        assert game_state is not None
                        game_state.add_bullet(new_bullet.state)
                if new_mine is not None:
                    mines.append(new_mine)
                    if not self.competition_safe_mode:
                        assert game_state is not None
                        game_state.add_mine(new_mine.state)
            # The bullet and mine that the ship shot will get updated from the start of the frame to the end
            for asteroid in asteroids:
                asteroid.update(self.delta_time, scenario.map_size)
            for bullet in bullets:
                bullet.update(self.delta_time)
            for mine in mines:
                mine.update(self.delta_time)

            # Update performance tracker
            if self.perf_tracker:
                perf_dict['physics_update'] += time.perf_counter() - prev
                prev = time.perf_counter()

            # --- CHECK FOR COLLISIONS AND ENQUEUE ---

            self.enqueue_bullet_asteroid_collisions(bullets, asteroids, self.delta_time)
            self.enqueue_mine_asteroid_collisions(mines, asteroids)
            self.enqueue_mine_ship_collisions(mines, ships)
            self.enqueue_ship_asteroid_collisions(ships, asteroids, self.delta_time)
            self.enqueue_ship_ship_collisions(ships)

            # --- Resolve collisions in the queue until it is empty ---
            # So earlier we advanced everything to the next frame, but then found when collisions happen.
            # This loop goes through the collision events one-by-one, rewinds both involved objects for each event,
            # and handles them. If new children asteroids get created, we advance those children to the end of the frame
            # and check for collisions, handling them recursively.
            # This way, all chain-reaction collisions get handled, and no events are missed.
            ships_to_cull.clear()
            asteroids_to_cull.clear()
            bullets_to_cull.clear()
            while self.collision_queue:
                event = self.collision_queue.pop(0)
                dt = event.time_offset
                match event.collision_type:
                    case CollisionType.BULLET_ASTEROID:
                        # Rewind the bullet and asteroid to the time of collision, and handle it.
                        # Check new asteroid splits for collisions and add to the queue
                        bul_idx = event.object_a_idx
                        ast_idx = event.object_b_idx

                        if bul_idx in bullets_to_cull or ast_idx in asteroids_to_cull:
                            continue

                        bullet = bullets[bul_idx]
                        asteroid = asteroids[ast_idx]
                        # Rewind
                        bullet.update(dt)
                        asteroid.update(dt, scenario.map_size)
                        # Handle collision
                        bullets_to_cull.append(bul_idx)
                        asteroids_to_cull.append(ast_idx)

                        bullet.owner.bullets_hit += 1
                        bullet.owner.asteroids_hit += 1

                        new_asteroids = asteroid.destruct(impactor=bullet, map_size=scenario.map_size, random_ast_split=self.random_ast_splits)
                        bullet.destruct()
                        for a in new_asteroids:
                            # This is a forward update, from the time of collision to the end of the frame!
                            a.update(-dt, scenario.map_size)
                        ast_idx_offset = len(asteroids)
                        asteroids.extend(new_asteroids)
                        if not self.competition_safe_mode:
                            assert game_state is not None
                            game_state.add_asteroids([a.state for a in new_asteroids])
                        # Take care of possible collision events from these children asteroids this frame
                        self.enqueue_bullet_asteroid_collisions(bullets, new_asteroids, -dt, ast_idx_offset)
                        self.enqueue_mine_asteroid_collisions(mines, new_asteroids, ast_idx_offset)
                        self.enqueue_ship_asteroid_collisions(ships, new_asteroids, -dt, ast_idx_offset)
                    case CollisionType.MINE_ASTEROID:
                        mine_idx = event.object_a_idx
                        ast_idx = event.object_b_idx

                        if ast_idx in asteroids_to_cull:
                            continue

                        mine = mines[mine_idx]
                        asteroid = asteroids[ast_idx]
                        # Rewind
                        # Since dt is 0.0 as mines drop and explode on frame boundaries, we do not need to rollback
                        #mine.update(dt)
                        #asteroid.update(dt)
                        # Handle collision
                        mine.owner.mines_hit += 1
                        mine.owner.asteroids_hit += 1

                        asteroids_to_cull.append(ast_idx)

                        new_asteroids = asteroid.destruct(impactor=mine, map_size=scenario.map_size, random_ast_split=self.random_ast_splits)
                        
                        #for a in new_asteroids:
                            # This is a forward update, from the time of collision to the end of the frame!
                        #    a.update(-dt)
                        asteroids.extend(new_asteroids)
                        if not self.competition_safe_mode:
                            assert game_state is not None
                            game_state.add_asteroids([a.state for a in new_asteroids])
                        # We do NOT enqueue new collisions, because we treat the mine explosions as basically the last thing that can happen
                        # If we enqueued this further, then the same mine would hit the asteroid, along with all of their children!
                        #self.enqueue_bullet_asteroid_collisions(bullets, new_asteroids)
                        #self.enqueue_mine_asteroid_collisions(mines, new_asteroids)
                        #self.enqueue_ship_asteroid_collisions(ships, new_asteroids)
                    case CollisionType.MINE_SHIP:
                        mine_idx = event.object_a_idx
                        ship_idx = event.object_b_idx

                        if ship_idx in ships_to_cull:
                            continue
                        
                        mine = mines[mine_idx]
                        ship = ships[ship_idx]

                        assert ship.alive
                        if ship.is_respawning:
                            continue
                        
                        assert dt == 0.0
                        #if dt != 0.0:
                        #    mine.update(dt)
                        #    ship.update(dt, scenario.map_size, False)

                        ship.destruct(map_size=scenario.map_size)
                        if not ship.alive:
                            ships_to_cull.append(ship_idx)
                        #elif dt != 0.0:
                        #    ship.update(-dt, scenario.map_size, False)
                    case CollisionType.SHIP_ASTEROID:
                        ship_idx = event.object_a_idx
                        ast_idx = event.object_b_idx

                        if ship_idx in ships_to_cull or ast_idx in asteroids_to_cull:
                            continue

                        ship = ships[ship_idx]
                        assert ship.alive
                        if ship.is_respawning:
                            continue
                        asteroid = asteroids[ast_idx]
                        # Rewind
                        assert abs(dt) <= self.delta_time
                        assert dt <= 0.0
                        ship.update(dt, scenario.map_size, False)
                        asteroid.update(dt, scenario.map_size)
                        # Handle collision
                        ship.asteroids_hit += 1

                        new_asteroids = asteroid.destruct(impactor=ship, map_size=scenario.map_size, random_ast_split=self.random_ast_splits)
                        ship.destruct(map_size=scenario.map_size)

                        for a in new_asteroids:
                            # This is a forward update, from the time of collision to the end of the frame!
                            a.update(-dt, scenario.map_size)
                        ast_idx_offset = len(asteroids)
                        asteroids.extend(new_asteroids)
                        if not self.competition_safe_mode:
                            assert game_state is not None
                            game_state.add_asteroids([a.state for a in new_asteroids])
                        self.enqueue_bullet_asteroid_collisions(bullets, new_asteroids, -dt, ast_idx_offset)
                        self.enqueue_mine_asteroid_collisions(mines, new_asteroids, ast_idx_offset)
                        self.enqueue_ship_asteroid_collisions(ships, new_asteroids, -dt, ast_idx_offset)

                        if ship.alive:
                            ship.update(-dt, scenario.map_size, False)
                        else:
                            ships_to_cull.append(ship_idx)
                        asteroids_to_cull.append(ast_idx)
                    case CollisionType.SHIP_SHIP:
                        ship1_idx = event.object_a_idx
                        ship2_idx = event.object_b_idx

                        if ship1_idx in ships_to_cull or ship2_idx in ships_to_cull:
                            continue

                        ship1 = ships[ship1_idx]
                        ship2 = ships[ship2_idx]

                        assert ship1.alive and ship2.alive
                        if ship1.is_respawning or ship2.is_respawning:
                            continue
                        # Rollback
                        ship1.update(dt, scenario.map_size, False)
                        ship2.update(dt, scenario.map_size, False)
                        # Handle collision
                        ship1.destruct(map_size=scenario.map_size)
                        ship2.destruct(map_size=scenario.map_size)
                        # Roll forward to the end of the frame again if alive
                        if ship1.alive:
                            ship1.update(-dt, scenario.map_size, False)
                        else:
                            ships_to_cull.append(ship1_idx)
                        if ship2.alive:
                            ship2.update(-dt, scenario.map_size, False)
                        else:
                            ships_to_cull.append(ship2_idx)

            # Now that all collisions are handled and resolved, the final step is to cull the removed objects
            # TODO: Sort as we go instead of at the end here. Probably faster.
            assert len(asteroids_to_cull) == len(set(asteroids_to_cull))
            for ast_idx in sorted(asteroids_to_cull, reverse=True):
                asteroids[ast_idx] = asteroids[-1]
                asteroids.pop()
                if not self.competition_safe_mode:
                    assert game_state is not None
                    game_state.remove_asteroid(ast_idx)
            
            assert len(bullets_to_cull) == len(set(bullets_to_cull))
            for bul_idx in sorted(bullets_to_cull, reverse=True):
                bullets[bul_idx] = bullets[-1]
                bullets.pop()
                if not self.competition_safe_mode:
                    assert game_state is not None
                    game_state.remove_bullet(bul_idx)
            
            mines_to_cull.clear()
            for mine_idx, mine in enumerate(mines):
                if mine.detonating:
                    mines_to_cull.append(mine_idx)
            mines_to_cull.reverse()
            for mine_idx in mines_to_cull:
                mines[mine_idx] = mines[-1]
                mines.pop()
                if not self.competition_safe_mode:
                    assert game_state is not None
                    game_state.remove_mine(mine_idx)

            # Cull ships if they are all out of lives
            # We don't cull a ship just because it took damage this frame! They may still have more lives.
            # TODO: Swap and pop this just like mines
            new_liveships = [ship for ship in liveships if ship.alive]
            if ships_to_cull:
                liveships = new_liveships
                if not self.competition_safe_mode:
                    assert game_state is not None
                    game_state.update_ships([ship.state for ship in liveships])

            # Update performance tracker with collisions timing
            if self.perf_tracker:
                perf_dict['collisions_check'] += time.perf_counter() - prev
                prev = time.perf_counter()

                # --- UPDATE SCORE CLASS -----------------------------------------------------------------------------------
                score.update(ships, sim_time, perf_dict['controller_times'])

                # Update performance tracker with score timing
                perf_dict['score_update'] += time.perf_counter() - prev
                prev = time.perf_counter()
            else:
                score.update(ships, sim_time)


            # --- UPDATE GRAPHICS --------------------------------------------------------------------------------------
            if sim_frame % self.frame_skip == 0:
                graphics.update(score, ships, asteroids, bullets, mines)

                # Update performance tracker with graphics timing
                if self.perf_tracker:
                    perf_dict['graphics_draw'] += time.perf_counter() - prev
                    prev = time.perf_counter()
            
            # --- CHECK STOP CONDITIONS --------------------------------------------------------------------------------
            if not asteroids:
                # No asteroids remain
                stop_reason = StopReason.no_asteroids
            elif not liveships and not (len(mines) > 0 or len(bullets) > 0):
                # No ships are alive and no mines exist and no bullets exist
                # Prevents unfairness where ship that dies before another gets score from its bullets as long as the other
                # is alive but the one that lives longer doesn't get the same benefit from its bullets/mines persisting
                # after it dies
                stop_reason = StopReason.no_ships
            elif not sum([ship.bullets_remaining for ship in liveships]) > 0 \
                    and not sum([ship.mines_remaining for ship in liveships])\
                    and not (len(bullets) > 0 or len(mines) > 0) \
                    and scenario.stop_if_no_ammo:
                # All live ships are out of bullets and no bullets are on map
                stop_reason = StopReason.out_of_bullets
            elif sim_frame >= ceil(time_limit * self.frequency):
                # Out of time
                stop_reason = StopReason.time_expired

            # --- FINISHING TIME STEP ----------------------------------------------------------------------------------
            # Get overall time step compute time
            if self.perf_tracker:
                perf_dict['total_frame_time'] += time.perf_counter() - step_start

            # Hold simulation so that it runs at realtime ratio if specified, else let it pass
            if self.realtime_multiplier != 0.0:
                time_dif = time.perf_counter() - step_start
                while time_dif < self.delta_time / self.realtime_multiplier:
                    time_dif = time.perf_counter() - step_start

        ############################################
        # Finalization after scenario has been run #
        ############################################

        # Close graphics display
        graphics.close()

        # Finalize score class before returning
        score.finalize(sim_time, stop_reason, ships)

        # Return the score and stop condition
        return score, perf_dict


class TrainerEnvironment(KesslerGame):
    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        """
        Instantiates a KesslerGame object with settings to optimize training time
        """
        if settings is None:
            settings = {}
        trainer_settings: SettingsDict = {
            'frequency': settings.get("frequency", 30.0),
            'perf_tracker': settings.get("perf_tracker", False),
            'prints_on': settings.get("prints_on", False),
            'graphics_type': GraphicsType.NoGraphics,
            'realtime_multiplier': 0.0,
            'time_limit': settings.get("time_limit", inf)
        }
        super().__init__(trainer_settings)
