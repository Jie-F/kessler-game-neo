# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from __future__ import annotations

import time
import warnings

from math import inf, nan, isfinite, isnan, ceil, sqrt, radians, sin, cos
from typing import Any, TypedDict, cast, ClassVar
from enum import Enum, IntEnum

from .scenario import Scenario
from .score import Score
from .controller import KesslerController
from .collisions import circle_line_collision_continuous, collision_time_interval, ship_asteroid_continuous_collision_time, ship_ship_continuous_collision_time, time_until_exit, time_until_enter
from .graphics import GraphicsType, GraphicsHandler, KesslerGraphics
from .mines import Mine
from .asteroid import Asteroid
from .ship import Ship
from .bullet import Bullet
from .settings_dicts import SettingsDict, UISettingsDict
from .state_models import GameState, ShipState
from .heapq_mypyc import heappush, heappop, heapify, heapreplace, merge, nlargest, nsmallest, heappushpop


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
    SHIP_ASTEROID = 1
    SHIP_SHIP = 2
    MINE_ASTEROID = 3
    MINE_SHIP = 4


class CollisionEvent:
    __slots__ = ("time_offset", "distance", "object_a_idx", "object_b_idx", "collision_type", "collision_specific_final_tiebreaker", "insertion_order")

    TOLERANCE: ClassVar[float] = 1e-10
    _counter: ClassVar[int] = 0  # class-level monotonic counter

    def __init__(self, time_offset: float, distance: float, object_a_idx: int, object_b_idx: int, collision_type: CollisionType, specific_tiebreaker: float = 0.0):
        """
        Represents a collision event between two game objects at a specific time.

        :param time_offset: Time (in seconds) relative to the end of the frame. Must be in [-dt, 0.0].
        :param distance: Squared wrapped distance between the colliding objects. Used to break ties
        :param object_a_idx: Index of first object involved in the collision.
        :param object_b_idx: Index of second object involved in the collision.
        :param collision_type: Type of collision as defined in CollisionType enum.
        """
        self.time_offset: float = time_offset # Time offset in seconds relative to frame end, e.g., -0.001
        self.distance: float = distance # Distance between centers of colliding objects. Squared!
        # Time offset should be in range [-delta_time, 0.0]
        self.object_a_idx = object_a_idx
        self.object_b_idx = object_b_idx
        self.collision_type = collision_type
        self.collision_specific_final_tiebreaker: float = specific_tiebreaker
        # The sort order is: time offset, collision type, distance, collision specific final tiebreaker, insertion order

        # Assign and increment insertion_order
        self.insertion_order = CollisionEvent._counter
        CollisionEvent._counter += 1

    # Hand-implement each of the comparison dunders for speed, instead of using one for the others

    # Tolerant float comparison helpers
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
        if self._greater(self.time_offset, other.time_offset):
            return False

        if self.collision_type < other.collision_type:
            return True
        if self.collision_type > other.collision_type:
            return False

        if self._less(self.distance, other.distance):
            return True
        if self._greater(self.distance, other.distance):
            return False

        if self._less(self.collision_specific_final_tiebreaker, other.collision_specific_final_tiebreaker):
            return True
        if self._greater(self.collision_specific_final_tiebreaker, other.collision_specific_final_tiebreaker):
            return False

        # Final tiebreaker: insertion order
        return self.insertion_order < other.insertion_order

    def __le__(self, other: CollisionEvent) -> bool:
        if self._less(self.time_offset, other.time_offset):
            return True
        if self._greater(self.time_offset, other.time_offset):
            return False

        if self.collision_type < other.collision_type:
            return True
        if self.collision_type > other.collision_type:
            return False

        if self._less(self.distance, other.distance):
            return True
        if self._greater(self.distance, other.distance):
            return False

        if self._less(self.collision_specific_final_tiebreaker, other.collision_specific_final_tiebreaker):
            return True
        if self._greater(self.collision_specific_final_tiebreaker, other.collision_specific_final_tiebreaker):
            return False

        return self.insertion_order <= other.insertion_order

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CollisionEvent):
            return NotImplemented
        return (
            self._equal(self.time_offset, other.time_offset)
            and self.collision_type == other.collision_type
            and self._equal(self.distance, other.distance)
            and self._equal(self.collision_specific_final_tiebreaker, other.collision_specific_final_tiebreaker)
            and self.insertion_order == other.insertion_order
        )

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, CollisionEvent):
            return NotImplemented
        return (
            not self._equal(self.time_offset, other.time_offset)
            or self.collision_type != other.collision_type
            or not self._equal(self.distance, other.distance)
            or not self._equal(self.collision_specific_final_tiebreaker, other.collision_specific_final_tiebreaker)
            or self.insertion_order != other.insertion_order
        )

    def __gt__(self, other: CollisionEvent) -> bool:
        if self._greater(self.time_offset, other.time_offset):
            return True
        if self._less(self.time_offset, other.time_offset):
            return False

        if self.collision_type > other.collision_type:
            return True
        if self.collision_type < other.collision_type:
            return False

        if self._greater(self.distance, other.distance):
            return True
        if self._less(self.distance, other.distance):
            return False

        if self._greater(self.collision_specific_final_tiebreaker, other.collision_specific_final_tiebreaker):
            return True
        if self._less(self.collision_specific_final_tiebreaker, other.collision_specific_final_tiebreaker):
            return False

        return self.insertion_order > other.insertion_order

    def __ge__(self, other: CollisionEvent) -> bool:
        if self._greater(self.time_offset, other.time_offset):
            return True
        if self._less(self.time_offset, other.time_offset):
            return False

        if self.collision_type > other.collision_type:
            return True
        if self.collision_type < other.collision_type:
            return False

        if self._greater(self.distance, other.distance):
            return True
        if self._less(self.distance, other.distance):
            return False

        if self._greater(self.collision_specific_final_tiebreaker, other.collision_specific_final_tiebreaker):
            return True
        if self._less(self.collision_specific_final_tiebreaker, other.collision_specific_final_tiebreaker):
            return False

        return self.insertion_order >= other.insertion_order

    def __repr__(self) -> str:
        return (
            f"<CollisionEvent time_offset={self.time_offset}s distance={self.distance} "
            f"type={self.collision_type} obj_a_idx={self.object_a_idx} obj_b_idx={self.object_b_idx} "
            f"collision_specific_final_tiebreaker={self.collision_specific_final_tiebreaker} "
            f"insertion_order={self.insertion_order}>"
        )


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

    def enqueue_bullet_asteroid_collisions(self, bullets: list[Bullet], asteroids: list[Asteroid], asteroid_past_time_clamp: float, asteroid_list_idx_offset: int = 0, already_a_heap: bool = False) -> None:
        # Collect all potential bullet-asteroid collisions
        # Since bullets do not wrap, we treat the bullet hitbox as being clamped at the map edge.
        # The way to calculate the collision time interval is elegant. It considers two virtual bullets,
        # the intersection of the two being the true bullet's position:
        # 1. The bullet that passes right through without getting clamped
        # 2. The bullet that is stationary, with its head right on the border, and the tail sticking into the map
        # The intersection of these two form the clamped bullet hitbox we're interested in, during the time
        # interval from when the bullet head first hits the edge, until the tail also leaves. This construction
        # is invalid before this time interval!

        # We might not be able to go back a full delta_time if the asteroid wasn't alive for that long yet!
        # So we clamp that time
        collision_past_time_clamp = min(asteroid_past_time_clamp, self.delta_time)

        for bul_idx, bullet in enumerate(bullets):
            # Find unclamped bullet path (as if bullet passes through the map edge)
            bullet_head_x = bullet.x
            bullet_head_y = bullet.y
            bullet_tail_x = bullet.x + bullet.tail_delta_x
            bullet_tail_y = bullet.y + bullet.tail_delta_y

            # Find when head and tail leave the visible map
            t_head_exit = time_until_exit(bullet_head_x, bullet_head_y, bullet.vx, bullet.vy, self.map_width, self.map_height)
            t_tail_exit = time_until_exit(bullet_tail_x, bullet_tail_y, bullet.vx, bullet.vy, self.map_width, self.map_height)
            assert t_head_exit <= t_tail_exit
            # Find when head and tail enter the visible map
            t_head_enter = time_until_enter(bullet_head_x, bullet_head_y, bullet.vx, bullet.vy, self.map_width, self.map_height)
            t_tail_enter = time_until_enter(bullet_tail_x, bullet_tail_y, bullet.vx, bullet.vy, self.map_width, self.map_height)
            assert t_head_enter <= t_tail_enter

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

                if t_head_exit >= 0.0 and t_tail_enter <= -collision_past_time_clamp:
                    # This means that for the whole duration we're checking, the bullet is entirely within the map's visible bounds!
                    # The tail has entered the map before the interval starts, and the head will not leave until after the interval ends.
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
                            warnings.warn("Numeric instability in quadratic solver? Real bullet collision time is NaN", RuntimeWarning)
                            continue
                        collision_time = max(-collision_past_time_clamp, collision_start_time)
                        assert -collision_past_time_clamp <= collision_time <= 0.0

                        # Calculate the distance between center of bullet and the asteroid at the time of collision, to use as a tiebreaker in ordering events
                        bul_x_mid_collision = 0.5 * (bullet_head_x + bullet_tail_x) + collision_time * bullet.vx
                        bul_y_mid_collision = 0.5 * (bullet_head_y + bullet_tail_y) + collision_time * bullet.vy
                        ast_x_collision = ast_x_centered + collision_time * asteroid.vx
                        ast_y_collision = ast_y_centered + collision_time * asteroid.vy
                        dx = ast_x_collision - bul_x_mid_collision
                        dy = ast_y_collision - bul_y_mid_collision
                        sq_dist = dx * dx + dy * dy

                        bul_head_x_collision = bullet_head_x + collision_time * bullet.vx
                        bul_head_y_collision = bullet_head_y + collision_time * bullet.vy
                        bul_tail_x_collision = bullet_tail_x + collision_time * bullet.vx
                        bul_tail_y_collision = bullet_tail_y + collision_time * bullet.vy
                        # Either the bullet head or tail should be inside the map bounds for this collision to be valid.
                        # This should be guaranteed, because at no point during this interval was the bullet expected to leave the map bound, and need to be clamped!
                        assert (((0.0 <= bul_head_x_collision <= self.map_width) and (0.0 <= bul_head_y_collision <= self.map_height))
                                or ((0.0 <= bul_tail_x_collision <= self.map_width) and (0.0 <= bul_tail_y_collision <= self.map_height)))

                        # It happens surprisingly frequently where an asteroid splits, and the three overlapping children asteroids get hit by a bullet. We need a tiebreaker for this situation!
                        # Or else we get weird random indeterminate behavior, and there goes our framerate independence.
                        dot_bullet_vel_ast_vel_tiebreaker = bullet.vx * asteroid.vx + bullet.vy * asteroid.vy
                        collision_event = CollisionEvent(collision_time, sq_dist, bul_idx, ast_idx + asteroid_list_idx_offset, CollisionType.BULLET_ASTEROID, dot_bullet_vel_ast_vel_tiebreaker)
                        if already_a_heap:
                            heappush(self.collision_queue, collision_event)
                        else:
                            self.collision_queue.append(collision_event)
                else:
                    # During the time interval we're checking, either the whole time or a part of the time,
                    # the bullet is at least partially out of bounds and has to have its hitbox clipped at the edge.
                    # To do this check, we create two virtual bullets! The first virtual bullet is just the regular bullet, but it's
                    # allowed to go out of bounds.
                    # The second virtual bullet is a stationary one, with the head at the map border where the bullet leaves, and the tail
                    # also on the border where the bullet would first enter the map
                    # If we find the collision time interval between the asteroid and these two virtual bullets, and take
                    # their intersections, we'll have the true collision interval of the clamped bullet.
                    
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
                        warnings.warn("Numeric instability in quadratic solver? VB1 collision time is NaN", RuntimeWarning)
                        continue

                    # Virtual bullet 2: stationary bullet, where head and tails are pinned at the map border, along the bullet's line of travel
                    # Remember that the t_clamp_start is a negative number. The bullet head at the end of the frame is already past bound.
                    pinned_head_x = bullet_head_x + bullet.vx * t_head_exit
                    pinned_head_y = bullet_head_y + bullet.vy * t_head_exit
                    # Stick the tail of the bullet on the map border where the bullet would enter the map
                    pinned_tail_x = bullet_tail_x + bullet.vx * t_tail_enter
                    pinned_tail_y = bullet_tail_y + bullet.vy * t_tail_enter

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
                        warnings.warn("Numeric instability in quadratic solver? VB2 collision time is NaN", RuntimeWarning)
                        continue

                    # Take the intersection of the intervals
                    # Use nested min/max instead of 3-arg min/max, because this is much faster for MyPyC compilation
                    t_start = max(max(max(t1_start, t2_start), -collision_past_time_clamp), t_head_enter)
                    t_end = min(min(min(t1_end, t2_end), 0.0), t_tail_exit)

                    if t_start <= t_end:
                        # The interval actually exists
                        collision_time = t_start
                        assert -collision_past_time_clamp <= collision_time <= 0.0

                        # Calculate the distance between center of bullet and the asteroid at the time of collision, to use as a tiebreaker in ordering events
                        # This does not consider that the bullet could be clamped at the edge, and just uses the bullet passing through the border anyway.
                        bul_x_mid_collision = 0.5 * (bullet_head_x + bullet_tail_x) + collision_time * bullet.vx
                        bul_y_mid_collision = 0.5 * (bullet_head_y + bullet_tail_y) + collision_time * bullet.vy
                        ast_x_collision = ast_x_centered + collision_time * asteroid.vx
                        ast_y_collision = ast_y_centered + collision_time * asteroid.vy
                        dx = ast_x_collision - bul_x_mid_collision
                        dy = ast_y_collision - bul_y_mid_collision
                        sq_dist = dx * dx + dy * dy

                        bul_head_x_collision = bullet_head_x + collision_time * bullet.vx
                        bul_head_y_collision = bullet_head_y + collision_time * bullet.vy
                        bul_tail_x_collision = bullet_tail_x + collision_time * bullet.vx
                        bul_tail_y_collision = bullet_tail_y + collision_time * bullet.vy

                        # It happens surprisingly frequently where an asteroid splits, and the three overlapping children asteroids get hit by a bullet. We need a tiebreaker for this situation!
                        # Or else we get weird random indeterminate behavior, and there goes our framerate independence.
                        dot_bullet_vel_ast_vel_tiebreaker = bullet.vx * asteroid.vx + bullet.vy * asteroid.vy
                        collision_event = CollisionEvent(collision_time, sq_dist, bul_idx, ast_idx + asteroid_list_idx_offset, CollisionType.BULLET_ASTEROID, dot_bullet_vel_ast_vel_tiebreaker)
                        if already_a_heap:
                            heappush(self.collision_queue, collision_event)
                        else:
                            self.collision_queue.append(collision_event)

    def enqueue_mine_asteroid_collisions(self, mines: list[Mine], asteroids: list[Asteroid], asteroid_list_idx_offset: int = 0, already_a_heap: bool = False) -> None:
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
                        if already_a_heap:
                            heappush(self.collision_queue, collision_event)
                        else:
                            self.collision_queue.append(collision_event)

    def enqueue_mine_ship_collisions(self, mines: list[Mine], ships: list[Ship]) -> None:
        for mine_idx, mine in enumerate(mines):
            if mine.detonating:
                # For each live, non-respawning ship, apply damage only from the closest mine within range
                for ship_idx, ship in enumerate(ships):
                    if ship.is_respawning_internal or not ship.alive:
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
                        self.collision_queue.append(collision_event)

    def enqueue_ship_asteroid_collisions(self, ships: list[Ship], asteroids: list[Asteroid], asteroid_past_time_clamp: float, asteroid_list_idx_offset: int = 0, already_a_heap: bool = False) -> None:
        for ship_idx, ship in enumerate(ships):
            if ship.is_respawning_internal or not ship.alive:
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
                assert ship.respawn_time_internal <= 1e-12, f"{ship.respawn_time_internal=}"
                collision_start_time = ship_asteroid_continuous_collision_time(
                    ship.x, ship.y, ship.radius, ship.speed, ship.integration_initial_states,
                    ast_x_centered_around_ship, ast_y_centered_around_ship, asteroid.vx, asteroid.vy, asteroid.radius, asteroid.speed,
                    max(-min(asteroid_past_time_clamp, self.delta_time), min(0.0, ship.respawn_time_internal)), 0.0 # Only check collisions starting from when the ship's respawn invincibility wore off
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

                    # Break ties in case everything else matches, including asteroid distance from the ship
                    ship_heading_collision = radians(ship.heading + ship.turn_rate * collision_start_time)
                    dot_product_tiebreaker_ship_ast = cos(ship_heading_collision) * asteroid.vx + sin(ship_heading_collision) * asteroid.vy
                    collision_event = CollisionEvent(collision_start_time, sq_dist, ship_idx, ast_idx + asteroid_list_idx_offset, CollisionType.SHIP_ASTEROID, dot_product_tiebreaker_ship_ast)
                    if already_a_heap:
                        heappush(self.collision_queue, collision_event)
                    else:
                        self.collision_queue.append(collision_event)

    def enqueue_ship_ship_collisions(self, ships: list[Ship]) -> None:
        num_ships = len(ships)
        for ship1_idx, ship1 in enumerate(ships):
            if ship1.alive and not ship1.is_respawning_internal:
                for ship2_idx in range(ship1_idx + 1, num_ships):
                    ship2 = ships[ship2_idx]
                    if ship2.alive and not ship2.is_respawning_internal:
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
                        
                        collision_check_interval_start = max(-self.delta_time, min(0.0, max(ship1.respawn_time_internal, ship2.respawn_time_internal)))
                        collision_start_time = ship_ship_continuous_collision_time(
                            ship1.x, ship1.y, ship1.radius, ship1.speed, ship1.integration_initial_states,
                            ship2_x_centered_around_ship1, ship2_y_centered_around_ship1, ship2.radius, ship2.speed, ship2.integration_initial_states,
                            collision_check_interval_start, 0.0 # Clamp to when ships are out of respawn. Double max/min calls is MUCH faster than calling max/min with 3-4 args, in MyPyC compiled code!
                        )
                        if not isnan(collision_start_time):
                            assert -self.delta_time <= collision_start_time <= 0.0 # Collision happened within past frame
                            ship1_past_x, ship1_past_y = ship1.get_past_position(collision_start_time, (self.map_width, self.map_height))
                            ship2_past_x, ship2_past_y = ship2.get_past_position(collision_start_time, (self.map_width, self.map_height))
                            dx = abs(ship1_past_x - ship2_past_x)
                            dy = abs(ship1_past_y - ship2_past_y)
                            if dx > 0.5 * self.map_width:
                                dx = self.map_width - dx
                            if dy > 0.5 * self.map_height:
                                dy = self.map_height - dy
                            sq_dist = dx * dx + dy * dy
                            
                            # This following test is to make sure that the ships end up in a definitely colliding state, and
                            # is good for ensuring framerate independence. So that at other framerates, the ships don't end up not actually colliding
                            # The root finder is meant to find the time that the ships begin colliding, but due to imprecision, it might find the time right before they start colliding.
                            radii_sum = ship1.radius + ship2.radius
                            radii_sum_sq = radii_sum * radii_sum
                            verified_collision: bool = True
                            if not (sq_dist + 1e-10 <= radii_sum_sq):
                                verified_collision = False
                                # Ships aren't colliding. Nudge the time forward and hope that makes them collide.
                                # Add an eps to REALLY make sure the ships are colliding!
                                # But we only want to do this if the root was found in the middle of the interval when the function
                                # dips down. If the interval start is negative, then we do NOT nudge this forward, or else wacky stuff
                                # will happen, and we get edge cases where the ship on the edge of respawn will have framerate dependence due
                                # to floating point error. Especially in mine-ship collisions which happen on integer seconds.
                                nudgification_factor = 1e-12
                                for i in range(1000):
                                    # Instead of using a while loop, it's better to use a for loop and have an upper bound on this,
                                    # just so we don't infinite loop here in some super weird case
                                    collision_start_time += nudgification_factor
                                    if collision_start_time > 0.0:
                                        # We reached the end of the interval, so this is hopeless. These ships are not colliding!
                                        break

                                    ship1_past_x, ship1_past_y = ship1.get_past_position(collision_start_time, (self.map_width, self.map_height))
                                    ship2_past_x, ship2_past_y = ship2.get_past_position(collision_start_time, (self.map_width, self.map_height))
                                    dx = abs(ship1_past_x - ship2_past_x)
                                    dy = abs(ship1_past_y - ship2_past_y)
                                    if dx > 0.5 * self.map_width:
                                        dx = self.map_width - dx
                                    if dy > 0.5 * self.map_height:
                                        dy = self.map_height - dy
                                    sq_dist = dx * dx + dy * dy
                                    if sq_dist + 1e-10 <= radii_sum_sq:
                                        verified_collision = True
                                        break
                                    nudgification_factor *= 2.0 # Exponentially increase the nudge

                            if verified_collision:
                                collision_event = CollisionEvent(collision_start_time, sq_dist, ship1_idx, ship2_idx, CollisionType.SHIP_SHIP)
                                self.collision_queue.append(collision_event)

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
                if not ship.alive:
                    continue

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
                
                # Default null action
                thrust, turn_rate, fire, drop_mine = 0.0, 0.0, False, False

                try:
                    # Attempt to get and validate controller actions
                    proposed = controllers[ship_idx].actions(ShipState(ship.ownstate), game_state_to_controller)

                    if not isinstance(proposed, (list, tuple)) or len(proposed) != 4:
                        raise ValueError(f"Controller {ship_idx} returned invalid action tuple: {proposed!r}")

                    raw_thrust, raw_turn_rate, raw_fire, raw_drop_mine = proposed

                    if not isinstance(raw_thrust, (int, float)) or not isfinite(float(raw_thrust)):
                        raise ValueError(f"Controller {ship_idx} thrust invalid: {raw_thrust!r}")
                    if not isinstance(raw_turn_rate, (int, float)) or not isfinite(float(raw_turn_rate)):
                        raise ValueError(f"Controller {ship_idx} turn_rate invalid: {raw_turn_rate!r}")
                    if not isinstance(raw_fire, bool):
                        raise TypeError(f"Controller {ship_idx} fire is not bool: {raw_fire!r}")
                    if not isinstance(raw_drop_mine, bool):
                        raise TypeError(f"Controller {ship_idx} drop_mine is not bool: {raw_drop_mine!r}")

                    # Only update if all checks passed
                    thrust = float(raw_thrust) # Upcast potential ints to float
                    turn_rate = float(raw_turn_rate) # Upcast potential ints to float
                    fire = raw_fire
                    drop_mine = raw_drop_mine
                except Exception as e:
                    if not self.competition_safe_mode:
                        raise  # In dev mode, fail loudly
                    # Log the error if needed
                    print(f"[Competition Safe Mode] Controller {ship_idx} error: {e!r}. Assigning null actions for frame {sim_frame}.")

                ship.thrust = thrust
                ship.turn_rate = turn_rate
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
                asteroid.update(self.delta_time, self.map_width, self.map_height)
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

            heapify(self.collision_queue) # Create priority queue in O(n)

            # --- Resolve collisions in the queue until it is empty ---
            # So earlier we advanced everything to the next frame, but then found when collisions happen.
            # This loop goes through the collision events one-by-one, rewinds both involved objects for each event,
            # and handles them. If new children asteroids get created, we advance those children to the end of the frame
            # and check for collisions, handling them recursively.
            # This way, all chain-reaction collisions get handled, and no events are missed.
            ships_to_cull.clear()
            asteroids_to_cull.clear()
            bullets_to_cull.clear()
            last_time_offset: float = -self.delta_time
            while self.collision_queue:
                event = heappop(self.collision_queue)
                dt = event.time_offset
                assert dt + 1e-12 >= last_time_offset, f"The collision events are not monotonic! Last offset={last_time_offset}, current offset={dt}"
                last_time_offset = dt
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
                        if abs(dt) > CollisionEvent.TOLERANCE:
                            bullet.update(dt)
                            asteroid.update(dt, self.map_width, self.map_height)
                        # Handle collision
                        bullets_to_cull.append(bul_idx)
                        asteroids_to_cull.append(ast_idx)

                        bullet.owner.bullets_hit += 1
                        bullet.owner.asteroids_hit += 1

                        new_asteroids = asteroid.destruct(impactor=bullet, map_width=self.map_width, map_height=self.map_height, random_ast_split=self.random_ast_splits)
                        bullet.destruct()
                        if abs(dt) > CollisionEvent.TOLERANCE:
                            for a in new_asteroids:
                                # This is a forward update, from the time of collision to the end of the frame!
                                a.update(-dt, self.map_width, self.map_height)
                        ast_idx_offset = len(asteroids)
                        asteroids.extend(new_asteroids)
                        if not self.competition_safe_mode:
                            assert game_state is not None
                            game_state.add_asteroids([a.state for a in new_asteroids])
                        # Take care of possible collision events from these children asteroids this frame
                        if abs(dt) > CollisionEvent.TOLERANCE:
                            # Only do this if we have time left, and the collision didn't happen at the very end of the frame
                            self.enqueue_bullet_asteroid_collisions(bullets, new_asteroids, -dt, ast_idx_offset, True)
                            self.enqueue_mine_asteroid_collisions(mines, new_asteroids, ast_idx_offset, True)
                            self.enqueue_ship_asteroid_collisions(ships, new_asteroids, -dt, ast_idx_offset, True)
                    case CollisionType.MINE_ASTEROID:
                        mine_idx = event.object_a_idx
                        ast_idx = event.object_b_idx

                        if ast_idx in asteroids_to_cull:
                            continue

                        mine = mines[mine_idx]
                        asteroid = asteroids[ast_idx]
                        # Rewind
                        # Since dt is 0.0 as mines drop and explode on frame boundaries, we do not need to rollback
                        assert dt == 0.0
                        if abs(dt) > CollisionEvent.TOLERANCE:
                            mine.update(dt)
                            asteroid.update(dt, self.map_width, self.map_height)
                        # Handle collision
                        mine.owner.mines_hit += 1
                        mine.owner.asteroids_hit += 1

                        asteroids_to_cull.append(ast_idx)

                        new_asteroids = asteroid.destruct(impactor=mine, map_width=self.map_width, map_height=self.map_height, random_ast_split=self.random_ast_splits)
                        
                        if abs(dt) > CollisionEvent.TOLERANCE:
                            for a in new_asteroids:
                                # This is a forward update, from the time of collision to the end of the frame!
                                a.update(-dt, self.map_width, self.map_height)
                        ast_idx_offset = len(asteroids)
                        asteroids.extend(new_asteroids)
                        if not self.competition_safe_mode:
                            assert game_state is not None
                            game_state.add_asteroids([a.state for a in new_asteroids])
                        # We do NOT enqueue new collisions, because we treat the mine explosions as basically the last thing that can happen
                        # If we enqueued this further, then the same mine would hit the asteroid, along with all of their children!
                        if abs(dt) > CollisionEvent.TOLERANCE:
                            self.enqueue_bullet_asteroid_collisions(bullets, new_asteroids, -dt, ast_idx_offset, True)
                            self.enqueue_mine_asteroid_collisions(mines, new_asteroids, ast_idx_offset, True)
                            self.enqueue_ship_asteroid_collisions(ships, new_asteroids, -dt, ast_idx_offset, True)
                    case CollisionType.MINE_SHIP:
                        mine_idx = event.object_a_idx
                        ship_idx = event.object_b_idx

                        if ship_idx in ships_to_cull:
                            continue
                        
                        mine = mines[mine_idx]
                        ship = ships[ship_idx]

                        assert ship.alive
                        if ship.is_respawning_internal:
                            continue
                        
                        assert dt == 0.0
                        if abs(dt) > CollisionEvent.TOLERANCE:
                            mine.update(dt)
                            ship.update(dt, scenario.map_size, False)

                        ship.destruct(map_size=scenario.map_size)
                        if not ship.alive:
                            ships_to_cull.append(ship_idx)
                        elif abs(dt) > CollisionEvent.TOLERANCE:
                            ship.update(-dt, scenario.map_size, False)
                    case CollisionType.SHIP_ASTEROID:
                        ship_idx = event.object_a_idx
                        ast_idx = event.object_b_idx

                        if ship_idx in ships_to_cull or ast_idx in asteroids_to_cull:
                            continue

                        ship = ships[ship_idx]
                        assert ship.alive
                        if ship.is_respawning_internal:
                            continue
                        asteroid = asteroids[ast_idx]
                        # Rewind
                        assert abs(dt) <= self.delta_time
                        assert dt <= 0.0
                        if abs(dt) > CollisionEvent.TOLERANCE:
                            ship.update(dt, scenario.map_size, False)
                            asteroid.update(dt, self.map_width, self.map_height)
                        # Handle collision
                        ship.asteroids_hit += 1

                        new_asteroids = asteroid.destruct(impactor=ship, map_width=self.map_width, map_height=self.map_height, random_ast_split=self.random_ast_splits)
                        ship.destruct(map_size=scenario.map_size)

                        if abs(dt) > CollisionEvent.TOLERANCE:
                            for a in new_asteroids:
                                # This is a forward update, from the time of collision to the end of the frame!
                                a.update(-dt, self.map_width, self.map_height)
                        ast_idx_offset = len(asteroids)
                        asteroids.extend(new_asteroids)
                        if not self.competition_safe_mode:
                            assert game_state is not None
                            game_state.add_asteroids([a.state for a in new_asteroids])
                        if abs(dt) > CollisionEvent.TOLERANCE:
                            self.enqueue_bullet_asteroid_collisions(bullets, new_asteroids, -dt, ast_idx_offset, True)
                            self.enqueue_mine_asteroid_collisions(mines, new_asteroids, ast_idx_offset, True)
                            self.enqueue_ship_asteroid_collisions(ships, new_asteroids, -dt, ast_idx_offset, True)

                        if ship.alive:
                            if abs(dt) > CollisionEvent.TOLERANCE:
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
                        if ship1.is_respawning_internal or ship2.is_respawning_internal:
                            continue
                        # Rollback
                        if abs(dt) > CollisionEvent.TOLERANCE:
                            ship1.update(dt, scenario.map_size, False)
                            ship2.update(dt, scenario.map_size, False)
                        # Handle collision
                        ship1.destruct(map_size=scenario.map_size)
                        ship2.destruct(map_size=scenario.map_size)
                        # Roll forward to the end of the frame again if alive
                        if ship1.alive:
                            if abs(dt) > CollisionEvent.TOLERANCE:
                                ship1.update(-dt, scenario.map_size, False)
                        else:
                            ships_to_cull.append(ship1_idx)
                        if ship2.alive:
                            if abs(dt) > CollisionEvent.TOLERANCE:
                                ship2.update(-dt, scenario.map_size, False)
                        else:
                            ships_to_cull.append(ship2_idx)

            # Now that all collisions are handled and resolved, the final step is to cull the removed objects
            assert len(asteroids_to_cull) == len(set(asteroids_to_cull))
            for ast_idx in sorted(asteroids_to_cull, reverse=True):
                asteroids[ast_idx] = asteroids[-1]
                asteroids.pop()
                if not self.competition_safe_mode:
                    assert game_state is not None
                    game_state.remove_asteroid(ast_idx)
            
            assert len(bullets_to_cull) == len(set(bullets_to_cull))

            # Cull bullets that are off the map
            # It might be tempting to cull a bullet if both the head and tail are out of bounds. And this was the original logic.
            # But this misses something. What if the tail and head were out of bounds, but the middle of the bullet was still inbounds in the corner of a map?
            # This is something that is geometrically plausible especially at higher framerates, and the ship is peeking its head into the map when shooting a bullet!
            for bul_idx, bullet in enumerate(bullets):
                if bul_idx in bullets_to_cull:
                    continue
                time_for_tail_to_leave = time_until_exit(bullet.x + bullet.tail_delta_x, bullet.y + bullet.tail_delta_y, bullet.vx, bullet.vy, self.map_width, self.map_height)
                if time_for_tail_to_leave <= 0.0:
                    # The bullet has left the map already
                    bullet.destruct()
                    bullets_to_cull.append(bul_idx)
            
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
            if scenario.stop_if_no_asteroids and not asteroids:
                # No asteroids remain
                stop_reason = StopReason.no_asteroids
            elif scenario.stop_if_no_ships and not liveships and not (mines or bullets):
                # No ships are alive and no mines exist and no bullets exist
                # Prevents unfairness where ship that dies before another gets score from its bullets as long as the other
                # is alive but the one that lives longer doesn't get the same benefit from its bullets/mines persisting
                # after it dies
                stop_reason = StopReason.no_ships
            elif (
                scenario.stop_if_no_ammo
                and sum(ship.bullets_remaining for ship in liveships) == 0
                and sum(ship.mines_remaining for ship in liveships) == 0
                and not (bullets or mines)
            ):
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
                while time_dif * self.realtime_multiplier < self.delta_time:
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
