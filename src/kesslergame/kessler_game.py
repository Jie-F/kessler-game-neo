# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from __future__ import annotations

import time

from math import inf, nan, isfinite, isnan
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
    def __init__(self, time_offset: float, object_a_idx: int, object_b_idx: int, collision_type: CollisionType):
        """
        Represents a collision event between two game objects at a specific time.

        :param time_offset: Time (in seconds) relative to the end of the frame. Must be in [-dt, 0.0].
        :param object_a_idx: Index of first object involved in the collision.
        :param object_b_idx: Index of second object involved in the collision.
        :param collision_type: Type of collision as defined in CollisionType enum.
        """
        self.time_offset = time_offset  # Time offset in seconds relative to frame end, e.g., -0.001
        # Time offset should be in range [-delta_time, 0.0]
        self.object_a_idx = object_a_idx
        self.object_b_idx = object_b_idx
        self.collision_type = collision_type

    def __lt__(self, other: CollisionEvent) -> bool:
        """Allow sorting events by time offset (earlier events come first)."""
        return self.time_offset < other.time_offset

    def __repr__(self):
        return f"<CollisionEvent time_offset={self.time_offset:.4f}s type={self.collision_type} obj_a_idx={self.object_a_idx} obj_b_idx={self.object_b_idx}>"


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

    def enqueue_bullet_asteroid_collisions(self, bullets: list[Bullet], asteroids: list[Asteroid]):
        # Collect all potential bullet-asteroid collisions
        for bul_idx, bullet in enumerate(bullets):
            for ast_idx, asteroid in enumerate(asteroids):
                if circle_line_collision_continuous(
                    bullet.x, bullet.y, bullet.x + bullet.tail_delta_x, bullet.y + bullet.tail_delta_y, bullet.vx, bullet.vy,
                    asteroid.x, asteroid.y, asteroid.vx, asteroid.vy, asteroid.radius, self.delta_time
                ):
                    collision_start_time, _ = collision_time_interval(
                        bullet.x, bullet.y,
                        bullet.x + bullet.tail_delta_x, bullet.y + bullet.tail_delta_y,
                        bullet.vx, bullet.vy,
                        asteroid.x, asteroid.y,
                        asteroid.vx, asteroid.vy,
                        asteroid.radius
                    )
                    if isnan(collision_start_time):
                        # This case should NEVER get hit since the circle_line_collision_continuous function
                        # already found that there would be a collision. But just in case of numerical instability causing
                        # these to return different results, this will prevent a crash
                        continue
                    collision_time = max(-self.delta_time, collision_start_time)
                    assert -self.delta_time <= collision_time <= 0.0
                    # Inline insertion to keep collisions sorted by time
                    i = len(self.collision_queue)
                    while i > 0 and self.collision_queue[i - 1].time_offset > collision_time:
                        i -= 1
                    self.collision_queue.insert(i, CollisionEvent(collision_time, bul_idx, ast_idx, CollisionType.BULLET_ASTEROID))

    def enqueue_mine_asteroid_collisions(self, mines: list[Mine], asteroids: list[Asteroid]):
        detonating_mines: list[Mine] = [mine for mine in mines if mine.detonating]
        if detonating_mines:
            for ast_idx, asteroid in enumerate(asteroids):
                for mine_idx, mine in enumerate(detonating_mines):
                    dx = asteroid.x - mine.x
                    dy = asteroid.y - mine.y
                    radius_sum = mine.blast_radius + asteroid.radius
                    sq_dist = dx * dx + dy * dy
                    if sq_dist <= radius_sum * radius_sum:
                        i = len(self.collision_queue)
                        collision_time = 0.0
                        while i > 0 and self.collision_queue[i - 1].time_offset > collision_time:
                            i -= 1
                        self.collision_queue.insert(i, CollisionEvent(collision_time, mine_idx, ast_idx, CollisionType.MINE_ASTEROID))

    def enqueue_mine_ship_collisions(self, mines: list[Mine], ships: list[Ship]):
        for mine_idx, mine in enumerate(mines):
            # For each live, non-respawning ship, apply damage only from the closest mine within range
            for ship_idx, ship in enumerate(ships):
                if ship.is_respawning or not ship.alive:
                    continue
                dx = ship.x - mine.x
                dy = ship.y - mine.y
                radius_sum = mine.blast_radius + ship.radius
                sq_dist = dx * dx + dy * dy
                if sq_dist <= radius_sum * radius_sum:
                    i = len(self.collision_queue)
                    collision_time = 0.0
                    while i > 0 and self.collision_queue[i - 1].time_offset > collision_time:
                        i -= 1
                    self.collision_queue.insert(i, CollisionEvent(collision_time, mine_idx, ship_idx, CollisionType.MINE_SHIP))

    def enqueue_ship_asteroid_collisions(self, ships: list[Ship], asteroids: list[Asteroid]):
        for ship_idx, ship in enumerate(ships):
            if ship.is_respawning or not ship.alive:
                continue
            for ast_idx, asteroid in enumerate(asteroids):
                # Check for collisions in time interval [t - delta_time, t]
                # TODO: Change interval to when the ship got out of respawn, until t.
                collision_start_time = ship_asteroid_continuous_collision_time(
                    ship.x, ship.y, ship.radius, ship.speed, ship.integration_initial_states,
                    asteroid.x, asteroid.y, asteroid.vx, asteroid.vy, asteroid.radius, asteroid.speed,
                    self.delta_time
                )
                if not isnan(collision_start_time):
                    assert -self.delta_time <= collision_start_time <= 0.0 # Collision happened within past frame
                    i = len(self.collision_queue)
                    while i > 0 and self.collision_queue[i - 1].time_offset > collision_start_time:
                        i -= 1
                    self.collision_queue.insert(i, CollisionEvent(collision_start_time, ship_idx, ast_idx, CollisionType.SHIP_ASTEROID))

    def enqueue_ship_ship_collisions(self, ships: list[Ship]):
        num_ships = len(ships)
        for ship1_idx, ship1 in enumerate(ships):
            if ship1.alive and not ship1.is_respawning:
                for ship2_idx in range(ship1_idx + 1, num_ships):
                    ship2 = ships[ship2_idx]
                    if ship2.alive and not ship2.is_respawning:
                        # Check for collisions in time interval [t - delta_time, t]
                        # TODO: Check for respawn invinc end to determine interval start
                        collision_start_time = ship_ship_continuous_collision_time(
                            ship1.x, ship1.y, ship1.radius, ship1.speed, ship1.integration_initial_states,
                            ship2.x, ship2.y, ship2.radius, ship2.speed, ship2.integration_initial_states,
                            self.delta_time
                        )
                        if not isnan(collision_start_time):
                            assert -self.delta_time <= collision_start_time <= 0.0 # Collision happened within past frame
                            # Insert chronologically
                            i = len(self.collision_queue)
                            while i > 0 and self.collision_queue[i - 1].time_offset > collision_start_time:
                                i -= 1
                            self.collision_queue.insert(i, CollisionEvent(collision_start_time, ship1_idx, ship2_idx, CollisionType.SHIP_SHIP))

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
        map_width, map_height = scenario.map_size

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

        new_asteroids: list[Asteroid] = []
        bullets_to_cull: list[int] = []
        asteroids_to_cull: list[int] = []

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

            # --- UPDATE STATE INFORMATION OF EACH OBJECT --------------------------------------------------------------

            # Update each Asteroid, Bullet, and Ship
            # Because the game_state stores a mutable reference to the internal states of the ship/asteroid/bullet/mine,
            # these updates automatically reflect in the game_state
            for ship in liveships:
                new_bullet, new_mine = ship.update(self.delta_time, scenario.map_size)
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

            self.enqueue_bullet_asteroid_collisions(bullets, asteroids)
            self.enqueue_mine_asteroid_collisions(mines, asteroids)
            self.enqueue_mine_ship_collisions(mines, ships)
            self.enqueue_ship_asteroid_collisions(ships, asteroids)
            self.enqueue_ship_ship_collisions(ships)

            # --- Resolve collisions in the queue until it is empty ---
            ships_to_cull: list[int] = []
            asteroids_to_cull: list[int] = []
            bullets_to_cull: list[int] = []
            mines_to_cull: list[int] = []
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

                        bullet.update(dt)
                        asteroid.update(dt)

                        bullets_to_cull.append(bul_idx)
                        asteroids_to_cull.append(ast_idx)

                        new_asteroids = asteroid.destruct()
                        for a in new_asteroids:
                            # This is a forward update, from the time of collision to the end of the frame!
                            a.update(-dt)
                        
                        self.enqueue_bullet_asteroid_collisions(bullets, new_asteroids)
                        self.enqueue_mine_asteroid_collisions(mines, new_asteroids)
                        self.enqueue_ship_asteroid_collisions(ships, new_asteroids)
                    case CollisionType.MINE_ASTEROID:
                        mine_idx = event.object_a_idx
                        ast_idx = event.object_b_idx

                        if mine_idx in mines_to_cull or ast_idx in asteroids_to_cull:
                            continue

                        mine = mines[mine_idx]
                        asteroid = asteroids[ast_idx]

                        mine.update(dt)
                        asteroid.update(dt)

                        new_asteroids = asteroid.destruct()
                        for a in new_asteroids:
                            # This is a forward update, from the time of collision to the end of the frame!
                            a.update(-dt)
                        
                        self.enqueue_bullet_asteroid_collisions(bullets, new_asteroids)
                        self.enqueue_mine_asteroid_collisions(mines, new_asteroids)
                        self.enqueue_ship_asteroid_collisions(ships, new_asteroids)

                        mines_to_cull.append(mine_idx)
                    case CollisionType.MINE_SHIP:
                        mine_idx = event.object_a_idx
                        ship_idx = event.object_b_idx

                        if mine_idx in mines_to_cull or ship_idx in ships_to_cull:
                            continue
                        
                        mine = mines[mine_idx]
                        ship = ships[ship_idx]

                        assert ship.alive
                        if ship.is_respawning:
                            continue

                        mine.update(dt)
                        mine.destruct()

                        ship.update(dt)
                        ship.destruct() # TODO: Add map size
                        if not ship.alive:
                            ships_to_cull.append(ship_idx)
                        else:
                            ship.update(-dt)
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

                        ship.update(dt)
                        asteroid.update(dt)

                        ship.destruct()
                        new_asteroids = asteroid.destruct()

                        ship.update(-dt)
                        for a in new_asteroids:
                            # This is a forward update, from the time of collision to the end of the frame!
                            a.update(-dt)
                        
                        self.enqueue_bullet_asteroid_collisions(bullets, new_asteroids)
                        self.enqueue_mine_asteroid_collisions(mines, new_asteroids)
                        self.enqueue_ship_asteroid_collisions(ships, new_asteroids)

                        if not ship.alive:
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

                        ship1.update(dt)
                        ship2.update(dt)

                        ship1.destruct()
                        ship2.destruct()

                        ship1.update(-dt)
                        ship2.update(-dt)

            # Now that all collisions are handled and resolved, the final step is to cull the removed objects
            # TODO: Sort as we go instead of at the end here. Probably faster.
            for ast_idx in sorted(asteroids_to_cull, reverse=True):
                asteroids[ast_idx] = asteroids[-1]
                asteroids.pop()
                if not self.competition_safe_mode:
                    game_state.remove_asteroid(ast_idx)
            
            for bul_idx in sorted(bullets_to_cull, reverse=True):
                bullets[bul_idx] = bullets[-1]
                bullets.pop()
                if not self.competition_safe_mode:
                    game_state.remove_bullet(bul_idx)
            
            for mine_idx in sorted(mines_to_cull, reverse=True):
                mines[bul_idx] = mines[-1]
                mines.pop()
                if not self.competition_safe_mode:
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
            sim_time += self.delta_time
            sim_frame += 1
            if not self.competition_safe_mode:
                assert game_state is not None
                game_state.time = sim_time
                game_state.frame = sim_frame

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
            elif sim_time > time_limit:
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
