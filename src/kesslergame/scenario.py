# -*- coding: utf-8 -*-
# Copyright © 2018-2020 Thales Avionics USA
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from typing import Any
import random
from math import isclose, inf, hypot, degrees, atan2

from .ship import Ship
from .asteroid import Asteroid

def wrap_asteroid(asteroid_dict: dict[str, Any], map_size: tuple[int, int]) -> dict[str, Any]:
    """
    Wrap the asteroid inbounds.
    Scenarios may be validly defined with asteroids out of bounds, and in these cases, the AI agent will see
    the asteroids out of bounds on frame 0, and inbounds (wrapped) on all subsequent frames.
    By wrapping the asteroids as a preprocessing stage, the agents can always assume asteroids are inbounds,
    reducing confusion, especially on the first critical frame of analyzing the map situation.
    """
    if "position" not in asteroid_dict:
        # Invalid asteroid, do not process
        return asteroid_dict
    x, y = asteroid_dict["position"]
    width, height = map_size
    x %= width
    y %= height
    asteroid_dict["position"] = (x, y)
    return asteroid_dict

def nudge_asteroid_away_from_border(asteroid_dict: dict[str, Any], map_size: tuple[int, int]) -> dict[str, Any]:
    """
    Due to the way the wrapping is done, it's possible for asteroids to oscillate between a boundary instead of smoothly passing through.
    For example, an asteroid with initial state {"position": (0, 0), "angle": 360.0, "speed": 100}
    in a 1000x800 map will cycle its y coordinate between 0.0 and 800.0

    This function preprocesses each asteroid state to avoid these initial states that cause oscillation,
    by taking asteroids exactly on the boundary and nudging them away.
    Not a perfect fix, and given enough time the asteroids may begin oscillating anyway, but this eliminates 99.9% of an issue which already was incredibly rare.
    """
    if "position" not in asteroid_dict:
        # Invalid asteroid, do not process
        return asteroid_dict
    x, y = asteroid_dict["position"]
    width, height = map_size
    EPS = 1e-10
    # Check and nudge X
    if isclose(x, 0.0, abs_tol=1e-14):
        x += EPS
    elif isclose(x, width, abs_tol=1e-14):
        x -= EPS

    # Check and nudge Y
    if isclose(y, 0.0, abs_tol=1e-14):
        y += EPS
    elif isclose(y, height, abs_tol=1e-14):
        y -= EPS

    asteroid_dict["position"] = (x, y)
    return asteroid_dict

class Scenario:
    def __init__(self, name: str = "Scenario", num_asteroids: int | None = None, asteroid_states: list[dict[str, Any]] | None = None,
                 ship_states: list[dict[str, Any]] | None = None, map_size: tuple[int, int] | None = None, seed: int | None = None,
                 time_limit: float | None = None, ammo_limit_multiplier: float | None = None, bullet_limit: int | None = None,
                 mine_limit: int | None = None, stop_if_no_ammo: bool | None = None, stop_if_no_asteroids: bool | None = None,
                 stop_if_no_ships: bool | None = None) -> None:
        """
        Specify the starting state of the environment, including map dimensions and optional features

        Make sure to only set either 'num_asteroids' or 'asteroid_states'.

        :param name: Optional, name of the scenario
        :param num_asteroids: Optional, Number of asteroids
        :param asteroid_states: Optional, Asteroid Starting states
        :param ship_states: Optional, Ship Starting states (list of dictionaries)
        :param game_map: Game Map using 'Map' object
        :param seed: Optional seeding value to pass to random.seed() which is called before asteroid creation
        :param time_limit: Optional value for limiting the total duration of the scenario, will be set to infinity if not defined
        :param ammo_limit_multiplier: Optional value for limiting the number of bullets each ship will have
        :param stop_if_no_ammo: Optional flag for stopping the scenario if all ships run out of ammo
        :param stop_if_no_asteroids: Optional flag for stopping the scenario if no asteroids remain
        :param stop_if_no_ships: Optional flag for stopping the scenario if no ships remain
        """
        # Protected variable for managing the name, through getter/setter interface
        self._name: str | None = None

        # Store name as string using setter
        self.name = name

        # Store map size
        if map_size is None:
            self.map_size = (1000, 800)
        else:
            if not (
                isinstance(map_size, tuple) and
                len(map_size) == 2 and
                all(isinstance(x, int) for x in map_size)
            ):
                raise ValueError(f"map_size must be a tuple of two integers, got {map_size!r}")
            self.map_size = map_size


        # Store ship states if not None, otherwise, create one ship at center
        self.ship_states = ship_states if ship_states is not None else [{"position": (self.map_size[0] / 2, self.map_size[1] / 2)}]

        # Set the time_limit to infinity if it is 0 or None
        self.time_limit = time_limit if time_limit is not None else inf

        # Store random seed
        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"seed must be an integer or None, got {type(seed).__name__}")
        self.seed = seed

        # Build asteroids list
        self.asteroid_states = list()
        # Check for mismatch between explicitly defined number of asteroids and tuple of states
        if num_asteroids is not None and asteroid_states is not None:
            raise ValueError("Both 'num_asteroids' and 'asteroid_positions' are specified for Scenario() constructor. Make sure to only define one of these arguments")
        elif asteroid_states is not None:
            # Store asteroid states
            self.asteroid_states = asteroid_states
        elif num_asteroids is not None:
            self.asteroid_states = [dict() for _ in range(num_asteroids)]
        else:
            raise (ValueError("Please define 'num_asteroids' or 'asteroid_states' to create valid custom starting states for the environment"))

        # Set the ammo limit multiplier
        if ammo_limit_multiplier is not None and ammo_limit_multiplier < 0.0:
            raise ValueError("Ammo limit multiplier must be > 0. If unlimited ammo is desired, use 0.0, or do not pass the ammo limit multiplier")

        if ammo_limit_multiplier is not None and bullet_limit is not None:
            raise ValueError("Both 'ammo_limit_multiplier' and 'bullet_limit' are specified for Scenario() constructor. Please define at most one of these arguments.")

        self._ammo_limit_multiplier = ammo_limit_multiplier

        # Validate bullet_limit
        if bullet_limit is not None:
            if not isinstance(bullet_limit, int):
                raise ValueError(f"bullet_limit must be an integer, got {type(bullet_limit).__name__}")
            if bullet_limit < -1:
                raise ValueError("bullet_limit must be -1 for unlimited, or a nonnegative integer")
        self.bullet_limit = bullet_limit

        # If using ammo_limit_multiplier, estimate bullets now
        if self._ammo_limit_multiplier is not None:
            assert self.bullet_limit is None
            estimated_asteroid_count = (
                sum([Scenario.count_asteroids(state.get("size", 3)) for state in self.asteroid_states])
            )
            self.bullet_limit = max(1, round(estimated_asteroid_count * self._ammo_limit_multiplier))

        # Validate mine_limit
        if mine_limit is not None:
            if not isinstance(mine_limit, int):
                raise ValueError(f"mine_limit must be an integer, got {type(mine_limit).__name__}")
            if mine_limit < -1:
                raise ValueError("mine_limit must be -1 for unlimited, or a nonnegative integer")
        self.mine_limit = mine_limit

        # Validate that bullets are specified either nowhere, in the ship state, or in the scenario. But not both!
        if any("bullets_remaining" in ship for ship in self.ship_states) and self.bullet_limit is not None:
            raise ValueError("Both 'bullets_remaining' in ship states, and 'bullet_limit' in the scenario are specified. Please only specify in one place or the other.")

        # Validate that mines are specified either nowhere, in the ship state, or in the scenario. But not both!
        if any("mines_remaining" in ship for ship in self.ship_states) and self.mine_limit is not None:
            raise ValueError("Both 'mines_remaining' in ship states, and 'mine_limit' in the scenario are specified. Please only specify in one place or the other.")

        # Inject the bullets and mines limit into the ship states, if specified
        if self.bullet_limit is not None:
            for ship in self.ship_states:
                if "bullets_remaining" not in ship:
                    ship["bullets_remaining"] = self.bullet_limit
        if self.mine_limit is not None:
            for ship in self.ship_states:
                if "mines_remaining" not in ship:
                    ship["mines_remaining"] = self.mine_limit

        if stop_if_no_ammo is not None and not isinstance(stop_if_no_ammo, bool):
            raise ValueError(f"stop_if_no_ammo must be a boolean or None, got {type(stop_if_no_ammo).__name__}")
        self.stop_if_no_ammo = stop_if_no_ammo if stop_if_no_ammo is not None else False

        if stop_if_no_asteroids is not None and not isinstance(stop_if_no_asteroids, bool):
            raise ValueError(f"stop_if_no_asteroids must be a boolean or None, got {type(stop_if_no_asteroids).__name__}")
        self.stop_if_no_asteroids = stop_if_no_asteroids if stop_if_no_asteroids is not None else True

        if stop_if_no_ships is not None and not isinstance(stop_if_no_ships, bool):
            raise ValueError(f"stop_if_no_ships must be a boolean or None, got {type(stop_if_no_ships).__name__}")
        self.stop_if_no_ships = stop_if_no_ships if stop_if_no_ships is not None else True

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise ValueError(f"Scenario name must be a string, got {type(name).__name__}")
        self._name = name

    @property
    def num_starting_asteroids(self) -> float:
        return len(self.asteroid_states)

    @property
    def is_random(self) -> bool:
        return not all(state for state in self.asteroid_states) if self.asteroid_states else True

    @property
    def max_asteroids(self) -> int:
        return sum([Scenario.count_asteroids(asteroid.size) for asteroid in self.asteroids()])

    @property
    def map_width(self) -> int:
        return self.map_size[0]

    @property
    def map_height(self) -> int:
        return self.map_size[1]

    @staticmethod
    def count_asteroids(asteroid_size: int) -> int:
        # Counting based off of each asteroid making 3 children when destroyed
        return sum([3 ** (size - 1) for size in range(1, asteroid_size + 1)])

    def asteroids(self) -> list[Asteroid]:
        """
        Create asteroid sprites
        :return: list of Asteroids
        """
        asteroids = []

        # Seed the random number generator via an optionally defined user seed
        if self.seed is not None:
            random.seed(self.seed)

        # Loop through and create AsteroidSprites based on starting state
        for asteroid_state in self.asteroid_states:
            if asteroid_state: # Not an empty dictionary
                # Copy to avoid mutating original input
                asteroid_state = dict(asteroid_state)

                has_velocity = "velocity" in asteroid_state
                has_speed = "speed" in asteroid_state
                has_angle = "angle" in asteroid_state

                if has_velocity:
                    if has_speed or has_angle:
                        raise ValueError(
                            "Asteroid state cannot contain both 'velocity' and 'speed' or 'angle'. "
                            "If specifying 'velocity', please omit 'speed/angle'"
                        )
                    vx, vy = asteroid_state.pop("velocity")
                    speed = hypot(vx, vy)
                    angle = degrees(atan2(vy, vx)) % 360.0
                    asteroid_state["speed"] = speed
                    asteroid_state["angle"] = angle

                # No need to change anything if velocity wasn't specified,
                # because the Asteroid constructor handles optional speed/angle

                # Apply position preprocessing as needed
                asteroid_state = wrap_asteroid(asteroid_state, self.map_size)
                asteroid_state = nudge_asteroid_away_from_border(asteroid_state, self.map_size)
                
                # Create the asteroid object
                asteroids.append(Asteroid(**asteroid_state))
            else:
                # Empty dict. Initialize a default random asteroid.
                asteroids.append(
                    Asteroid(position=(random.randrange(0, self.map_size[0]),
                                       random.randrange(0, self.map_size[1])),
                                   ))

        return asteroids

    def ships(self) -> list[Ship]:
        """
        Create ship sprites
        :param frequency: Operating frequency of the game
        :return: list of ShipSprites
        """
        # Loop through and create ShipSprites based on starting state
        return [Ship(idx + 1, **ship_state) for idx, ship_state in enumerate(self.ship_states)]
