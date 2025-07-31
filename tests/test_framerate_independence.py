import random
from kesslergame import Scenario, KesslerGame, GraphicsType, KesslerController, StopReason
from math import sqrt, inf
TRIALS = 100000000000
GRAPHICS = False
rand_seed = None
TIME_LIMIT_OVERRIDE = inf # Nvm it's not actually an override. Just used if the scenario has no time limit defined.
COMPETITION_SAFE_MODE = False
WIDTH = 1000
HEIGHT = 800

thrust_range = (-480.0, 480.0)
turn_rate_range = (-180.0, 180.0)

class FramerateIndependentController(KesslerController):
    def __init__(self, actions_list: list[tuple[float, float, bool, bool]]):
        self.actions_list = actions_list

    def actions(self, ship_state: object, game_state: object) -> tuple[float, float, bool, bool]:
        """
        This controller executes predefined actions from the actions_list.
        Each second, it will execute a different action. And the controller can only shoot or drop mines once per second.
        """
        time_s = game_state.time
        time_f = game_state.frame
        framerate = game_state.frame_rate
        thrust = None
        turn_rate = None
        fire = False
        drop_mine = False
        for second, action in enumerate(self.actions_list):
            # second is an integer, where if the floor of the time in seconds is equal to it, it will do that action
            if time_f == second * int(framerate):
                fire = action[2]
                drop_mine = action[3]
            if second * int(framerate) <= time_f < (second + 1) * int(framerate):
                thrust = action[0]
                turn_rate = action[1]
                break
        if thrust is None or turn_rate is None:
            # Ran out of planned actions. Just use the final one for the rest of time.
            thrust = self.actions_list[-1][0]
            turn_rate = self.actions_list[-1][1]
            fire = self.actions_list[-1][2]
            drop_mine = self.actions_list[-1][3]
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "FPS Indep Test"

def random_ship_states(number: int) -> list[dict]:
    ship_states = []
    for i in range(number):
        state = {"position": (random.uniform(0.0, WIDTH), random.uniform(0.0, HEIGHT)),
                 "angle": random.uniform(0.0, 360.0),
                 "lives": random.randint(30, 1000),
                 "team": random.randint(1, 2),
                 #"bullets_remaining": random.randint(1, 5000),
                 "mines_remaining": random.randint(10, 100)}
        ship_states.append(state)
    return ship_states

def randomly_initialized_controllers(number: int) -> list[FramerateIndependentController]:
    controllers = []
    for i in range(number):
        actions_list = [(random.uniform(thrust_range[0], thrust_range[1]),
                        random.uniform(turn_rate_range[0], turn_rate_range[1]),
                        random.choice([True, False]) if i != 0 else False,
                        random.choice([True, False]) if i != 0 else False) for i in range(1000)]
        controllers.append(FramerateIndependentController(actions_list))
    return controllers

for i in range(TRIALS):
    if rand_seed is None:
        random.seed()
        seed = random.randint(0, 1_000_000_000)
    else:
        seed = rand_seed
    random.seed(seed)
    framerate1 = random.randint(10, 60)
    framerate2 = framerate1
    while framerate1 == framerate2:
        framerate2 = random.randint(10, 60)
    print(f"Trial={i}, seed={seed}, framerates: {framerate1} and {framerate2}")
    num_ships = random.randint(1, 5)
    scenario = Scenario(name=f"Trial {i}",
                        num_asteroids=random.randint(1, 3),
                        ship_states=random_ship_states(num_ships),
                        map_size=(WIDTH, HEIGHT),
                        seed=seed,
                        ammo_limit_multiplier=random.uniform(0.0, 2.0),
                        time_limit=float(random.randint(6, 16)))
    controllers = randomly_initialized_controllers(num_ships)

    game_settings_1 = {'perf_tracker': True,
                    'graphics_type': GraphicsType.NoGraphics if not GRAPHICS else GraphicsType.Tkinter,
                    'realtime_multiplier': 1.0 if GRAPHICS else 0.0,
                    'frame_skip': 1,
                    'graphics_obj': None,
                    'frequency': framerate1,
                    'time_limit': TIME_LIMIT_OVERRIDE,
                    "competition_safe_mode": COMPETITION_SAFE_MODE,
                    'UI_settings': {'ships': True, 'lives_remaining': True, 'accuracy': True,
                                    'asteroids_hit': True, 'shots_fired': True, 'bullets_remaining': True,
                                    'controller_name': True, 'scale': 2.0}}
    game_1 = KesslerGame(settings=game_settings_1)
    score_1, perf_data_1 = game_1.run(scenario, controllers)

    game_settings_2 = {'perf_tracker': True,
                    'graphics_type': GraphicsType.NoGraphics if not GRAPHICS else GraphicsType.Tkinter,
                    'realtime_multiplier': 1.0 if GRAPHICS else 0.0,
                    'frame_skip': 1,
                    'graphics_obj': None,
                    'frequency': framerate2,
                    'time_limit': TIME_LIMIT_OVERRIDE,
                    "competition_safe_mode": COMPETITION_SAFE_MODE,
                    'UI_settings': {'ships': True, 'lives_remaining': True, 'accuracy': True,
                                    'asteroids_hit': True, 'shots_fired': True, 'bullets_remaining': True,
                                    'controller_name': True, 'scale': 2.0}}
    game_2 = KesslerGame(settings=game_settings_2)
    score_2, perf_data_2 = game_2.run(scenario, controllers)

    print(score_1)
    print()
    print(score_2)
    print()
    if score_1 != score_2:
        if score_1.stop_reason == StopReason.no_asteroids and score_2.stop_reason == StopReason.no_asteroids:
            # If no asteroids, the scores won't match because the sim time at the end can be slightly off.
            # Just make sure everything else matches.
            if len(score_1.teams) != len(score_2.teams):
                print("MISMATCH!")
                break
            for team_self, team_other in zip(score_1.teams, score_2.teams):
                if team_self != team_other:
                    print("MISMATCH!")
                    break
        else:
            print("MISMATCH!")
            break
