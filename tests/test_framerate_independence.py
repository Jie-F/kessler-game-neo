import random
import logging
import os
from kesslergame import Scenario, KesslerGame, GraphicsType, KesslerController, StopReason
from math import sqrt, inf
import argparse

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Run Kessler game simulations.')
parser.add_argument('-seed', type=int, help='Seed value to use for the simulation (enables graphics by default).')
parser.add_argument('--nogui', action='store_true', help='Disable graphics, even if seed is specified.')
parser.add_argument('-trials', type=int, help='Specify max number of trials to run.')
args = parser.parse_args()

# Set seed and graphics flag
rand_seed = args.seed
GRAPHICS = rand_seed is not None and not args.nogui

TRIALS = args.trials if args.trials is not None else 100000000000
TIME_LIMIT_OVERRIDE = inf # Nvm it's not actually an override. Just used if the scenario has no time limit defined.
COMPETITION_SAFE_MODE = False
WIDTH = 1000
HEIGHT = 800

thrust_range = (-480.0, 480.0)
turn_rate_range = (-180.0, 180.0)

# Setup logging
log_filename = f'mismatches_{os.getpid()}.log'

class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SEED: %(message)s',
    handlers=[FlushFileHandler(log_filename, mode='w')]
)

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
            thrust, turn_rate, fire, drop_mine = self.actions_list[-1]
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "FPS Indep Test"

def random_ship_states(number: int) -> list[dict]:
    ship_states = []
    for _ in range(number):
        state = {"position": (random.uniform(0.0, WIDTH), random.uniform(0.0, HEIGHT)),
                 "angle": random.uniform(0.0, 360.0),
                 "lives": random.randint(1, 50),
                 "team": random.randint(1, 2),
                 #"bullets_remaining": random.randint(1, 5000),
                 "mines_remaining": random.randint(0, 20)}
        ship_states.append(state)
    return ship_states

def randomly_initialized_controllers(number: int) -> list[FramerateIndependentController]:
    controllers = []
    for _ in range(number):
        actions_list = [(random.uniform(*thrust_range),
                         random.uniform(*turn_rate_range),
                         random.choice([True, False]),
                         random.choice([True, False])) for _ in range(62)]
        #actions_list[0] = (actions_list[0][0], actions_list[0][1], False, False)  # No fire/drop on first
        controllers.append(FramerateIndependentController(actions_list))
    return controllers

def check_scores(score_1, score_2, seed) -> bool:
    if score_1 != score_2:
        if (score_1.stop_reason == StopReason.no_asteroids and score_2.stop_reason == StopReason.no_asteroids) or (score_1.stop_reason == StopReason.no_ships and score_2.stop_reason == StopReason.no_ships):
            if len(score_1.teams) != len(score_2.teams):
                logging.info(f'{seed} - team length mismatch: {len(score_1.teams)} vs {len(score_2.teams)}')
                return False
            for t1, t2 in zip(score_1.teams, score_2.teams):
                if t1 != t2:
                    logging.info(f'{seed} - team data mismatch: {t1} vs {t2}')
                    return False
            return True
        else:
            logging.info(f'{seed} - score mismatch\nScore1: {score_1}\nScore2: {score_2}')
            return False
    return True

# Main loop
for i in range(TRIALS):
    if rand_seed is None:
        random.seed()
        seed = random.randint(0, 100_000_000_000)
    else:
        seed = rand_seed
    random.seed(seed)

    framerate1 = random.randint(5, 60)
    framerate2 = framerate1
    while framerate1 == framerate2:
        framerate2 = random.randint(5, 60)

    print(f"Trial={i}, seed={seed}, framerates: {framerate1} and {framerate2}")

    num_ships = random.randint(1, 6)
    scenario = Scenario(name=f"Trial {i}",
                        num_asteroids=random.randint(1, 10),
                        ship_states=random_ship_states(num_ships),
                        map_size=(WIDTH, HEIGHT),
                        seed=seed,
                        ammo_limit_multiplier=random.uniform(0.0, 2.0),
                        stop_if_no_ammo=False,
                        stop_if_no_asteroids=False,
                        stop_if_no_ships=False,
                        time_limit=float(random.randint(2, 20)))
    controllers = randomly_initialized_controllers(num_ships)

    settings_base = {
        'perf_tracker': True,
        'graphics_type': GraphicsType.NoGraphics if not GRAPHICS else GraphicsType.Tkinter,
        'realtime_multiplier': 1.0 if GRAPHICS else 0.0,
        'frame_skip': 1,
        'graphics_obj': None,
        'time_limit': TIME_LIMIT_OVERRIDE,
        'perf_tracker': False,
        "competition_safe_mode": COMPETITION_SAFE_MODE,
        'UI_settings': {'ships': True, 'lives_remaining': True, 'accuracy': True,
                        'asteroids_hit': True, 'shots_fired': True, 'bullets_remaining': True,
                        'controller_name': True, 'scale': 2.0}
    }

    settings_1 = settings_base | {'frequency': framerate1}
    settings_2 = settings_base | {'frequency': framerate2}

    game_1 = KesslerGame(settings=settings_1)
    score_1, _ = game_1.run(scenario, controllers)

    game_2 = KesslerGame(settings=settings_2)
    score_2, _ = game_2.run(scenario, controllers)

    if not check_scores(score_1, score_2, seed):
        print(f"Mismatch found. Seed logged: {seed}")
