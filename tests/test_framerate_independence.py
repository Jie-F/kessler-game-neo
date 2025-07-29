import random
from kesslergame import Scenario, KesslerGame, GraphicsType, KesslerController

TRIALS = 1
GRAPHICS = True
COMPETITION_SAFE_MODE = True
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
        if time_f == 0:
            print()
        for second, action in enumerate(self.actions_list):
            # second is an integer, where if the floor of the time in seconds is equal to it, it will do that action
            if time_f == second * int(framerate):
                if float(time_f + framerate) + 1e-6 >= framerate*game_state.time_limit:
                    print(f"Frame {time_f}, pos=({ship_state.x}, {ship_state.y}), heading={ship_state.heading}, speed={ship_state.speed}")
                fire = action[2]
                drop_mine = action[3]
            if second * int(framerate) <= time_f < (second + 1) * int(framerate):
                thrust = action[0]
                turn_rate = action[1]
                break
        #print(f"Frame {time_f}, pos=({ship_state.x}, {ship_state.y}), heading={ship_state.heading}, speed={ship_state.speed}")
        if thrust is None or turn_rate is None:
            # Ran out of planned actions. Just use the final one for the rest of time.
            thrust = self.actions_list[-1][0]
            turn_rate = self.actions_list[-1][1]
            fire = self.actions_list[-1][2]
            drop_mine = self.actions_list[-1][3]
        
        #print(f"Frame={time_f}, thrust={thrust}, turn_rate={turn_rate}, fire={fire}, drop_mine={drop_mine}")
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Framerate Independent Controller"

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
                        random.choice([False, False]),
                        random.choice([False, False])) for _ in range(1000)]
        controllers.append(FramerateIndependentController(actions_list))
    return controllers

for i in range(TRIALS):
    seed = random.randint(0, 1_000_000)
    seed = 373426
    random.seed(seed)
    framerate1 = 20#random.randint(2, 60)
    framerate2 = 40#framerate1
    while framerate1 == framerate2:
        framerate2 = random.randint(2, 60)
    print(f"Trial={i}, seed={seed}, framerates: {framerate1} and {framerate2}")
    num_ships = random.randint(3, 3)
    scenario = Scenario(name=f"Trial {i}",
                        num_asteroids=random.randint(5, 15),
                        ship_states=random_ship_states(num_ships),
                        map_size=(WIDTH, HEIGHT),
                        seed=seed,
                        ammo_limit_multiplier=random.uniform(0.0, 2.0),
                        time_limit=float(random.randint(10, 30)))
    controllers = randomly_initialized_controllers(num_ships)

    game_settings_1 = {'perf_tracker': True,
                    'graphics_type': GraphicsType.NoGraphics if not GRAPHICS else GraphicsType.Tkinter,
                    'realtime_multiplier': 1.0,
                    'frame_skip': 2,
                    'graphics_obj': None,
                    'frequency': framerate1,
                    "competition_safe_mode": COMPETITION_SAFE_MODE,
                    'UI_settings': {'ships': True, 'lives_remaining': True, 'accuracy': True,
                                    'asteroids_hit': True, 'shots_fired': True, 'bullets_remaining': True,
                                    'controller_name': True, 'scale': 2.0}}
    game_1 = KesslerGame(settings=game_settings_1)
    score_1, perf_data_1 = game_1.run(scenario, controllers)

    game_settings_2 = {'perf_tracker': True,
                    'graphics_type': GraphicsType.NoGraphics if not GRAPHICS else GraphicsType.Tkinter,
                    'realtime_multiplier': 1.0,
                    'frame_skip': 2,
                    'graphics_obj': None,
                    'frequency': framerate2,
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
        print("MISMATCH!")
        break
