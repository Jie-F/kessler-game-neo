import time
import random
import numpy as np
import cProfile
import sys
from scenarios.xfc_2021_scenarios import *
from scenarios.xfc_2024_scenarios import *
from scenarios.xfc_2023_replica_scenarios import *
from scenarios.custom_scenarios import *
import argparse

from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1) # Fixes blurriness when a scale factor is used in Windows

ASTEROID_COUNT_LOOKUP = (0, 1, 4, 13, 40)

from src.kesslergame import Scenario, KesslerGame, GraphicsType
from src.kesslergame.controller_gamepad import GamepadController
#from src.neo_controller import NeoController
from neo_controller import NeoController
#from src.neo_controller_cont_working import NeoController 
from src.neo_controller_wcci_bench import NeoController as NeoControllerWCCI
from benchmark_controller import BenchmarkController

BENCHMARK_TIME_LIMIT = 60.0
#BENCHMARK_TIME_LIMIT = 3.0
JUMP_IND = 2000
global color_text
color_text = True

TRIALS = 1000000

GRAPHICS = False

def color_print(text='', color='white', style='normal', same=False, previous=False) -> None:
    global color_text
    global colors
    global styles

    if color_text and 'colorama' not in sys.modules:
        import colorama
        colorama.init()

        colors = {
            'black': colorama.Fore.BLACK,
            'red': colorama.Fore.RED,
            'green': colorama.Fore.GREEN,
            'yellow': colorama.Fore.YELLOW,
            'blue': colorama.Fore.BLUE,
            'magenta': colorama.Fore.MAGENTA,
            'cyan': colorama.Fore.CYAN,
            'white': colorama.Fore.WHITE,
        }

        styles = {
            'dim': colorama.Style.DIM,
            'normal': colorama.Style.NORMAL,
            'bright': colorama.Style.BRIGHT,
        }
    elif color_text:
        import colorama

    if same:
        end = ''
    else:
        end = '\n'
    if color_text:
        print(previous*'\033[A' + colors[color] + styles[style] + str(text) + colorama.Style.RESET_ALL, end=end)
    else:
        print(str(text), end=end)

width, height = (1000, 800)



#controllers_used = [NeoController(), NeoControllerWCCI()]


# Define Game Settings
game_settings = {'perf_tracker': True,
                 'graphics_type': GraphicsType.NoGraphics if not GRAPHICS else GraphicsType.Tkinter,#UnrealEngine,Tkinter,NoGraphics
                 'realtime_multiplier': 0.0,
                 'graphics_obj': None,
                 'frequency': 30.0,
                 'UI_settings': 'all'}

game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario

# Evaluate the game

xfc_2021_portfolio = [
    threat_test_1,
    threat_test_2,
    threat_test_3,
    threat_test_4,
    accuracy_test_1,
    accuracy_test_2,
    accuracy_test_3,
    accuracy_test_4,
    accuracy_test_5,
    accuracy_test_6,
    accuracy_test_7,
    accuracy_test_8,
    accuracy_test_9,
    accuracy_test_10,
    wall_left_easy,
    wall_right_easy,
    wall_top_easy,
    wall_bottom_easy,
    ring_closing,
    ring_static_left,
    ring_static_right,
    ring_static_top,
    ring_static_bottom,

    wall_right_wrap_1,
    wall_right_wrap_2,
    wall_right_wrap_3,
    wall_right_wrap_4,
    wall_left_wrap_1,
    wall_left_wrap_2,
    wall_left_wrap_3,
    wall_left_wrap_4,
    wall_top_wrap_1,
    wall_top_wrap_2,
    wall_top_wrap_3,
    wall_top_wrap_4,
    wall_bottom_wrap_1,
    wall_bottom_wrap_2,
    wall_bottom_wrap_3,
    wall_bottom_wrap_4,
]

show_portfolio = [
    threat_test_1,
    threat_test_2,
    threat_test_3,
    threat_test_4,
    accuracy_test_5,
    accuracy_test_6,
    accuracy_test_7,
    accuracy_test_8,
    accuracy_test_9,
    accuracy_test_10,
    wall_left_easy,
    wall_right_easy,
    wall_top_easy,
    wall_bottom_easy,
    ring_closing,
    ring_static_left,
    ring_static_right,
    ring_static_top,
    ring_static_bottom,
    wall_right_wrap_3,
    wall_right_wrap_4,
    wall_left_wrap_3,
    wall_left_wrap_4,
    wall_top_wrap_3,
    wall_top_wrap_4,
    wall_bottom_wrap_3,
    wall_bottom_wrap_4,
]

alternate_scenarios = [
    corridor_left,
    corridor_right,
    corridor_top,
    corridor_bottom,

    # May have to cut these
    moving_corridor_1,
    moving_corridor_2,
    moving_corridor_3,
    moving_corridor_4,
    moving_corridor_angled_1,
    moving_corridor_angled_2,
    moving_corridor_curve_1,
    moving_corridor_curve_2,

    scenario_small_box,
    scenario_big_box,
    scenario_2_still_corridors,
]

# xfc2023 = [
#     ex_adv_four_corners_pt1,
#     ex_adv_four_corners_pt2,
#     ex_adv_asteroids_down_up_pt1,
#     ex_adv_asteroids_down_up_pt2,
#     ex_adv_direct_facing,
#     ex_adv_two_asteroids_pt1,
#     ex_adv_two_asteroids_pt2,
#     ex_adv_ring_pt1,
#     adv_random_big_1,
#     adv_random_big_2,
#     adv_random_big_3,
#     adv_random_big_4,
#     adv_multi_wall_bottom_hard_1,
#     adv_multi_wall_right_hard_1,
#     adv_multi_ring_closing_left,
#     adv_multi_ring_closing_right,
#     adv_multi_two_rings_closing,
#     avg_multi_ring_closing_both2,
#     adv_multi_ring_closing_both_inside,
#     adv_multi_ring_closing_both_inside_fast
# ]

xfc2024 = [
    adv_random_small_1,
    adv_random_small_1_2,
    adv_multi_wall_left_easy,
    adv_multi_four_corners,
    adv_multi_wall_top_easy,
    adv_multi_2wall_closing,
    adv_wall_bottom_staggered,
    adv_multi_wall_right_hard,
    adv_moving_corridor_angled_1,
    adv_moving_corridor_angled_1_mines,
    adv_multi_ring_closing_left,
    adv_multi_ring_closing_left2,
    adv_multi_ring_closing_both2,
    adv_multi_ring_closing_both_inside_fast,
    adv_multi_two_rings_closing
]

custom_scenarios = [
    target_priority_optimization1,
    closing_ring_scenario,
    easy_closing_ring_scenario,
    more_intense_closing_ring_scenario,
    rotating_square_scenario,
    rotating_square_2_overlap,
    falling_leaves_scenario,
    zigzag_motion_scenario,
    shearing_pattern_scenario,
    super_hard_wrap,
    wonky_ring,
    moving_ring_scenario,
    shifting_square_scenario,
    delayed_closing_ring_scenario,
    spiral_assault_scenario,
    dancing_ring,
    dancing_ring_2,
    intersecting_lines_scenario,
    exploding_grid_scenario,
    grid_formation_explosion_scenario,
    aspect_ratio_grid_formation_scenario,
    adv_asteroid_stealing,
    wrapping_nightmare,
    wrapping_nightmare_fast,
    purgatory,
    cross,
    fight_for_asteroid,
    shot_pred_test,
    shredder,
    diagonal_shredder,
    out_of_bound_mine,
    explainability_1,
    explainability_2,
    split_forecasting,
    minefield_maze_scenario,
    wrap_collision_test
]

score = None
died = False
team_1_hits = 0
team_2_hits = 0
team_1_deaths = 0
team_2_deaths = 0
team_1_wins = 0
team_2_wins = 0
team_1_shot_efficiency = 0
team_2_shot_efficiency = 0
team_1_shot_efficiency_including_mines = 0
team_2_shot_efficiency_including_mines = 0
team_1_bullets_hit = 0
team_1_shots_fired = 0
team_2_bullets_hit = 0
team_2_shots_fired = 0

selected_portfolio = [None]

randseed = 0
color_print(f'\nUsing seed {randseed}', 'green')
random.seed(randseed)

benchmark_scenario = Scenario(name="Benchmark Scenario",
                                num_asteroids=200,
                                ship_states=[
                                    {'position': (width/2.0, height/2.0), 'angle': 0.0, 'lives': 1000000, 'team': 1, 'mines_remaining': 1000000},
                                ],
                                map_size=(width, height),
                                seed=0,
                                time_limit=BENCHMARK_TIME_LIMIT)

scenario_to_run = benchmark_scenario

run_times = []
missed = False
for i in range(JUMP_IND, JUMP_IND + TRIALS):
    random.seed(i)
    print()
    print(f"Trial {i}")
    print()
    benchmark_scenario = Scenario(name="Benchmark Scenario",
                                    num_asteroids=60,
                                    ship_states=[
                                        {'position': (width/2.0, height/2.0), 'angle': 0.0, 'lives': 1000000, 'team': 1, 'mines_remaining': 1000000},
                                    ],
                                    map_size=(width, height),
                                    seed=i,
                                    time_limit=BENCHMARK_TIME_LIMIT)
    controllers_used = [NeoController()]
    scenario_to_run = benchmark_scenario
    if scenario_to_run is not None:
        print(f"Evaluating scenario {scenario_to_run.name}")
    pre_time = time.perf_counter()
    score, perf_data = game.run(scenario=scenario_to_run, controllers=controllers_used)
    post_time = time.perf_counter()
    run_times.append(post_time - pre_time)
    # Print out some general info about the result
    num_teams = len(score.teams)
    if score:
        team1 = score.teams[0]
        if num_teams > 1:
            team2 = score.teams[1]
        asts_hit = [team.asteroids_hit for team in score.teams]
        color_print('Scenario eval time: '+str(run_times[-1]), 'green')
        color_print(score.stop_reason, 'green')
        color_print(f"Scenario in-game time: {score.sim_time:.02f} s", 'green')
        color_print('Asteroids hit: ' + str(asts_hit), 'green')
        team_1_hits += asts_hit[0]
        if num_teams > 1:
            team_2_hits += asts_hit[1]
            if asts_hit[0] > asts_hit[1]:
                team_1_wins += 1
            elif asts_hit[0] < asts_hit[1]:
                team_2_wins += 1
        else:
            team_1_wins += 1
        team_deaths = [team.deaths for team in score.teams]
        team_1_deaths += team_deaths[0]
        if num_teams > 1:
            team_2_deaths += team_deaths[1]
        color_print('Deaths: ' + str(team_deaths), 'green')
        if team_deaths[0] >= 1:
            died = True
        else:
            died = False
        color_print('Accuracy: ' + str([team.accuracy for team in score.teams]), 'green')
        color_print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]), 'green')
        if score.teams[0].accuracy < 1:
            color_print('NEO MISSED SDIOFJSDI(FJSDIOJFIOSDJFIODSJFIOJSDIOFJSDIOFJOSDIJFISJFOSDJFOJSDIOFJOSDIJFDSJFI)SDFJHSUJFIOSJFIOSJIOFJSDIOFJIOSDFOSDF\n\n', 'red')
            missed = True
        else:
            missed = False
        team_1_shot_efficiency = (team1.bullets_hit/score.sim_time)/(1/(1/10))
        team_1_shot_efficiency_including_mines = (team1.asteroids_hit/score.sim_time)/(1/(1/10))
        if num_teams > 1:
            team_2_shot_efficiency = (team2.bullets_hit/score.sim_time)/(1/(1/10))
            team_2_shot_efficiency_including_mines = (team2.asteroids_hit/score.sim_time)/(1/(1/10))
            team_1_bullets_hit += team1.bullets_hit
            team_2_bullets_hit += team2.bullets_hit
            team_1_shots_fired += team1.shots_fired
            team_2_shots_fired += team2.shots_fired
    print(f"Team 1, 2 hits: ({team_1_hits}, {team_2_hits})")
    print(f"Team 1, 2 wins: ({team_1_wins}, {team_2_wins})")
    print(f"Team 1, 2 deaths: ({team_1_deaths}, {team_2_deaths})")
    print(f"Team 1, 2 accuracies: ({team_1_bullets_hit/(team_1_shots_fired + 0.000000000000001)}, {team_2_bullets_hit/(team_2_shots_fired + 0.000000000000001)})")
    print(f"Team 1, 2 shot efficiencies: ({team_1_shot_efficiency:.02%}, {team_2_shot_efficiency:.02%})")
    print(f"Team 1, 2 shot efficiencies inc. mines/ship hits: ({team_1_shot_efficiency_including_mines:.02%}, {team_2_shot_efficiency_including_mines:.02%})")
    if missed:
        break
print(f"Run times are: {run_times}")
print(f"The average time over {TRIALS} trials is {sum(run_times)/len(run_times)} s")
assert len(run_times) == TRIALS, f"Looks like not all trials completed!"
