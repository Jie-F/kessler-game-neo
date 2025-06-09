// _____   __           
// ___  | / /__________ 
// __   |/ /_  _ \  __ \
// _  /|  / /  __/ /_/ /
// /_/ |_/  \___/\____/ 

// Kessler controller
// Jie Fan (jie.f@pm.me)

// TODO: Show stats at the end
// DONE: Verify that frontrun protection's working (shot stealing protection), because it still feels like it's not totally working!
// DONE: Make it so during a respawn maneuver, if I'm no longer gonna hit anything, I can begin to shoot!
// TODO: Use the tolerance in the shot for the target selection so I don't just aim for the center all the time
// KINDA DONE: Add error handling as a catch-all
// TODO: Analyze each base state, and store analysis results. Like the heuristic FIS, except use more random search. Density affects the movement speed and cruise timesteps. Tune stuff much better.
// TODO: Add error checks, so that if Neo thinks its done but there's still asteroids left, it'll realize that and re-ingest the updated state and finish off its job. This should NEVER happen though, but in the 1/1000000 chance a new bug happens during the competition, this would catch it.
// DONE: Tune gc to maybe speed stuff up
// DONE: Match collision checks with Kessler, including <= vs <
// DONE: Add iteration boosting algorithm to do more iterations in critical moments
// DONE: If we're chilling, have mines and lives, go and do some damage!
// WON'T FIX: Optimally, the target selection will consider mines blowing up asteroids, and having forecasted asteroids there. But this is a super specific condition and it's very complex to implement correctly, so maybe let's just skip this lol. It's probably not worth spending 50 hours implementing something that will rarely come up, and there's plenty of other asteroids I can shoot, and not just ones coming off of a mine blast.
// WON'T FIX: Remove unnecessary class attributes such as ship thrust range, asteroid mass, to speed up class creation and copying
// DONE: Differentiate between my mine and adversary's mine, to know whether to shoot size 1's or not
// TODO: Mine FIS currently doesn't take into account if an asteroid will ALREADY get hit by a mine, and drop another one anyway
// DONE: When validating a sim is good when there's another ship, make sure the shots hit! The other ship might have shot those asteroids already.
// DONE: Try a wider random search for maneuvers
// KINDA DONE: GA to beat the random maneuver search, and narrow down the search space. In crowded areas, don't cruise for as long, for example!
// TRIED, not faster: Add per-timestep velocities to asteroids and bullets and stuff to save a multiplication
// TODO: Revisit the aimbot and improve things more
// NO NEED TO FIX because Kessler changed it so that it'll wait out the bullet, as long as there's still time remaining: If we're gonna die and we're the only ship left, don't shoot a bullet if it doesn't land before I die, because it'll count as a miss
// EHH WON'T MAKE THIS CHANGE BECAUSE IT'S KINDA BAD AND COMPLEX: Use math to see how the bullet lines up with the asteroid, to predict whether it's gonna hit before doing the bullet sim
// DONE: get_next_extrapolated_asteroid_collision_time doesn't handle the edges properly! Collisions can't go through the edge but this can predict that. Do the checks!
// DONE: Improve targeting during maneuvers to maintain random maneuvers. Each sim should target differently, not just always aim at the closest asteroid! But we need to maintain cohesion in the targets from one iteration to the next, so Neo doesn't switch targets randomly within the sim.
// PROBABLY NOT WORTH ADDING: Add corner camping hardcoded logic
// PROBABLY NOT WORTH ADDING: Add closing ring freezing
// TODO: If nearing the end of a scenario, it might be worth getting closer to asteroids so the bullets take less time to shoot them, and I might be able to get an extra point or two before I start withholding shots


// POST-XFC 2024 IMPROVEMENT IDEAS:
// DONE (I THINK): Make it so that if the other ship steals my shots, I realize sooner and avoid shooting.
// Currently Neo takes up to 2 planning periods to realize, but if I do a second pass check of the planned actions and make sure all shots land,
// then I can reduce this down to up to 1 planning period of delay. Neo can be only 96% accurate with a good adversary, so hopefully this can be bumped to like 98%.
// This is VERY important in scenarios with a bullet limit since a missed shot is a missed point

// DONE: Inspired by OMUlettes taking out the Fuzzifiers by crashing into them, I want to implement the following hard-coded logic:
// if the other ship is on its last life and I have at least 2 lives:
//     SLAM INTO THE OTHER SHIP AND TAKE THEIR LAST LIFE

// TODO: Improve avoiding shooting size-1 asteroids within the blast radius of my own mine.
// Currently it avoids targeting these, but it could still accidentally hit such asteroids in the way of their intended target

// DONE: Improve handling low bullet limits. Currently Neo just kinda chills and doesn't use its remaining mines effectively.
// Improving its behavior here can help get a bit more score. Even sacrificing lives to get a couple more hits could be a good strat.

// TODO: Ration mines better. Some scenarios Neo uses them too sparingly, and sometimes it dumps them all at the start, and doesn't have any left to use.
// The former is a larger issue. If there's only 3 seconds left, Neo can dump a mine and it could get a few more hits at basically zero cost
// The rationing issue might be solvable by scaling up and down what is considered a good number of asteroids within a blast radius. Currently these are hardcoded.

// DONE: Print out a build date at the start of each run, so during the competition I can make sure the correct version of my controller is run.

// WON'T FIX: Remove my training wheels artificial limitation of placing mines 3 seconds apart.
// I can place them as low as 1 second apart, so removing this might add more strategic options.
// But this will also give Neo more opportunities to bomb itself, so this is hard to implement well.

// TODO: Consider the time limit better, and adapt my strategy based on that. Currently it's not considered (other than avoiding shooting a bullet that won't land before the time's up).
// For example, if there's a very long time limit and I have unlimited bullets, then I probably want to focus on killing the other ship first.
// Otherwise if there's a short time limit, I shouldn't waste time killing the other ship because even if I do,
// I don't have time to hit all the asteroids myself, and it's better to just shoot asteroids and trust that I'm gaining score quicker than the other team is able to gain score,
// or at least I'm no worse than the other team

// DONE: Investigate why Neo just stayed on top of the mine near the end of the scenario for the closing double rings scenario

// DONE: Don't place a mine if it can't explode before the time runs out

// DONE: Dump mines if it can hit stuff right before the end of the scenario

// DONE: Check for time running out condition to handle stuff like that better

// Standard Library
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <iomanip>

// Third-party Library: pybind11
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

constexpr double pi = std::numbers::pi;

constexpr double inf = std::numeric_limits<double>::infinity();
//constexpr double nan = std::numeric_limits<double>::quiet_NaN();

// Build Info
constexpr const char* BUILD_NUMBER = "2025-06-05 Neo";

// Output Config
constexpr bool DEBUG_MODE = false;
constexpr bool PRINT_EXPLANATIONS = false;
constexpr double EXPLANATION_MESSAGE_SILENCE_INTERVAL_S = 2.0;

// Safety&Performance Flags
constexpr bool STATE_CONSISTENCY_CHECK_AND_RECOVERY = true;
constexpr bool CLEAN_UP_STATE_FOR_SUBSEQUENT_SCENARIO_RUNS = true;
constexpr bool ENABLE_SANITY_CHECKS = false;
constexpr bool PRUNE_SIM_STATE_SEQUENCE = true;
constexpr bool VALIDATE_SIMULATED_KEY_STATES = false;
constexpr bool VALIDATE_ALL_SIMULATED_STATES = false;
constexpr bool VERIFY_AST_TRACKING = false;
constexpr bool RESEED_RNG = false;
constexpr bool ENABLE_UNWRAP_CACHE = false; // This is slightly slower than not using the cache lmao

// Strategic/algorithm switches
constexpr bool CONTINUOUS_LOOKAHEAD_PLANNING = true;
constexpr bool USE_HEURISTIC_MANEUVER = false;
constexpr int64_t END_OF_SCENARIO_DONT_CARE_TIMESTEPS = 8;
constexpr int64_t ADVERSARY_ROTATION_TIMESTEP_FUDGE = 20;
constexpr double UNWRAP_ASTEROID_COLLISION_FORECAST_TIME_HORIZON = 6.0;
constexpr double UNWRAP_ASTEROID_TARGET_SELECTION_TIME_HORIZON = 2.3;

// Asteroid priorities
constexpr std::array<double, 5> ASTEROID_SIZE_SHOT_PRIORITY = {std::numeric_limits<double>::quiet_NaN(), 1, 2, 3, 4};

// Optional weights for fitness function
std::optional<std::array<double, 9>> fitness_function_weights = std::nullopt;

// Mine settings
constexpr int64_t MINE_DROP_COOLDOWN_FUDGE_TS = 61;
constexpr double MINE_ASTEROID_COUNT_FUDGE_DISTANCE = 50.0;
constexpr int64_t MINE_OPPORTUNITY_CHECK_INTERVAL_TS = 10;
constexpr double MINE_OTHER_SHIP_RADIUS_FUDGE = 40.0;
constexpr int64_t MINE_OTHER_SHIP_ASTEROID_COUNT_EQUIVALENT = 10;
constexpr double TARGETING_AIMING_UNDERTURN_ALLOWANCE_DEG = 6.0;

// Fitness Weights (default)
constexpr std::array<double, 9> DEFAULT_FITNESS_WEIGHTS = {0.0, 0.13359801675028146, 0.1488417344765523, 0.0, 0.06974293843076491, 0.20559835937182916, 0.12775194210275548, 0.14357775694291458, 0.17088925192490204};

// Angle cone/culling parameters
constexpr double MANEUVER_CONVENIENT_SHOT_CHECKER_CONE_WIDTH_ANGLE_HALF = 45.0;
// TODO: Make this constexpr by making a constexpr cos/sin function
const double MANEUVER_CONVENIENT_SHOT_CHECKER_CONE_WIDTH_ANGLE_HALF_COSINE = std::cos(MANEUVER_CONVENIENT_SHOT_CHECKER_CONE_WIDTH_ANGLE_HALF * pi / 180.0);

constexpr double MANEUVER_BULLET_SIM_CULLING_CONE_WIDTH_ANGLE_HALF = 60.0;
const double MANEUVER_BULLET_SIM_CULLING_CONE_WIDTH_ANGLE_HALF_COSINE = std::cos(MANEUVER_BULLET_SIM_CULLING_CONE_WIDTH_ANGLE_HALF * pi / 180.0);

constexpr double MAX_CRUISE_TIMESTEPS = 30.0;
constexpr int64_t MANEUVER_TUPLE_LEARNING_ROLLING_AVG_PERIOD = 10;
constexpr int64_t OVERALL_FITNESS_ROLLING_AVERAGE_PERIOD = 5;
constexpr double AIMING_CONE_FITNESS_CONE_WIDTH_HALF = 18.0;
const double AIMING_CONE_FITNESS_CONE_WIDTH_HALF_COSINE = std::cos(AIMING_CONE_FITNESS_CONE_WIDTH_HALF * pi / 180.0);

constexpr int64_t MANEUVER_SIM_DISALLOW_TARGETING_FOR_START_TIMESTEPS_AMOUNT = 10;
constexpr double ASTEROID_AIM_BUFFER_PIXELS = 1.0;
constexpr double COORDINATE_BOUND_CHECK_PADDING = 1.0;
constexpr int SHIP_AVOIDANCE_PADDING = 25;
constexpr double SHIP_AVOIDANCE_SPEED_PADDING_RATIO = 1.0/100.0;
constexpr int64_t PERFORMANCE_CONTROLLER_ROLLING_AVERAGE_FRAME_INTERVAL = 10;
constexpr int64_t RANDOM_WALK_SCHEDULE_LENGTH = 3;
constexpr double PERFORMANCE_CONTROLLER_PUSHING_THE_ENVELOPE_FUDGE_MULTIPLIER = 0.55;
constexpr double MINIMUM_DELTA_TIME_FRACTION_BUDGET = 0.55;
constexpr bool ENABLE_PERFORMANCE_CONTROLLER = false;

constexpr int64_t MAX_RESPAWN_PER_TIMESTEP_SEARCH_ITERATIONS = 10;
constexpr int64_t MAX_MANEUVER_PER_TIMESTEP_SEARCH_ITERATIONS = 10;

// Per-lives/per-fitness LUTs
constexpr std::array<std::array<int64_t, 3>, 10> MIN_RESPAWN_PER_TIMESTEP_SEARCH_ITERATIONS_LUT = {{
    {{80, 55, 14}},
    {{70, 40, 13}},
    {{60, 28, 12}},
    {{50, 26, 11}},
    {{45, 14, 10}},
    {{16, 12, 9}},
    {{15, 11, 8}},
    {{14, 10, 7}},
    {{13, 9, 6}},
    {{12, 8, 5}}
}};

constexpr std::array<std::array<int64_t, 3>, 10> MIN_RESPAWN_PER_PERIOD_SEARCH_ITERATIONS_LUT = {{
    {{100, 90, 44}},
    {{95, 81, 43}},
    {{92, 78, 42}},
    {{90, 73, 41}},
    {{85, 71, 40}},
    {{81, 68, 39}},
    {{79, 66, 38}},
    {{76, 64, 37}},
    {{73, 62, 36}},
    {{70, 60, 35}}
}};

constexpr std::array<std::array<int64_t, 3>, 10> MIN_MANEUVER_PER_TIMESTEP_SEARCH_ITERATIONS_LUT = {{
    {{85, 65, 30}},
    {{65, 52, 25}},
    {{55, 40, 20}},
    {{45, 25, 15}},
    {{25, 12, 9}},
    {{20, 9, 6}},
    {{14, 7, 5}},
    {{8, 5, 4}},
    {{7, 4, 3}},
    {{6, 3, 2}}
}};

constexpr std::array<std::array<int64_t, 3>, 10> MIN_MANEUVER_PER_PERIOD_SEARCH_ITERATIONS_LUT = {{
    {{100, 77, 35}},
    {{77, 62, 29}},
    {{64, 47, 23}},
    {{53, 29, 18}},
    {{29, 14, 11}},
    {{19, 9, 7}},
    {{12, 8, 6}},
    {{8, 6, 5}},
    {{5, 4, 3}},
    {{4, 4, 2}}
}};

constexpr std::array<std::array<int64_t, 3>, 10> MIN_MANEUVER_PER_PERIOD_SEARCH_ITERATIONS_IF_WILL_DIE_LUT = {{
    {{172, 136, 68}},
    {{166, 132, 66}},
    {{160, 128, 64}},
    {{154, 124, 62}},
    {{148, 120, 60}},
    {{142, 116, 58}},
    {{138, 112, 56}},
    {{132, 108, 54}},
    {{126, 104, 52}},
    {{120, 100, 50}}
}};


// State dumping for debug
constexpr bool PLOT_MANEUVER_TRACES = false;
constexpr int64_t PLOT_MANEUVER_MIN_TRACE_FOR_PLOT = 30;
constexpr bool REALITY_STATE_DUMP = false;
constexpr bool SIMULATION_STATE_DUMP = false;
constexpr bool KEY_STATE_DUMP = false;
constexpr bool GAMESTATE_PLOTTING = false;
constexpr bool BULLET_SIM_PLOTTING = false;
constexpr bool NEXT_TARGET_PLOTTING = false;
constexpr bool MANEUVER_SIM_PLOTTING = false;
constexpr double START_GAMESTATE_PLOTTING_AT_SECOND = 0.0;
constexpr double NEW_TARGET_PLOT_PAUSE_TIME_S = 0.5;
constexpr double SLOW_DOWN_GAME_AFTER_SECOND = inf;
constexpr double SLOW_DOWN_GAME_PAUSE_TIME = 2.0;

// Debug settings
constexpr bool ENABLE_BAD_LUCK_EXCEPTION = false;
constexpr double BAD_LUCK_EXCEPTION_PROBABILITY = 0.001;

// Quantities
constexpr double TAD = 0.1;
constexpr double GRAIN = 0.001;
constexpr double EPS = 1e-10;
constexpr int64_t INT_NEG_INF = -1000000;
constexpr int64_t INT_INF = 1000000;
constexpr double RAD_TO_DEG = 180.0/pi;
constexpr double DEG_TO_RAD = pi/180.0;
constexpr double TAU = 2.0*pi;

// Kessler game constants
constexpr int64_t FIRE_COOLDOWN_TS = 3;
constexpr int64_t MINE_COOLDOWN_TS = 30;
constexpr double FPS = 30.0;
constexpr double DELTA_TIME = 1.0/FPS;
constexpr double SHIP_FIRE_TIME = 1.0/10.0; // seconds
constexpr double BULLET_SPEED = 800.0;
constexpr double BULLET_MASS = 1.0;
constexpr double BULLET_LENGTH = 12.0;
constexpr double BULLET_LENGTH_RECIPROCAL = 1.0/BULLET_LENGTH;
constexpr double TWICE_BULLET_LENGTH_RECIPROCAL = 2.0/BULLET_LENGTH;
constexpr double SHIP_MAX_TURN_RATE = 180.0;
constexpr double SHIP_MAX_TURN_RATE_RAD = DEG_TO_RAD*SHIP_MAX_TURN_RATE;
constexpr double SHIP_MAX_TURN_RATE_RAD_RECIPROCAL = 1.0/SHIP_MAX_TURN_RATE_RAD;
constexpr double SHIP_MAX_TURN_RATE_DEG_TS = DELTA_TIME*SHIP_MAX_TURN_RATE;
constexpr double SHIP_MAX_TURN_RATE_RAD_TS = DEG_TO_RAD*SHIP_MAX_TURN_RATE_DEG_TS;
constexpr double SHIP_MAX_THRUST = 480.0;
constexpr double SHIP_DRAG = 80.0;
constexpr double SHIP_MAX_SPEED = 240.0;
constexpr double SHIP_RADIUS = 20.0;
constexpr double SHIP_MASS = 300.0;
const int64_t TIMESTEPS_UNTIL_SHIP_ACHIEVES_MAX_SPEED = static_cast<int64_t>(std::ceil(SHIP_MAX_SPEED/(SHIP_MAX_THRUST - SHIP_DRAG)*FPS));
constexpr double MINE_BLAST_RADIUS = 150.0;
constexpr double MINE_RADIUS = 12.0;
constexpr double MINE_BLAST_PRESSURE = 2000.0;
constexpr double MINE_FUSE_TIME = 3.0;
constexpr double MINE_MASS = 25.0;
// Asteroid radii lookup
constexpr std::array<double, 5> ASTEROID_RADII_LOOKUP = {0, 8, 16, 24, 32};
// Asteroid area lookup (pi * r^2)
constexpr std::array<double, 5> ASTEROID_AREA_LOOKUP = {
    pi * ASTEROID_RADII_LOOKUP[0] * ASTEROID_RADII_LOOKUP[0],
    pi * ASTEROID_RADII_LOOKUP[1] * ASTEROID_RADII_LOOKUP[1],
    pi * ASTEROID_RADII_LOOKUP[2] * ASTEROID_RADII_LOOKUP[2],
    pi * ASTEROID_RADII_LOOKUP[3] * ASTEROID_RADII_LOOKUP[3],
    pi * ASTEROID_RADII_LOOKUP[4] * ASTEROID_RADII_LOOKUP[4]
};
// Asteroid mass lookup (0.25 * pi * (8 * i)^2), expanded without std::pow
constexpr std::array<double, 5> ASTEROID_MASS_LOOKUP = {
    0.25 * pi * (8 * 0) * (8 * 0),
    0.25 * pi * (8 * 1) * (8 * 1),
    0.25 * pi * (8 * 2) * (8 * 2),
    0.25 * pi * (8 * 3) * (8 * 3),
    0.25 * pi * (8 * 4) * (8 * 4)
};
constexpr double RESPAWN_INVINCIBILITY_TIME_S = 3.0;
constexpr std::array<int64_t, 5> ASTEROID_COUNT_LOOKUP = {0, 1, 4, 13, 40};
constexpr double DEGREES_BETWEEN_SHOTS = double(FIRE_COOLDOWN_TS)*SHIP_MAX_TURN_RATE*DELTA_TIME;
constexpr double DEGREES_TURNED_PER_TIMESTEP = SHIP_MAX_TURN_RATE*DELTA_TIME;
constexpr double SHIP_RADIUS_PLUS_SIZE_4_ASTEROID_RADIUS = SHIP_RADIUS + ASTEROID_RADII_LOOKUP[4];
constexpr double TIMESTEPS_IT_TAKES_SHIP_TO_COME_TO_DEAD_STOP_FROM_FULL_SPEED = SHIP_MAX_SPEED / (SHIP_MAX_THRUST + SHIP_DRAG) * FPS;
const double TIMESTEPS_IT_TAKES_SHIP_TO_ACCELERATE_TO_FULL_SPEED_FROM_DEAD_STOP = std::ceil(SHIP_MAX_SPEED / (SHIP_MAX_THRUST - SHIP_DRAG) * FPS);

// FIS Settings
constexpr int64_t ASTEROIDS_HIT_VERY_GOOD = 65;
constexpr int64_t ASTEROIDS_HIT_OKAY_CENTER = 23;

// Dirty globals - reset these if sim re-initialized
std::unordered_map<std::string, int64_t> explanation_messages_with_timestamps;
std::vector<double> abs_cruise_speeds = {SHIP_MAX_SPEED/2.0};
std::vector<int64_t> cruise_timesteps_global_history = {static_cast<int64_t>(std::round(MAX_CRUISE_TIMESTEPS/2))};
std::vector<double> overall_fitness_record;
int64_t total_sim_timesteps = 0;

template <typename T>
void print_vector(const std::vector<T>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << "Index " << i << ": " << vec[i] << "\n";
    }
}

void print_sorted_dict(const std::unordered_map<int64_t, int64_t>& umap) {
    // Use std::map to automatically sort keys
    std::map<int64_t, int64_t> sorted_map(umap.begin(), umap.end());

    std::cout << "{";
    bool first = true;
    for (const auto& pair : sorted_map) {
        if (!first) {
            std::cout << ", ";
        }
        std::cout << pair.first << ": " << pair.second;
        first = false;
    }
    std::cout << "}" << std::endl;
}

// ---------------------------------- CLASSES ----------------------------------
struct Asteroid {
    double x = 0, y = 0, vx = 0, vy = 0;
    int64_t size = 0;
    double mass = 0, radius = 0;
    int64_t timesteps_until_appearance = 0;
    bool alive = true;

    Asteroid() = default;
    Asteroid(double x, double y, double vx, double vy, int64_t size, double mass, double radius, int64_t t = 0)
        : x(x), y(y), vx(vx), vy(vy), size(size), mass(mass), radius(radius), timesteps_until_appearance(t), alive(true) {}

    std::string str() const {
        return "Asteroid(position=(" + std::to_string(x) + ", " + std::to_string(y)
            + "), velocity=(" + std::to_string(vx) + ", " + std::to_string(vy) + "), size=" + std::to_string(size)
            + ", mass=" + std::to_string(mass) + ", radius=" + std::to_string(radius)
            + ", timesteps_until_appearance=" + std::to_string(timesteps_until_appearance)
            + ", alive=" + std::to_string(alive) + ")";
    }
    std::string repr() const { return str(); }
    bool operator==(const Asteroid& other) const {
        //return x == other.x && y == other.y && vx == other.vx && vy == other.vy && size == other.size && mass == other.mass && radius == other.radius && timesteps_until_appearance == other.timesteps_until_appearance;
        // Simplified equality, should hold
        return x == other.x && y == other.y && vx == other.vx && vy == other.vy && size == other.size && timesteps_until_appearance == other.timesteps_until_appearance;
    }
    std::size_t hash() const {
        double combined = x + 0.4266548291679171*y + 0.8164926348982552*vx + 0.8397584399461026*vy;
        double scaled = combined * 1000000000.0;
        return static_cast<std::size_t>(scaled) + static_cast<std::size_t>(size);
    }
    double float_hash() const {
        return x + 0.4266548291679171*y + 0.8164926348982552*vx + 0.8397584399461026*vy;
    }
    int64_t int_hash() const {
        return static_cast<int64_t>(1000000000.0*float_hash());
    }
};

// Unwrap cache
std::unordered_map<int64_t, std::vector<Asteroid>> unwrap_cache;

struct Ship {
    bool is_respawning = false;
    double x = 0, y = 0, vx = 0, vy = 0;
    double speed = 0, heading = 0, mass = 0, radius = 0;
    int64_t id = 0;
    std::string team;
    int64_t lives_remaining = 0, bullets_remaining = 0, mines_remaining = 0;
    bool can_fire = true, can_deploy_mine = true;
    double fire_rate = 0.0, mine_deploy_rate = 0.0;
    std::pair<double, double> thrust_range = {-SHIP_MAX_THRUST, SHIP_MAX_THRUST};
    std::pair<double, double> turn_rate_range = {-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE};
    double max_speed = SHIP_MAX_SPEED, drag = SHIP_DRAG;

    Ship() = default;
    Ship(bool is_respawning, double x, double y, double vx, double vy, double speed, double heading, double mass, double radius,
         int64_t id, std::string team, int64_t lives_remaining, int64_t bullets_remaining, int64_t mines_remaining, bool can_fire, double fire_rate,
         bool can_deploy_mine, double mine_deploy_rate, std::pair<double, double> thrust_range, std::pair<double, double> turn_rate_range,
         double max_speed, double drag)
        : is_respawning(is_respawning), x(x), y(y), vx(vx), vy(vy), speed(speed), heading(heading),
          mass(mass), radius(radius), id(id), team(team), lives_remaining(lives_remaining),
          bullets_remaining(bullets_remaining), mines_remaining(mines_remaining), can_fire(can_fire),
          fire_rate(fire_rate), can_deploy_mine(can_deploy_mine), mine_deploy_rate(mine_deploy_rate),
          thrust_range(thrust_range), turn_rate_range(turn_rate_range),
          max_speed(max_speed), drag(drag) {}
    std::string str() const {
        return "Ship(is_respawning=" + std::to_string(is_respawning) + ", position=(" + std::to_string(x) + ", " + std::to_string(y)
            + "), velocity=(" + std::to_string(vx) + ", " + std::to_string(vy) + "), speed=" + std::to_string(speed)
            + ", heading=" + std::to_string(heading) + ", mass=" + std::to_string(mass) + ", radius=" + std::to_string(radius)
            + ", id=" + std::to_string(id) + ", team=\"" + team + "\", lives_remaining=" + std::to_string(lives_remaining)
            + ", bullets_remaining=" + std::to_string(bullets_remaining) + ", mines_remaining=" + std::to_string(mines_remaining)
            + ", can_fire=" + std::to_string(can_fire) + ", fire_rate=" + std::to_string(fire_rate)
            + ", can_deploy_mine=" + std::to_string(can_deploy_mine) + ", mine_deploy_rate=" + std::to_string(mine_deploy_rate)
            + ", thrust_range=(" + std::to_string(thrust_range.first) + ", " + std::to_string(thrust_range.second) + ")"
            + ", turn_rate_range=(" + std::to_string(turn_rate_range.first) + ", " + std::to_string(turn_rate_range.second) + ")"
            + ", max_speed=" + std::to_string(max_speed) + ", drag=" + std::to_string(drag) + ")";
    }
    std::string repr() const { return str(); }
    bool operator==(const Ship& other) const {
        return is_respawning == other.is_respawning
            && x == other.x && y == other.y && vx == other.vx && vy == other.vy
            && speed == other.speed && heading == other.heading && mass == other.mass && radius == other.radius
            && id == other.id && team == other.team && lives_remaining == other.lives_remaining
            && bullets_remaining == other.bullets_remaining && mines_remaining == other.mines_remaining
            //&& can_fire == other.can_fire
            && fire_rate == other.fire_rate
            //&& can_deploy_mine == other.can_deploy_mine
            && mine_deploy_rate == other.mine_deploy_rate
            && thrust_range == other.thrust_range
            && turn_rate_range == other.turn_rate_range
            && max_speed == other.max_speed && drag == other.drag;
    }
};

struct Mine {
    double x = 0, y = 0, mass = 0, fuse_time = 0, remaining_time = 0;
    bool alive = true;

    Mine() = default;
    Mine(double x, double y, double mass, double fuse_time, double remaining_time)
        : x(x), y(y), mass(mass), fuse_time(fuse_time), remaining_time(remaining_time), alive(true) {}
    std::string str() const {
        return "Mine(position=(" + std::to_string(x) + ", " + std::to_string(y)
            + "), mass=" + std::to_string(mass) + ", fuse_time=" + std::to_string(fuse_time)
            + ", remaining_time=" + std::to_string(remaining_time) + ")";
    }
    std::string repr() const { return str(); }
    bool operator==(const Mine& other) const {
        //return x == other.x && y == other.y && mass == other.mass && fuse_time == other.fuse_time && remaining_time == other.remaining_time;
        return x == other.x && y == other.y && remaining_time == other.remaining_time;
    }
};

struct Bullet {
    double x = 0, y = 0, vx = 0, vy = 0, heading = 0, mass = BULLET_MASS, tail_delta_x = 0, tail_delta_y = 0;
    bool alive = true;

    Bullet() = default;
    Bullet(double x, double y, double vx, double vy, double heading, double mass = BULLET_MASS, double tail_delta_x = 0, double tail_delta_y = 0)
        : x(x), y(y), vx(vx), vy(vy), heading(heading), mass(mass), tail_delta_x(tail_delta_x), tail_delta_y(tail_delta_y), alive(true) {}
    std::string str() const {
        return "Bullet(position=(" + std::to_string(x) + ", " + std::to_string(y)
            + "), velocity=(" + std::to_string(vx) + ", " + std::to_string(vy)
            + "), heading=" + std::to_string(heading) + ", mass=" + std::to_string(mass)
            + ", tail_delta=(" + std::to_string(tail_delta_x) + ", " + std::to_string(tail_delta_y) + "))";
    }
    std::string repr() const { return str(); }
    bool operator==(const Bullet& other) const {
        //return x == other.x && y == other.y && vx == other.vx && vy == other.vy && heading == other.heading && mass == other.mass && tail_delta_x == other.tail_delta_x && tail_delta_y == other.tail_delta_y;
        return x == other.x && y == other.y && vx == other.vx && vy == other.vy;
    }
};

struct GameState {
    std::vector<Asteroid> asteroids;
    std::vector<Ship> ships;
    std::vector<Bullet> bullets;
    std::vector<Mine> mines;

    double map_size_x = 0, map_size_y = 0;
    double time = 0, delta_time = 0;
    int64_t sim_frame = 0;
    double time_limit = 0;

    GameState() = default;
    GameState(const std::vector<Asteroid>& asteroids, const std::vector<Ship>& ships,
              const std::vector<Bullet>& bullets, const std::vector<Mine>& mines,
              double map_size_x, double map_size_y, double time, double delta_time,
              int64_t sim_frame, double time_limit)
        : asteroids(asteroids), ships(ships), bullets(bullets), mines(mines),
          map_size_x(map_size_x), map_size_y(map_size_y), time(time), delta_time(delta_time),
          sim_frame(sim_frame), time_limit(time_limit) {}
    /*
    std::string str() const {
        return "GameState(asteroids=" + std::to_string(asteroids.size())
            + ", ships=" + std::to_string(ships.size())
            + ", bullets=" + std::to_string(bullets.size())
            + ", mines=" + std::to_string(mines.size())
            + ", map_size=(" + std::to_string(map_size_x) + ", " + std::to_string(map_size_y)
            + "), time=" + std::to_string(time)
            + ", delta_time=" + std::to_string(delta_time)
            + ", sim_frame=" + std::to_string(sim_frame)
            + ", time_limit=" + std::to_string(time_limit) + ")";
    }*/
    std::string str() const {
        std::string result = "GameState(\n";
        result += "  asteroids=[\n";
        for (const auto& a : asteroids)
            result += "    " + a.str() + ",\n";
        result += "  ],\n  ships=[\n";
        for (const auto& s : ships)
            result += "    " + s.str() + ",\n";
        result += "  ],\n  bullets=[\n";
        for (const auto& b : bullets)
            result += "    " + b.str() + ",\n";
        result += "  ],\n  mines=[\n";
        for (const auto& m : mines)
            result += "    " + m.str() + ",\n";
        result += "  ],\n";
        result += "  map_size=(" + std::to_string(map_size_x) + ", " + std::to_string(map_size_y) + "),\n";
        result += "  time=" + std::to_string(time) + ",\n";
        result += "  delta_time=" + std::to_string(delta_time) + ",\n";
        result += "  sim_frame=" + std::to_string(sim_frame) + ",\n";
        result += "  time_limit=" + std::to_string(time_limit) + "\n";
        result += ")";
        return result;
    }
    std::string repr() const { return str(); }
    GameState copy() const {
        std::vector<Asteroid> alive_asteroids;
        alive_asteroids.reserve(asteroids.size());
        for (const auto& a : asteroids)
            if (a.alive)
                alive_asteroids.push_back(a);

        std::vector<Bullet> alive_bullets;
        alive_bullets.reserve(bullets.size());
        for (const auto& b : bullets)
            if (b.alive)
                alive_bullets.push_back(b);

        std::vector<Mine> alive_mines;
        alive_mines.reserve(mines.size());
        for (const auto& m : mines)
            if (m.alive)
                alive_mines.push_back(m);

        std::vector<Ship> ships_copy;
        ships_copy.reserve(ships.size());
        for (const auto& s : ships)
            ships_copy.push_back(s);

        return GameState(
            std::move(alive_asteroids),
            std::move(ships_copy),
            std::move(alive_bullets),
            std::move(alive_mines),
            map_size_x, map_size_y,
            time, delta_time,
            sim_frame, time_limit
        );
    }
    bool operator==(const GameState& other) const {
        if (asteroids.size() != other.asteroids.size())
            return false;
        for (size_t i = 0; i < asteroids.size(); ++i)
            if (!(asteroids[i] == other.asteroids[i])) return false;
        if (bullets.size() != other.bullets.size())
            return false;
        for (size_t i = 0; i < bullets.size(); ++i)
            if (!(bullets[i] == other.bullets[i])) return false;
        if (mines.size() != other.mines.size())
            return false;
        for (size_t i = 0; i < mines.size(); ++i)
            if (!(mines[i] == other.mines[i])) return false;
        // Ships comparison commented for parity with Python
        return true;
    }
};

struct Target {
    Asteroid asteroid;
    bool feasible = false;
    double shooting_angle_error_deg = 0.0;
    int64_t aiming_timesteps_required = 0;
    double interception_time_s = 0.0;
    double intercept_x = 0.0, intercept_y = 0.0;
    double asteroid_dist_during_interception = 0.0;
    double imminent_collision_time_s = 0.0;
    bool asteroid_will_get_hit_by_my_mine = false, asteroid_will_get_hit_by_their_mine = false;

    Target() = default;
    Target(const Asteroid& asteroid, bool feasible = false, double shooting_angle_error_deg = 0.0, int64_t aiming_timesteps_required = 0, double interception_time_s = 0.0, double intercept_x = 0.0, double intercept_y = 0.0, double asteroid_dist_during_interception = 0.0, double imminent_collision_time_s = 0.0, bool asteroid_will_get_hit_by_my_mine = false, bool asteroid_will_get_hit_by_their_mine = false)
        : asteroid(asteroid), feasible(feasible), shooting_angle_error_deg(shooting_angle_error_deg), aiming_timesteps_required(aiming_timesteps_required), interception_time_s(interception_time_s), intercept_x(intercept_x), intercept_y(intercept_y), asteroid_dist_during_interception(asteroid_dist_during_interception), imminent_collision_time_s(imminent_collision_time_s), asteroid_will_get_hit_by_my_mine(asteroid_will_get_hit_by_my_mine), asteroid_will_get_hit_by_their_mine(asteroid_will_get_hit_by_their_mine) {}

    std::string str() const {
        return "Target(" + asteroid.str() + ", feasible=" + std::to_string(feasible)
            + ", shooting_angle_error_deg=" + std::to_string(shooting_angle_error_deg)
            + ", aiming_timesteps_required=" + std::to_string(aiming_timesteps_required)
            + ", interception_time_s=" + std::to_string(interception_time_s)
            + ", intercept_x=" + std::to_string(intercept_x)
            + ", intercept_y=" + std::to_string(intercept_y)
            + ", asteroid_dist_during_interception=" + std::to_string(asteroid_dist_during_interception)
            + ", imminent_collision_time_s=" + std::to_string(imminent_collision_time_s)
            + ", asteroid_will_get_hit_by_my_mine=" + std::to_string(asteroid_will_get_hit_by_my_mine)
            + ", asteroid_will_get_hit_by_their_mine=" + std::to_string(asteroid_will_get_hit_by_their_mine) + ")";
    }
    std::string repr() const { return str(); }
    //Target copy() const { return *this; }
};

struct Action {
    double thrust = 0.0;
    double turn_rate = 0.0;
    bool fire = false;
    bool drop_mine = false;
    int64_t timestep = 0;

    Action() = default;

    Action(double thrust, double turn_rate, bool fire, bool drop_mine, int64_t timestep)
        : thrust(thrust), turn_rate(turn_rate), fire(fire), drop_mine(drop_mine), timestep(timestep) {
        validate();
    }

    Action(double thrust, double turn_rate, bool fire)
        : thrust(thrust), turn_rate(turn_rate), fire(fire), drop_mine(false), timestep(0) {
        validate();
    }

    Action(double thrust, double turn_rate)
        : thrust(thrust), turn_rate(turn_rate), fire(false), drop_mine(false), timestep(0) {
        validate();
    }

    std::string str() const {
        return "Action(thrust=" + std::to_string(thrust)
            + ", turn_rate=" + std::to_string(turn_rate)
            + ", fire=" + std::to_string(fire)
            + ", drop_mine=" + std::to_string(drop_mine)
            + ", timestep=" + std::to_string(timestep) + ")";
    }

    std::string repr() const {
        return str();
    }

    //Action copy() const {
    //    return *this;
    //}

private:
    void validate() const {
        assert(thrust >= -SHIP_MAX_THRUST && thrust <= SHIP_MAX_THRUST && "Thrust out of bounds");
        assert(turn_rate >= -SHIP_MAX_TURN_RATE && turn_rate <= SHIP_MAX_TURN_RATE && "Turn rate out of bounds");
        assert(timestep >= 0 && "Timestep is negative");
    }
};


struct SimState {
    int64_t timestep = 0;
    Ship ship_state;
    std::optional<GameState> game_state;
    std::optional<std::unordered_map<int64_t, std::vector<Asteroid>>> asteroids_pending_death;
    std::optional<std::vector<Asteroid>> forecasted_asteroid_splits;

    SimState() = default;
    SimState(int64_t timestep, Ship ship_state, std::optional<GameState> game_state = std::nullopt, std::optional<std::unordered_map<int64_t, std::vector<Asteroid>>> asteroids_pending_death = std::nullopt, std::optional<std::vector<Asteroid>> forecasted_asteroid_splits = std::nullopt)
        : timestep(timestep), ship_state(ship_state), game_state(game_state),
          asteroids_pending_death(asteroids_pending_death),
          forecasted_asteroid_splits(forecasted_asteroid_splits) {}

    std::string str() const {
        return "SimState(timestep=" + std::to_string(timestep)
            + ", ship_state=" + ship_state.str()
            + ", game_state=" + (game_state ? game_state->str() : "None")
            + ", asteroids_pending_death=" + (asteroids_pending_death ? "..." : "None")
            + ", forecasted_asteroid_splits=" + (forecasted_asteroid_splits ? "..." : "None") + ")";
    }
    std::string repr() const { return str(); }
    SimState copy() const {
        return SimState(
            timestep,
            ship_state,
            game_state ? std::optional<GameState>{game_state->copy()} : std::nullopt,
            asteroids_pending_death, // Shallow copy; for full deep copy you can implement as needed
            forecasted_asteroid_splits ? std::optional<std::vector<Asteroid>>{std::vector<Asteroid>(forecasted_asteroid_splits->begin(), forecasted_asteroid_splits->end())} : std::nullopt
        );
    }
};

inline std::ostream& operator<<(std::ostream& os, const Asteroid& a) { return os << a.str(); }
inline std::ostream& operator<<(std::ostream& os, const Ship& s) { return os << s.str(); }
inline std::ostream& operator<<(std::ostream& os, const Bullet& b) { return os << b.str(); }
inline std::ostream& operator<<(std::ostream& os, const Mine& m) { return os << m.str(); }
inline std::ostream& operator<<(std::ostream& os, const GameState& gs) { return os << gs.str(); }
inline std::ostream& operator<<(std::ostream& os, const Action& act) { return os << act.str(); }
inline std::ostream& operator<<(std::ostream& os, const SimState& ss) { return os << ss.str(); }
inline std::ostream& operator<<(std::ostream& os, const Target& t) { return os << t.str(); }

// ------------------------------- TYPEDDICT EQUIVALENTS -------------------------------

Asteroid create_asteroid_from_dict(py::dict d) {
    auto pos = d["position"].cast<std::pair<double, double>>();
    auto vel = d["velocity"].cast<std::pair<double, double>>();
    return Asteroid(
        pos.first, pos.second,
        vel.first, vel.second,
        d["size"].cast<int64_t>(),
        d["mass"].cast<double>(),
        d["radius"].cast<double>());
}

Ship create_ship_from_dict(py::dict d) {
    auto pos = d.contains("position") ? d["position"].cast<std::pair<double, double>>() : std::make_pair(0.0, 0.0);
    auto vel = d.contains("velocity") ? d["velocity"].cast<std::pair<double, double>>() : std::make_pair(0.0, 0.0);
    auto thrust_range = d.contains("thrust_range") ? d["thrust_range"].cast<std::pair<double, double>>() : std::make_pair(-SHIP_MAX_THRUST, SHIP_MAX_THRUST);
    auto turn_range = d.contains("turn_rate_range") ? d["turn_rate_range"].cast<std::pair<double, double>>() : std::make_pair(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE);
    return Ship(
        d.contains("is_respawning") ? d["is_respawning"].cast<bool>() : false,
        pos.first, pos.second, vel.first, vel.second,
        d.contains("speed") ? d["speed"].cast<double>() : 0.0,
        d.contains("heading") ? d["heading"].cast<double>() : 0.0,
        d.contains("mass") ? d["mass"].cast<double>() : 0.0,
        d.contains("radius") ? d["radius"].cast<double>() : 0.0,
        d.contains("id") ? d["id"].cast<int64_t>() : 0,
        d.contains("team") ? d["team"].cast<std::string>() : "",
        d.contains("lives_remaining") ? d["lives_remaining"].cast<int64_t>() : 0,
        d.contains("bullets_remaining") ? d["bullets_remaining"].cast<int64_t>() : 0,
        d.contains("mines_remaining") ? d["mines_remaining"].cast<int64_t>() : 0,
        d.contains("can_fire") ? d["can_fire"].cast<bool>() : true,
        d.contains("fire_rate") ? d["fire_rate"].cast<double>() : 10.0,
        d.contains("can_deploy_mine") ? d["can_deploy_mine"].cast<bool>() : true,
        d.contains("mine_deploy_rate") ? d["mine_deploy_rate"].cast<double>() : 1.0,
        thrust_range, turn_range,
        d.contains("max_speed") ? d["max_speed"].cast<double>() : SHIP_MAX_SPEED,
        d.contains("drag") ? d["drag"].cast<double>() : SHIP_DRAG
    );
}

Mine create_mine_from_dict(py::dict d) {
    auto pos = d["position"].cast<std::pair<double, double>>();
    return Mine(pos.first, pos.second, d["mass"].cast<double>(), d["fuse_time"].cast<double>(), d["remaining_time"].cast<double>());
}

Bullet create_bullet_from_dict(py::dict d) {
    auto pos = d["position"].cast<std::pair<double, double>>();
    auto vel = d["velocity"].cast<std::pair<double, double>>();
    double heading = d["heading"].cast<double>();
    double heading_rad = heading*DEG_TO_RAD;
    return Bullet(pos.first, pos.second, vel.first, vel.second, heading, d["mass"].cast<double>(), -BULLET_LENGTH*cos(heading_rad), -BULLET_LENGTH*sin(heading_rad));
}

GameState create_game_state_from_dict(py::dict game_state_dict) {
    // Asteroids
    py::list asteroid_list = game_state_dict["asteroids"].cast<py::list>();
    std::vector<Asteroid> asteroids;
    asteroids.reserve(asteroid_list.size());
    for (auto a : asteroid_list)
        asteroids.push_back(create_asteroid_from_dict(a.cast<py::dict>()));

    // Ships
    py::list ship_list = game_state_dict["ships"].cast<py::list>();
    std::vector<Ship> ships;
    ships.reserve(ship_list.size());
    for (auto s : ship_list)
        ships.push_back(create_ship_from_dict(s.cast<py::dict>()));

    // Bullets
    py::list bullet_list = game_state_dict["bullets"].cast<py::list>();
    std::vector<Bullet> bullets;
    bullets.reserve(bullet_list.size());
    for (auto b : bullet_list)
        bullets.push_back(create_bullet_from_dict(b.cast<py::dict>()));

    // Mines
    py::list mine_list = game_state_dict["mines"].cast<py::list>();
    std::vector<Mine> mines;
    mines.reserve(mine_list.size());
    for (auto m : mine_list)
        mines.push_back(create_mine_from_dict(m.cast<py::dict>()));

    // Construct GameState
    auto map_size = game_state_dict["map_size"].cast<std::pair<double, double>>();
    return GameState(
        asteroids, ships, bullets, mines,
        map_size.first, map_size.second,
        game_state_dict["time"].cast<double>(),
        game_state_dict["delta_time"].cast<double>(),
        game_state_dict["sim_frame"].cast<int64_t>(),
        game_state_dict["time_limit"].cast<double>());
}


struct BasePlanningGameState {
    int64_t timestep;
    bool respawning;
    Ship ship_state;
    GameState game_state;
    double ship_respawn_timer;
    std::unordered_map<int64_t, std::vector<Asteroid>> asteroids_pending_death;
    std::vector<Asteroid> forecasted_asteroid_splits;
    int64_t last_timestep_fired;
    int64_t last_timestep_mined;
    std::set<std::pair<double, double>> mine_positions_placed;
    bool fire_next_timestep_flag;
};


// Thread-safe random (can adjust as needed for your codebase)
//inline static thread_local std::mt19937 rng(std::random_device{}());
inline static thread_local std::mt19937 rng(1);
inline static thread_local std::uniform_real_distribution<> std_uniform(0.0, 1.0);

inline double pymod(double x, double y)
{
    //return std::fmod(std::fmod(x, y) + y, y);
    // Or, equivalently and more numerically stable:
    // double result = std::fmod(x, y);
    // if (result < 0) result += y;
    // return result;
    return x - y * std::floor(x / y);
}

inline void reseed_rng(unsigned int seed) {
    rng.seed(seed);
}

inline double random_double() {
    return std_uniform(rng);
}

// ------ Angle & Math Utilities ------

inline double degrees(double x) {
    // Convert radians to degrees
    return x * RAD_TO_DEG;
}

inline double radians(double x) {
    // Convert degrees to radians
    return x * DEG_TO_RAD;
}

inline double sign(double x) {
    return (x >= 0.0) ? 1.0 : -1.0;
}

inline int64_t randint(int64_t a, int64_t b) {
    // Generate uniform random in [a, b]
    return a + static_cast<int64_t>(std::floor((b - a + 1) * random_double()));
}

inline double rand_uniform(double a, double b) {
    return a + (b - a) * random_double();
}

inline double rand_triangular(double low, double high, double mode) {
    double u = random_double();
    double c = (mode - low) / (high - low);
    if (u < c) {
        return low + std::sqrt(u * (high - low) * (mode - low));
    } else {
        return high - std::sqrt((1.0 - u) * (high - low) * (high - mode));
    }
}

inline double dist(double p1x, double p1y, double p2x, double p2y) {
    double dx = p1x - p2x;
    double dy = p1y - p2y;
    return std::sqrt(dx * dx + dy * dy);
}

inline bool is_close(double x, double y) {
    return std::abs(x - y) <= EPS;
}

inline bool is_kinda_close(double x, double y) {
    return std::abs(x - y) <= GRAIN;
}

inline bool is_close_to_zero(double x) {
    return std::abs(x) <= EPS;
}

inline bool is_kinda_close_to_zero(double x) {
    return std::abs(x) <= GRAIN;
}

// ------ Fast and SuperFast Trig Functions ------

inline double super_fast_acos(double x) {
    return (-0.69813170079773212 * x * x - 0.87266462599716477) * x + 1.5707963267948966;
}

inline double fast_acos(double x) {
    double negate = static_cast<double>(x < 0);
    x = std::abs(x);
    double ret = (((-0.0187293 * x + 0.0742610) * x - 0.2121144) * x + 1.5707288) * std::sqrt(1.0 - x);
    return negate * pi + ret - 2.0 * negate * ret;
}


inline double super_fast_asin(double x) {
    double x_square = x * x;
    return x * (0.9678828 + x_square * (0.8698691 - x_square * (2.166373 - x_square * 1.848968)));
}


inline double fast_asin(double x) {
    double negate = static_cast<double>(x < 0);
    x = std::abs(x);
    double ret = (((-0.0187293 * x + 0.0742610) * x - 0.2121144) * x + 1.5707288);
    ret = 0.5 * pi - std::sqrt(1.0 - x) * ret;
    return ret - 2.0 * negate*ret;
}

inline double super_fast_atan2(double y, double x) {
    // Handle edge cases for 0 inputs
    if (x == 0.0) {
        if (y == 0.0) {
            return 0.0; // atan2(0, 0) is undefined, return 0 for simplicity
        } else {
            return (y > 0.0 ? 0.5 * pi : -0.5 * pi);
        }
    }
    if (y == 0.0) {
        if (x > 0.0) {
            return 0.0;
        } else {
            return pi;
        }
    }
    bool swap = false;
    double atan_input;
    if (std::abs(x) < std::abs(y)) {
        swap = true;
        atan_input = x / y;
    } else {
        swap = false;
        atan_input = y / x;
    }
    double x_sq = atan_input * atan_input;
    double atan_result = atan_input * (0.995354 - x_sq * (0.288679 - 0.079331 * x_sq));
    if (swap) {
        if (atan_input >= 0.0) {
            atan_result = 0.5 * pi - atan_result;
        } else {
            atan_result = -0.5 * pi - atan_result;
        }
    }
    if (x < 0.0) {
        if (y >= 0.0) {
            atan_result += pi;
        } else {
            atan_result += -pi;
        }
    }
    return atan_result;
}

inline double fast_atan2(double y, double x) {
    // Handle edge cases for 0 inputs
    if (x == 0.0) {
        if (y == 0.0) {
            return 0.0; // atan2(0, 0) is undefined, return 0
        } else {
            return (y > 0.0 ? 0.5 * pi : -0.5 * pi);
        }
    }
    if (y == 0.0) {
        if (x > 0.0) {
            return 0.0;
        } else {
            return pi;
        }
    }
    bool swap = false;
    double atan_input;
    if (std::abs(x) < std::abs(y)) {
        swap = true;
        atan_input = x / y;
    } else {
        swap = false;
        atan_input = y / x;
    }
    double x_sq = atan_input * atan_input;
    double atan_result = atan_input * (0.99997726 - x_sq * (0.33262347 - x_sq * (0.19354346 - x_sq * (0.11643287 - x_sq * (0.05265332 - x_sq * 0.01172120)))));
    if (swap) {
        if (atan_input >= 0.0) {
            atan_result = 0.5 * pi - atan_result;
        } else {
            atan_result = -0.5 * pi - atan_result;
        }
    }
    if (x < 0.0) {
        if (y >= 0.0) {
            atan_result += pi;
        } else {
            atan_result += -pi;
        }
    }
    return atan_result;
}

// Returns true if the absolute heading difference between vector "a" (angle in radians) and
// vector "b" (x, y components) is less than or equal to the arccos(cos_threshold).
inline bool heading_diff_within_threshold(double a_vec_theta_rad, double b_vec_x, double b_vec_y, double cos_threshold)
{
    // a_vec_theta_rad: Heading angle (radians)
    // b_vec_x, b_vec_y: Direction vector to compare against
    // cos_threshold: cosine(angle threshold)
    // This avoids explicit angle math and uses only dot and magnitude with cos threshold.

    double a_vec_x = std::cos(a_vec_theta_rad);
    double a_vec_y = std::sin(a_vec_theta_rad);
    double dot_product = a_vec_x * b_vec_x + a_vec_y * b_vec_y;
    double magnitude = std::sqrt(b_vec_x * b_vec_x + b_vec_y * b_vec_y);
    if (magnitude != 0.0) {
        double cos_theta = dot_product / magnitude;
        return cos_theta >= cos_threshold;
    } else {
        // Zero magnitude means the "other" vector has no direction; treat as always "within"
        return true;
    }
}

inline int64_t get_min_respawn_per_timestep_search_iterations(int64_t lives, double average_fitness) {
    assert(0.0 <= average_fitness && average_fitness < 1.0);
    size_t lives_lookup_index = static_cast<size_t>(std::min<int64_t>(3, lives));
    size_t fitness_lookup_index = static_cast<size_t>(std::floor(average_fitness * 10.0));
    return MIN_RESPAWN_PER_TIMESTEP_SEARCH_ITERATIONS_LUT.at(fitness_lookup_index).at(lives_lookup_index - 1);
}

inline int64_t get_min_respawn_per_period_search_iterations(int64_t lives, double average_fitness) {
    assert(0.0 <= average_fitness && average_fitness < 1.0);
    size_t lives_lookup_index = static_cast<size_t>(std::min<int64_t>(3, lives));
    size_t fitness_lookup_index = static_cast<size_t>(std::floor(average_fitness * 10.0));
    return MIN_RESPAWN_PER_PERIOD_SEARCH_ITERATIONS_LUT.at(fitness_lookup_index).at(lives_lookup_index - 1);
}

inline int64_t get_min_maneuver_per_timestep_search_iterations(int64_t lives, double average_fitness) {
    assert(0.0 <= average_fitness && average_fitness < 1.0);
    size_t lives_lookup_index = static_cast<size_t>(std::min<int64_t>(3, lives));
    size_t fitness_lookup_index = static_cast<size_t>(std::floor(average_fitness * 10.0));
    return MIN_MANEUVER_PER_TIMESTEP_SEARCH_ITERATIONS_LUT.at(fitness_lookup_index).at(lives_lookup_index - 1);
}

inline int64_t get_min_maneuver_per_period_search_iterations(int64_t lives, double average_fitness) {
    assert(0.0 <= average_fitness && average_fitness < 1.0);
    size_t lives_lookup_index = static_cast<size_t>(std::min<int64_t>(3, lives));
    size_t fitness_lookup_index = static_cast<size_t>(std::floor(average_fitness * 10.0));
    return MIN_MANEUVER_PER_PERIOD_SEARCH_ITERATIONS_LUT.at(fitness_lookup_index).at(lives_lookup_index - 1);
}

inline int64_t get_min_maneuver_per_period_search_iterations_if_will_die(int64_t lives, double average_fitness) {
    assert(0.0 <= average_fitness && average_fitness < 1.0);
    size_t lives_lookup_index = static_cast<size_t>(std::min<int64_t>(3, lives));
    size_t fitness_lookup_index = static_cast<size_t>(std::floor(average_fitness * 10.0));
    return MIN_MANEUVER_PER_PERIOD_SEARCH_ITERATIONS_IF_WILL_DIE_LUT.at(fitness_lookup_index).at(lives_lookup_index - 1);
}

// Mine FIS stuff is not included

// Forward declarations for helpers/constants assumed as globals or methods somewhere:
// int count_asteroids_in_mine_blast_radius(const GameState&, double x, double y, int timesteps);
// bool mine_fis(int64_t mines_remaining, int64_t lives_remaining, int64_t mine_ast_count);
// const double MINE_BLAST_RADIUS, MINE_OTHER_SHIP_RADIUS_FUDGE;
// const int64_t MINE_OTHER_SHIP_ASTEROID_COUNT_EQUIVALENT;
// const double MINE_FUSE_TIME, FPS;

inline int64_t count_asteroids_in_mine_blast_radius(const GameState& game_state, double mine_x, double mine_y, int64_t future_timesteps) {
    int64_t count = 0;
    for (const Asteroid& a : game_state.asteroids) {
        if (a.alive) {
            // Project asteroid position into future (with correct wrapping)
            double asteroid_future_pos_x = pymod(a.x + static_cast<double>(future_timesteps) * a.vx * DELTA_TIME, game_state.map_size_x);
            double asteroid_future_pos_y = pymod(a.y + static_cast<double>(future_timesteps) * a.vy * DELTA_TIME, game_state.map_size_y);
            // Fast bounding check (no function call)
            double delta_x = asteroid_future_pos_x - mine_x;
            double delta_y = asteroid_future_pos_y - mine_y;
            double separation = a.radius + (MINE_BLAST_RADIUS - MINE_ASTEROID_COUNT_FUDGE_DISTANCE);
            if (std::abs(delta_x) <= separation && std::abs(delta_y) <= separation && delta_x * delta_x + delta_y * delta_y <= separation * separation)
            {
                ++count;
            }
        }
    }
    return count;
}

inline bool mine_fis(int64_t mines_remaining, int64_t lives_remaining, int64_t mine_ast_count) {
    int64_t blah = mines_remaining + lives_remaining + mine_ast_count;
    return true;
}

inline bool check_mine_opportunity(const Ship& ship_state, const GameState& game_state, const std::vector<Ship>& other_ships) {
    // If there's already more than one mine on the field, don't consider laying another
    if (game_state.mines.size() > 1) {
        return false;
    }

    int64_t mine_ast_count = count_asteroids_in_mine_blast_radius(game_state, ship_state.x, ship_state.y, static_cast<int>(std::round(MINE_FUSE_TIME * FPS)));
    int64_t lives_fudge = 0;

    for (const auto& other_ship : other_ships) {
        double delta_x = ship_state.x - other_ship.x;
        double delta_y = ship_state.y - other_ship.y;
        double separation = (MINE_BLAST_RADIUS - MINE_OTHER_SHIP_RADIUS_FUDGE) + other_ship.radius;
        // Fast circular-rect bound test before full circle
        if (std::abs(delta_x) <= separation && std::abs(delta_y) <= separation &&
            (delta_x * delta_x + delta_y * delta_y <= separation * separation))
        {
            // Like bombing the other ship, count as bonus "asteroids"
            mine_ast_count += MINE_OTHER_SHIP_ASTEROID_COUNT_EQUIVALENT;
        }
    }

    if (ship_state.bullets_remaining == 0) {
        // Fudge mine count, encourage mining when out of ammo
        mine_ast_count *= 10;
        if (game_state.mines.size() > 0) {
            // If any mine is already present, avoid wasting a mine
            return false;
        }
        lives_fudge = 2;
    }
    // return value from mine_fis (fuzzy logic function) for opportunity confirmation
    return mine_fis(ship_state.mines_remaining, ship_state.lives_remaining + lives_fudge, mine_ast_count);
}

// Sigmoid
inline double sigmoid(double x, double k=1.0, double x0=0.0) {
    // Logistic sigmoid with scaling and shift
    return 1.0/(1.0 + std::exp(-k*(x - x0)));
}

// Linear interpolation (with clamping)
inline double linear(double x, double x1, double y1, double x2, double y2) {
    assert(x1 < x2);
    if (x <= x1) {
        return y1;
    } else if (x >= x2) {
        return y2;
    } else {
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1);
    }
}

// Weighted average
template<typename T>
double weighted_average(const std::vector<T>& numbers, const std::vector<T>* weights=nullptr) {
    if (weights) {
        if (weights->size() != numbers.size())
            throw std::invalid_argument("Length of weights must match length of numbers.");
        double total_weight = std::accumulate(weights->begin(), weights->end(), 0.0);
        double total_weighted = 0.0;
        for (size_t i = 0; i < numbers.size(); ++i)
            total_weighted += static_cast<double>(numbers[i]) * static_cast<double>((*weights)[i]);
        return total_weight ? (total_weighted/total_weight) : 0.0;
    } else {
        // Regular average
        if (numbers.empty()) return 0.0;
        return std::accumulate(numbers.begin(), numbers.end(), 0.0) / numbers.size();
    }
}

template <typename... Args>
std::vector<double> tuple_to_vector(const std::tuple<Args...>& tpl) {
    std::vector<double> vec;
    vec.reserve(sizeof...(Args));
    std::apply([&vec](const Args&... args) {
        (vec.push_back(args), ...);
    }, tpl);
    return vec;
}

// Weighted harmonic mean
inline double weighted_harmonic_mean(const std::vector<double>& numbers, const std::vector<double>* weights=nullptr, double offset = 0.0) {
    if (numbers.empty()) return 0.0;
    if (weights) {
        if (weights->size() != numbers.size())
            throw std::invalid_argument("Length of weights must match length of numbers.");
        double weight_sum = std::accumulate(weights->begin(), weights->end(), 0.0);
        double weighted_reciprocals_sum = 0.0;
        for (size_t i = 0; i < numbers.size(); ++i)
            weighted_reciprocals_sum += (*weights)[i] / std::max(numbers[i] + offset, TAD);
        double whmean = weight_sum / weighted_reciprocals_sum - offset;
        return whmean;
    } else {
        double weight_sum = static_cast<double>(numbers.size());
        double weighted_reciprocals_sum = 0.0;
        for (double x : numbers)
            weighted_reciprocals_sum += 1.0 / std::max(x + offset, TAD);
        double whmean = weight_sum / weighted_reciprocals_sum - offset;
        return whmean;
    }
}

// print_explanation
inline void print_explanation(const std::string& message, int64_t current_timestep)
{
    if (!PRINT_EXPLANATIONS) return;

    // Search for last print time
    int64_t last_timestep_printed = INT_NEG_INF;
    auto iter = explanation_messages_with_timestamps.find(message);
    if (iter != explanation_messages_with_timestamps.end())
        last_timestep_printed = iter->second;
    // Only print if time window elapsed
    if (current_timestep - last_timestep_printed >= static_cast<int64_t>(EXPLANATION_MESSAGE_SILENCE_INTERVAL_S * FPS)) {
        std::cout << message << std::endl;
        //log_explanation(message, current_timestep);    // Uncomment if logging wanted by default
        explanation_messages_with_timestamps[message] = current_timestep;
    }
}

// log_explanation
/*
inline void log_explanation(const std::string& message, int64_t current_timestep, const std::string& log_file = "Neo Explanations.txt")
{
    try {
        std::ofstream file(log_file, std::ios::app);
        if (!file) throw std::runtime_error("Could not open log file.");
        file << "Timestep " << current_timestep << " - " << message << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred when trying to log explanation: " << e.what() << std::endl;
    }
}*/

// debug_print
template <typename... Args>
inline void debug_print(Args&&... messages)
{
    if constexpr (sizeof...(messages) > 0) {
        if (DEBUG_MODE) {
            ((std::cout << messages << " "), ...) << std::endl;
        }
    }
}

inline std::pair<int64_t, int64_t> asteroid_counter(const std::vector<Asteroid>& asteroids) {
    int64_t current_count = static_cast<int64_t>(asteroids.size());
    int64_t total_count = 0;
    for (const Asteroid& a : asteroids) {
        total_count += ASTEROID_COUNT_LOOKUP.at(a.size);
    }
    return {total_count, current_count};
}

void inspect_scenario(const GameState& game_state, const Ship& ship_state) {
    const auto& asteroids = game_state.asteroids;
    double width = game_state.map_size_x;
    double height = game_state.map_size_y;
    //int64_t asteroids_count, current_count;
    auto [asteroids_count, current_count] = asteroid_counter(asteroids);
    if (current_count == 0) {
        print_explanation("There's no asteroids on the screen! I'm lonely.", 0);
        return;
    }
    print_explanation("The starting field has " + std::to_string(current_count) +
                      " asteroids on the screen, with a total of " + std::to_string(asteroids_count) +
                      " counting splits.", 0);
    print_explanation("At my max shot rate, it'll take " +
                      std::to_string(static_cast<double>(asteroids_count)*SHIP_FIRE_TIME) +
                      " seconds to clear the field.", 0);

    if (ship_state.bullets_remaining == -1) {
        print_explanation("Yay I have unlimited bullets!", 0);
    } else if (ship_state.bullets_remaining == 0) {
        print_explanation("Oh no, I haven't been given any bullets. I'll just hopefully put on a good show and dodge asteroids until the end of time.", 0);
    } else {
        std::ostringstream oss;
        double percent = static_cast<double>(ship_state.bullets_remaining)/std::max(int64_t(1), asteroids_count);
        oss.precision(0);
        oss << std::fixed;
        oss << "Bullets are limited to letting me shoot " << (percent*100.0)
            << "% of the asteroids. If there's another ship, I'll be careful not to let them steal my shots! Otherwise, I'll shoot away!";
        print_explanation(oss.str(), 0);
    }

    // --- Local helper lambdas for statistics (not used, but ported for completeness) ---

    auto asteroid_density = [&]() -> double {
        double total_asteroid_coverage_area = 0.0;
        for (const auto& a : asteroids) {
            // a.size used as index for lookup
            total_asteroid_coverage_area += ASTEROID_AREA_LOOKUP.at(a.size);
        }
        double total_screen_size = width * height;
        if (total_screen_size == 0.0)
            return 0.0;
        else
            return total_asteroid_coverage_area / total_screen_size;
    };

    auto average_velocity = [&]() -> std::pair<double, double> {
        double total_x_velocity = 0.0;
        double total_y_velocity = 0.0;
        for (const Asteroid& a : asteroids) {
            total_x_velocity += a.vx;
            total_y_velocity += a.vy;
        }
        int num_asteroids = static_cast<int>(asteroids.size());
        if (num_asteroids == 0)
            return {0.0, 0.0};
        else
            return {total_x_velocity / num_asteroids, total_y_velocity / num_asteroids};
    };

    auto average_speed = [&]() -> double {
        double total_speed = 0.0;
        for (const Asteroid& a : asteroids) {
            total_speed += std::sqrt(a.vx * a.vx + a.vy * a.vy);
        }
        int num_asteroids = static_cast<int>(asteroids.size());
        if (num_asteroids == 0)
            return 0.0;
        else
            return total_speed / num_asteroids;
    };

    // Uncomment for extra scenario output:
    // double average_density = asteroid_density();
    // auto [curr_ast, tot_ast] = asteroid_counter(asteroids);
    // auto avg_vel = average_velocity();
    // double avg_speed = average_speed();
    // std::cout << "Average asteroid density: " << average_density
    //           << ", average vel: (" << avg_vel.first << ", " << avg_vel.second
    //           << "), average speed: " << avg_speed << std::endl;
}

// Get all ships except self
inline std::vector<Ship> get_other_ships(const GameState& game_state, int64_t self_ship_id) {
    std::vector<Ship> result;
    //result.reserve(game_state.ships.size());
    result.reserve(2);
    for (const auto& ship : game_state.ships) {
        if (ship.id != self_ship_id)
            result.push_back(ship);
    }
    return result;
}

// Angle difference (radians): wraps to [-pi, +pi]
inline double angle_difference_rad(double angle1, double angle2) {
    double diff = std::fmod(angle1 - angle2 + pi, TAU);
    // fmod may return negative, wrap up by tau if so
    if (diff < 0.0) {
        diff += TAU;
    }
    return diff - pi;
}

// Angle difference (degrees): wraps to [-180, +180]
inline double angle_difference_deg(double angle1, double angle2) {
    double diff = std::fmod(angle1 - angle2 + 180.0, 360.0);
    if (diff < 0.0) {
        diff += 360.0;
    }
    return diff - 180.0;
}

inline std::vector<Action>
get_ship_maneuver_move_sequence(double ship_heading_angle, double ship_cruise_speed, double ship_accel_turn_rate, int64_t ship_cruise_timesteps, double ship_cruise_turn_rate, double ship_starting_speed = 0.0) {
    std::vector<Action> move_sequence;
    double ship_speed = ship_starting_speed;

    // --- update helper ---
    auto update = [&](double thrust, double turn_rate) {
        // Apply drag, stop at zero if needed
        double drag_amount = SHIP_DRAG * DELTA_TIME;
        if (drag_amount > std::abs(ship_speed)) {
            ship_speed = 0.0;
        } else {
            ship_speed -= drag_amount * sign(ship_speed);
        }
        // Limit thrust
        thrust = std::clamp(thrust, -SHIP_MAX_THRUST, SHIP_MAX_THRUST);
        // Apply thrust
        ship_speed += thrust * DELTA_TIME;
        // Clamp speed
        if (ship_speed > SHIP_MAX_SPEED)
            ship_speed = SHIP_MAX_SPEED;
        else if (ship_speed < -SHIP_MAX_SPEED)
            ship_speed = -SHIP_MAX_SPEED;

        move_sequence.emplace_back(thrust, turn_rate, false);
    };

    // --- rotate_heading helper ---
    auto rotate_heading = [&](double heading_difference_deg) {
        if (std::abs(heading_difference_deg) < GRAIN)
            return;
        double still_need_to_turn = heading_difference_deg;
        double turn_max = SHIP_MAX_TURN_RATE * DELTA_TIME;
        while (std::abs(still_need_to_turn) > turn_max) {
            // assert(-SHIP_MAX_TURN_RATE <= SHIP_MAX_TURN_RATE*sign(heading_difference_deg) <= SHIP_MAX_TURN_RATE)
            update(0.0, SHIP_MAX_TURN_RATE * sign(heading_difference_deg));
            still_need_to_turn -= SHIP_MAX_TURN_RATE * sign(heading_difference_deg) * DELTA_TIME;
        }
        if (std::abs(still_need_to_turn) > EPS) {
            // assert(-SHIP_MAX_TURN_RATE <= still_need_to_turn*FPS <= SHIP_MAX_TURN_RATE)
            update(0.0, still_need_to_turn * FPS);
        }
    };

    // --- accelerate helper ---
    auto accelerate = [&](double target_speed, double turn_rate) {
        while (std::abs(target_speed - ship_speed) > EPS) {
            double drag = -SHIP_DRAG * sign(ship_speed);
            double drag_amount = SHIP_DRAG * DELTA_TIME;
            if (drag_amount > std::abs(ship_speed)) {
                double adjust_drag_by = std::abs((drag_amount - std::abs(ship_speed)) * FPS);
                drag -= adjust_drag_by * sign(drag);
            }
            double delta_speed_to_target = target_speed - ship_speed;
            double thrust_amount = delta_speed_to_target * FPS - drag;
            thrust_amount = std::clamp(thrust_amount, -SHIP_MAX_THRUST, SHIP_MAX_THRUST);
            update(thrust_amount, turn_rate);
        }
    };

    // --- cruise helper ---
    auto cruise = [&](int64_t cruise_timesteps, double cruise_turn_rate) {
        for (int64_t i = 0; i < cruise_timesteps; ++i) {
            update(sign(ship_speed) * SHIP_DRAG, cruise_turn_rate);
        }
    };

    // --- Run maneuver sequence ---
    rotate_heading(ship_heading_angle);
    accelerate(ship_cruise_speed, ship_accel_turn_rate);
    cruise(ship_cruise_timesteps, ship_cruise_turn_rate);
    accelerate(0.0, 0.0);

    // Guarantee at least one action
    if (move_sequence.empty())
        move_sequence.emplace_back(0.0, 0.0, false);

    // --- Special: scan for “bad” move sequence; diagnostic printout ---
    bool flag = false;
    for (const auto& a : move_sequence) {
        if (is_close(a.turn_rate, 81.69680070842742))
            flag = true;
    }
    if (is_close(ship_accel_turn_rate, 81.69680070842742))
        flag = true;
    if (flag) {
        std::cout << "\nFOUND THE BAD MOVE SEQUENCE doing rotate to ship_heading_angle=" << ship_heading_angle
                << " accelerate to ship_cruise_speed=" << ship_cruise_speed
                << " at turn rate of ship_accel_turn_rate=" << ship_accel_turn_rate
                << " and cruise for ship_cruise_timesteps=" << ship_cruise_timesteps
                << " at tr of ship_cruise_turn_rate=" << ship_cruise_turn_rate << '\n';
        for (const auto& a : move_sequence) {
            std::cout << "Action(thrust=" << a.thrust << ", turn_rate=" << a.turn_rate << ", fire=" << a.fire << ")\n";
        }
        std::cout << std::endl;
    }

    return move_sequence;
}

// =============== 2. calculate_border_crossings ========================
// Returns a vector of (universe_x, universe_y) (int,int) pairs in order
inline std::vector<std::pair<int64_t, int64_t>> calculate_border_crossings(
    double pos_x, double pos_y, double vel_x, double vel_y,
    double width, double height, double time_horizon)
{
    std::vector<double> x_crossings_times;
    std::vector<double> y_crossings_times;
    int x_crossings = 0, y_crossings = 0;
    double abs_vx = std::abs(vel_x);
    if (abs_vx > EPS) {
        double x_crossing_interval = width / abs_vx;
        double time_to_first_x_crossing = (vel_x > 0.0) ? (width - pos_x) / vel_x : -pos_x / vel_x;
        x_crossings_times.push_back(time_to_first_x_crossing);
        ++x_crossings;
        while ((x_crossings_times.back() + x_crossing_interval) <= time_horizon) {
            x_crossings_times.push_back(x_crossings_times.back() + x_crossing_interval);
            ++x_crossings;
        }
    }
    double abs_vy = std::abs(vel_y);
    if (abs_vy > EPS) {
        double y_crossing_interval = height / abs_vy;
        double time_to_first_y_crossing = (vel_y > 0.0) ? (height - pos_y) / vel_y : -pos_y / vel_y;
        y_crossings_times.push_back(time_to_first_y_crossing);
        ++y_crossings;
        while ((y_crossings_times.back() + y_crossing_interval) <= time_horizon) {
            y_crossings_times.push_back(y_crossings_times.back() + y_crossing_interval);
            ++y_crossings;
        }
    }
    // Merge x/y crossing times into a sequence of which border (true: x, false: y)
    std::vector<bool> border_crossing_sequence;
    int i = 0, j = 0;
    while (i < x_crossings && j < y_crossings) {
        if (x_crossings_times[i] < y_crossings_times[j]) {
            border_crossing_sequence.push_back(true);
            ++i;
        } else {
            border_crossing_sequence.push_back(false);
            ++j;
        }
    }
    while (i < x_crossings) { border_crossing_sequence.push_back(true); ++i; }
    while (j < y_crossings) { border_crossing_sequence.push_back(false); ++j; }

    // Now step through the sequence, tracking which universes we visit
    int64_t current_universe_x = 0, current_universe_y = 0;
    int64_t universe_increment_direction_x = (vel_x > 0.0) ? 1 : -1;
    int64_t universe_increment_direction_y = (vel_y > 0.0) ? 1 : -1;
    std::vector<std::pair<int64_t, int64_t>> universes;
    for (bool crossing : border_crossing_sequence) {
        if (crossing) {
            current_universe_x += universe_increment_direction_x;
        } else {
            current_universe_y += universe_increment_direction_y;
        }
        universes.emplace_back(current_universe_x, current_universe_y);
    }
    return universes;
}

inline bool coordinates_in_same_wrap(const double &pos1x, const double &pos1y, const double &pos2x, const double &pos2y, const double &map_size_x, const double &map_size_y) {
    // Checks whether the coordinates are in the same universe
    // Cast to int to mimic Python's floor division, since double//double in Python is floor division
    // TODO: Honestly this is kinda sketch and using modulo is probably more accurate.
    double x_wrap1 = std::floor(pos1x / map_size_x);
    double x_wrap2 = std::floor(pos2x / map_size_x);
    if (x_wrap1 != x_wrap2) {
        return false;
    }
    double y_wrap1 = std::floor(pos1y / map_size_y);
    double y_wrap2 = std::floor(pos2y / map_size_y);
    return (y_wrap1 == y_wrap2);
}

// ============= unwrap_asteroid =====================
// Use copy constructor and int_hash (see Asteroid definition).
inline std::vector<Asteroid> unwrap_asteroid(const Asteroid& asteroid, double max_x, double max_y, double time_horizon_s = 10.0, bool use_cache = true) {
    // Compute hash
    int64_t ast_hash;
    if constexpr (ENABLE_UNWRAP_CACHE) {
        ast_hash = asteroid.int_hash();
        if (use_cache) {
            auto it = unwrap_cache.find(ast_hash);
            if (it != unwrap_cache.end())
                // CACHE HIT!
                return it->second;
        }
    }
    // Gotta calculate it. Not in the cache.
    /*
    thread_local static std::vector<Asteroid> unwrapped_asteroids = [] {
        std::vector<Asteroid> a;
        a.reserve(4);
        return a;
    }();
    unwrapped_asteroids.clear();*/
    std::vector<Asteroid> unwrapped_asteroids;
    //unwrapped_asteroids.reserve(3);
    unwrapped_asteroids.push_back(asteroid);
    if (is_close_to_zero(asteroid.vx) && is_close_to_zero(asteroid.vy)) {
        // An asteroid that is stationary will never move across borders and wrap
        if constexpr (ENABLE_UNWRAP_CACHE) {
            if (use_cache) {
                unwrap_cache[ast_hash] = unwrapped_asteroids; // Cache this
            }
        }
        return unwrapped_asteroids;
    }
    /*
    if (coordinates_in_same_wrap(asteroid.x, asteroid.y, asteroid.x + asteroid.vx * time_horizon_s, asteroid.y + asteroid.vy * time_horizon_s, max_x, max_y)) {
        // After the asteroid travels the time horizon, it's still in the same wrap! So we can just return the one asteroid lol
        if constexpr (ENABLE_UNWRAP_CACHE) {
            if (use_cache) {
                unwrap_cache[ast_hash] = unwrapped_asteroids;
            }
        }
        return unwrapped_asteroids;
    }*/
    // Find where the asteroid ends up, universe-wise after the time horizon is up, and use this to short-circuit some computation without having to do the full mathy stuff
    if constexpr (ENABLE_SANITY_CHECKS) {
        double x_wrap1 = std::floor(asteroid.x / max_x);
        double y_wrap1 = std::floor(asteroid.y / max_y);
        assert(x_wrap1 == 0.0);
        assert(y_wrap1 == 0.0);
    }
    double x_wrap2 = std::floor((asteroid.x + asteroid.vx * time_horizon_s) / max_x);
    double y_wrap2 = std::floor((asteroid.y + asteroid.vy * time_horizon_s) / max_y);

    //std::cout << x_wrap1 << y_wrap1 << x_wrap2 << y_wrap2 << std::endl;
    if (x_wrap2 == 0.0) {
        // Asteroid does not wrap horizontally
        if (y_wrap2 == 0.0) {
            // Does not wrap vertically either
            // After the time horizon passes, the asteroid is still in the same wrap! So no unwrapped asteroids appear
            return unwrapped_asteroids;
        } else {
            // The asteroid wraps around vertically but not horizontally
            if (y_wrap2 == -1.0) {
                unwrapped_asteroids.emplace_back(
                    asteroid.x,
                    asteroid.y + max_y,
                    asteroid.vx,
                    asteroid.vy,
                    asteroid.size,
                    asteroid.mass,
                    asteroid.radius,
                    asteroid.timesteps_until_appearance
                );
            } else if (y_wrap2 == 1.0) {
                unwrapped_asteroids.emplace_back(
                    asteroid.x,
                    asteroid.y - max_y,
                    asteroid.vx,
                    asteroid.vy,
                    asteroid.size,
                    asteroid.mass,
                    asteroid.radius,
                    asteroid.timesteps_until_appearance
                );
            }
        }
    } else if (y_wrap2 == 0.0) {
        // The asteroid only wraps around horizontally
        if (x_wrap2 == -1.0) {
            unwrapped_asteroids.emplace_back(
                asteroid.x + max_x,
                asteroid.y,
                asteroid.vx,
                asteroid.vy,
                asteroid.size,
                asteroid.mass,
                asteroid.radius,
                asteroid.timesteps_until_appearance
            );
        } else if (x_wrap2 == 1.0) {
            unwrapped_asteroids.emplace_back(
                asteroid.x - max_x,
                asteroid.y,
                asteroid.vx,
                asteroid.vy,
                asteroid.size,
                asteroid.mass,
                asteroid.radius,
                asteroid.timesteps_until_appearance
            );
        }
    }
    // The asteroid wraps both horizontally AND vertically, so we do some complicated math to figure out exactly in what sequence it does the wraps

    for (const auto& universe : calculate_border_crossings(asteroid.x, asteroid.y, asteroid.vx, asteroid.vy, max_x, max_y, time_horizon_s)) {
        // We move the asteroid the opposite direction virtually, from the direction it actually went! Hence the negative signs.
        double dx = -static_cast<double>(universe.first) * max_x;
        double dy = -static_cast<double>(universe.second) * max_y;
        unwrapped_asteroids.emplace_back(
            asteroid.x + dx,
            asteroid.y + dy,
            asteroid.vx,
            asteroid.vy,
            asteroid.size,
            asteroid.mass,
            asteroid.radius,
            asteroid.timesteps_until_appearance
        );
    }
    if constexpr (ENABLE_UNWRAP_CACHE) {
        if (use_cache) {
            unwrap_cache[ast_hash] = unwrapped_asteroids;
        }
    }
    return unwrapped_asteroids;
}

// --- check_coordinate_bounds ---
inline bool check_coordinate_bounds(const GameState& game_state, double x, double y) {
    return (0.0 <= x && x <= game_state.map_size_x && 0.0 <= y && y <= game_state.map_size_y);
}

// --- check_coordinate_bounds_exact ---
inline bool check_coordinate_bounds_exact(const GameState& game_state, double x, double y) {
    double x_wrapped = pymod(x, game_state.map_size_x);
    double y_wrapped = pymod(y, game_state.map_size_y);
    if (is_close(x, x_wrapped) && is_close(y, y_wrapped))
        return true;
    else
        return false;
}

// --- solve_quadratic ---
// Solves a*x^2 + b*x + c = 0 for real roots. Returns (x1, x2) or (nan, nan) if no real solution.
inline std::pair<double, double> solve_quadratic(double a, double b, double c) {
    if (a == 0.0) {
        // Linear
        if (b == 0.0) {
            if (c == 0.0)
                return {0.0, 0.0};
            else
                return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
        } else {
            double x = -c / b;
            return {x, x};
        }
    }
    double discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) {
        // No real solutions
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    double sqrt_disc = std::sqrt(discriminant);
    double q = -0.5 * (b + std::copysign(sqrt_disc, b));
    if (c == 0.0) {
        double x1 = -b / a;
        if (x1 < 0.0) {
            return {x1, 0.0};
        } else {
            return {0.0, x1};
        }
    }
    // q cannot be 0 here
    double x1 = q / a;
    double x2 = c / q;
    if (x1 <= x2) {
        return {x1, x2};
    } else {
        return {x2, x1};
    }
}

// Returns: {t_enter, t_exit} if potentially colliding, or {nan, nan} if no collision in future.
inline std::pair<double, double> collision_prediction_slow(double ax, double ay, double vax, double vay, double ra, double bx, double by, double vbx, double vby, double rb) {
    double separation = ra + rb;
    double delta_x = ax - bx;
    double delta_y = ay - by;

    if (is_close_to_zero(vax) && is_close_to_zero(vay) && is_close_to_zero(vbx) && is_close_to_zero(vby)) {
        // Both stationary – just check overlap now:
        if (std::abs(delta_x) <= separation && std::abs(delta_y) <= separation && (delta_x * delta_x + delta_y * delta_y <= separation * separation))
        {
            // "collide now and forever"
            return std::make_pair(-inf, inf);
        } 
        else 
        {
            // Never collide
            return std::make_pair(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
        }
    } else {
        // Relative velocity
        double vel_delta_x = vax - vbx;
        double vel_delta_y = vay - vby;
        double a = vel_delta_x * vel_delta_x + vel_delta_y * vel_delta_y;
        double b = 2.0 * (delta_x * vel_delta_x + delta_y * vel_delta_y);
        double c = delta_x * delta_x + delta_y * delta_y - separation * separation;
        return solve_quadratic(a, b, c);
    }
}

// Returns: {t_enter, t_exit} if potentially colliding, or {nan, nan} if no collision in future. 
inline std::pair<double, double> collision_prediction(
    double ax, double ay, double vax, double vay, double ra,
    double bx, double by, double vbx, double vby, double rb
) {
    // Super fast geometric way to do this using trig
    // The derivation's a bit wacky but just trust me on this one that this is equivalent
    // This is about 1.4X the speed of the older version of this function that requires solving a quadratic without simplifying or early exiting, benchmarked on a mix of no collisions and collisions test cases
    double separation = ra + rb;

    double dx = ax - bx;
    double dy = ay - by;
    double dvx = vax - vbx;
    double dvy = vay - vby;

    double dist_sq = dx * dx + dy * dy;
    double speed_sq = dvx * dvx + dvy * dvy;
    double dot = dx * dvx + dy * dvy;
    double sep_sq = separation * separation;

    // Both stationary
    if (is_close_to_zero(speed_sq)) {
        if (dist_sq <= sep_sq) {
            return std::make_pair(-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()); // Overlapping forever
        } else {
            return std::make_pair(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()); // Never collide
        }
    }

    if (dot >= 0.0 && dist_sq > sep_sq) {
        return std::make_pair(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()); // Moving apart or tangent
    }

    double cos_theta_sq = (dot * dot) / (dist_sq * speed_sq);
    double sin_theta_sq = 1.0 - cos_theta_sq;
    double min_sin_sq = sep_sq / dist_sq;

    if (sin_theta_sq > min_sin_sq) {
        return std::make_pair(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()); // Will miss each other
    }

    double root_term = std::sqrt((sep_sq - dist_sq * sin_theta_sq) / speed_sq);
    double t_mid = -dot / speed_sq;

    double t_enter = t_mid - root_term;
    double t_exit  = t_mid + root_term;
    return std::make_pair(t_enter, t_exit);
}


// === 1. find_time_interval_in_which_unwrapped_asteroid_is_within_main_wrap ===
inline std::pair<double, double>
find_time_interval_in_which_unwrapped_asteroid_is_within_main_wrap(double ast_pos_x, double ast_pos_y, double ast_vel_x, double ast_vel_y, const GameState& game_state) {
    std::pair<double, double> x_interval, y_interval;

    if (is_close_to_zero(ast_vel_x)) {
        if (check_coordinate_bounds(game_state, ast_pos_x, 0.0)) {
            x_interval = {-inf, inf};
        } else {
            return { std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN() };
        }
    } else {
        if (ast_vel_x > 0.0) {
            x_interval = {-ast_pos_x / ast_vel_x, (game_state.map_size_x - ast_pos_x) / ast_vel_x};
        } else {
            x_interval = {(game_state.map_size_x - ast_pos_x) / ast_vel_x, -ast_pos_x / ast_vel_x};
        }
    }

    if (is_close_to_zero(ast_vel_y)) {
        if (check_coordinate_bounds(game_state, 0.0, ast_pos_y)) {
            y_interval = { -inf, inf };
        } else {
            return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
        }
    } else {
        if (ast_vel_y > 0.0) {
            y_interval = {-ast_pos_y / ast_vel_y, (game_state.map_size_y - ast_pos_y) / ast_vel_y};
        } else {
            y_interval = {(game_state.map_size_y - ast_pos_y) / ast_vel_y, -ast_pos_y / ast_vel_y};
        }
    }

    assert(x_interval.first <= x_interval.second);
    assert(y_interval.first <= y_interval.second);

    // Take the intersection of intervals
    if (x_interval.second < y_interval.first || y_interval.second < x_interval.first) {
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    } else {
        double start = std::max(x_interval.first, y_interval.first);
        double end = std::min(x_interval.second, y_interval.second);
        return {start, end};
    }
}

inline double predict_next_imminent_collision_time_with_asteroid(
    double ship_pos_x, double ship_pos_y, double ship_vel_x, double ship_vel_y, double ship_r,
    double ast_pos_x, double ast_pos_y, double ast_vel_x, double ast_vel_y, double ast_radius,
    const GameState& game_state)
{
    assert(is_close_to_zero(ship_vel_x) && is_close_to_zero(ship_vel_y));
    assert(check_coordinate_bounds(game_state, ship_pos_x, ship_pos_y));

    auto [start_collision_time, end_collision_time] = collision_prediction(
        ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_r,
        ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius
    );

    // Optional sanity check
    if constexpr (ENABLE_SANITY_CHECKS) {
        auto [start_old, end_old] = collision_prediction_slow(
            ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_r,
            ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius
        );

        auto check_equal = [](double a, double b) -> bool {
            if (std::isnan(a) && std::isnan(b)) return true;
            if (std::isinf(a) && std::isinf(b)) return (std::signbit(a) == std::signbit(b));
            return is_kinda_close(a, b);
        };
        
        if (end_old < 0.0) {
            start_old = std::numeric_limits<double>::quiet_NaN();
            end_old = std::numeric_limits<double>::quiet_NaN();
        }

        if (!check_equal(start_collision_time, start_old) || !check_equal(end_collision_time, end_old)) {
            std::cerr << "Sanity check failed!\n";
            std::cerr << std::setprecision(15) << "New: [" << start_collision_time << ", " << end_collision_time << "]\n";
            std::cerr << std::setprecision(15) << "Old: [" << start_old << ", " << end_old << "]\n";
            assert(false && "collision_prediction mismatch");
        }
    }

    if (std::isnan(start_collision_time) || std::isnan(end_collision_time)) {
        return inf;
    } else {
        assert(start_collision_time <= end_collision_time);
        if (!(SHIP_RADIUS_PLUS_SIZE_4_ASTEROID_RADIUS < ship_pos_x && ship_pos_x < game_state.map_size_x - SHIP_RADIUS_PLUS_SIZE_4_ASTEROID_RADIUS
           && SHIP_RADIUS_PLUS_SIZE_4_ASTEROID_RADIUS < ship_pos_y && ship_pos_y < game_state.map_size_y - SHIP_RADIUS_PLUS_SIZE_4_ASTEROID_RADIUS))
        {
            // Asteroid could be outside main area--clip collision time to existence in main wrap
            auto [t1, t2] = find_time_interval_in_which_unwrapped_asteroid_is_within_main_wrap(
                ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, game_state
            );
            // Intersect collision interval with asteroid-in-world interval
            if (end_collision_time < t1 || start_collision_time > t2) {
                return inf;
            } else {
                start_collision_time = std::max(start_collision_time, t1);
                end_collision_time = std::min(end_collision_time, t2);
            }
        }
        // Intersection with [0, inf)
        if (end_collision_time < 0.0) {
            return inf;
        } else if (start_collision_time <= 0.0) {
            return 0.0;
        } else {
            return start_collision_time;
        }
    }
}

std::tuple<double, double, double, double, int64_t, double, int64_t, int64_t>
analyze_gamestate_for_heuristic_maneuver(const GameState& game_state, const Ship& ship_state) {
    // This is a helper function to analyze and prepare the gamestate, to give
    // the maneuver FIS useful information, to heuristically command a maneuver to try out

    auto calculate_angular_width = [](double radius, double distance) -> double {
        // From the ship's point of view, find the angular width of an asteroid
        if (distance == 0.0) return TAU;
        double sin_theta = radius / distance;
        if (sin_theta >= -1.0 && sin_theta <= 1.0) {
            return 2.0 * super_fast_asin(sin_theta);
        } else {
            return TAU;
        }
    };

    auto average_velocity = [](const std::vector<Asteroid>& asteroids) -> std::pair<double, double> {
        double total_x_velocity = 0.0;
        double total_y_velocity = 0.0;
        for (const Asteroid& a : asteroids) {
            assert(a.alive);
            total_x_velocity += a.vx;
            total_y_velocity += a.vy;
        }
        size_t num_asteroids = asteroids.size();
        if (num_asteroids == 0)
            return {0.0, 0.0};
        else
            return {total_x_velocity / num_asteroids, total_y_velocity / num_asteroids};
    };

    auto find_largest_gap = [&](const std::vector<Asteroid>& asteroids, std::pair<double, double> ship_position) -> std::pair<double, double> {
        // Find the largest angular gap around the ship, and this is the gap I'll try escaping through
        if (asteroids.empty()) {
            // No asteroids mean the entire space is a gap.
            return {0.0, TAU};
        }
        std::vector<std::pair<double, bool>> angles;
        int64_t initial_cover_count = 0; // Counter for asteroids covering angle 0

        for (const Asteroid& asteroid : asteroids) {
            assert(asteroid.alive);
            double x = asteroid.x - ship_position.first;
            double y = asteroid.y - ship_position.second;
            double distance = std::sqrt(x * x + y * y);
            double angle = std::fmod(super_fast_atan2(y, x), TAU);
            double angular_width = calculate_angular_width(asteroid.radius, distance);
            double start_angle = std::fmod(angle - 0.5 * angular_width + TAU, TAU);
            double end_angle = std::fmod(angle + 0.5 * angular_width + TAU, TAU);

            // Check if this asteroid covers the angle 0 (or equivalently, 2π)
            if (start_angle > end_angle)  // wraps around angle 0
                initial_cover_count++;

            // Add angles in original and offset positions
            // True is for start and False for end
            angles.emplace_back(start_angle, true);
            angles.emplace_back(end_angle, false);
            angles.emplace_back(start_angle + TAU, true);
            angles.emplace_back(end_angle + TAU, false);
        }

        // Sort by angle
        std::sort(angles.begin(), angles.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

        // Initialize counter with the number of asteroids covering angle 0
        int counter = int(initial_cover_count);
        double largest_gap_midpoint = 0.0;
        double largest_gap = 0.0;
        double gap_start = std::numeric_limits<double>::quiet_NaN();

        for (const auto& [angle, marker] : angles) {
            //assert(counter >= 0); TODO reenable this!
            if (marker) {
                // Start
                if (counter == 0 && !std::isnan(gap_start)) {
                    double gap = angle - gap_start;
                    assert(gap >= 0.0);
                    if (gap > largest_gap) {
                        largest_gap = gap;
                        largest_gap_midpoint = std::fmod(0.5 * (gap_start + angle), TAU);
                    }
                }
                counter++;
            } else {
                // End
                counter--;
                if (counter == 0) {
                    gap_start = angle;
                }
            }
        }
        // No need to adjust for wraparound explicitly due to "doubling" the angles list
        return {largest_gap_midpoint, largest_gap};
    };

    // --- Main function body ---
    std::vector<Asteroid> asteroids(game_state.asteroids.begin(), game_state.asteroids.end());

    // Convert other ships to pseudo-asteroids:
    for (const Ship& ship : game_state.ships) {
        if (ship.id != ship_state.id) {
            asteroids.emplace_back(Asteroid{ship.x, ship.y, 0.0, 0.0, 0, 0.0, ship.radius
            });
        }
    }
    double ship_pos_x = ship_state.x, ship_pos_y = ship_state.y, ship_vel_x = ship_state.vx, ship_vel_y = ship_state.vy;
    double most_imminent_collision_time_s = std::numeric_limits<double>::infinity();
    std::optional<Asteroid> most_imminent_asteroid;
    std::optional<double> most_imminent_asteroid_speed;
    double nearby_asteroid_total_speed = 0.0;
    int64_t nearby_asteroid_count = 0;
    double nearby_threshold_square = 40000.0; // 200.0**2
    std::vector<Asteroid> nearby_asteroids;

    for (const Asteroid& asteroid : asteroids) {
        assert(asteroid.alive);
        for (const Asteroid& a : unwrap_asteroid(asteroid, game_state.map_size_x, game_state.map_size_y, UNWRAP_ASTEROID_COLLISION_FORECAST_TIME_HORIZON, false)) {
            double imminent_collision_time_s;
            if (is_close_to_zero(ship_vel_x) && is_close_to_zero(ship_vel_y)) {
                imminent_collision_time_s = predict_next_imminent_collision_time_with_asteroid(
                    ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, SHIP_RADIUS,
                    a.x, a.y, a.vx, a.vy, a.radius, game_state
                );
            } else {
                imminent_collision_time_s = std::numeric_limits<double>::infinity();
            }

            double delta_x = a.x - ship_pos_x;
            double delta_y = a.y - ship_pos_y;
            std::optional<double> asteroid_speed;
            if (delta_x * delta_x + delta_y * delta_y <= nearby_threshold_square) {
                asteroid_speed = std::sqrt(a.vx * a.vx + a.vy * a.vy);
                nearby_asteroid_total_speed += *asteroid_speed;
                ++nearby_asteroid_count;
                nearby_asteroids.push_back(a);
            }

            if (imminent_collision_time_s < most_imminent_collision_time_s) {
                most_imminent_collision_time_s = imminent_collision_time_s;
                most_imminent_asteroid = a;
                if (asteroid_speed.has_value())
                    most_imminent_asteroid_speed = *asteroid_speed;
                else
                    most_imminent_asteroid_speed.reset();
            }
        }
    }

    double most_imminent_asteroid_speed_val = 0.0;
    double imminent_asteroid_relative_heading_deg = 0.0;

    if (!most_imminent_asteroid.has_value()) {
        most_imminent_asteroid_speed_val = 0.0;
        imminent_asteroid_relative_heading_deg = 0.0;
    } else {
        if (!most_imminent_asteroid_speed.has_value()) {
            most_imminent_asteroid_speed_val =
                std::sqrt(most_imminent_asteroid->vx * most_imminent_asteroid->vx + most_imminent_asteroid->vy * most_imminent_asteroid->vy);
        } else {
            most_imminent_asteroid_speed_val = *most_imminent_asteroid_speed;
        }
        imminent_asteroid_relative_heading_deg = degrees(super_fast_atan2(
            most_imminent_asteroid->y - ship_pos_y, most_imminent_asteroid->x - ship_pos_x));
    }

    auto [largest_gap_absolute_heading_rad, _] = find_largest_gap(nearby_asteroids, {ship_pos_x, ship_pos_y});
    double largest_gap_absolute_heading_deg = degrees(largest_gap_absolute_heading_rad);
    double largest_gap_relative_heading_deg = std::fmod(largest_gap_absolute_heading_deg - ship_state.heading + TAU, TAU);
    double nearby_asteroid_average_speed = (nearby_asteroid_count == 0 ? 0.0 : nearby_asteroid_total_speed / nearby_asteroid_count);

    auto average_directional_velocity = average_velocity(asteroids);
    double average_directional_speed = std::sqrt(average_directional_velocity.first * average_directional_velocity.first + average_directional_velocity.second * average_directional_velocity.second);

    int64_t total_asteroid_count, current_asteroids_count;
    std::tie(total_asteroid_count, current_asteroids_count) = asteroid_counter(asteroids);

    return {most_imminent_asteroid_speed_val, imminent_asteroid_relative_heading_deg, largest_gap_relative_heading_deg, nearby_asteroid_average_speed, nearby_asteroid_count, average_directional_speed, total_asteroid_count, current_asteroids_count};
}

inline bool check_collision(double a_x, double a_y, double a_r, double b_x, double b_y, double b_r) {
    double delta_x = a_x - b_x;
    double delta_y = a_y - b_y;
    double separation = a_r + b_r;
    // Fast bounding-box rejection, then distance^2 check
    if (std::abs(delta_x) <= separation &&
        std::abs(delta_y) <= separation &&
        (delta_x * delta_x + delta_y * delta_y <= separation * separation)) {
        return true;
    } else {
        return false;
    }
}

inline std::tuple<
    bool,         // feasible
    double,       // shot_heading_error_rad
    double,       // shot_heading_tolerance_rad
    double,       // interception_time_s
    double,       // intercept_x
    double,       // intercept_y
    double        // asteroid_dist_during_interception
> calculate_interception(
    double ship_pos_x, double ship_pos_y,
    double asteroid_pos_x, double asteroid_pos_y,
    double asteroid_vel_x, double asteroid_vel_y, double asteroid_r,
    double ship_heading_deg,
    const GameState& game_state,
    int64_t future_shooting_timesteps = 0
)
{
    // t_0 = (SHIP_RADIUS - 0.5*BULLET_LENGTH)/BULLET_SPEED
    // Your code uses t_0=0.0175 hardcoded; you can replace with the above line if desired.
    double t_0 = 0.0175;
    double origin_x = ship_pos_x;
    double origin_y = ship_pos_y;
    double avx = asteroid_vel_x, avy = asteroid_vel_y;

    // Project asteroid one timestep ahead.
    double ax = asteroid_pos_x - origin_x + avx * DELTA_TIME;
    double ay = asteroid_pos_y - origin_y + avy * DELTA_TIME;

    double vb = BULLET_SPEED;
    double vb_sq = vb * vb;
    double theta_0 = ship_heading_deg * DEG_TO_RAD;

    // Quadratic coefficients for interception time
    double a = avx * avx + avy * avy - vb_sq;

    double time_until_can_fire_s = static_cast<double>(future_shooting_timesteps) * DELTA_TIME;
    double ax_delayed = ax + time_until_can_fire_s * avx;
    double ay_delayed = ay + time_until_can_fire_s * avy;

    double b = 2.0 * (ax_delayed * avx + ay_delayed * avy - vb_sq * t_0);
    double c = ax_delayed * ax_delayed + ay_delayed * ay_delayed - vb_sq * t_0 * t_0;

    auto roots = solve_quadratic(a, b, c);
    for (int i = 0; i < 2; ++i) {
        double t = (i == 0) ? std::get<0>(roots) : std::get<1>(roots);
        if (std::isnan(t) || t < 0.0) {
            continue;
        }
        double x = ax_delayed + t * avx;
        double y = ay_delayed + t * avy;
        double theta = fast_atan2(y, x);

        // Interception position in world/absolute coordinates
        double intercept_x = x + origin_x;
        double intercept_y = y + origin_y;
        bool feasible = check_coordinate_bounds(game_state, intercept_x, intercept_y);
        if (!feasible) {
            continue;
        }

        double asteroid_dist = std::sqrt(x * x + y * y);
        double shot_heading_tolerance_rad;
        if (asteroid_r < asteroid_dist) {
            shot_heading_tolerance_rad = super_fast_asin((asteroid_r - ASTEROID_AIM_BUFFER_PIXELS) / asteroid_dist);
        } else {
            shot_heading_tolerance_rad = 0.5 * pi;
        }
        return std::make_tuple(feasible,
            angle_difference_rad(theta, theta_0),
            shot_heading_tolerance_rad,
            t,
            intercept_x,
            intercept_y,
            asteroid_dist
        );
    }
    // No feasible solution found.
    return std::make_tuple(false, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
}

// Actual splitting logic
inline std::tuple<Asteroid, Asteroid, Asteroid>
forecast_asteroid_splits(const Asteroid& a, int64_t timesteps_until_appearance, double vfx, double vfy, double v, double split_angle, const GameState& game_state) {
    double theta = std::atan2(vfy, vfx) * RAD_TO_DEG;
    int64_t new_size = a.size - 1;
    double new_mass = ASTEROID_MASS_LOOKUP[new_size];
    double new_radius = ASTEROID_RADII_LOOKUP[new_size];

    double angle_left = (theta + split_angle) * DEG_TO_RAD;
    double angle_center = theta * DEG_TO_RAD;
    double angle_right = (theta - split_angle) * DEG_TO_RAD;

    double cos_angle_left = std::cos(angle_left);
    double sin_angle_left = std::sin(angle_left);
    double cos_angle_center = std::cos(angle_center);
    double sin_angle_center = std::sin(angle_center);
    double cos_angle_right = std::cos(angle_right);
    double sin_angle_right = std::sin(angle_right);

    if (timesteps_until_appearance == 0) {
        return std::make_tuple(
            Asteroid(a.x, a.y, v * cos_angle_left, v * sin_angle_left, new_size, new_mass, new_radius, 0),
            Asteroid(a.x, a.y, v * cos_angle_center, v * sin_angle_center, new_size, new_mass, new_radius, 0),
            Asteroid(a.x, a.y, v * cos_angle_right, v * sin_angle_right, new_size, new_mass, new_radius, 0)
        );
    } else {
        double dt = DELTA_TIME * static_cast<double>(timesteps_until_appearance);
        // For fmod; ensure positive modulus results like Python’s %
        auto wrap = [](double x, double mod) {
            double r = std::fmod(x, mod);
            return r < 0 ? r + mod : r;
        };
        return std::make_tuple(
            Asteroid(
                wrap(a.x + a.vx * dt - dt * cos_angle_left * v, game_state.map_size_x),
                wrap(a.y + a.vy * dt - dt * sin_angle_left * v, game_state.map_size_y),
                v * cos_angle_left, v * sin_angle_left,
                new_size, new_mass, new_radius, timesteps_until_appearance),
            Asteroid(
                wrap(a.x + a.vx * dt - dt * cos_angle_center * v, game_state.map_size_x),
                wrap(a.y + a.vy * dt - dt * sin_angle_center * v, game_state.map_size_y),
                v * cos_angle_center, v * sin_angle_center,
                new_size, new_mass, new_radius, timesteps_until_appearance),
            Asteroid(
                wrap(a.x + a.vx * dt - dt * cos_angle_right * v, game_state.map_size_x),
                wrap(a.y + a.vy * dt - dt * sin_angle_right * v, game_state.map_size_y),
                v * cos_angle_right, v * sin_angle_right,
                new_size, new_mass, new_radius, timesteps_until_appearance)
        );
    }
}

// Heading-based bullet splits
inline std::tuple<Asteroid, Asteroid, Asteroid>
forecast_asteroid_bullet_splits_from_heading(const Asteroid& a, int64_t timesteps_until_appearance, double bullet_heading_deg, const GameState& game_state) {
    double bullet_heading_rad = bullet_heading_deg * DEG_TO_RAD;
    double bullet_vel_x = std::cos(bullet_heading_rad) * BULLET_SPEED;
    double bullet_vel_y = std::sin(bullet_heading_rad) * BULLET_SPEED;
    double vfx = (1.0 / (BULLET_MASS + a.mass)) * (BULLET_MASS * bullet_vel_x + a.mass * a.vx);
    double vfy = (1.0 / (BULLET_MASS + a.mass)) * (BULLET_MASS * bullet_vel_y + a.mass * a.vy);
    double v = std::sqrt(vfx * vfx + vfy * vfy);
    return forecast_asteroid_splits(a, timesteps_until_appearance, vfx, vfy, v, 15.0, game_state);
}

// Instantaneous (velocity) bullet splits
inline std::tuple<Asteroid, Asteroid, Asteroid>
forecast_instantaneous_asteroid_bullet_splits_from_velocity(const Asteroid& a, double bullet_vx, double bullet_vy, const GameState& game_state) {
    double vfx = (1.0 / (BULLET_MASS + a.mass)) * (BULLET_MASS * bullet_vx + a.mass * a.vx);
    double vfy = (1.0 / (BULLET_MASS + a.mass)) * (BULLET_MASS * bullet_vy + a.mass * a.vy);
    double v = std::sqrt(vfx * vfx + vfy * vfy);
    return forecast_asteroid_splits(a, 0, vfx, vfy, v, 15.0, game_state);
}

// Mine splits
inline std::tuple<Asteroid, Asteroid, Asteroid>
forecast_asteroid_mine_instantaneous_splits(const Asteroid& asteroid, const Mine& mine, const GameState& game_state) {
    double delta_x = mine.x - asteroid.x;
    double delta_y = mine.y - asteroid.y;
    double dist = std::sqrt(delta_x * delta_x + delta_y * delta_y);
    double F = (-dist / MINE_BLAST_RADIUS + 1.0) * MINE_BLAST_PRESSURE * 2.0 * asteroid.radius;
    double a_accel = F / asteroid.mass;
    double vfx, vfy, v, split_angle;
    if (dist != 0.0) {
        double cos_theta = (asteroid.x - mine.x) / dist;
        double sin_theta = (asteroid.y - mine.y) / dist;
        vfx = asteroid.vx + a_accel * cos_theta;
        vfy = asteroid.vy + a_accel * sin_theta;
        v = std::sqrt(vfx * vfx + vfy * vfy);
        split_angle = 15.0;
    } else {
        vfx = asteroid.vx;
        vfy = asteroid.vy;
        v = std::sqrt(vfx * vfx + vfy * vfy + a_accel * a_accel);
        split_angle = 120.0;
    }
    return forecast_asteroid_splits(asteroid, 0, vfx, vfy, v, split_angle, game_state);
}

// Ship splits
inline std::tuple<Asteroid, Asteroid, Asteroid> forecast_asteroid_ship_splits(
    const Asteroid& asteroid, int64_t timesteps_until_appearance, double ship_vx, double ship_vy, const GameState& game_state)
{
    double vfx = (1.0 / (SHIP_MASS + asteroid.mass)) * (SHIP_MASS * ship_vx + asteroid.mass * asteroid.vx);
    double vfy = (1.0 / (SHIP_MASS + asteroid.mass)) * (SHIP_MASS * ship_vy + asteroid.mass * asteroid.vy);
    double v = std::sqrt(vfx * vfx + vfy * vfy);
    return forecast_asteroid_splits(asteroid, timesteps_until_appearance, vfx, vfy, v, 15.0, game_state);
}

// Maintain split asteroids' forecast
inline std::vector<Asteroid> maintain_forecasted_asteroids(const std::vector<Asteroid>& forecasted_asteroid_splits, const GameState& game_state) {
    std::vector<Asteroid> updated_asteroids;
    for (const Asteroid& forecasted_asteroid : forecasted_asteroid_splits) {
        if (forecasted_asteroid.timesteps_until_appearance > 1) {
            updated_asteroids.emplace_back(
                pymod(forecasted_asteroid.x + forecasted_asteroid.vx * DELTA_TIME, game_state.map_size_x),
                pymod(forecasted_asteroid.y + forecasted_asteroid.vy * DELTA_TIME, game_state.map_size_y),
                forecasted_asteroid.vx,
                forecasted_asteroid.vy,
                forecasted_asteroid.size,
                forecasted_asteroid.mass,
                forecasted_asteroid.radius,
                forecasted_asteroid.timesteps_until_appearance - 1
            );
        }
    }
    return updated_asteroids;
}

// --- Asteroid fuzzy equality in list, including wrap handling ---
inline bool is_asteroid_in_list(const std::vector<Asteroid>& list_of_asteroids, const Asteroid& asteroid, const GameState& game_state) {
    assert(asteroid.alive);
    // Since floating point comparison isn't a good idea, break apart the asteroid dict and compare each element manually in a fuzzy way
    for (const Asteroid& a : list_of_asteroids) {
        assert(a.alive);
        // Compare with fuzzy position (including wrap-around), velocity, and size
        // The reason we do the seemingly redundant checks for position, is that we need to account for wrap. If the game field was 1000 pixels wide, and one asteroid is at 0.0000000001 and the other is at 999.9999999999, they're basically the same asteroid, so we need to realize that.
        if ((is_close(a.x, asteroid.x) || is_close_to_zero(game_state.map_size_x - std::abs(a.x - asteroid.x))) &&
            (is_close(a.y, asteroid.y) || is_close_to_zero(game_state.map_size_y - std::abs(a.y - asteroid.y))) &&
            is_close(a.vx, asteroid.vx) &&
            is_close(a.vy, asteroid.vy) &&
            a.size == asteroid.size) {
            return true;
        }
    }
    return false;
}

inline double predict_ship_mine_collision(double ship_pos_x, double ship_pos_y, const Mine& mine, int64_t future_timesteps = 0) {
    // If mine hasn't exploded yet by that time horizon
    if (mine.remaining_time >= static_cast<double>(future_timesteps) * DELTA_TIME) {
        double delta_x = ship_pos_x - mine.x;
        double delta_y = ship_pos_y - mine.y;
        double separation = SHIP_RADIUS + MINE_BLAST_RADIUS;
        if (std::abs(delta_x) <= separation &&
            std::abs(delta_y) <= separation &&
            delta_x * delta_x + delta_y * delta_y <= separation * separation)
        {
            return mine.remaining_time;
        } else {
            return inf;
        }
    } else {
        // Mine exploded in the past
        return inf;
    }
}

inline int64_t calculate_timesteps_until_bullet_hits_asteroid(double time_until_asteroid_center_s, double asteroid_radius)
{
    // Add 1 for the initial timestep (see Python)
    double travel_distance = time_until_asteroid_center_s * BULLET_SPEED - asteroid_radius - SHIP_RADIUS;
    double timesteps = (travel_distance / BULLET_SPEED) * FPS;
    return 1 + static_cast<int64_t>(std::ceil(timesteps));
}

inline bool asteroid_bullet_collision(
    double bullet_head_x, double bullet_head_y,
    double bullet_tail_x, double bullet_tail_y,
    double asteroid_x, double asteroid_y,
    double asteroid_radius)
{
    // This is an optimized version of circle_line_collision() from the Kessler source code
    // First, do a rough check if there's no chance the collision can occur
    // Avoid the use of min/max because it should be a bit faster
    double x_min, x_max, y_min, y_max;
    if (bullet_head_x < bullet_tail_x) {
        x_min = bullet_head_x - asteroid_radius;
        if (asteroid_x < x_min) return false;
        x_max = bullet_tail_x + asteroid_radius;
    } else {
        x_min = bullet_tail_x - asteroid_radius;
        if (asteroid_x < x_min) return false;
        x_max = bullet_head_x + asteroid_radius;
    }
    if (asteroid_x > x_max) return false;

    if (bullet_head_y < bullet_tail_y) {
        y_min = bullet_head_y - asteroid_radius;
        if (asteroid_y < y_min) return false;
        y_max = bullet_tail_y + asteroid_radius;
    } else {
        y_min = bullet_tail_y - asteroid_radius;
        if (asteroid_y < y_min) return false;
        y_max = bullet_head_y + asteroid_radius;
    }
    if (asteroid_y > y_max) return false;

    // A collision is possible.
    // Create a triangle between the center of the asteroid, and the two ends of the bullet.
    // Inlined calculation
    // Compute distances from asteroid center to bullet head and tail
    // a = dist(asteroid_x, asteroid_y, bullet_head_x, bullet_head_y)
    double bhdx = bullet_head_x - asteroid_x;
    double bhdy = bullet_head_y - asteroid_y;
    double a = std::sqrt(bhdx*bhdx + bhdy*bhdy);
    // b = dist(asteroid_x, asteroid_y, bullet_tail_x, bullet_tail_y)
    double btdx = bullet_tail_x - asteroid_x;
    double btdy = bullet_tail_y - asteroid_y;
    double b = std::sqrt(btdx*btdx + btdy*btdy);
    // c = BULLET_LENGTH
    double s = 0.5 * (a + b + BULLET_LENGTH);
    double squared_area = s * (s - a) * (s - b) * (s - BULLET_LENGTH);

    // Heron's height of triangle from asteroid center to bullet line
    double triangle_height = TWICE_BULLET_LENGTH_RECIPROCAL * std::sqrt(std::max(0.0, squared_area));

    return triangle_height < asteroid_radius;
}

inline std::tuple<
    bool,    // feasible
    double,  // shooting_angle_error_deg
    int64_t, // aiming_timesteps_required
    double,  // interception_time_s
    double,  // intercept_x
    double,  // intercept_y
    double   // asteroid_dist_during_interception
> solve_interception(const Asteroid& asteroid, const Ship& ship_state, const GameState& game_state, int64_t timesteps_until_can_fire = 0)
{
    double t_0 = 0.0175; // (SHIP_RADIUS - 0.5*BULLET_LENGTH)/BULLET_SPEED (hardcoded for parity)

    double ship_position_x = ship_state.x;
    double ship_position_y = ship_state.y;
    double origin_x = ship_position_x;
    double origin_y = ship_position_y;
    double avx = asteroid.vx;
    double avy = asteroid.vy;
    double ax = asteroid.x - origin_x + avx * DELTA_TIME;
    double ay = asteroid.y - origin_y + avy * DELTA_TIME;

    double vb = BULLET_SPEED;
    double vb_sq = vb * vb;
    double theta_0 = ship_state.heading * DEG_TO_RAD;

    double a = avx*avx + avy*avy - vb_sq;

    double k1 = ay*vb - avy*vb*t_0;
    double k2 = ax*vb - avx*vb*t_0;
    double k3 = avy*ax - avx*ay;

    // --- nested helpers (now as lambdas or inline fns) ---

    auto naive_desired_heading_calc = [&](int64_t timesteps_until_fire = 0) -> std::tuple<double, double, int64_t, double, double, double> {
        double time_until_can_fire_s = double(timesteps_until_fire) * DELTA_TIME;
        double ax_delayed = ax + time_until_can_fire_s * avx;
        double ay_delayed = ay + time_until_can_fire_s * avy;
        double b = 2.0 * (ax_delayed * avx + ay_delayed * avy - vb_sq * t_0);
        double c = ax_delayed * ax_delayed + ay_delayed * ay_delayed - vb_sq * t_0 * t_0;
        auto quadratic_roots = solve_quadratic(a, b, c);
        for (int i = 0; i < 2; ++i) {
            double t = (i == 0) ? quadratic_roots.first : quadratic_roots.second;
            if (std::isnan(t) || t < 0.0)
                continue;
            double x = ax_delayed + t * avx;
            double y = ay_delayed + t * avy;
            double theta = fast_atan2(y, x);
            double intercept_x = x + origin_x;
            double intercept_y = y + origin_y;
            return std::make_tuple(
                t, angle_difference_rad(theta, theta_0), timesteps_until_fire, intercept_x, intercept_y, dist(ship_position_x, ship_position_y, intercept_x, intercept_y)
            );
        }
        return std::make_tuple(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), 0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    };

    auto root_function = [&](double theta) -> double {
        theta += theta_0;
        if (!(theta_0 - pi <= theta && theta <= theta_0 + pi))
            theta = std::fmod(theta - theta_0 + pi, TAU) - pi + theta_0;
        double abs_delta_theta = std::abs(theta - theta_0);
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        double sinusoidal_component = k1 * cos_theta - k2 * sin_theta + k3;
        double wacky_component = vb * abs_delta_theta / pi * (avy * cos_theta - avx * sin_theta);
        return sinusoidal_component + wacky_component;
    };
    auto root_function_derivative = [&](double theta) -> double {
        theta += theta_0;
        if (!(theta_0 - pi <= theta && theta <= theta_0 + pi))
            theta = std::fmod(theta - theta_0 + pi, TAU) - pi + theta_0;
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        double sinusoidal_component = -k1 * sin_theta - k2 * cos_theta;
        double wacky_component = -vb * sign(theta - theta_0) / pi *
            (avx * sin_theta - avy * cos_theta + (theta - theta_0) * (avx * cos_theta + avy * sin_theta));
        return sinusoidal_component + wacky_component;
    };
    auto root_function_second_derivative = [&](double theta) -> double {
        theta += theta_0;
        if (!(theta_0 - pi <= theta && theta <= theta_0 + pi))
            theta = std::fmod(theta - theta_0 + pi, TAU) - pi + theta_0;
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        double sinusoidal_component = -k1 * cos_theta + k2 * sin_theta;
        double wacky_component = -vb * sign(theta - theta_0) / pi *
            (2.0 * (avx * cos_theta + avy * sin_theta) - (theta - theta_0) * (avx * sin_theta - avy * cos_theta));
        return sinusoidal_component + wacky_component;
    };

    auto turbo_rootinator_5000 = [&](double initial_guess, double tolerance = EPS, int64_t max_iterations = 4) -> double {
        double theta_old = initial_guess, theta_new, func_value, initial_func_value = std::numeric_limits<double>::quiet_NaN();
        for (int64_t j = 0; j < max_iterations; ++j) {
            func_value = root_function(theta_old);
            if (std::abs(func_value) < TAD)
                return theta_old;
            if (std::isnan(initial_func_value))
                initial_func_value = func_value;
            double derivative_value = root_function_derivative(theta_old);
            double second_derivative_value = root_function_second_derivative(theta_old);
            double denominator = 2.0 * derivative_value * derivative_value - func_value * second_derivative_value;
            if (denominator == 0.0) return std::numeric_limits<double>::quiet_NaN();
            theta_new = theta_old - (2.0 * func_value * derivative_value) / denominator;
            if (theta_new < -pi) theta_new = pi - GRAIN;
            else if (theta_new > pi) theta_new = -pi + GRAIN;
            else if (-pi <= theta_old && theta_old <= 0.0 && 0.0 <= theta_new && theta_new <= pi)
                theta_new = GRAIN;
            else if (-pi <= theta_new && theta_new <= 0.0 && 0.0 <= theta_old && theta_old <= pi)
                theta_new = -GRAIN;
            if (std::abs(theta_new - theta_old) < tolerance && std::abs(func_value) < 0.1 * std::abs(initial_func_value))
                return theta_new;
            theta_old = theta_new;
        }
        return std::numeric_limits<double>::quiet_NaN();
    };

    auto rotation_time = [](double delta_theta_rad) -> double {
        return std::abs(delta_theta_rad) * SHIP_MAX_TURN_RATE_RAD_RECIPROCAL;
    };
    auto bullet_travel_time = [&](double theta, double t_rot) -> double {
        theta += theta_0;
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        double denominator_x = avx - vb * cos_theta;
        double denominator_y = avy - vb * sin_theta;
        if (denominator_x == 0.0 && denominator_y == 0.0)
            return inf;
        double t_bul;
        if (std::abs(denominator_x) > std::abs(denominator_y)) {
            t_bul = (vb * t_0 * cos_theta - ax - avx * t_rot) / denominator_x;
        } else {
            t_bul = (vb * t_0 * sin_theta - ay - avy * t_rot) / denominator_y;
        }
        return t_bul;
    };

    double amount_we_can_turn_before_we_can_shoot_rad = double(timesteps_until_can_fire) * SHIP_MAX_TURN_RATE_RAD_TS;
    auto naive_solution = naive_desired_heading_calc(timesteps_until_can_fire);
    double naive_angle = std::get<1>(naive_solution);
    if (std::abs(naive_angle) <= amount_we_can_turn_before_we_can_shoot_rad + EPS) {
        // The naive solution works because there's no turning delay
        double n_intercept_x = std::get<3>(naive_solution);
        double n_intercept_y = std::get<4>(naive_solution);
        if (check_coordinate_bounds(game_state, n_intercept_x, n_intercept_y)) {
            return std::make_tuple(
                true,
                naive_angle * RAD_TO_DEG,
                timesteps_until_can_fire,
                std::get<0>(naive_solution),
                n_intercept_x,
                n_intercept_y,
                std::get<5>(naive_solution)
            );
        }
    } else {
        double delta_theta_solution = std::numeric_limits<double>::quiet_NaN();
        if (std::abs(avx) < GRAIN && std::abs(avy) < GRAIN) {
            delta_theta_solution = std::get<1>(naive_solution);
        } else {
            delta_theta_solution = turbo_rootinator_5000(std::get<1>(naive_solution), TAD, 4);
        }

        if (std::isnan(delta_theta_solution)) {
            return std::make_tuple(false, std::numeric_limits<double>::quiet_NaN(), -1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
        }
        double absolute_theta_solution = delta_theta_solution + theta_0;
        assert(-pi <= delta_theta_solution && delta_theta_solution <= pi);

        double delta_theta_solution_deg = delta_theta_solution * RAD_TO_DEG;
        double t_rot = rotation_time(delta_theta_solution);

        assert(is_close(t_rot, std::abs(delta_theta_solution_deg) / SHIP_MAX_TURN_RATE));

        double t_bullet = bullet_travel_time(delta_theta_solution, t_rot);
        if (t_bullet < 0)
            return std::make_tuple(false, std::numeric_limits<double>::quiet_NaN(), -1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());

        double bullet_travel_dist = vb * (t_bullet + t_0);
        double intercept_x = origin_x + bullet_travel_dist * std::cos(absolute_theta_solution);
        double intercept_y = origin_y + bullet_travel_dist * std::sin(absolute_theta_solution);

        if (check_coordinate_bounds(game_state, intercept_x, intercept_y)) {
            int64_t t_rot_ts = std::max(
                timesteps_until_can_fire,
                static_cast<int64_t>(std::ceil(t_rot * FPS))
            );
            auto discrete_solution = naive_desired_heading_calc(t_rot_ts);
            if (!std::isnan(std::get<0>(discrete_solution))) {
                if (!(std::abs(std::get<1>(discrete_solution) * RAD_TO_DEG) - EPS <= double(t_rot_ts) * SHIP_MAX_TURN_RATE_DEG_TS))
                    return std::make_tuple(false, std::numeric_limits<double>::quiet_NaN(), -1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
                assert(t_rot_ts == std::get<2>(discrete_solution));
                if (check_coordinate_bounds(game_state, std::get<3>(discrete_solution), std::get<4>(discrete_solution))) {
                    return std::make_tuple(
                        true,
                        std::get<1>(discrete_solution) * RAD_TO_DEG,
                        t_rot_ts,
                        std::get<0>(discrete_solution),
                        std::get<3>(discrete_solution),
                        std::get<4>(discrete_solution),
                        std::get<5>(discrete_solution)
                    );
                }
            }
        }
    }
    return std::make_tuple(false, std::numeric_limits<double>::quiet_NaN(), -1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
}

inline double get_adversary_interception_time_lower_bound(
    const Asteroid& asteroid,
    const std::vector<Ship>& adversary_ships,
    const GameState& game_state,
    int64_t adversary_rotation_timestep_fudge = ADVERSARY_ROTATION_TIMESTEP_FUDGE)
{
    if (adversary_ships.empty())
        return inf;

    // See your Python: let _2 = aiming_timesteps_required and _3 = interception_time_s
    bool feasible;
    double _1;
    int64_t aiming_timesteps_required;
    double interception_time_s;
    double _4, _5, _6;

    std::tie(
        feasible, _1,
        aiming_timesteps_required, interception_time_s,
        _4, _5, _6
    ) = solve_interception(
        asteroid, adversary_ships[0], game_state, 0
    );

    if (feasible) {
        return std::max(0.0, interception_time_s + double(aiming_timesteps_required - adversary_rotation_timestep_fudge) * DELTA_TIME);
    } else {
        return inf;
    }
}

// For debugging
void print_asteroids_pending_death(const std::unordered_map<int64_t, std::vector<Asteroid>>& asteroids_pending_death) {
    std::cout << "Asteroids pending death:" << "\n";
    // Extract and sort the keys
    std::vector<int64_t> keys;
    for (const auto& pair : asteroids_pending_death) {
        keys.push_back(pair.first);
    }
    std::sort(keys.begin(), keys.end());

    // Print the values in key order
    for (int64_t key : keys) {
        std::cout << "Key: " << key << "\n";
        const auto& asteroid_list = asteroids_pending_death.at(key);
        for (const auto& asteroid : asteroid_list) {
            std::cout << "  - " << asteroid.str() << "\n";
        }
    }
}

// Checks if this asteroid already has a pending shot tracked
bool check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(
    const std::unordered_map<int64_t, std::vector<Asteroid>>& asteroids_pending_death,
    int64_t current_timestep,
    const GameState& game_state,
    const Asteroid& asteroid
) {
    // Helper lambda for checks:
    auto verify_asteroid_does_not_appear_in_wrong_timestep = [&](const Asteroid& a) -> bool {
        for (const auto& kv : asteroids_pending_death) {
            int64_t timestep = kv.first;
            const auto& asts_list = kv.second;
            if (is_asteroid_in_list(asts_list, a, game_state)) {
                double delta = static_cast<double>(timestep - current_timestep);
                if (std::abs(delta) <= 120) {
                    double movex = std::abs(delta) * DELTA_TIME * a.vx;
                    double movey = std::abs(delta) * DELTA_TIME * a.vy;
                    bool periodic_x = is_close_to_zero(fmod(movex, game_state.map_size_x)) ||
                        is_close(fmod(movex, game_state.map_size_x), game_state.map_size_x);
                    bool periodic_y = is_close_to_zero(fmod(movey, game_state.map_size_y)) ||
                        is_close(fmod(movey, game_state.map_size_y), game_state.map_size_y);

                    if (!(periodic_x && periodic_y)) {
                        throw std::runtime_error(
                            "Asteroid " + a.str() +
                            " from actual ts " + std::to_string(current_timestep) +
                            " appears in list on ts " + std::to_string(timestep) +
                            " with a delta of " + std::to_string(delta) + "!"
                        );
                    }
                }
                return false;
            }
        }
        return true;
    };

    // Sanity check
    if constexpr (ENABLE_SANITY_CHECKS) {
        assert(check_coordinate_bounds(game_state, asteroid.x, asteroid.y) || current_timestep == 0 && "Asteroid out of bounds!");
        if (!check_coordinate_bounds(game_state, asteroid.x, asteroid.y)) {
            std::cout << "WARNING, the scenario started with the asteroids out of bounds!" << std::endl;
        }
    }

    // See if asteroid for this timestep is tracked
    auto it = asteroids_pending_death.find(current_timestep);
    if (it != asteroids_pending_death.end()) {
        if (VERIFY_AST_TRACKING) {
            if (!is_asteroid_in_list(it->second, asteroid, game_state)) {
                return verify_asteroid_does_not_appear_in_wrong_timestep(asteroid);
            }
        }
        // If we found a list for this ts, return whether asteroid is not present
        return !is_asteroid_in_list(it->second, asteroid, game_state);
    } else {
        if (VERIFY_AST_TRACKING) {
            return verify_asteroid_does_not_appear_in_wrong_timestep(asteroid);
        }
        return true;
    }
}

void track_asteroid_we_shot_at(
    std::unordered_map<int64_t, std::vector<Asteroid>>& asteroids_pending_death,
    int64_t current_timestep,
    const GameState& game_state,
    int64_t bullet_travel_timesteps,
    const Asteroid& original_asteroid
) {
    /*
    asteroids_pending_death: For each timestep, the value is a list of asteroids that appear on that timestep, which already have a bullet heading toward it
    current_timestep: The current timestep at the time of fire, for which we begin the tracking
    bullet_travel_timesteps: The number of timesteps it takes for the bullet to reach the asteroid. We have to track for this many timesteps
    original_asteroid: This is the asteroid we're tracking, at the time of the bullet fire, at present time
    */
    //debug_print("Tracking asteroid we shot at. Asts pending death: ", ", current_timestep=", current_timestep, ", bullet_travel_timesteps=", bullet_travel_timesteps, ", original_asteroid=", original_asteroid.str());

    if constexpr (ENABLE_SANITY_CHECKS) {
        assert(check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(
            asteroids_pending_death, current_timestep, game_state, original_asteroid
        ));
    }

    // Create a copy of the asteroid so we don't mess up the original object
    Asteroid asteroid = original_asteroid;

    // Wrap asteroid position
    asteroid.x = pymod(asteroid.x, game_state.map_size_x);
    asteroid.y = pymod(asteroid.y, game_state.map_size_y);

    // Project asteroid into future positions for each timestep
    for (int64_t future_timesteps = 0; future_timesteps <= bullet_travel_timesteps; ++future_timesteps) {
        int64_t timestep = current_timestep + future_timesteps;
        auto& list_for_timestep = asteroids_pending_death[timestep];
        if constexpr (ENABLE_SANITY_CHECKS) {
            if (is_asteroid_in_list(list_for_timestep, asteroid, game_state)) {
                std::cout << "ABOUT TO FAIL ASSERTION, we are in the future by " << future_timesteps
                    << " timesteps from the current ts " << current_timestep << ", this asteroid is "
                    << asteroid.str() << " and LIST FOR THIS TS IS:\n";
                for (const auto& a : list_for_timestep)
                    std::cout << "  " << a.str() << "\n";
                for (const auto& pair : asteroids_pending_death) {
                    std::cout << "ts: " << pair.first << "\n";
                    for (const auto& a : pair.second) std::cout << "  " << a.str() << "\n";
                }
            }
            assert(!is_asteroid_in_list(list_for_timestep, asteroid, game_state) &&
                    ("The asteroid " + asteroid.str() +
                    " appeared in the list of pending death when it wasn't supposed to! I'm on future ts " +
                    std::to_string(future_timesteps) + " when tracking. This probably means we're reshooting at the same asteroid we already shot at!").c_str());
        }
        list_for_timestep.push_back(asteroid);
        // Advance the asteroid to the next position, unless last iteration
        if (future_timesteps != bullet_travel_timesteps) {
            asteroid.x = pymod(asteroid.x + asteroid.vx * DELTA_TIME, game_state.map_size_x);
            asteroid.y = pymod(asteroid.y + asteroid.vy * DELTA_TIME, game_state.map_size_y);
        }
    }
    //print_asteroids_pending_death(asteroids_pending_death);
}

Asteroid time_travel_asteroid(const Asteroid& asteroid, int64_t timesteps, const GameState& game_state) {
    // Project an asteroid forward or backward in time, with automatic position wrapping
    return Asteroid(
        pymod(asteroid.x + static_cast<double>(timesteps) * asteroid.vx * DELTA_TIME, game_state.map_size_x),
        pymod(asteroid.y + static_cast<double>(timesteps) * asteroid.vy * DELTA_TIME, game_state.map_size_y),
        asteroid.vx,
        asteroid.vy,
        asteroid.size,
        asteroid.mass,
        asteroid.radius,
        asteroid.timesteps_until_appearance
    );
}

Asteroid time_travel_asteroid_s(const Asteroid& asteroid, double time, const GameState& game_state) {
    // Project an asteroid forward or backward in time, with automatic position wrapping
    return Asteroid(
        pymod(asteroid.x + time * asteroid.vx, game_state.map_size_x),
        pymod(asteroid.y + time * asteroid.vy, game_state.map_size_y),
        asteroid.vx,
        asteroid.vy,
        asteroid.size,
        asteroid.mass,
        asteroid.radius,
        asteroid.timesteps_until_appearance
    );
}

class Matrix {
public:
    // Member variables
    int64_t initial_timestep;
    int64_t future_timesteps;
    int64_t last_timestep_fired;
    int64_t last_timestep_mined;
    GameState game_state;
    Ship ship_state;
    std::vector<Ship> other_ships;
    std::vector<Action> ship_move_sequence;
    std::vector<SimState> state_sequence;
    int64_t asteroids_shot;
    std::unordered_map<int64_t, std::vector<Asteroid>> asteroids_pending_death;
    std::unordered_map<int64_t, std::unordered_map<int64_t, std::vector<Asteroid>>> asteroids_pending_death_history;
    std::vector<Asteroid> forecasted_asteroid_splits;
    std::vector<std::vector<Asteroid>> forecasted_asteroid_splits_history;
    bool halt_shooting;
    bool fire_next_timestep_flag;
    bool fire_first_timestep;
    //std::optional<GameStatePlotter> game_state_plotter;
    int64_t sim_id;
    std::vector<std::string> explanation_messages;
    std::vector<std::string> safety_messages; 
    double respawn_timer;
    std::vector<double> respawn_timer_history;
    bool plot_this_sim;
    bool ship_crashed;
    std::optional<GameState> backed_up_game_state_before_post_mutation;
    std::optional<std::array<double, 9>> fitness_breakdown;
    bool cancel_firing_first_timestep;
    bool verify_first_shot;
    std::vector<Action> intended_move_sequence;
    bool sim_placed_a_mine;
    bool verify_maneuver_shots;
    std::set<std::pair<double, double>> mine_positions_placed;
    std::unordered_map<int64_t, std::set<std::pair<double, double>>> mine_positions_placed_history;
    int64_t last_timestep_colliding;
    int64_t respawn_maneuver_pass_number;
    std::vector<bool> random_walk_schedule;

    // Constructor
    Matrix() {}

    Matrix(
        const GameState& game_state_,
        const Ship& ship_state_,
        int64_t initial_timestep_,
        double respawn_timer_ = 0.0,
        const std::unordered_map<int64_t, std::vector<Asteroid>>* asteroids_pending_death_ = nullptr,
        const std::vector<Asteroid>* forecasted_asteroid_splits_ = nullptr,
        int64_t last_timestep_fired_ = INT_NEG_INF,
        int64_t last_timestep_mined_ = INT_NEG_INF,
        const std::set<std::pair<double, double>>* mine_positions_placed_ = nullptr,
        bool halt_shooting_ = false,
        bool fire_first_timestep_ = false,
        bool verify_first_shot_ = false,
        bool verify_maneuver_shots_ = true,
        int64_t last_timestep_colliding_ = -1,
        int64_t respawn_maneuver_pass_ = 0
        //std::optional<GameStatePlotter> game_state_plotter_ = std::nullopt
    )
        : initial_timestep(initial_timestep_),
          future_timesteps(0),
          last_timestep_fired(last_timestep_fired_),
          last_timestep_mined(last_timestep_mined_),
          halt_shooting(halt_shooting_),
          fire_first_timestep(fire_first_timestep_),
          //game_state_plotter(game_state_plotter_),
          sim_id(randint(1, 100000)),
          respawn_timer(respawn_timer_),
          plot_this_sim(false),
          ship_crashed(false),
          cancel_firing_first_timestep(false),
          verify_first_shot(verify_first_shot_),
          sim_placed_a_mine(false),
          verify_maneuver_shots(verify_maneuver_shots_),
          // mine_positions_placed is initialized below
          last_timestep_colliding(last_timestep_colliding_ != -1 ? last_timestep_colliding_ : initial_timestep_ - 1),
          respawn_maneuver_pass_number(respawn_maneuver_pass_)
    {
        std::unordered_map<int64_t, std::vector<Asteroid>> local_asteroids_pending_death;
        if (asteroids_pending_death_ == nullptr) {
            local_asteroids_pending_death = std::unordered_map<int64_t, std::vector<Asteroid>>();
        } else {
            local_asteroids_pending_death = *asteroids_pending_death_;
        }

        std::vector<Asteroid> local_forecasted_asteroid_splits;
        if (forecasted_asteroid_splits_ == nullptr) {
            local_forecasted_asteroid_splits = std::vector<Asteroid>();
        } else {
            local_forecasted_asteroid_splits = *forecasted_asteroid_splits_;
        }

        if constexpr (ENABLE_SANITY_CHECKS) {
            assert(static_cast<bool>(ship_state_.is_respawning) == (respawn_timer_ != 0.0));
        }

        game_state = game_state_.copy();
        ship_state = ship_state_;

        // Deep copy game state asteroids/bullets/mines/ships
        // TODO: This is probably unnecessary! This seems super slow.
        game_state.asteroids.clear();
        for (const auto& a : game_state_.asteroids) {
            game_state.asteroids.push_back(a);
        }
        for (const auto& a : game_state.asteroids) {
            assert(a.alive);
        }
        game_state.ships.clear();
        for (const auto& s : game_state_.ships) {
            game_state.ships.push_back(s);
        }
        game_state.bullets.clear();
        for (const auto& b : game_state_.bullets) {
            game_state.bullets.push_back(b);
        }
        for (const auto& b : game_state.bullets) {
            assert(b.alive);
        }
        game_state.mines.clear();
        for (const auto& m : game_state_.mines) {
            game_state.mines.push_back(m);
        }
        for (const auto& m : game_state.mines) {
            assert(m.alive);
        }
        other_ships = get_other_ships(game_state, ship_state.id);
        if constexpr (ENABLE_SANITY_CHECKS) {
            assert(0 <= static_cast<int>(other_ships.size()) && static_cast<int>(other_ships.size()) <= 1);
        }

        ship_move_sequence.clear();
        state_sequence.clear();
        asteroids_shot = 0;

        // asteroids_pending_death: deep copy
        for (const auto& kv : local_asteroids_pending_death) {
            std::vector<Asteroid> tmp;
            for (const auto& ast : kv.second) {
                tmp.push_back(ast);
            }
            asteroids_pending_death[kv.first] = tmp;
        }
        asteroids_pending_death_history.clear();

        // forecasted_asteroid_splits: deep copy
        forecasted_asteroid_splits.clear();
        for (const auto& a : local_forecasted_asteroid_splits) {
            forecasted_asteroid_splits.push_back(a);
        }
        for (const auto& a : forecasted_asteroid_splits) {
            assert(a.alive);
        }
        forecasted_asteroid_splits_history.clear();

        fire_next_timestep_flag = false;
        explanation_messages.clear();
        safety_messages.clear();
        respawn_timer_history.clear();
        backed_up_game_state_before_post_mutation = std::nullopt;
        fitness_breakdown = std::nullopt;
        intended_move_sequence.clear();
        mine_positions_placed_history.clear();

        if (mine_positions_placed_ != nullptr) {
            mine_positions_placed = *mine_positions_placed_;
        } else {
            mine_positions_placed = std::set<std::pair<double, double>>();
        }

        // 0 - Not a respawn maneuver, 1 - First pass respawn, 2 - Second pass respawn
        if (!halt_shooting && last_timestep_colliding_ == -1) {
            assert(respawn_maneuver_pass_number == 0);
        } else if (last_timestep_colliding_ == -1) {
            assert(respawn_maneuver_pass_number == 1);
        } else {
            assert(respawn_maneuver_pass_number == 2);
        }

        if constexpr (ENABLE_SANITY_CHECKS) {
            if (respawn_maneuver_pass_number == 1) {
                assert(halt_shooting);
                assert(!fire_first_timestep);
            } else if (respawn_maneuver_pass_number == 2) {
                assert(halt_shooting);
                assert(!fire_first_timestep);
            }
        }

        // Define random walk schedule
        double bias = random_double(); // Random number between 0.0 and 1.0
        random_walk_schedule.clear();
        for (int i = 0; i < RANDOM_WALK_SCHEDULE_LENGTH; ++i) {
            random_walk_schedule.push_back(random_double() < bias);
        }

        if (sim_id == 1234567) {
            std::cout << "Starting sim " << sim_id << " with ship state " << ship_state_.str()
                << ", fire_next_timestep_flag=" << fire_next_timestep_flag
                << ", fire_first_timestep=" << fire_first_timestep
                << ", last_timestep_fired=" << last_timestep_fired << std::endl;
        }
    }

    // ---- getters ----

    int64_t get_last_timestep_colliding() const {
        return last_timestep_colliding;
    }

    std::set<std::pair<double, double>> get_mine_positions_placed() const {
        return mine_positions_placed;
    }

    bool get_cancel_firing_first_timestep() const {
        return cancel_firing_first_timestep;
    }

    std::vector<std::string> get_explanations() const {
        return explanation_messages;
    }

    std::vector<std::string> get_safety_messages() const {
        return safety_messages;
    }

    int64_t get_sim_id() const {
        return sim_id;
    }

    double get_respawn_timer() const {
        return respawn_timer;
    }

    int64_t get_respawn_maneuver_pass_number() const {
        return this->respawn_maneuver_pass_number;
    }

    std::unordered_map<int64_t, double> get_respawn_timer_history() const {
        std::unordered_map<int64_t, double> respawn_timer_history_dict;
        for (size_t i = 0; i < respawn_timer_history.size(); ++i) {
            respawn_timer_history_dict[initial_timestep + static_cast<int64_t>(i) + 1] = respawn_timer_history[i];
        }
        return respawn_timer_history_dict;
    }

    Ship get_ship_state() const {
        return ship_state;
    }

    GameState get_game_state() const {
        if (backed_up_game_state_before_post_mutation.has_value()) {
            return backed_up_game_state_before_post_mutation.value().copy();
        }
        return game_state.copy();
    }

    bool get_fire_next_timestep_flag() const {
        return fire_next_timestep_flag;
    }

    void set_fire_next_timestep_flag(bool flag) {
        this->fire_next_timestep_flag = flag;
    }

    std::unordered_map<int64_t, std::vector<Asteroid>> get_asteroids_pending_death() const {
        return asteroids_pending_death;
    }

    std::unordered_map<int64_t, std::unordered_map<int64_t, std::vector<Asteroid>>> get_asteroids_pending_death_history() const {
        // This is a doozy. First of all, if we never shot any asteroids during this sim, then this history dict will be empty!
        // But each time we shoot an asteroid, we add to this dict the timestep the thing was updated, but we plop down the PREVIOUS version!
        // For example we have versions A and B. 0:A, 1:A, 2:A, 3:B, 4:B is how the thing changes. So the dict would say: {3: A}. And then B is held in the variable.
        std::unordered_map<int64_t, std::unordered_map<int64_t, std::vector<Asteroid>>> asteroids_pending_death_history_dict;
        // Use a pointer to avoid copying large map repeatedly
        const std::unordered_map<int64_t, std::vector<Asteroid>>* latest_version = &asteroids_pending_death;
        // TODO: Make sure there's no off-by-one error in these bounds and indices!
        for (int64_t t = initial_timestep + future_timesteps; t > initial_timestep; --t) {
            // We add one because it's the frame after this one that we're updating the state to be, and that we read this variable from
            asteroids_pending_death_history_dict[t + 1] = *latest_version;  // Dereference to copy into result
            auto it = asteroids_pending_death_history.find(t);
            if (it != asteroids_pending_death_history.end()) {
                latest_version = &it->second;
            }
        }
        return asteroids_pending_death_history_dict;
    }


    std::vector<Asteroid> get_forecasted_asteroid_splits() const {
        return forecasted_asteroid_splits;
    }

    std::unordered_map<int64_t, std::vector<Asteroid>> get_forecasted_asteroid_splits_history() const {
        std::unordered_map<int64_t, std::vector<Asteroid>> forecasted_asteroid_splits_dict;
        for (size_t i = 0; i < forecasted_asteroid_splits_history.size(); ++i) {
            forecasted_asteroid_splits_dict[initial_timestep + static_cast<int64_t>(i)] = forecasted_asteroid_splits_history[i];
        }
        assert(static_cast<int64_t>(forecasted_asteroid_splits_history.size()) == future_timesteps);
        forecasted_asteroid_splits_dict[initial_timestep + static_cast<int64_t>(forecasted_asteroid_splits_history.size())] = forecasted_asteroid_splits;
        return forecasted_asteroid_splits_dict;
    }

    std::unordered_map<int64_t, std::set<std::pair<double, double>>> get_mine_positions_placed_history() const {
        // This is a doozy. First of all, if we never shot any asteroids during this sim, then this history dict will be empty!
        // But each time we shoot an asteroid, we add to this dict the timestep the thing was updated, but we plop down the PREVIOUS version!
        // For example we have versions A and B. 0:A, 1:A, 2:A, 3:B, 4:B is how the thing changes. So the dict would say: {3: A}. And then B is held in the variable.
        std::unordered_map<int64_t, std::set<std::pair<double, double>>> mine_positions_placed_history_dict;
        // Use pointer to avoid unnecessary copying of the set
        const std::set<std::pair<double, double>>* latest_version = &mine_positions_placed;
        // TODO: Make sure there's no off-by-one error in these bounds and indices!
        for (int64_t t = initial_timestep + future_timesteps; t > initial_timestep; --t) {
            // We add one because it's the frame after this one that we're updating the state to be, and that we read this variable from
            mine_positions_placed_history_dict[t + 1] = *latest_version;  // Copy the current version into the result
            auto it = mine_positions_placed_history.find(t);
            if (it != mine_positions_placed_history.end()) {
                latest_version = &it->second;
            }
        }
        return mine_positions_placed_history_dict;
    }

    bool get_instantaneous_asteroid_collision(const std::vector<Asteroid>* asteroids = nullptr, const std::pair<double, double>* ship_position = nullptr) const {
        // UNUSED
        double position_x, position_y;
        if (ship_position != nullptr) {
            position_x = ship_position->first;
            position_y = ship_position->second;
        } else {
            position_x = ship_state.x;
            position_y = ship_state.y;
        }
        const std::vector<Asteroid>& ast_ref = (asteroids != nullptr) ? *asteroids : game_state.asteroids;
        for (const auto& a : ast_ref) {
            if (check_collision(position_x, position_y, SHIP_RADIUS, a.x, a.y, a.radius)) {
                return true;
            }
        }
        return false;
    }

    /*
    bool get_instantaneous_ship_collision() const {
        // UNUSED. This is too inaccurate, and there's better ways to handle avoiding the other ship rather than being overconfident in giving a binary yes you will collide/no you will not collide
        for (const auto& ship : other_ships) {
            double padding = SHIP_AVOIDANCE_PADDING + std::sqrt(ship.vx*ship.vx + ship.vy*ship.vy)*SHIP_AVOIDANCE_SPEED_PADDING_RATIO;
            // The faster the other ship is going, the bigger of a bubble around it I'm going to draw, since they can deviate from their path very quickly and run into me even though I thought I was in the clear
            if (check_collision(ship_state.x, ship_state.y, SHIP_RADIUS, ship.x, ship.y, ship.radius + SHIP_AVOIDANCE_PADDING + sqrt(ship.vx**2 + ship.vy**2)*SHIP_AVOIDANCE_SPEED_PADDING_RATIO)) {
                return true;
            }
        }
        return false;
    }*/

    bool get_instantaneous_mine_collision() {
        // UNUSED
        bool mine_collision = false;
        std::vector<size_t> mine_remove_idxs;
        for (size_t i = 0; i < game_state.mines.size(); ++i) {
            const auto& m = game_state.mines[i];
            if (m.remaining_time < EPS) {
                if (check_collision(ship_state.x, ship_state.y, SHIP_RADIUS, m.x, m.y, MINE_BLAST_RADIUS)) {
                    mine_collision = true;
                }
                mine_remove_idxs.push_back(i);
            }
        }
        if (!mine_remove_idxs.empty()) {
            std::vector<Mine> new_mines;
            for (size_t idx = 0; idx < game_state.mines.size(); ++idx) {
                if (std::find(mine_remove_idxs.begin(), mine_remove_idxs.end(), idx) == mine_remove_idxs.end()) {
                    new_mines.push_back(game_state.mines[idx]);
                }
            }
            game_state.mines = new_mines;
        }
        return mine_collision;
    }

    double get_next_extrapolated_asteroid_collision_time(int64_t additional_timesteps_to_blow_up_mines = 0) const {
        double next_imminent_asteroid_collision_time = std::numeric_limits<double>::infinity();
        // The asteroids from the game state could have been from the future since we waited out the mines, but the forecasted splits are from present time, so we need to treat them differently and only back-extrapolate the existing asteroids and not the forecasted ones
        
        auto process_asteroid = [&](const Asteroid& asteroid, bool asteroid_is_born) {
            if (asteroid.alive) {
                // For each unwrapped image
                std::vector<Asteroid> unwrapped_asteroids = unwrap_asteroid(
                    asteroid, game_state.map_size_x, game_state.map_size_y,
                    UNWRAP_ASTEROID_COLLISION_FORECAST_TIME_HORIZON, false
                );
                for (const Asteroid& a : unwrapped_asteroids) {
                    // Sanity check: ship should not be moving
                    assert(is_close_to_zero(ship_state.vx) && is_close_to_zero(ship_state.vy));

                    double predicted_collision_time_from_future =
                        predict_next_imminent_collision_time_with_asteroid(
                            ship_state.x, ship_state.y, ship_state.vx, ship_state.vy, SHIP_RADIUS,
                            a.x, a.y, a.vx, a.vy, a.radius, game_state
                        );
                    //double predicted_collision_time;
                    /*
                    if (asteroid_is_born) {
                        predicted_collision_time = predicted_collision_time_from_future + DELTA_TIME * static_cast<double>(additional_timesteps_to_blow_up_mines);
                    } else {
                        predicted_collision_time = predicted_collision_time_from_future;
                    }*/
                    double predicted_collision_time = predicted_collision_time_from_future + asteroid_is_born * (DELTA_TIME * static_cast<double>(additional_timesteps_to_blow_up_mines));

                    if (std::isinf(predicted_collision_time)) {
                        continue;
                    }
                    if constexpr (ENABLE_SANITY_CHECKS) {
                        assert(predicted_collision_time >= 0.0);
                    }
                    // The predicted collision time is finite and after the end of the sim
                    // TODO: Verify there isn't an off by one error
                    // Only consider asteroids that exist, or will exist before possible collision time
                    if (!(asteroid.timesteps_until_appearance > 0 &&
                        static_cast<double>(asteroid.timesteps_until_appearance) * DELTA_TIME > predicted_collision_time + EPS)) {
                        // The asteroid either exists, or will come into existence before our collision time
                        // Check the canonical asteroid, and not the unwrapped one!
                        Asteroid ast_to_check;
                        if (asteroid_is_born && additional_timesteps_to_blow_up_mines != 0) {
                            ast_to_check = time_travel_asteroid(asteroid, -additional_timesteps_to_blow_up_mines, game_state);
                        } else {
                            ast_to_check = asteroid;
                        }

                        if (!check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(
                                asteroids_pending_death,
                                initial_timestep + future_timesteps,
                                game_state, ast_to_check)) {

                            // We're already shooting the asteroid. Check whether the imminent collision time is before or after the asteroid is eliminated
                            int64_t predicted_collision_ts = static_cast<int64_t>(std::floor(predicted_collision_time * FPS));
                            Asteroid future_asteroid_during_imminent_collision_time;
                            if (asteroid_is_born) {
                                future_asteroid_during_imminent_collision_time = time_travel_asteroid(a, predicted_collision_ts - additional_timesteps_to_blow_up_mines, game_state);
                            } else {
                                future_asteroid_during_imminent_collision_time = time_travel_asteroid(a, predicted_collision_ts, game_state);
                            }

                            if (check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(
                                    asteroids_pending_death,
                                    initial_timestep + future_timesteps + predicted_collision_ts,
                                    game_state, future_asteroid_during_imminent_collision_time)) {
                                // Already eliminated by future time; skip
                                // In the future time the asteroid has already been eliminated, so there won't actually be a collision
                                // debug_print("In the future time the asteroid has already been eliminated, so there won't actually be a collision")
                                return; // continue to next asteroid
                            } else {
                                next_imminent_asteroid_collision_time = std::min(
                                    next_imminent_asteroid_collision_time, predicted_collision_time
                                );
                            }
                        } else {
                            // We're not eliminating this asteroid, and if it was forecasted, it comes into existence before our collision time. Therefore our collision is real and should be considered.
                            // We're not eliminating it, or it's forecasted, so collision is real
                            next_imminent_asteroid_collision_time = std::min(
                                next_imminent_asteroid_collision_time, predicted_collision_time
                            );
                        }
                    }
                    // else: asteroid is born after the predicted collision time
                    // There is no collision since the asteroid is born after our predicted collision time, and an unborn asteroid can't collide with anything
                }
            }
        };

        // We need to iterate over game_state.asteroids and then forecasted_asteroid_splits,
        // treating their indices as in Python's `enumerate(chain(...))`.
        size_t ast_idx = 0;
        // Iterate over game_state.asteroids
        for (const Asteroid& a : game_state.asteroids) {
            process_asteroid(a, true);
            ++ast_idx;
        }
        // ...then over forecasted_asteroid_splits
        for (const Asteroid& a : forecasted_asteroid_splits) {
            process_asteroid(a, false);
            ++ast_idx;
        }
        return next_imminent_asteroid_collision_time;
    }

    std::vector<std::pair<double, std::pair<double, double>>>
    get_next_extrapolated_mine_collision_times_and_pos() const {
        std::vector<std::pair<double, std::pair<double, double>>> times_and_mine_pos;
        for (const Mine& m : game_state.mines) {
            assert(m.alive);
            double next_imminent_mine_collision_time = predict_ship_mine_collision(
                ship_state.x, ship_state.y, m, 0
            );
            if (!std::isinf(next_imminent_mine_collision_time)) {
                times_and_mine_pos.emplace_back(
                    next_imminent_mine_collision_time,
                    std::make_pair(m.x, m.y)
                );
            }
        }
        return times_and_mine_pos;
    }

    double get_fitness() {
        // if (sim_id == 15869 || sim_id == 73186) {
        //     // Debug print trigger
        // }
        if (fitness_breakdown.has_value()) {
            throw std::runtime_error("Do not call get_fitness twice!");
        }

        // This is meant to be the last method called from this class. This is rather destructive!
        if (!is_close_to_zero(ship_state.speed)) {
            std::cerr << "ship_state.speed = " << ship_state.speed << std::endl;
            assert(false);
        }


        // This will return a scalar number representing how good of an action/state sequence we just went through
        // If these moves will keep us alive for a long time and shoot many asteroids along the way, then the fitness is good
        // If these moves result in us getting into a dangerous spot, or if we don't shoot many asteroids at all, then the fitness will be bad
        // The HIGHER the fitness score, the BETTER!
        // This fitness function is unified to consider all three types of moves. Stationary targeting, maneuvers, and respawn maneuvers.


        // ========== NESTED LAMBDAS ==========

        // Asteroid safe time fitness
        auto get_asteroid_safe_time_fitness = [&](double next_extrapolated_asteroid_collision_time, double displacement, double move_sequence_length_s) -> double {
            // NOTE THAT the move sequence length is discounted, because if we're deciding between maneuvers, we really mostly care about how long you're safe for after the maneuver is done!
            if (!std::isinf(game_state.time_limit) 
                && initial_timestep + future_timesteps + END_OF_SCENARIO_DONT_CARE_TIMESTEPS >= static_cast<int64_t>(std::floor(FPS * game_state.time_limit))) {
                return 1.0;
            } else if (ship_state.bullets_remaining == 0 && ship_state.mines_remaining == 0) {
                return 1.0;
            } else {
                if (displacement < EPS) {
                    return sigmoid(next_extrapolated_asteroid_collision_time + move_sequence_length_s * 0.25, 1.4, 3.0);
                } else {
                    return sigmoid(next_extrapolated_asteroid_collision_time + move_sequence_length_s * 0.25, 1.4, 3.0);
                }
            }
        };

        // How frequently did we shoot asteroids
        auto get_asteroid_shot_frequency_fitness = [&](int64_t asteroids_shot, double move_sequence_length_s) -> double {
            if (asteroids_shot < 0) {
                return -0.9;
            } else {
                double fudged_asteroids_shot = (asteroids_shot == 0) ? 0.1 : static_cast<double>(asteroids_shot);
                double time_per_asteroids_shot = move_sequence_length_s / fudged_asteroids_shot;
                double asteroids_fitness = sigmoid(time_per_asteroids_shot, -0.5 * FPS, 10.8*DELTA_TIME);
                return asteroids_fitness;
            }
        };

        // Mine safety
        auto get_mine_safety_fitness = [&](const std::vector<std::pair<double, std::pair<double, double>>>& next_extrapolated_mine_collision_times)
            -> std::pair<double, double> {
            if (next_extrapolated_mine_collision_times.empty()) {
                return std::make_pair(1.0, std::numeric_limits<double>::infinity());
            }
            if (!std::isinf(game_state.time_limit)
                && initial_timestep + future_timesteps + END_OF_SCENARIO_DONT_CARE_TIMESTEPS >= static_cast<int64_t>(std::floor(FPS * game_state.time_limit))) {
                return std::make_pair(1.0, std::numeric_limits<double>::infinity());
            }
            double mines_threat_level = 0.0;
            double next_extrapolated_mine_collision_time = std::numeric_limits<double>::infinity();
            for (const auto& mc_pair : next_extrapolated_mine_collision_times) {
                double mine_collision_time = mc_pair.first;
                const std::pair<double, double>& mine_pos = mc_pair.second;
                next_extrapolated_mine_collision_time = std::min(next_extrapolated_mine_collision_time, mine_collision_time);
                if constexpr (ENABLE_SANITY_CHECKS) {
                    assert(-EPS <= mine_collision_time && mine_collision_time <= MINE_FUSE_TIME + EPS);
                }
                double dist_to_ground_zero = dist(ship_state.x, ship_state.y, mine_pos.first, mine_pos.second);
                double mine_ground_zero_fudge = linear(dist_to_ground_zero, 0.0, 1.0, MINE_BLAST_RADIUS + SHIP_RADIUS, 0.6);
                mines_threat_level += std::pow(MINE_FUSE_TIME - next_extrapolated_mine_collision_time, 2.0) / 9.0 * mine_ground_zero_fudge;
            }
            double mine_safe_time_fitness = sigmoid(mines_threat_level, -6.8, 0.232);
            return std::make_pair(mine_safe_time_fitness, next_extrapolated_mine_collision_time);
        };

        // Asteroid aiming cone fitness
        auto get_asteroid_aiming_cone_fitness = [&]() -> double {
            if (!std::isinf(game_state.time_limit)
                && initial_timestep + future_timesteps + END_OF_SCENARIO_DONT_CARE_TIMESTEPS >= static_cast<int64_t>(std::floor(FPS * game_state.time_limit))) {
                return 1.0;
            } else if (ship_state.bullets_remaining == 0 && ship_state.mines_remaining == 0) {
                return 1.0;
            }
            double ship_heading_rad = ship_state.heading * DEG_TO_RAD;
            int asts_within_cone = 0;
            // Chain together game_state.asteroids and forecasted_asteroid_splits
            for (const auto& a : game_state.asteroids) {
                if (a.alive) {
                    if (heading_diff_within_threshold(
                            ship_heading_rad, a.x - ship_state.x, a.y - ship_state.y,
                            AIMING_CONE_FITNESS_CONE_WIDTH_HALF_COSINE)) {
                        ++asts_within_cone;
                    }
                }
            }
            for (const auto& a : forecasted_asteroid_splits) {
                if (a.alive) {
                    if (heading_diff_within_threshold(
                            ship_heading_rad, a.x - ship_state.x, a.y - ship_state.y,
                            AIMING_CONE_FITNESS_CONE_WIDTH_HALF_COSINE)) {
                        ++asts_within_cone;
                    }
                }
            }
            return sigmoid(asts_within_cone, 1.0, 2.4);
        };

        // Crash fitness
        auto get_crash_fitness = [&]() -> double {
            double crash_fitness = 1.0;
            if (!std::isinf(game_state.time_limit)
                && initial_timestep + future_timesteps + END_OF_SCENARIO_DONT_CARE_TIMESTEPS >= static_cast<int64_t>(std::floor(FPS * game_state.time_limit))) {
                crash_fitness = ship_crashed ? 1.0 : 0.0;
            } else {
                if (ship_crashed) {
                    if (ship_state.bullets_remaining == 0 && ship_state.mines_remaining == 0) {
                        crash_fitness = 1.0;
                    } else {
                        if (ship_state.lives_remaining >= 2) {
                            crash_fitness = 0.5;
                        } else if (ship_state.lives_remaining >= 1) {
                            crash_fitness = 0.2;
                        } else {
                            crash_fitness = 0.0;
                        }
                    }
                } else {
                    if (ship_state.bullets_remaining == 0 && ship_state.mines_remaining == 0) {
                        crash_fitness = 0.0;
                    } else {
                        crash_fitness = 1.0;
                    }
                }
            }
            return crash_fitness;
        };

        // Sequence length fitness
        auto get_sequence_length_fitness = [&](double move_sequence_length_s, double displacement) -> double {
            if (respawn_maneuver_pass_number > 0) {
                return sigmoid(move_sequence_length_s, -2.8, 1.7);
            } else {
                if (displacement < EPS) {
                    return sigmoid(move_sequence_length_s, -2.8, 1.7);
                } else {
                    return sigmoid(move_sequence_length_s, -5.7, 0.8);
                }
            }
        };

        // Other ship proximity
        auto get_other_ship_proximity_fitness = [&](const std::vector<std::pair<double, double>>& self_positions) -> double {
            bool invert_ship_affinity = false;
            // Complex logic for invert affinity -- assign as appropriate for your situation.
            if ((ship_state.bullets_remaining == 0 && ship_state.mines_remaining == 0)
                || (!other_ships.empty() &&
                    ((weighted_average(overall_fitness_record) > 0.55 && other_ships[0].lives_remaining == 1 && ship_state.lives_remaining >= 3)
                    || (weighted_average(overall_fitness_record) > 0.7 && other_ships[0].lives_remaining < ship_state.lives_remaining)))
            ) {
                invert_ship_affinity = true;
                explanation_messages.emplace_back("I'm either out of bullets/mines or the other ship is about to die, so I'm gonna try to crash into the other ship mwahahahaha");
            }

            for (const Ship& other_ship : other_ships) {
                double other_ship_speed = std::sqrt(other_ship.vx * other_ship.vx + other_ship.vy * other_ship.vy);
                double other_ship_speed_dist_mul = linear(other_ship_speed, 0.0, 1.0, SHIP_MAX_SPEED, 0.3);
                std::vector<double> separation_dists;
                for (const auto& self_pos : self_positions) {
                    double self_pos_x = self_pos.first;
                    double self_pos_y = self_pos.second;
                    double abs_sep_x = std::abs(self_pos_x - other_ship.x);
                    double abs_sep_y = std::abs(self_pos_y - other_ship.y);
                    double sep_x = std::min(abs_sep_x, game_state.map_size_x - abs_sep_x);
                    double sep_y = std::min(abs_sep_y, game_state.map_size_y - abs_sep_y);
                    double separation_dist = std::sqrt(sep_x*sep_x + sep_y*sep_y);
                    separation_dists.push_back(separation_dist);
                }
                double mean_separation_dist = weighted_harmonic_mean(separation_dists);
                double fit_val = invert_ship_affinity ?
                    1.0 - sigmoid(mean_separation_dist * other_ship_speed_dist_mul, 0.032, 120.0)
                    :      sigmoid(mean_separation_dist * other_ship_speed_dist_mul, 0.032, 120.0);
                return fit_val;
            }
            return 1.0;
        };

        // ========== MAIN BODY ==========

        // Only alive mines
        std::vector<Mine> filtered_mines;
        for (const auto& m : game_state.mines) {
            if (m.alive) filtered_mines.push_back(m);
        }
        game_state.mines = filtered_mines;

        // Get states before mutations
        std::vector<SimState> states = this->get_state_sequence();
        assert(states.size() > 0 && "States is empty! WTH?");

        double move_sequence_length_s = static_cast<double>(this->get_sequence_length() - 1) * DELTA_TIME;
        std::vector<std::pair<double, std::pair<double, double>>> next_extrapolated_mine_collision_times = this->get_next_extrapolated_mine_collision_times_and_pos();

        double mine_safe_time_fitness, next_extrapolated_mine_collision_time;
        std::tie(mine_safe_time_fitness, next_extrapolated_mine_collision_time) = get_mine_safety_fitness(next_extrapolated_mine_collision_times);

        double asteroids_fitness = get_asteroid_shot_frequency_fitness(asteroids_shot, move_sequence_length_s);
        double asteroid_aiming_cone_fitness = get_asteroid_aiming_cone_fitness();

        int64_t additional_timesteps_to_blow_up_mines = 0;
        double next_extrapolated_asteroid_collision_time = this->get_next_extrapolated_asteroid_collision_time();

        // If we have mines, simulate blowing them up
        if (!game_state.mines.empty()) {
            backed_up_game_state_before_post_mutation = game_state.copy();
            backed_up_game_state_before_post_mutation->asteroids.clear();
            for (const auto& a : game_state.asteroids) if (a.alive) backed_up_game_state_before_post_mutation->asteroids.push_back(a);
            backed_up_game_state_before_post_mutation->mines.clear();
            for (const auto& m : game_state.mines) if (m.alive) backed_up_game_state_before_post_mutation->mines.push_back(m);
            backed_up_game_state_before_post_mutation->bullets.clear();
            for (const auto& b : game_state.bullets) if (b.alive) backed_up_game_state_before_post_mutation->bullets.push_back(b);

            while (!game_state.mines.empty()) {
                additional_timesteps_to_blow_up_mines += 1;
                // Step simulation forward one tick at a time (disable shooting etc)
                this->update(0.0, 0.0, false, false, std::nullopt, true);
            }
        }

        double safe_time_after_maneuver_s;
        if (additional_timesteps_to_blow_up_mines == 0) {
            assert(std::isinf(next_extrapolated_mine_collision_time));
            safe_time_after_maneuver_s = std::min(next_extrapolated_asteroid_collision_time, next_extrapolated_mine_collision_time);
        } else {
            double next_extrapolated_asteroid_collision_time_after_mines = this->get_next_extrapolated_asteroid_collision_time(additional_timesteps_to_blow_up_mines);
            if (!std::isinf(next_extrapolated_asteroid_collision_time) &&
                next_extrapolated_asteroid_collision_time <= additional_timesteps_to_blow_up_mines * DELTA_TIME) {
                safe_time_after_maneuver_s = std::min({
                    next_extrapolated_asteroid_collision_time,
                    next_extrapolated_asteroid_collision_time_after_mines,
                    next_extrapolated_mine_collision_time
                });
            } else {
                safe_time_after_maneuver_s = std::min(
                    next_extrapolated_asteroid_collision_time_after_mines,
                    next_extrapolated_mine_collision_time
                );
            }
        }
        double overall_safe_time_fitness = sigmoid(safe_time_after_maneuver_s, 2.9, 1.4);

        // Ship path geometry
        double ship_start_position_x = states.front().ship_state.x;
        double ship_start_position_y = states.front().ship_state.y;
        double ship_end_position_x = states.back().ship_state.x;
        double ship_end_position_y = states.back().ship_state.y;

        double displacement;
        if (states.size() >= 2) {
            displacement = dist(ship_start_position_x, ship_start_position_y, ship_end_position_x, ship_end_position_y);
        } else {
            displacement = 0.0;
        }

        double asteroid_safe_time_fitness = get_asteroid_safe_time_fitness(next_extrapolated_asteroid_collision_time, displacement, move_sequence_length_s);

        // Determine "ship positions" for other ship proximity check
        std::vector<std::pair<double, double>> self_ship_positions;
        if (displacement < EPS || respawn_maneuver_pass_number > 0) {
            self_ship_positions.emplace_back(ship_end_position_x, ship_end_position_y);
        } else {
            for (const SimState& s : states) {
                self_ship_positions.emplace_back(s.ship_state.x, s.ship_state.y);
            }
        }

        double other_ship_proximity_fitness = get_other_ship_proximity_fitness(self_ship_positions);
        double sequence_length_fitness = get_sequence_length_fitness(move_sequence_length_s, displacement);
        double crash_fitness = get_crash_fitness();
        double placed_mine_fitness = 0.0;
        if (sim_placed_a_mine) {
            // Uncommented logic from Py, can adjust if necessary.
            if (ship_state.lives_remaining >= 3) {
                placed_mine_fitness = 1.0;
            } else {
                placed_mine_fitness = mine_safe_time_fitness;
            }
        }

        // ====== STATUS/SAFETY MESSAGES ======
        if (asteroid_safe_time_fitness < 0.1) {
            safety_messages.push_back("I'm dangerously close to being hit by asteroids if I stay here (Imminent collision in " + std::to_string(next_extrapolated_asteroid_collision_time) + "s). Trying my hardest to maneuver out of this situation.");
        } else if (asteroid_safe_time_fitness < 0.4) {
            safety_messages.push_back("I'm close to being hit by asteroids if I stay here (Imminent collision in " + std::to_string(next_extrapolated_asteroid_collision_time) + "s).");
        } else if (asteroid_safe_time_fitness < 0.8) {
            safety_messages.push_back("I'll eventually get hit by asteroids if I stay here (Imminent collision in " + std::to_string(next_extrapolated_asteroid_collision_time) + "s). Keeping my eye out for a dodge maneuver.");
        }
        if (mine_safe_time_fitness < 0.1) {
            safety_messages.push_back("I'm dangerously close to being kablooied by a mine (Imminent blast in " + std::to_string(next_extrapolated_mine_collision_time) + "s). Trying my hardest to maneuver out of this situation.");
        } else if (mine_safe_time_fitness < 0.4) {
            safety_messages.push_back("I'm close to being boomed by a mine (Imminent blast in " + std::to_string(next_extrapolated_mine_collision_time) + "s).");
        } else if (mine_safe_time_fitness < 0.9) {
            safety_messages.push_back("I'm within the radius of a mine (Imminent blast in " + std::to_string(next_extrapolated_mine_collision_time) + "s).");
        }
        if (other_ship_proximity_fitness < 0.2) {
            safety_messages.push_back("I'm dangerously close to the other ship. Get away from me!");
        } else if (other_ship_proximity_fitness < 0.5) {
            safety_messages.push_back("I'm near the other ship. Being cautious.");
        }
        // ================================

        // Final assertion checks
        if constexpr (ENABLE_SANITY_CHECKS) {
            assert(asteroid_safe_time_fitness >= 0.0 && asteroid_safe_time_fitness <= 1.0);
            assert(mine_safe_time_fitness >= 0.0 && mine_safe_time_fitness <= 1.0);
            assert(asteroids_fitness >= -1.0 && asteroids_fitness <= 1.0);
            assert(sequence_length_fitness >= 0.0 && sequence_length_fitness <= 1.0);
            assert(other_ship_proximity_fitness >= 0.0 && other_ship_proximity_fitness <= 1.0);
            assert(crash_fitness >= 0.0 && crash_fitness <= 1.0);
            assert(asteroid_aiming_cone_fitness >= 0.0 && asteroid_aiming_cone_fitness <= 1.0);
            assert(placed_mine_fitness >= 0.0 && placed_mine_fitness <= 1.0);
        }
        //assert(1 == 2 && "we got to the 1==2lmao");
        // Store the breakdown, and compute the overall fitness
        this->fitness_breakdown = {
            asteroid_safe_time_fitness,
            mine_safe_time_fitness,
            asteroids_fitness,
            sequence_length_fitness,
            other_ship_proximity_fitness,
            crash_fitness,
            asteroid_aiming_cone_fitness,
            placed_mine_fitness,
            overall_safe_time_fitness
        };

        assert(fitness_breakdown.has_value() && "fitness_breakdown should have just been set!");

        std::vector<double> fitness_breakdown_vector(fitness_breakdown.value().begin(), fitness_breakdown.value().end());

        // Pick weights
        std::vector<double> fitness_weights = fitness_function_weights.has_value()
            ? std::vector<double>(fitness_function_weights->begin(), fitness_function_weights->end())
            : std::vector<double>(DEFAULT_FITNESS_WEIGHTS.begin(), DEFAULT_FITNESS_WEIGHTS.end());

        double overall_fitness = weighted_harmonic_mean(fitness_breakdown_vector, &fitness_weights, 1.0);
        assert((overall_fitness >= 0.0 && overall_fitness <= 1.0) || asteroids_fitness < 0.0);

        if (overall_fitness > 0.9) {
            safety_messages.push_back("I'm safe and chilling");
        }
        return overall_fitness;
    }

    std::array<double, 9>
    get_fitness_breakdown() const {
        assert(fitness_breakdown.has_value() && "fitness_breakdown must not be null");
        return fitness_breakdown.value();
    }

    std::optional<Target>
    find_extreme_shooting_angle_error(const std::vector<Target>& asteroid_list, double threshold, const std::string& mode = "largest_below") const {
        // Extract the angle values
        std::vector<double> shooting_angles;
        shooting_angles.reserve(asteroid_list.size());
        for (const Target& d : asteroid_list) {
            shooting_angles.push_back(d.shooting_angle_error_deg);
        }

        size_t idx = 0;
        if (mode == "largest_below") {
            // std::lower_bound returns the iterator to the first element >= threshold
            auto it = std::lower_bound(shooting_angles.begin(), shooting_angles.end(), threshold);
            idx = static_cast<size_t>(it - shooting_angles.begin());
            if (idx > 0) {
                --idx;
            } else {
                return std::nullopt;
            }
        } else if (mode == "smallest_above") {
            // std::upper_bound returns iterator to first element > threshold
            auto it = std::upper_bound(shooting_angles.begin(), shooting_angles.end(), threshold);
            idx = static_cast<size_t>(it - shooting_angles.begin());
            if (idx >= asteroid_list.size()) {
                return std::nullopt;
            }
        } else {
            throw std::invalid_argument("Invalid mode. Choose 'largest_below' or 'smallest_above'");
        }

        return asteroid_list[idx];
    }

    bool target_selection() {
        // The job of this method is to calculate how to hit each asteroid, and then pick the best one to try to target

        // --- Simulate shooting at a particular target, returning all the information the inner Python function did ---
        auto simulate_shooting_at_target =
            [&](const Asteroid& target_asteroid_original, double target_asteroid_shooting_angle_error_deg, double target_asteroid_interception_time_s, int64_t target_asteroid_turning_timesteps)
                -> std::tuple<std::optional<Asteroid>, std::vector<Action>, Asteroid, double, double, int64_t, std::optional<int64_t>, Ship>
        {
            std::vector<Action> aiming_move_sequence = get_rotate_heading_move_sequence(target_asteroid_shooting_angle_error_deg);

            int64_t timesteps_until_can_fire;
            if (fire_first_timestep) {
                timesteps_until_can_fire = std::max(int64_t(0), int64_t(FIRE_COOLDOWN_TS) - int64_t(aiming_move_sequence.size()));
            } else {
                timesteps_until_can_fire = std::max(int64_t(0), int64_t(FIRE_COOLDOWN_TS) -
                    (int64_t(initial_timestep) + int64_t(future_timesteps) + int64_t(aiming_move_sequence.size()) - int64_t(last_timestep_fired)));
            }
            for (int64_t i = 0; i < timesteps_until_can_fire; ++i)
                aiming_move_sequence.push_back(Action{0.0, 0.0, false, false, 0});

            int64_t asteroid_advance_timesteps = int64_t(aiming_move_sequence.size());
            if constexpr (ENABLE_SANITY_CHECKS) {
                assert(asteroid_advance_timesteps <= target_asteroid_turning_timesteps || target_asteroid_turning_timesteps == 0);
            }
            if (asteroid_advance_timesteps < target_asteroid_turning_timesteps) {
                for (int64_t i = 0; i < (target_asteroid_turning_timesteps - asteroid_advance_timesteps); ++i)
                    aiming_move_sequence.push_back(Action{0.0, 0.0, false, false, 0});
            }
            Asteroid target_asteroid = time_travel_asteroid(target_asteroid_original, asteroid_advance_timesteps, game_state);

            Ship ship_state_after_aiming = get_ship_state();
            ship_state_after_aiming.heading = std::fmod(ship_state_after_aiming.heading + target_asteroid_shooting_angle_error_deg, 360.0);

            //std::optional<Asteroid> actual_asteroid_hit;
            //std::optional<int64_t> timesteps_until_bullet_hit_asteroid;
            //bool ignored_bool;
            auto [actual_asteroid_hit, timesteps_until_bullet_hit_asteroid, ignored_bool] =
                bullet_sim(ship_state_after_aiming, fire_first_timestep, aiming_move_sequence.size());
            return std::make_tuple(
                actual_asteroid_hit,
                aiming_move_sequence,
                target_asteroid,
                target_asteroid_shooting_angle_error_deg,
                target_asteroid_interception_time_s,
                target_asteroid_turning_timesteps,
                timesteps_until_bullet_hit_asteroid,
                ship_state_after_aiming
            );
        };

        // --- Target acquisition loop ---

        std::vector<Target> target_asteroids_list;
        Ship dummy_ship_state(
            /*is_respawning=*/false,
            /*x=*/ship_state.x,
            /*y=*/ship_state.y,
            /*vx=*/0.0,
            /*vy=*/0.0,
            /*speed=*/0.0,
            /*heading=*/ship_state.heading,
            /*mass=*/300.0,
            /*radius=*/20.0,
            /*id=*/ship_state.id,
            /*team=*/ship_state.team,
            /*lives_remaining=*/123,
            /*bullets_remaining=*/0,
            /*mines_remaining=*/0,
            /*can_fire=*/ship_state.can_fire,
            /*fire_rate=*/10.0,
            /*can_deploy_mine=*/false, /* note: need to set properly if used */
            /*mine_deploy_rate=*/0.0,
            /*thrust_range=*/std::make_pair(-480.0, 480.0),
            /*turn_rate_range=*/std::make_pair(-180.0, 180.0),
            /*max_speed=*/240,
            /*drag=*/80.0
        );

        int64_t timesteps_until_can_fire;
        if (fire_first_timestep) {
            timesteps_until_can_fire = FIRE_COOLDOWN_TS;
        } else {
            timesteps_until_can_fire = std::max<int64_t>(0, FIRE_COOLDOWN_TS - (initial_timestep + future_timesteps - last_timestep_fired));
        }

        bool most_imminent_asteroid_exists = false;
        bool asteroids_still_exist = false;
        std::vector<const Asteroid*> chained_asteroids;
        // TODO: Optimize this
        for (const auto& a : game_state.asteroids) chained_asteroids.push_back(&a);
        for (const auto& a : forecasted_asteroid_splits) chained_asteroids.push_back(&a);

        for (const Asteroid* astptr : chained_asteroids) {
            const Asteroid& asteroid = *astptr;
            if (asteroid.alive) {
                if (check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(asteroids_pending_death, initial_timestep + future_timesteps, game_state, asteroid)) {
                    asteroids_still_exist = true;
                    std::optional<std::tuple<bool, double, int64_t, double, double, double, double>> best_feasible_unwrapped_target;
                    bool asteroid_will_get_hit_by_my_mine = false, asteroid_will_get_hit_by_their_mine = false;

                    for (const auto& m : game_state.mines) {
                        assert(m.alive);
                        Asteroid asteroid_when_mine_explodes = time_travel_asteroid_s(asteroid, m.remaining_time, game_state);
                        double delta_x = asteroid_when_mine_explodes.x - m.x;
                        double delta_y = asteroid_when_mine_explodes.y - m.y;
                        double separation = asteroid_when_mine_explodes.radius + MINE_BLAST_RADIUS;
                        if (std::abs(delta_x) <= separation && std::abs(delta_y) <= separation && delta_x*delta_x + delta_y*delta_y <= separation*separation) {
                            if (mine_positions_placed.count(std::make_pair(m.x, m.y))) {
                                asteroid_will_get_hit_by_my_mine = true;
                                if (asteroid_will_get_hit_by_their_mine) break;
                            } else {
                                asteroid_will_get_hit_by_their_mine = true;
                                if (asteroid_will_get_hit_by_my_mine) break;
                            }
                        }
                    }
                    std::vector<Asteroid> unwrapped_asteroids = unwrap_asteroid(asteroid, game_state.map_size_x, game_state.map_size_y, UNWRAP_ASTEROID_TARGET_SELECTION_TIME_HORIZON, true);
                    for (const Asteroid& a : unwrapped_asteroids) {
                        bool feasible;
                        double shooting_angle_error_deg, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception;
                        int64_t aiming_timesteps_required;
                        std::tie(feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception)
                            = solve_interception(a, dummy_ship_state, game_state, timesteps_until_can_fire);
                        if (feasible) {
                            assert(aiming_timesteps_required >= 0);
                            if (!best_feasible_unwrapped_target.has_value() || aiming_timesteps_required < std::get<2>(best_feasible_unwrapped_target.value())) {
                                best_feasible_unwrapped_target = std::make_tuple(feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception);
                            }
                        }
                    }
                    if (best_feasible_unwrapped_target.has_value()) {
                        bool feasible; double shooting_angle_error_deg, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception; int64_t aiming_timesteps_required;
                        std::tie(feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception) = best_feasible_unwrapped_target.value();
                        double imminent_collision_time_s = std::numeric_limits<double>::infinity();
                        assert(is_close_to_zero(ship_state.vx) && is_close_to_zero(ship_state.vy));
                        for (const Asteroid& a : unwrapped_asteroids) {
                            imminent_collision_time_s = std::min(imminent_collision_time_s,
                                predict_next_imminent_collision_time_with_asteroid(
                                    ship_state.x, ship_state.y, ship_state.vx, ship_state.vy, SHIP_RADIUS,
                                    a.x, a.y, a.vx, a.vy, a.radius, game_state));
                        }
                        target_asteroids_list.emplace_back(
                            Target{
                                asteroid, feasible, shooting_angle_error_deg, aiming_timesteps_required,
                                interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception,
                                imminent_collision_time_s, asteroid_will_get_hit_by_my_mine, asteroid_will_get_hit_by_their_mine
                            });
                        if (imminent_collision_time_s < std::numeric_limits<double>::infinity())
                            most_imminent_asteroid_exists = true;
                    }
                }
            }
        }

        double turn_angle_deg_until_can_fire = double(timesteps_until_can_fire)*SHIP_MAX_TURN_RATE*DELTA_TIME;
        std::optional<Asteroid> actual_asteroid_hit;
        std::vector<Action> aiming_move_sequence;
        Asteroid target_asteroid, target_asteroid_when_firing, actual_asteroid_hit_when_firing, actual_asteroid_hit_at_present_time;
        double target_asteroid_shooting_angle_error_deg = 0, target_asteroid_interception_time_s = 0;
        int64_t target_asteroid_turning_timesteps = 0;
        std::optional<int64_t> timesteps_until_bullet_hit_asteroid;
        Ship ship_state_after_aiming;

        if (most_imminent_asteroid_exists) {
            std::vector<Target> sorted_imminent_targets = target_asteroids_list;
            if (!other_ships.empty()) {
                double frontrun_score_multiplier = ship_state.bullets_remaining > 0 ? 4.0 : 3.0;
                std::sort(sorted_imminent_targets.begin(), sorted_imminent_targets.end(), [&](const Target& t1, const Target& t2) {
                    auto score = [&](const Target& t) {
                        return std::min(10.0, t.imminent_collision_time_s) +
                            ASTEROID_SIZE_SHOT_PRIORITY[t.asteroid.size]*0.25 +
                            t.interception_time_s +
                            t.asteroid_dist_during_interception/400.0 +
                            frontrun_score_multiplier*std::min(0.5, std::max(0.0, t.interception_time_s - get_adversary_interception_time_lower_bound(t.asteroid, other_ships, game_state))) +
                            ((t.asteroid.size == 1 ? 5.0 : -5.0) * (t.asteroid_will_get_hit_by_my_mine ? 1.0 : 0.0)) +
                            ((t.asteroid.size != 1 ? 3.0 : -3.0) * (t.asteroid_will_get_hit_by_their_mine ? 1.0 : 0.0));
                    };
                    return score(t1) < score(t2);
                });
            } else {
                std::sort(sorted_imminent_targets.begin(), sorted_imminent_targets.end(), [&](const Target& t1, const Target& t2) {
                    auto score = [&](const Target& t) {
                        return std::min(10.0, t.imminent_collision_time_s) +
                            ASTEROID_SIZE_SHOT_PRIORITY[t.asteroid.size]*0.25 +
                            t.asteroid_dist_during_interception/400.0 +
                            ((t.asteroid.size == 1 ? 5.0 : -5.0) * (t.asteroid_will_get_hit_by_my_mine ? 1.0 : 0.0));
                    };
                    return score(t1) < score(t2);
                });
            }
            for (const Target& candidate_target : sorted_imminent_targets) {
                if (std::isinf(candidate_target.imminent_collision_time_s)) break;
                int64_t most_imminent_asteroid_aiming_timesteps = candidate_target.aiming_timesteps_required;
                Asteroid most_imminent_asteroid = candidate_target.asteroid;
                double most_imminent_asteroid_shooting_angle_error_deg = candidate_target.shooting_angle_error_deg;
                double most_imminent_asteroid_interception_time_s = candidate_target.interception_time_s;

                if (most_imminent_asteroid_aiming_timesteps <= timesteps_until_can_fire) {
                    if constexpr (ENABLE_SANITY_CHECKS) {
                        assert(most_imminent_asteroid_aiming_timesteps == timesteps_until_can_fire);
                    }
                    std::tie(actual_asteroid_hit, aiming_move_sequence, target_asteroid,
                        target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s,
                        target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid, ship_state_after_aiming) =
                            simulate_shooting_at_target(most_imminent_asteroid, most_imminent_asteroid_shooting_angle_error_deg, most_imminent_asteroid_interception_time_s, most_imminent_asteroid_aiming_timesteps);
                    if (actual_asteroid_hit.has_value()) {
                        assert(timesteps_until_bullet_hit_asteroid.has_value());
                        int64_t len_aiming_move_sequence = aiming_move_sequence.size();
                        actual_asteroid_hit_when_firing = time_travel_asteroid(actual_asteroid_hit.value(), len_aiming_move_sequence - timesteps_until_bullet_hit_asteroid.value(), game_state);
                        if (check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(asteroids_pending_death, initial_timestep + future_timesteps + len_aiming_move_sequence, game_state, actual_asteroid_hit_when_firing)) {
                            break;
                        }
                    }
                } else {
                    // Not enough time to aim; try for "closest along the way"
                    std::vector<Target> sorted_targets = target_asteroids_list;
                    std::sort(sorted_targets.begin(), sorted_targets.end(), [](const Target& t1, const Target& t2) {
                        int a1 = int(std::round(t1.shooting_angle_error_deg)), a2 = int(std::round(t2.shooting_angle_error_deg));
                        auto p1 = std::make_pair(a1, ASTEROID_SIZE_SHOT_PRIORITY[t1.asteroid.size]);
                        auto p2 = std::make_pair(a2, ASTEROID_SIZE_SHOT_PRIORITY[t2.asteroid.size]);
                        return p1 < p2;
                    });
                    std::optional<Target> target;
                    if (most_imminent_asteroid_shooting_angle_error_deg > 0.0) {
                        target = find_extreme_shooting_angle_error(sorted_targets, turn_angle_deg_until_can_fire, "largest_below");
                        if (!target.has_value() || target->shooting_angle_error_deg < 0.0 ||
                            target->shooting_angle_error_deg < turn_angle_deg_until_can_fire - TARGETING_AIMING_UNDERTURN_ALLOWANCE_DEG) {
                            target = find_extreme_shooting_angle_error(sorted_targets, turn_angle_deg_until_can_fire, "smallest_above");
                        }
                    } else {
                        target = find_extreme_shooting_angle_error(sorted_targets, -turn_angle_deg_until_can_fire, "smallest_above");
                        if (!target.has_value() || target->shooting_angle_error_deg > 0.0 ||
                            target->shooting_angle_error_deg > -turn_angle_deg_until_can_fire + TARGETING_AIMING_UNDERTURN_ALLOWANCE_DEG) {
                            target = find_extreme_shooting_angle_error(sorted_targets, -turn_angle_deg_until_can_fire, "largest_below");
                        }
                    }
                    if (target.has_value()) {
                        std::tie(actual_asteroid_hit, aiming_move_sequence, target_asteroid,
                            target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s,
                            target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid, ship_state_after_aiming) =
                            simulate_shooting_at_target(target->asteroid, target->shooting_angle_error_deg, target->interception_time_s, target->aiming_timesteps_required);
                        if (actual_asteroid_hit.has_value()) {
                            assert(timesteps_until_bullet_hit_asteroid.has_value());
                            int64_t len_aiming_move_sequence = aiming_move_sequence.size();
                            actual_asteroid_hit_when_firing = time_travel_asteroid(actual_asteroid_hit.value(), len_aiming_move_sequence - timesteps_until_bullet_hit_asteroid.value(), game_state);
                            if (check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(asteroids_pending_death, initial_timestep + future_timesteps + len_aiming_move_sequence, game_state, actual_asteroid_hit_when_firing)) {
                                break;
                            }
                        }
                    } else {
                        // Waste shot: turn all the way
                        std::tie(actual_asteroid_hit, aiming_move_sequence, target_asteroid,
                            target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s,
                            target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid, ship_state_after_aiming) =
                            simulate_shooting_at_target(most_imminent_asteroid, most_imminent_asteroid_shooting_angle_error_deg, most_imminent_asteroid_interception_time_s, most_imminent_asteroid_aiming_timesteps);
                        if (actual_asteroid_hit.has_value()) {
                            assert(timesteps_until_bullet_hit_asteroid.has_value());
                            int64_t len_aiming_move_sequence = aiming_move_sequence.size();
                            actual_asteroid_hit_when_firing = time_travel_asteroid(actual_asteroid_hit.value(), len_aiming_move_sequence - timesteps_until_bullet_hit_asteroid.value(), game_state);
                            if (check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(asteroids_pending_death, initial_timestep + future_timesteps + len_aiming_move_sequence, game_state, actual_asteroid_hit_when_firing)) {
                                break;
                            }
                        }
                    }
                }
            }
        }

        if (!actual_asteroid_hit.has_value()) {
            // Try for least aiming delay targets
            if (!target_asteroids_list.empty()) {
                explanation_messages.push_back("No asteroids on collision course with me. Shooting at asteroids with least turning delay.");
                std::vector<Target> sorted_targets = target_asteroids_list;
                if (!other_ships.empty()) {
                    double frontrun_score_multiplier = ship_state.bullets_remaining > 0 ? 25.0 : 15.0;
                    std::sort(sorted_targets.begin(), sorted_targets.end(), [&](const Target& t1, const Target& t2) {
                        auto score = [&](const Target& t) {
                            return double(t.aiming_timesteps_required)*2.0 +
                                ASTEROID_SIZE_SHOT_PRIORITY[t.asteroid.size] +
                                t.interception_time_s +
                                t.asteroid_dist_during_interception/400.0 +
                                frontrun_score_multiplier*std::min(0.5, std::max(0.0, t.interception_time_s - get_adversary_interception_time_lower_bound(t.asteroid, other_ships, game_state))) +
                                ((t.asteroid.size == 1 ? 20.0 : -20.0) * (t.asteroid_will_get_hit_by_my_mine ? 1.0 : 0.0)) +
                                ((t.asteroid.size != 1 ? 20.0 : -20.0) * (t.asteroid_will_get_hit_by_their_mine ? 1.0 : 0.0));
                        };
                        return score(t1) < score(t2);
                    });
                } else {
                    std::sort(sorted_targets.begin(), sorted_targets.end(), [&](const Target& t1, const Target& t2) {
                        auto score = [&](const Target& t) {
                            return double(t.aiming_timesteps_required)*2.0
                                + ASTEROID_SIZE_SHOT_PRIORITY[t.asteroid.size]
                                + t.asteroid_dist_during_interception/400.0
                                + ((t.asteroid.size == 1 ? 20.0 : -20.0) * (t.asteroid_will_get_hit_by_my_mine ? 1.0 : 0.0));
                        };
                        return score(t1) < score(t2);
                    });
                }
                for (const Target& confirmed_target : sorted_targets) {
                    Asteroid least_shot_delay_asteroid = confirmed_target.asteroid;
                    double least_shot_delay_asteroid_shooting_angle_error_deg = confirmed_target.shooting_angle_error_deg;
                    double least_shot_delay_asteroid_interception_time_s = confirmed_target.interception_time_s;
                    int64_t least_shot_delay_asteroid_aiming_timesteps = confirmed_target.aiming_timesteps_required;

                    std::tie(actual_asteroid_hit, aiming_move_sequence, target_asteroid,
                        target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s,
                        target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid, ship_state_after_aiming) =
                        simulate_shooting_at_target(least_shot_delay_asteroid, least_shot_delay_asteroid_shooting_angle_error_deg,
                            least_shot_delay_asteroid_interception_time_s, least_shot_delay_asteroid_aiming_timesteps);
                    if (actual_asteroid_hit.has_value()) {
                        assert(timesteps_until_bullet_hit_asteroid.has_value());
                        int64_t len_aiming_move_sequence = aiming_move_sequence.size();
                        actual_asteroid_hit_when_firing = time_travel_asteroid(actual_asteroid_hit.value(), len_aiming_move_sequence - timesteps_until_bullet_hit_asteroid.value(), game_state);
                        if (check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(asteroids_pending_death, initial_timestep + future_timesteps + len_aiming_move_sequence, game_state, actual_asteroid_hit_when_firing)) {
                            break;
                        }
                    }
                }
            } else {
                explanation_messages.push_back("There's nothing I can feasibly shoot at!");
                int turn_direction = 0; double idle_thrust = 0.0;
                if (asteroids_still_exist) {
                    if (randint(1, 10) == 1) {
                        explanation_messages.push_back("Asteroids exist but we can't hit them. Moving around a bit randomly.");
                        asteroids_shot -= 1;
                    }
                }
                bool sim_complete_without_crash = update(idle_thrust, SHIP_MAX_TURN_RATE*double(turn_direction), false, false);
                assert(is_close_to_zero(ship_state.speed));
                return sim_complete_without_crash;
            }
        }

        if (!actual_asteroid_hit.has_value() ||
            !check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(
                asteroids_pending_death,
                initial_timestep + future_timesteps + aiming_move_sequence.size(),
                game_state,
                time_travel_asteroid((actual_asteroid_hit.has_value() ? actual_asteroid_hit.value() : Asteroid()), aiming_move_sequence.size() - (timesteps_until_bullet_hit_asteroid.has_value() ? timesteps_until_bullet_hit_asteroid.value() : 0), game_state))) {
            this->fire_next_timestep_flag = false;

            int turn_direction = 0; double idle_thrust = 0.0;
            if (asteroids_still_exist) {
                if (randint(1, 10) == 1) {
                    explanation_messages.push_back("Asteroids exist but we can't hit them. Moving around a bit randomly.");
                    asteroids_shot -= 1;
                }
            }
            bool sim_complete_without_crash = update(idle_thrust, SHIP_MAX_TURN_RATE*double(turn_direction), false, false);
            assert(is_close_to_zero(ship_state.speed));
            return sim_complete_without_crash;
        } else {
            assert(timesteps_until_bullet_hit_asteroid.has_value());
            if constexpr (ENABLE_SANITY_CHECKS) {
                assert(check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(
                    asteroids_pending_death, initial_timestep + future_timesteps + timesteps_until_bullet_hit_asteroid.value(), game_state, actual_asteroid_hit.value()));
                assert(check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(
                    asteroids_pending_death, initial_timestep + future_timesteps + aiming_move_sequence.size(), game_state,
                    time_travel_asteroid(actual_asteroid_hit.value(), aiming_move_sequence.size() - timesteps_until_bullet_hit_asteroid.value(), game_state)));
            }
            actual_asteroid_hit_at_present_time = time_travel_asteroid(actual_asteroid_hit.value(), -timesteps_until_bullet_hit_asteroid.value(), game_state);
            //if (game_state_plotter.has_value() && GAMESTATE_PLOTTING && NEXT_TARGET_PLOTTING && (!START_GAMESTATE_PLOTTING_AT_SECOND.has_value() || START_GAMESTATE_PLOTTING_AT_SECOND.value()*FPS <= double(initial_timestep + future_timesteps))) {
            //    actual_asteroid_hit_at_present_time = time_travel_asteroid(actual_asteroid_hit.value(), -timesteps_until_bullet_hit_asteroid.value() - 1, game_state);
            //    game_state_plotter.value().update_plot(nullptr, nullptr, nullptr, nullptr,std::vector<Asteroid>{actual_asteroid_hit_at_present_time},nullptr, nullptr, nullptr, false, NEW_TARGET_PLOT_PAUSE_TIME_S, "FEASIBLE TARGETS");
            //}
            if (actual_asteroid_hit_at_present_time.size != 1) {
                std::apply([&](const auto&... asteroids) {
                    (forecasted_asteroid_splits.push_back(asteroids), ...);
                }, forecast_asteroid_bullet_splits_from_heading(
                    actual_asteroid_hit_at_present_time,
                    timesteps_until_bullet_hit_asteroid.value(),
                    ship_state_after_aiming.heading,
                    game_state));
            }
            bool sim_complete_without_crash = apply_move_sequence(aiming_move_sequence);
            if (sim_complete_without_crash) {
                asteroids_shot += 1;
                this->fire_next_timestep_flag = true;
                assert(future_timesteps == future_timesteps); // Remove if wrong
                asteroids_pending_death_history[initial_timestep + future_timesteps] = asteroids_pending_death;
                //std::cout << "Tracking from targ sel in sim number " << std::to_string(this->sim_id) << std::endl;
                track_asteroid_we_shot_at(asteroids_pending_death, initial_timestep + future_timesteps, game_state, timesteps_until_bullet_hit_asteroid.value() - aiming_move_sequence.size(), actual_asteroid_hit_when_firing);
            } else {
                this->fire_next_timestep_flag = false;
            }
            assert(is_close_to_zero(ship_state.speed));
            return sim_complete_without_crash;
        }
    }

    std::tuple<std::optional<Asteroid>, int64_t, bool>
    bullet_sim(
        const std::optional<Ship>& ship_state = std::nullopt,
        bool fire_first_timestep = false,
        int64_t fire_after_timesteps = 0,
        bool skip_half_of_first_cycle = false,
        const std::optional<int64_t>& current_move_index = std::nullopt,
        const std::optional<std::vector<Action>>& whole_move_sequence = std::nullopt,
        int64_t timestep_limit = INT_INF,
        const std::optional<std::vector<Asteroid>>& asteroids_to_check = std::nullopt
        ) const
    {
        // This simulates shooting at an asteroid to tell us whether we'll hit it, when we hit it, and which asteroid we hit

        // Copy/shallow copy
        // TODO: See whether this copy and alive checks are necessary!
        std::vector<Asteroid> asteroids;
        if (asteroids_to_check.has_value()) {
            for (const auto& a : asteroids_to_check.value())
                if (a.alive) asteroids.push_back(a);
        } else {
            for (const auto& a : game_state.asteroids)
                if (a.alive) asteroids.push_back(a);
        }
        std::vector<Mine> mines;
        for (const auto& m : game_state.mines)
            if (m.alive) mines.push_back(m);
        std::vector<Bullet> bullets;
        for (const auto& b : game_state.bullets)
            if (b.alive) bullets.push_back(b);

        Ship initial_ship_state = get_ship_state();
        if constexpr (ENABLE_SANITY_CHECKS) {
            if (ship_state.has_value()) {
                assert(check_coordinate_bounds(game_state, ship_state.value().x, ship_state.value().y));
            }
        }

        std::optional<Ship> bullet_sim_ship_state = std::nullopt;
        if (whole_move_sequence.has_value())
            bullet_sim_ship_state = get_ship_state();

        std::optional<Bullet> my_bullet = std::nullopt;
        bool ship_not_collided_with_asteroid = true;
        int64_t timesteps_until_bullet_hit_asteroid = skip_half_of_first_cycle ? int64_t(-1) : int64_t(0);
        //std::set<int64_t> asteroid_remove_idxs;
        double new_bullet_x, new_bullet_y;
        double rad_heading, cos_heading, sin_heading;
        double thrust, turn_rate, drag_amount;
        double bullet_fired_from_ship_heading, bullet_fired_from_ship_position_x, bullet_fired_from_ship_position_y;
        double delta_x, delta_y, separation;
        double ship_position_x, ship_position_y;
        std::vector<Asteroid> new_asteroids;
        new_asteroids.reserve(3);
        while (true) {
            // Simplified update() simulation loop
            // Step the simulation.
            timesteps_until_bullet_hit_asteroid += 1;
            if (timesteps_until_bullet_hit_asteroid > timestep_limit) {
                return std::make_tuple(std::nullopt, int64_t(-1), ship_not_collided_with_asteroid);
            }

            // (plotting code skipped for clarity but you can add it here)

            // Advance bullets
            if (!(skip_half_of_first_cycle && timesteps_until_bullet_hit_asteroid == 0)) {
                for (Bullet& b : bullets) {
                    if (b.alive) {
                        new_bullet_x = b.x + b.vx * DELTA_TIME;
                        new_bullet_y = b.y + b.vy * DELTA_TIME;
                        //if check_coordinate_bounds(self.game_state, new_bullet_x, new_bullet_y):
                        if (0.0 <= new_bullet_x && new_bullet_x <= game_state.map_size_x && 0.0 <= new_bullet_y && new_bullet_y <= game_state.map_size_y) {
                            b.x = new_bullet_x;
                            b.y = new_bullet_y;
                        } else {
                            b.alive = false;
                        }
                    }
                }
                if (my_bullet.has_value()) {
                    new_bullet_x = my_bullet->x + my_bullet->vx * DELTA_TIME;
                    new_bullet_y = my_bullet->y + my_bullet->vy * DELTA_TIME;
                    if (0.0 <= new_bullet_x && new_bullet_x <= game_state.map_size_x && 0.0 <= new_bullet_y && new_bullet_y <= game_state.map_size_y) {
                        my_bullet->x = new_bullet_x;
                        my_bullet->y = new_bullet_y;
                    } else {
                        // The bullet got shot into the void without hitting anything :(
                        return std::make_tuple(std::nullopt, int64_t(-1), ship_not_collided_with_asteroid);
                    }
                }

                for (Mine& m : mines)
                    if (m.alive)
                        m.remaining_time -= DELTA_TIME;
                
                for (Asteroid& a : asteroids) {
                    if (a.alive) {
                        a.x = pymod(a.x + a.vx * DELTA_TIME, game_state.map_size_x);
                        a.y = pymod(a.y + a.vy * DELTA_TIME, game_state.map_size_y);
                    }
                }
            }

            // Create the initial bullet we fire, if we're locked in
            if (fire_first_timestep && timesteps_until_bullet_hit_asteroid + (skip_half_of_first_cycle ? 0 : -1) == 0) {
                rad_heading = initial_ship_state.heading * DEG_TO_RAD;
                cos_heading = std::cos(rad_heading);
                sin_heading = std::sin(rad_heading);
                new_bullet_x = initial_ship_state.x + SHIP_RADIUS * cos_heading;
                new_bullet_y = initial_ship_state.y + SHIP_RADIUS * sin_heading;
                if (0.0 <= new_bullet_x && new_bullet_x <= game_state.map_size_x && 0.0 <= new_bullet_y && new_bullet_y <= game_state.map_size_y) {
                    Bullet initial_timestep_fire_bullet(
                        new_bullet_x, new_bullet_y, BULLET_SPEED*cos_heading, BULLET_SPEED*sin_heading,
                        initial_ship_state.heading, BULLET_MASS, -BULLET_LENGTH*cos_heading, -BULLET_LENGTH*sin_heading
                    );
                    bullets.push_back(initial_timestep_fire_bullet);
                }
            }

            if (!my_bullet.has_value() && timesteps_until_bullet_hit_asteroid + (skip_half_of_first_cycle ? 0 : -1) == fire_after_timesteps) {
                if (ship_state.has_value()) {
                    bullet_fired_from_ship_heading = ship_state->heading;
                    bullet_fired_from_ship_position_x = ship_state->x;
                    bullet_fired_from_ship_position_y = ship_state->y;
                } else {
                    bullet_fired_from_ship_heading = this->ship_state.heading;
                    bullet_fired_from_ship_position_x = this->ship_state.x;
                    bullet_fired_from_ship_position_y = this->ship_state.y;
                }
                rad_heading = bullet_fired_from_ship_heading * DEG_TO_RAD;
                cos_heading = std::cos(rad_heading);
                sin_heading = std::sin(rad_heading);
                new_bullet_x = bullet_fired_from_ship_position_x + SHIP_RADIUS * cos_heading;
                new_bullet_y = bullet_fired_from_ship_position_y + SHIP_RADIUS * sin_heading;
                // Make sure my bullet isn't being fired out into the void
                if (!(0.0 <= new_bullet_x && new_bullet_x <= game_state.map_size_x && 0.0 <= new_bullet_y && new_bullet_y <= game_state.map_size_y)) {
                    // My bullet got shot into the void without hitting anything :(
                    return std::make_tuple(std::nullopt, int64_t(-1), ship_not_collided_with_asteroid);
                }
                my_bullet = Bullet(new_bullet_x, new_bullet_y, BULLET_SPEED * cos_heading, BULLET_SPEED * sin_heading,
                                bullet_fired_from_ship_heading, BULLET_MASS, -BULLET_LENGTH * cos_heading, -BULLET_LENGTH * sin_heading);
            }

            if (whole_move_sequence.has_value()) {
                assert(current_move_index.has_value());
                assert(bullet_sim_ship_state.has_value());
                int64_t idx = current_move_index.value() + timesteps_until_bullet_hit_asteroid + (skip_half_of_first_cycle ? 0 : -1);
                if (idx < static_cast<int64_t>(whole_move_sequence->size())) {
                    // Simulate ship dynamics, if we have the full future list of moves to go off of
                    thrust = (*whole_move_sequence)[idx].thrust;
                    turn_rate = (*whole_move_sequence)[idx].turn_rate;
                    drag_amount = SHIP_DRAG * DELTA_TIME;
                    if (drag_amount > std::abs(bullet_sim_ship_state->speed)) {
                        bullet_sim_ship_state->speed = 0.0;
                    } else {
                        bullet_sim_ship_state->speed -= drag_amount * sign(bullet_sim_ship_state->speed);
                    }
                    if constexpr (ENABLE_SANITY_CHECKS) {
                        assert(-SHIP_MAX_THRUST <= thrust && thrust <= SHIP_MAX_THRUST);
                    }
                    // Apply thrust
                    bullet_sim_ship_state->speed += thrust*DELTA_TIME;
                    // Clamping omitted, could add if desired
                    // bullet_sim_ship_state.speed = min(max(-SHIP_MAX_SPEED, bullet_sim_ship_state.speed), SHIP_MAX_SPEED)
                    if constexpr (ENABLE_SANITY_CHECKS) {
                        assert(-SHIP_MAX_TURN_RATE <= turn_rate && turn_rate <= SHIP_MAX_TURN_RATE);
                    }
                    // Update the angle based on turning rate
                    bullet_sim_ship_state->heading += turn_rate*DELTA_TIME;
                    // Keep the angle within (0, 360)
                    bullet_sim_ship_state->heading = std::fmod(bullet_sim_ship_state->heading + 360.0, 360.0);
                    // Use speed magnitude to get velocity vector
                    rad_heading = bullet_sim_ship_state->heading * DEG_TO_RAD;
                    bullet_sim_ship_state->vx = std::cos(rad_heading) * bullet_sim_ship_state->speed;
                    bullet_sim_ship_state->vy = std::sin(rad_heading) * bullet_sim_ship_state->speed;
                    // Update the position based off the velocities
                    // Do the wrap in the same operation
                    bullet_sim_ship_state->x = pymod(bullet_sim_ship_state->x + bullet_sim_ship_state->vx * DELTA_TIME, game_state.map_size_x);
                    bullet_sim_ship_state->y = pymod(bullet_sim_ship_state->y + bullet_sim_ship_state->vy * DELTA_TIME, game_state.map_size_y);
                }
            }

            // Check bullet/asteroid collisions
            size_t len_bullets = bullets.size();

            // Helper lambda to process bullet/asteroid collision logic
            auto process_bullet = [&](Bullet& b, size_t b_idx){
                if (b.alive) {
                    double b_tail_x = b.x + b.tail_delta_x;
                    double b_tail_y = b.y + b.tail_delta_y;
                    new_asteroids.clear();
                    for (auto& a : asteroids) {
                        if (a.alive) {
                            if (asteroid_bullet_collision(b.x, b.y, b_tail_x, b_tail_y, a.x, a.y, a.radius)) {
                                if (b_idx == len_bullets) {
                                    // This bullet is my bullet!
                                    return std::optional<std::tuple<std::optional<Asteroid>, int64_t, bool>>(
                                        std::make_tuple(std::optional<Asteroid>(a), timesteps_until_bullet_hit_asteroid, ship_not_collided_with_asteroid)
                                    );
                                } else {
                                    // Kill bullet
                                    b.alive = false;

                                    // Create asteroid splits and mark for removal
                                    if (a.size != 1) {
                                        auto splits = forecast_instantaneous_asteroid_bullet_splits_from_velocity(a, b.vx, b.vy, game_state);
                                        std::apply([&](const auto&... new_ast) {
                                            (new_asteroids.push_back(new_ast), ...);
                                        }, splits);
                                    }

                                    a.alive = false;
                                    break;  // Stop checking this bullet
                                }
                            }
                        }
                    }

                    // Add new asteroids safely after loop
                    asteroids.insert(
                        asteroids.end(),
                        new_asteroids.begin(),
                        new_asteroids.end()
                    );

                }
                return std::optional<std::tuple<std::optional<Asteroid>, int64_t, bool>>{};
            };

            // First, process main bullets
            for (size_t b_idx = 0; b_idx < bullets.size(); ++b_idx) {
                auto result = process_bullet(bullets[b_idx], b_idx);
                if (result) return *result;
            }

            // Then, process your bullet (if it exists)
            if (my_bullet.has_value()) {
                auto result = process_bullet(my_bullet.value(), len_bullets);
                if (result) return *result;
            }

            // Check mine/asteroid collisions
            new_asteroids.clear();
            for (Mine& mine : mines) {
                if (mine.alive && mine.remaining_time < EPS) {
                    mine.alive = false;
                    for (Asteroid& asteroid : asteroids) {
                        if (asteroid.alive) {
                            delta_x = asteroid.x - mine.x;
                            delta_y = asteroid.y - mine.y;
                            separation = asteroid.radius + MINE_BLAST_RADIUS;
                            if (std::abs(delta_x) <= separation && std::abs(delta_y) <= separation && delta_x*delta_x + delta_y*delta_y <= separation*separation) {
                                if (asteroid.size != 1) {
                                    std::apply([&](const auto&... sp) {
                                        (new_asteroids.push_back(sp), ...);
                                    }, forecast_asteroid_mine_instantaneous_splits(asteroid, mine, game_state));
                                }
                                asteroid.alive = false;
                            }
                        }
                    }
                }
            }
            asteroids.insert(asteroids.end(), new_asteroids.begin(), new_asteroids.end());

            // Check ship/asteroid collisions
            if (ship_not_collided_with_asteroid) {
                if (whole_move_sequence.has_value() && bullet_sim_ship_state.has_value()) {
                    ship_position_x = bullet_sim_ship_state->x;
                    ship_position_y = bullet_sim_ship_state->y;
                } else if (ship_state.has_value()) {
                    ship_position_x = ship_state->x;
                    ship_position_y = ship_state->y;
                } else {
                    ship_position_x = this->ship_state.x;
                    ship_position_y = this->ship_state.y;
                }
                new_asteroids.clear();
                for (Asteroid& asteroid : asteroids) {
                    if (asteroid.alive) {
                        delta_x = ship_position_x - asteroid.x;
                        delta_y = ship_position_y - asteroid.y;
                        separation = SHIP_RADIUS + asteroid.radius;

                        if (std::abs(delta_x) <= separation &&
                            std::abs(delta_y) <= separation &&
                            delta_x * delta_x + delta_y * delta_y <= separation * separation) {

                            if (asteroid.size != 1) {
                                auto splits = forecast_asteroid_ship_splits(asteroid, 0, 0.0, 0.0, game_state);
                                std::apply([&](const auto&... ast) {
                                    (new_asteroids.push_back(ast), ...);
                                }, splits);
                            }

                            asteroid.alive = false;
                            ship_not_collided_with_asteroid = false;
                            break;
                        }
                    }
                }
                // Append new asteroids after iteration
                asteroids.insert(
                    asteroids.end(),
                    new_asteroids.begin(),
                    new_asteroids.end()
                );
            }
        }
        // unreachable
        assert(false);
    }

    bool apply_move_sequence(const std::vector<Action>& move_sequence, bool allow_free_firing = false, bool allow_free_mining = false) {
        /*
        Applies a sequence of moves to the ship, updating its state step-by-step.

        This method is typically used to simulate actual execution of a move sequence,
        and is slightly more permissive — it records whether the sequence was fully safe
        but does not immediately exit on failure. It also does not record the intended
        move sequence or pass the full sequence to the `update` method.

        Arguments:
            move_sequence (list[Action]): The sequence of actions to apply.
            allow_free_firing (bool): If True, disables firing for the purpose of the sim.
            allow_free_mining (bool): If True, disables mine-dropping for the sim.

        Returns:
            bool: True if the sequence was applied safely (all moves succeeded), else False.
        */
        if (sim_id == 1234567) {
            std::cout << "Applying move seq in 24111 ship_state=" << ship_state.str()
                    << " move_sequence.size=" << move_sequence.size()
                    << " allow_free_firing=" << allow_free_firing
                    << " allow_free_mining=" << allow_free_mining << std::endl;
        }
        // assert is_close_to_zero(ship_state.speed)
        bool sim_was_safe = true;
        double thrust, turn_rate;
        std::optional<bool> fire, drop_mine;
        for (const Action& move : move_sequence) {
            thrust = move.thrust;
            turn_rate = move.turn_rate;
            fire = allow_free_firing ? std::nullopt : std::optional<bool>(move.fire);
            drop_mine = allow_free_mining ? std::nullopt : std::optional<bool>(move.drop_mine);

            // Call update
            if (!update(thrust, turn_rate, fire, drop_mine)) {
                sim_was_safe = false;
                break;
            }
            // print after-thrust speed if desired (debug)
        }
        if constexpr (ENABLE_SANITY_CHECKS) {
            assert(is_close_to_zero(ship_state.speed)
                /* If you want a string message, you can add: */
                && "When returning in apply move sequence, the ship speed is not zero!"
            );
        }
        if (sim_id == 1234567) {
            std::cout << "24111 sim_was_safe=" << sim_was_safe << std::endl;
        }
        return sim_was_safe;
    }

    bool simulate_maneuver(const std::vector<Action>& move_sequence, bool allow_free_firing, bool allow_free_mining) {
        /*
        Simulates a maneuver by applying a sequence of moves to the ship, step-by-step.

        This method is primarily used for predictive or planning purposes. It records the
        intended move sequence for later inspection (e.g., for debugging or analysis) and
        passes the entire move sequence into the `update` method. It returns immediately
        upon detecting a failed move, rather than continuing.

        Arguments:
            move_sequence (list[Action]): The sequence of actions to simulate.
            allow_free_firing (bool): If True, disables firing for the purpose of the sim.
            allow_free_mining (bool): If True, disables mine-dropping for the sim.

        Returns:
            bool: True if all moves were simulated safely, else False.
        */
        intended_move_sequence = move_sequence; // Record intended move sequence in case maneuvers are interrupted

        //flag = false;
        //if (!is_close_to_zero(ship_state.speed) && sim_id == 333) {
        //    std::cout << "When starting in simulate maneuver where the sim was safe, the ship speed is not zero! ship_state.speed=" << ship_state.speed
        //              << ", ship_state.vx=" << ship_state.vx << ", ship_state.vy=" << ship_state.vy
        //              << ". The whole move sequence is REDACTED move_sequence" << std::endl;
        //    flag = true;
        //}
        double thrust, turn_rate;
        std::optional<bool> fire, drop_mine;
        for (const Action& move : move_sequence) {
            thrust = move.thrust;
            turn_rate = move.turn_rate;
            fire = allow_free_firing ? std::nullopt : std::optional<bool>(fire);
            drop_mine = allow_free_mining ? std::nullopt : std::optional<bool>(drop_mine);
            // if (sim_id == 23215) {
            //     std::cout << "Calling update from sim maneuver: with thrust=" << thrust << ", turn_rate=" << turn_rate << ", allow_firing=" << allow_free_firing << std::endl;
            // }
            if (!update(thrust,
                        turn_rate,
                        fire,
                        drop_mine,
                        move_sequence))
            {
                return false;
            }
            //if (flag) { ... }
            //if (sim_id == 333) {
            //    std::cout << "In sim " << sim_id << " After thrusting by " << thrust << " the true simmed ship speed is " << ship_state.speed << std::endl;
            //}
        }
        if constexpr (ENABLE_SANITY_CHECKS) {
            assert(is_close_to_zero(ship_state.speed) &&
                "When returning in simulate maneuver where the sim was safe, the ship speed is not zero!");
        }
        return true;
    }

    bool update(
        double thrust = 0.0,
        double turn_rate = 0.0,
        std::optional<bool> fire = std::nullopt,
        std::optional<bool> drop_mine = std::nullopt,
        std::optional<std::vector<Action>> whole_move_sequence = std::nullopt,
        bool wait_out_mines = false) {
        //total_sim_timesteps += 1;

        if (this->sim_id == 1235467) {
            std::cout << this->game_state << std::endl;
        }

        if constexpr (ENABLE_BAD_LUCK_EXCEPTION && random_double() < BAD_LUCK_EXCEPTION_PROBABILITY) {
            throw std::runtime_error("Bad luck exception!");
        }
        // This should exactly match what kessler_game.py does.
        // Being even one timestep off is the difference between life and death!!!
        std::optional<bool> return_value = std::nullopt;

        // Track state sequence and forecasted split history
        if (!wait_out_mines) {
            std::vector<Asteroid> forecasted_splits_copy;
            for (const auto& a : forecasted_asteroid_splits) forecasted_splits_copy.push_back(a);
            forecasted_asteroid_splits_history.push_back(forecasted_splits_copy);
            if constexpr (ENABLE_SANITY_CHECKS) {
                for (const auto& a : forecasted_asteroid_splits_history.back()) { assert(a.alive); }
            }
            if (PRUNE_SIM_STATE_SEQUENCE && future_timesteps != 0) {
                // Create a super lightweight state that omits unnecessary stuff
                state_sequence.push_back(SimState(
                    initial_timestep + future_timesteps,
                    ship_state
                    // lightweight mode: skip game_state, asteroids_pending_death, etc.
                    // Assuming game_state and other attributes are optional or have default values in SimState definition
                ));
            } else {
                state_sequence.push_back(SimState(
                    initial_timestep + future_timesteps,
                    ship_state,
                    get_game_state(),
                    asteroids_pending_death, // shallow copy is fine here
                    forecasted_splits_copy
                ));
            }
        }

        // The simulation starts by evaluating actions and dynamics of the current present timestep, and then steps into the future
        // The game state we're given is actually what we had at the end of the previous timestep
        // The game will take the previous state, and apply current actions and then update to get the result of this timestep
        if constexpr (ENABLE_SANITY_CHECKS) {
            if (whole_move_sequence) {
                const Action& action = (*whole_move_sequence)[future_timesteps];
                assert(action.thrust == thrust);
                assert(action.turn_rate == turn_rate);
                // Could also assert fire/drop_mine if you wish.
            }
        }

        // Simulation order:
        // Ships are given the game state from after the previous timestep. Ships then decide the inputs.
        // Update bullets/mines/asteroids.
        // Ship has inputs applied and updated. Any new bullets and mines that the ship creates is added to the list.
        // Bullets past the map edge get culled
        // Ships and asteroids beyond the map edge get wrapped
        // Check for bullet/asteroid collisions. Checked for each bullet in list order, check for each asteroid in list order. Bullets and asteroids are removed here, and any new asteroids created are added to the list now.
        // Check mine collisions with asteroids/ships. For each mine in list order, check whether it is detonating and if it collides with first asteroids in list order (and add new asteroids to list), and then ships.
        // Check for asteroid/ship collisions. For each ship in list order, check collisions with each asteroid in list order. New asteroids are added now. Ships and asteroids get culled if they collide.
        // Check ship/ship collisions and cull them.

        // Plotting (REMOVE_FOR_COMPETITION)
        /*
        if (plot_this_sim && game_state_plotter.has_value()) {
            std::vector<Asteroid> flattened_asteroids_pending_death;
            for (const auto& kv : asteroids_pending_death)
                for (const auto& ast : kv.second)
                    flattened_asteroids_pending_death.push_back(ast);
            game_state_plotter->update_plot(
                game_state.asteroids, ship_state, game_state.bullets, {}, {},
                flattened_asteroids_pending_death, forecasted_asteroid_splits, game_state.mines,
                true, 0.1, "SIM UPDATE TS " + std::to_string(initial_timestep + future_timesteps));
        }*/

        // Simulate dynamics of bullets
        // Kessler will move bullets and cull them in different steps, but we combine them in one operation here
        // So we need to detect when the bullets are crossing the boundary, and delete them if they try to
        // Enumerate and track indices to delete

        // Bullets step (advance and out-of-bounds cull)
        for (Bullet& b : game_state.bullets) {
            if (b.alive) {
                double nx = b.x + b.vx * DELTA_TIME;
                double ny = b.y + b.vy * DELTA_TIME;
                // if check_coordinate_bounds(self.game_state, new_bullet_x, new_bullet_y):
                if (0.0 <= nx && nx <= game_state.map_size_x && 0.0 <= ny && ny <= game_state.map_size_y) {
                    b.x = nx;
                    b.y = ny;
                } else {
                    b.alive = false;
                }
            }
        }
        // Mines step
        for (auto& m : game_state.mines) {
            if (m.alive) { // Might not be faster to do this, since it's not much computation I'm skipping
                if constexpr (ENABLE_SANITY_CHECKS) {
                    assert(m.remaining_time > EPS - DELTA_TIME);
                }
                m.remaining_time -= DELTA_TIME;
            }
            // If the timer is below eps, it'll detonate this timestep
        }
        // Simulate dynamics of asteroids
        // Wrap the asteroid positions in the same operation
        // Between when the asteroids get moved and when the future timesteps gets incremented, these asteroids exist at time (self.initial_timestep + self.future_timesteps + 1) instead of (self.initial_timestep + self.future_timesteps)!
        
        // Asteroids step (with wrap)
        for (Asteroid& a : game_state.asteroids) {
            if (a.alive) {
                a.x = pymod(a.x + a.vx * DELTA_TIME, game_state.map_size_x);
                a.y = pymod(a.y + a.vy * DELTA_TIME, game_state.map_size_y);
            }
        }

        // ==========================
        //           SHIP
        // ==========================
        bool fire_this_timestep = false;
        bool drop_mine_this_timestep = false;
        if (!wait_out_mines) {
            forecasted_asteroid_splits = maintain_forecasted_asteroids(forecasted_asteroid_splits, game_state);
            //forecasted_asteroid_splits.clear(); // DEBUG
            // Simulate the ship!
            // Bullet firing happens before we turn the ship
            // Check whether we want to shoot a simulated bullet
            // ================ HANDLE FIRING ================
            if (ship_state.bullets_remaining != 0) {
                if (fire_first_timestep && future_timesteps == 0) {
                    assert(respawn_maneuver_pass_number == 0 || (respawn_maneuver_pass_number == 2 && initial_timestep + future_timesteps > last_timestep_colliding));
                    // In theory we should be able to hit the target, however if we're in multiagent mode, the other ship could muddle with things in this time making me miss my shot, so let's just confirm that it's going to land before we fire for real!
                    if (verify_first_shot) {
                        std::optional<Asteroid> actual_asteroid_hit;
                        int64_t timesteps_until_bullet_hit_asteroid;
                        bool ship_was_safe;
                        std::tie(actual_asteroid_hit, timesteps_until_bullet_hit_asteroid, ship_was_safe) = bullet_sim(
                            std::nullopt, false, 0, true, future_timesteps, whole_move_sequence, INT_INF, std::nullopt
                        );
                        // Originally I wrongly asserted that if there is no other ship, then my shot will land
                        // I think this assertion doesn't work right after a ship dies, because their bullet can still be travelling in the air
                        // There's also the bullet skipping issue...
                        if (!actual_asteroid_hit.has_value()) {
                            fire_this_timestep = false;
                            cancel_firing_first_timestep = true;
                        } else {
                            fire_this_timestep = true;
                        }
                    } else {
                        fire_this_timestep = true;
                    }
                } else if (!fire.has_value()) {
                    // ----------- BEGIN BULLET "CONVENIENT FIRE" LOGIC -----------
                    // We're able to decide whether we want to fire any convenient shots we can get
                    int64_t timesteps_until_can_fire = std::max<int64_t>(0, FIRE_COOLDOWN_TS - (initial_timestep + future_timesteps - last_timestep_fired));
                    fire_this_timestep = false;
                    double ship_heading_rad = ship_state.heading * DEG_TO_RAD;
                    bool feasible_targets_exist = false;
                    
                    std::vector<Asteroid> culled_targets_for_simulation;
                    std::vector<int64_t> culled_target_idxs_for_simulation;
                    double max_interception_time = 0.0;

                    double min_positive_shot_heading_error_rad = INFINITY, second_min_positive_shot_heading_error_rad = INFINITY;
                    double min_negative_shot_heading_error_rad = -INFINITY, second_min_negative_shot_heading_error_rad = -INFINITY, min_shot_heading_error_rad = std::numeric_limits<double>::quiet_NaN(), second_min_shot_heading_error_rad = std::numeric_limits<double>::quiet_NaN();
                    size_t len_asteroids = game_state.asteroids.size();
                    bool avoid_targeting_this_asteroid = false;
                    bool check_next_asteroid = false;
                    if (!halt_shooting || (respawn_maneuver_pass_number == 2 && (initial_timestep + future_timesteps > last_timestep_colliding))) {
                        if (timesteps_until_can_fire == 0) {
                            // We can shoot this timestep! Loop through all asteroids and see which asteroids we can feasibly hit if we shoot at this angle, and take those and simulate with the bullet sim to see which we'll hit
                            // If mines exist, then we can't cull any asteroids since asteroids can be hit by the mine and get flung into the path of my shooting
                            
                            // Keep track of these so we can begin turning roughly toward our next target
                            // Keep track of both the positive (left turn) and negative (right turn) ones so we can follow our "random walk" turn schedule
                            for (size_t ast_idx = 0; ast_idx < game_state.asteroids.size() + forecasted_asteroid_splits.size(); ++ast_idx) {
                                // Loop through ALL asteroids and make sure at least one asteroid is a valid target
                                // Get the length of time the longest asteroid would take to hit, and that'll be the upper bound of the bullet sim's timesteps
                                // Avoid shooting my size 1 asteroids that are about to get mined by my mine
                                const Asteroid* asteroid;
                                bool is_forecasted = false;
                                if (ast_idx < game_state.asteroids.size()) {
                                    asteroid = &game_state.asteroids[ast_idx];
                                } else {
                                    asteroid = &forecasted_asteroid_splits[ast_idx - game_state.asteroids.size()];
                                    is_forecasted = true;
                                }
                                if (!asteroid->alive) continue;

                                avoid_targeting_this_asteroid = false;
                                if (asteroid->size == 1) {
                                    for (const Mine& m : game_state.mines) {
                                        if (m.alive && mine_positions_placed.count(std::make_pair(m.x, m.y))) {
                                            // This mine is mine
                                            //project_asteroid_by_timesteps_num = round(m.remaining_time*FPS)
                                            //asteroid_when_mine_explodes = time_travel_asteroid(asteroid, project_asteroid_by_timesteps_num, self.game_state)
                                            Asteroid asteroid_when_mine_explodes = time_travel_asteroid_s(*asteroid, m.remaining_time, game_state);
                                            //if check_collision(asteroid_when_mine_explodes.x, asteroid_when_mine_explodes.y, asteroid_when_mine_explodes.radius, m.x, m.y, MINE_BLAST_RADIUS):
                                            double delta_x = asteroid_when_mine_explodes.x - m.x;
                                            double delta_y = asteroid_when_mine_explodes.y - m.y;
                                            double separation = asteroid_when_mine_explodes.radius + MINE_BLAST_RADIUS;
                                            if (std::abs(delta_x) <= separation && std::abs(delta_y) <= separation && delta_x*delta_x + delta_y*delta_y <= separation*separation) {
                                                avoid_targeting_this_asteroid = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if (avoid_targeting_this_asteroid) continue;

                                //bool in_culling_cone = false;
                                if (ast_idx < len_asteroids && heading_diff_within_threshold(ship_heading_rad, asteroid->x - ship_state.x, asteroid->y - ship_state.y, MANEUVER_BULLET_SIM_CULLING_CONE_WIDTH_ANGLE_HALF_COSINE)) {
                                    //ast_angle = super_fast_atan2(asteroid.y - self.ship_state.y, asteroid.x - self.ship_state.x)
                                    //if abs(angle_difference_deg(degrees(ast_angle), self.ship_state.heading)) <= MANEUVER_BULLET_SIM_CULLING_CONE_WIDTH_ANGLE_HALF:
                                    // We also want to add the surrounding asteroids into the bullet sim, just in case any of them aren't added later in the feasible shots
                                    // The reasons for them not being added later, is that maybe we already shot at it, so we skipped over it.
                                    // We should be including all the asteroids we shot at, but unfortunately even including all the asteroids in a cone doesn't guarantee that, so this system still isn't perfect!!
                                    culled_target_idxs_for_simulation.push_back(ast_idx);
                                    //in_culling_cone = true;
                                }

                                check_next_asteroid = false;
                                if (check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(asteroids_pending_death, initial_timestep + future_timesteps + 1, game_state, *asteroid)) {
                                    std::vector<Asteroid> unwrapped_asteroids = unwrap_asteroid(*asteroid, game_state.map_size_x, game_state.map_size_y, UNWRAP_ASTEROID_TARGET_SELECTION_TIME_HORIZON, true);
                                    for (const Asteroid& a : unwrapped_asteroids) {
                                        if (check_next_asteroid) {
                                            break;
                                        }
                                        // Since we need to find the minimum shot heading errors, we can't break out of this loop early. We should just go through them all.
                                        //unwrapped_ast_angle = super_fast_atan2(a.y - self.ship_state.y, a.x - self.ship_state.x)
                                        //if abs(angle_difference_deg(degrees(unwrapped_ast_angle), self.ship_state.heading)) > MANEUVER_CONVENIENT_SHOT_CHECKER_CONE_WIDTH_ANGLE_HALF:
                                        if (!heading_diff_within_threshold(ship_heading_rad, a.x - ship_state.x, a.y - ship_state.y, MANEUVER_CONVENIENT_SHOT_CHECKER_CONE_WIDTH_ANGLE_HALF_COSINE)) {
                                            continue;
                                        }

                                        // Now, do interception math
                                        bool feasible;
                                        double shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, intercept_x, intercept_y, asteroid_dist_during_interception;
                                        std::tie(feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, intercept_x, intercept_y, asteroid_dist_during_interception) =
                                            calculate_interception(ship_state.x, ship_state.y, a.x, a.y, a.vx, a.vy, a.radius, ship_state.heading, game_state);

                                        if (feasible) {
                                            // Regardless of whether our heading is close enough to shooting this asteroid, keep track of this, just in case no asteroids are within shooting range this timestep, but we can begin to turn toward it next timestep!
                                            // if abs(shot_heading_error_rad) < abs(min_shot_heading_error_rad):
                                            //     second_min_shot_heading_error_rad = min_shot_heading_error_rad
                                            //     min_shot_heading_error_rad = shot_heading_error_rad
                                            if (shot_heading_error_rad >= 0.0) {
                                                if (shot_heading_error_rad < min_positive_shot_heading_error_rad) {
                                                    second_min_positive_shot_heading_error_rad = min_positive_shot_heading_error_rad;
                                                    min_positive_shot_heading_error_rad = shot_heading_error_rad;
                                                }
                                            } else {
                                                if (shot_heading_error_rad > min_negative_shot_heading_error_rad) {
                                                    second_min_negative_shot_heading_error_rad = min_negative_shot_heading_error_rad;
                                                    min_negative_shot_heading_error_rad = shot_heading_error_rad;
                                                }
                                            }
                                            if (std::abs(shot_heading_error_rad) <= shot_heading_tolerance_rad) {
                                                // If we shoot at our current heading, this asteroid can be hit!
                                                if (ast_idx < len_asteroids) {
                                                    // Only add real asteroids to the set of asteroids we simulate! Don't simulate the asteroids that don't exist yet.
                                                    //culled_targets_for_simulation.append(asteroid)
                                                    if (culled_target_idxs_for_simulation.empty() || culled_target_idxs_for_simulation.back() != static_cast<int64_t>(ast_idx))
                                                        culled_target_idxs_for_simulation.push_back(ast_idx);
                                                }
                                                feasible_targets_exist = true;
                                                if (interception_time > max_interception_time)
                                                    max_interception_time = interception_time;
                                                check_next_asteroid = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }

                            if (feasible_targets_exist) {
                                // Use the bullet sim to confirm that this will hit something
                                // There's technically a chance for culled_targets_for_simulation to be empty at this point, if we're purely shooting asteroids that haven't come into existence yet.
                                // In that case, this will detect that and will avoid doing the culling, and do the full sim. This should be rare.
                                culled_targets_for_simulation.clear();
                                for (auto idx : culled_target_idxs_for_simulation)
                                    if (idx < static_cast<int64_t>(game_state.asteroids.size()))
                                        culled_targets_for_simulation.push_back(game_state.asteroids[idx]);
                                int bullet_sim_timestep_limit = static_cast<int>(std::ceil(max_interception_time*FPS)) + 1;
                                std::optional<Asteroid> actual_asteroid_hit;
                                int64_t timesteps_until_bullet_hit_asteroid;
                                bool ship_was_safe;
                                std::tie(actual_asteroid_hit, timesteps_until_bullet_hit_asteroid, ship_was_safe)
                                    = bullet_sim(std::nullopt, false, 0, true, future_timesteps, whole_move_sequence,
                                                bullet_sim_timestep_limit,
                                                (!culled_targets_for_simulation.empty() && game_state.mines.empty()) ?
                                                    std::optional<std::vector<Asteroid>>(culled_targets_for_simulation) : std::nullopt);
                                if (actual_asteroid_hit.has_value() && ship_was_safe) {
                                    // Confirmed that the shot will land
                                    assert(timesteps_until_bullet_hit_asteroid >= 0);
                                    Asteroid actual_asteroid_hit_at_fire_time = time_travel_asteroid(
                                        actual_asteroid_hit.value(), -timesteps_until_bullet_hit_asteroid, game_state);
                                    if (check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(asteroids_pending_death, initial_timestep + future_timesteps + 1, game_state, actual_asteroid_hit_at_fire_time)) {
                                        fire_this_timestep = true;
                                        ++asteroids_shot;
                                        explanation_messages.push_back("During the maneuver, I conveniently shot asteroids.");
                                        if (actual_asteroid_hit_at_fire_time.size != 1)
                                            std::apply([&](const auto&... sp) {
                                                (forecasted_asteroid_splits.push_back(sp), ...);
                                            }, forecast_asteroid_bullet_splits_from_heading(actual_asteroid_hit_at_fire_time, timesteps_until_bullet_hit_asteroid, ship_state.heading, game_state));
                                        // The reason we add one to the timestep we track on, is that once we updated the asteroids' position in the update loop, it's technically the asteroid positions in the game state of the next timestep that gets passed to the controllers!
                                        // So the asteroid positions at a certain timestep is before their positions get updated. After updating, it's the next timestep.
                                        asteroids_pending_death_history[initial_timestep + future_timesteps + 1] = asteroids_pending_death;
                                        //std::cout << "Tracking from sim number " << std::to_string(this->sim_id) << std::endl;
                                        track_asteroid_we_shot_at(asteroids_pending_death, initial_timestep + future_timesteps + 1, game_state, timesteps_until_bullet_hit_asteroid, actual_asteroid_hit_at_fire_time);
                                    }
                                    if (fire_this_timestep && !std::isinf(game_state.time_limit) && initial_timestep + future_timesteps + timesteps_until_bullet_hit_asteroid > std::floor(FPS*game_state.time_limit)) {
                                        // Added one to the timesteps to prevent off by one :P
                                        fire_this_timestep = false;
                                        --asteroids_shot;
                                    }
                                }
                            }
                            assert(asteroids_shot >= 0);

                            // Below: aiming maneuver for next shot (random walk schedule)
                            if (respawn_maneuver_pass_number == 0 && (future_timesteps >= MANEUVER_SIM_DISALLOW_TARGETING_FOR_START_TIMESTEPS_AMOUNT)) {
                                // Might as well start turning toward our next target!
                                if (asteroids_shot >= RANDOM_WALK_SCHEDULE_LENGTH) {
                                    // Can turn either left or right
                                    // Find the min absolute shot heading error
                                    if (min_positive_shot_heading_error_rad + min_negative_shot_heading_error_rad >= 0.0)
                                        min_shot_heading_error_rad = min_negative_shot_heading_error_rad;
                                    else
                                        min_shot_heading_error_rad = min_positive_shot_heading_error_rad;
                                    if (second_min_positive_shot_heading_error_rad + second_min_negative_shot_heading_error_rad >= 0.0)
                                        second_min_shot_heading_error_rad = second_min_negative_shot_heading_error_rad;
                                    else
                                        second_min_shot_heading_error_rad = second_min_positive_shot_heading_error_rad;
                                } else if (random_walk_schedule[asteroids_shot]) {
                                    // We want to turn left
                                    min_shot_heading_error_rad = min_positive_shot_heading_error_rad;
                                    second_min_shot_heading_error_rad = second_min_positive_shot_heading_error_rad;
                                } else {
                                    // We want to turn right
                                    min_shot_heading_error_rad = min_negative_shot_heading_error_rad;
                                    second_min_shot_heading_error_rad = second_min_negative_shot_heading_error_rad;
                                }
                                double next_target_heading_error = std::numeric_limits<double>::quiet_NaN();
                                if (!fire_this_timestep && !std::isinf(min_shot_heading_error_rad)) {
                                    // We didn't fire this timestep, so we can use the min shot heading error rad to turn toward the same target and try again on the next timestep
                                    next_target_heading_error = min_shot_heading_error_rad; // This is where we're aiming for the next timestep!
                                }
                                else if (fire_this_timestep && !std::isinf(second_min_shot_heading_error_rad)) {
                                    next_target_heading_error = second_min_shot_heading_error_rad;
                                }
                                // The assumption is that the target that was hit wasn't the second smallest heading diff. THIS IS NOT TRUE IN GENERAL. This can be wrong! But whatever, it's not a big deal and probably not worth fixing/taking the extra compute to track this.
                                if (!std::isnan(next_target_heading_error)) {
                                    double min_shot_heading_error_deg = next_target_heading_error*RAD_TO_DEG;
                                    double altered_turn_command =
                                        (std::abs(min_shot_heading_error_deg) <= DEGREES_TURNED_PER_TIMESTEP)
                                        ? min_shot_heading_error_deg*FPS
                                        : SHIP_MAX_TURN_RATE*sign(min_shot_heading_error_rad);
                                    turn_rate = altered_turn_command;
                                    if (whole_move_sequence) {
                                        (whole_move_sequence.value())[future_timesteps].turn_rate = altered_turn_command;
                                    }
                                }
                            }
                        }
                    } else if (respawn_maneuver_pass_number == 0 && (future_timesteps >= MANEUVER_SIM_DISALLOW_TARGETING_FOR_START_TIMESTEPS_AMOUNT || timesteps_until_can_fire == 1)) {
                        // timesteps_until_can_fire is 1, 2, or 3
                        // On the next timestep, hopefully we'd be aimed at the asteroid and then the above if case will kick in and we will shoot it!
                        // This makes the shot efficiency during maneuvering a lot better because we're not only dodging, but we're also targetting and firing at the same time!
                        // If there's more timesteps to turn before we can fire, then we can still begin to turn toward the closest target
                        
                        // Next-shot aiming, even before firing is legal
                        bool locked_in = false;
                        double asteroid_least_shot_heading_error_deg = INFINITY;
                        double asteroid_least_shot_heading_tolerance_deg = std::numeric_limits<double>::quiet_NaN();
                        // Roughly predict the ship's position on the next timestep using its current heading. This isn't 100% correct but whatever, it's better than nothing.
                        
                        double ship_pred_speed = ship_state.speed;
                        double drag_amount = SHIP_DRAG*DELTA_TIME;
                        if (drag_amount > std::abs(ship_pred_speed)) {
                            ship_pred_speed = 0.0;
                        } else {
                            ship_pred_speed -= drag_amount*sign(ship_pred_speed);
                        }
                        // Apply thrust
                        ship_pred_speed += std::min(std::max(-SHIP_MAX_THRUST, thrust), SHIP_MAX_THRUST)*DELTA_TIME;
                        if (ship_pred_speed > SHIP_MAX_SPEED) {
                            ship_pred_speed = SHIP_MAX_SPEED;
                        }
                        if (ship_pred_speed < -SHIP_MAX_SPEED) {
                            ship_pred_speed = -SHIP_MAX_SPEED;
                        }
                        double rad_heading = ship_state.heading*DEG_TO_RAD;
                        double ship_speed_ts = DELTA_TIME*(double)timesteps_until_can_fire*ship_pred_speed;
                        double ship_predicted_pos_x = ship_state.x + ship_speed_ts*cos(rad_heading);
                        double ship_predicted_pos_y = ship_state.y + ship_speed_ts*sin(rad_heading);

                        // For both actual and forecasted asteroids
                        for (size_t i = 0; i < game_state.asteroids.size() + forecasted_asteroid_splits.size(); ++i) {
                            const Asteroid* asteroid;
                            if (i < game_state.asteroids.size())
                                asteroid = &game_state.asteroids[i];
                            else
                                asteroid = &forecasted_asteroid_splits[i - game_state.asteroids.size()];
                            if (!asteroid->alive) continue;
                            // Avoid shooting my size 1 asteroids that are about to get mined by my mine
                            avoid_targeting_this_asteroid = false;
                            if (asteroid->size == 1) {
                                for (const Mine& m : game_state.mines) {
                                    if (m.alive && mine_positions_placed.count(std::make_pair(m.x, m.y))) {
                                        // This mine is mine
                                        //project_asteroid_by_timesteps_num = round(m.remaining_time*FPS)
                                        //asteroid_when_mine_explodes = time_travel_asteroid(asteroid, project_asteroid_by_timesteps_num, self.game_state)
                                        Asteroid asteroid_when_mine_explodes = time_travel_asteroid_s(*asteroid, m.remaining_time, game_state);
                                        //if check_collision(asteroid_when_mine_explodes.x, asteroid_when_mine_explodes.y, asteroid_when_mine_explodes.radius, m.x, m.y, MINE_BLAST_RADIUS):
                                        double delta_x = asteroid_when_mine_explodes.x - m.x;
                                        double delta_y = asteroid_when_mine_explodes.y - m.y;
                                        double separation = asteroid_when_mine_explodes.radius + MINE_BLAST_RADIUS;
                                        if (std::abs(delta_x) <= separation && std::abs(delta_y) <= separation && delta_x*delta_x + delta_y*delta_y <= separation*separation) {
                                            avoid_targeting_this_asteroid = true;
                                            break;
                                        }
                                    }
                                }
                            }
                            if (avoid_targeting_this_asteroid) continue;
                            if (check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(asteroids_pending_death, initial_timestep + future_timesteps + 1, game_state, *asteroid)) {
                                std::vector<Asteroid> unwrapped_asteroids = unwrap_asteroid(*asteroid, game_state.map_size_x, game_state.map_size_y, UNWRAP_ASTEROID_TARGET_SELECTION_TIME_HORIZON, true);
                                for (const Asteroid& a : unwrapped_asteroids) {
                                    bool feasible;
                                    double shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, intercept_x, intercept_y, asteroid_dist;
                                    std::tie(feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, intercept_x, intercept_y, asteroid_dist) =
                                        calculate_interception(ship_predicted_pos_x, ship_predicted_pos_y, a.x, a.y, a.vx, a.vy, a.radius, ship_state.heading, game_state, timesteps_until_can_fire);
                                    if (feasible &&
                                        (asteroids_shot >= RANDOM_WALK_SCHEDULE_LENGTH ||
                                        (random_walk_schedule[asteroids_shot] && shot_heading_error_rad >= 0.0) ||
                                        (!random_walk_schedule[asteroids_shot] && shot_heading_error_rad <= 0.0))) {
                                        double shot_heading_error_deg = shot_heading_error_rad*RAD_TO_DEG;
                                        double shot_heading_tolerance_deg = shot_heading_tolerance_rad*RAD_TO_DEG;
                                        if (std::abs(shot_heading_error_deg) - shot_heading_tolerance_deg < std::abs(asteroid_least_shot_heading_error_deg)) {
                                            asteroid_least_shot_heading_error_deg = shot_heading_error_deg;
                                            asteroid_least_shot_heading_tolerance_deg = shot_heading_tolerance_deg;
                                        }
                                        assert(shot_heading_tolerance_deg >= 0.0);
                                        if (std::abs(shot_heading_error_deg) - shot_heading_tolerance_deg <= DEGREES_TURNED_PER_TIMESTEP) {
                                            locked_in = true;
                                            double altered_turn_command;
                                            if (std::abs(shot_heading_error_deg) <= DEGREES_TURNED_PER_TIMESTEP) {
                                                altered_turn_command = shot_heading_error_deg*FPS;
                                                assert(std::abs(altered_turn_command) <= SHIP_MAX_TURN_RATE);
                                            } else {
                                                altered_turn_command = SHIP_MAX_TURN_RATE*sign(shot_heading_error_deg);
                                            }
                                            turn_rate = altered_turn_command;
                                            if (whole_move_sequence)
                                                (whole_move_sequence.value())[future_timesteps].turn_rate = altered_turn_command;
                                            break;
                                        }
                                    }
                                }
                            }
                            if (!locked_in && !std::isinf(asteroid_least_shot_heading_error_deg) && !std::isnan(asteroid_least_shot_heading_tolerance_deg)) {
                                double altered_turn_command = SHIP_MAX_TURN_RATE*sign(asteroid_least_shot_heading_error_deg);
                                turn_rate = altered_turn_command;
                                if (whole_move_sequence)
                                    (whole_move_sequence.value())[future_timesteps].turn_rate = altered_turn_command;
                            }
                        }
                    }
                }
                else {
                    // Prescribed fire (via input sequence)
                    if (verify_maneuver_shots && fire.value()) {
                        std::optional<Asteroid> actual_asteroid_hit;
                        int64_t timesteps_until_bullet_hit_asteroid;
                        bool ship_was_safe;
                        std::tie(actual_asteroid_hit, timesteps_until_bullet_hit_asteroid, ship_was_safe) = bullet_sim(
                            std::nullopt, false, 0, true, future_timesteps, whole_move_sequence, INT_INF, std::nullopt
                        );
                        if (!actual_asteroid_hit.has_value()) {
                            debug_print("Didn't hit anything; not firing.");
                            fire_this_timestep = false;
                        } else {
                            debug_print("VERIFIED THE SHOT WORKS");
                            fire_this_timestep = true;
                            asteroids_shot += 1;
                        }
                    } else {
                        fire_this_timestep = fire.value();
                        if (fire.value()) asteroids_shot += 1;
                    }
                }
            }
            else {
                fire_this_timestep = false;
            }

            // Create bullet if needed
            if (fire_this_timestep) {
                this->last_timestep_fired = initial_timestep + future_timesteps;
                ship_state.is_respawning = false;
                respawn_timer = 0.0;
                if (ship_state.bullets_remaining != -1) --ship_state.bullets_remaining;
                double rad_heading = ship_state.heading * DEG_TO_RAD;
                double cos_head = cos(rad_heading), sin_head = sin(rad_heading);
                double bullet_x = ship_state.x + SHIP_RADIUS * cos_head;
                double bullet_y = ship_state.y + SHIP_RADIUS * sin_head;
                if (0.0 <= bullet_x && bullet_x <= game_state.map_size_x && 0.0 <= bullet_y && bullet_y <= game_state.map_size_y) {
                    Bullet new_bullet(
                        bullet_x, bullet_y, BULLET_SPEED*cos_head, BULLET_SPEED*sin_head,
                        ship_state.heading, BULLET_MASS, -BULLET_LENGTH*cos_head, -BULLET_LENGTH*sin_head
                    );
                    game_state.bullets.push_back(new_bullet);
                }
            }

            // --- Drop mine logic
            if (ship_state.mines_remaining != 0 && last_timestep_mined <= initial_timestep + future_timesteps - MINE_COOLDOWN_TS) {
                if (!drop_mine.has_value()) {
                    // Determine whether we want to drop a mine
                    bool should_drop_a_mine = false;

                    if (last_timestep_mined <= initial_timestep + future_timesteps - MINE_COOLDOWN_TS - MINE_DROP_COOLDOWN_FUDGE_TS &&
                        !halt_shooting && future_timesteps % MINE_OPPORTUNITY_CHECK_INTERVAL_TS == 0)
                        should_drop_a_mine = check_mine_opportunity(ship_state, game_state, other_ships);

                    if (!std::isinf(game_state.time_limit)) {
                        if (initial_timestep + future_timesteps + 90 > std::floor(FPS * game_state.time_limit)) should_drop_a_mine = false;
                        if (!halt_shooting && initial_timestep + future_timesteps + 90 == std::floor(FPS * game_state.time_limit) &&
                            count_asteroids_in_mine_blast_radius(game_state, ship_state.x, ship_state.y, lround(MINE_FUSE_TIME * FPS)) > 0) should_drop_a_mine = true;
                    }
                    drop_mine_this_timestep = should_drop_a_mine;
                } else {
                    drop_mine_this_timestep = drop_mine.value();
                }

                if (drop_mine_this_timestep) {
                    sim_placed_a_mine = true;
                    last_timestep_mined = initial_timestep + future_timesteps;
                    explanation_messages.push_back(
                        "This is a good chance to drop a mine to hit some asteroids and even the other ship. Bombs away!");
                    ship_state.is_respawning = false;
                    respawn_timer = 0.0;
                    Mine new_mine(ship_state.x, ship_state.y, MINE_MASS, MINE_FUSE_TIME, MINE_FUSE_TIME);
                    mine_positions_placed_history[initial_timestep + future_timesteps + 1] = mine_positions_placed;
                    mine_positions_placed.insert(std::make_pair(ship_state.x, ship_state.y));
                    game_state.mines.push_back(new_mine);
                    --ship_state.mines_remaining;
                    if constexpr (ENABLE_SANITY_CHECKS) assert(ship_state.mines_remaining >= 0);
                }
            }
            else {
                assert(!drop_mine.has_value() || (drop_mine == false));
                drop_mine_this_timestep = false;
            }

            // Respawn timer update
            if (respawn_timer <= 0) respawn_timer = 0.0;
            else respawn_timer -= DELTA_TIME;
            if (!respawn_timer) {
                ship_state.is_respawning = false;
                assert(respawn_timer == 0.0);
            }
            // Ship dynamics
            double drag_amount = SHIP_DRAG * DELTA_TIME;
            if (std::abs(ship_state.speed) < drag_amount) ship_state.speed = 0.0;
            else ship_state.speed -= drag_amount * sign(ship_state.speed);

            if constexpr (ENABLE_SANITY_CHECKS) { assert(-SHIP_MAX_THRUST <= thrust && thrust <= SHIP_MAX_THRUST); }
            ship_state.speed += thrust * DELTA_TIME;
            if constexpr (ENABLE_SANITY_CHECKS) {
                assert(-SHIP_MAX_SPEED-EPS <= ship_state.speed && ship_state.speed <= SHIP_MAX_SPEED+EPS);
                if (ship_state.speed > SHIP_MAX_SPEED) ship_state.speed = SHIP_MAX_SPEED;
                else if (ship_state.speed < -SHIP_MAX_SPEED) ship_state.speed = -SHIP_MAX_SPEED;
                assert(-SHIP_MAX_SPEED <= ship_state.speed && ship_state.speed <= SHIP_MAX_SPEED);
                assert(-SHIP_MAX_TURN_RATE <= turn_rate && turn_rate <= SHIP_MAX_TURN_RATE);
            }
            ship_state.heading += turn_rate * DELTA_TIME;
            ship_state.heading = fmod(ship_state.heading + 360.0, 360.0);
            double rad_heading = ship_state.heading * DEG_TO_RAD;
            ship_state.vx = cos(rad_heading) * ship_state.speed;
            ship_state.vy = sin(rad_heading) * ship_state.speed;
            ship_state.x = fmod(ship_state.x + ship_state.vx * DELTA_TIME + game_state.map_size_x, game_state.map_size_x);
            ship_state.y = fmod(ship_state.y + ship_state.vy * DELTA_TIME + game_state.map_size_y, game_state.map_size_y);
        }

        // --- Bullet/Asteroid collisions ---
        std::vector<Asteroid> new_asteroids;
        new_asteroids.reserve(3);
        for (Bullet& b : game_state.bullets) {
            if (b.alive) {
                for (Asteroid& a : game_state.asteroids) {
                    if (a.alive && asteroid_bullet_collision(
                        b.x, b.y,
                        b.x + b.tail_delta_x, b.y + b.tail_delta_y,
                        a.x, a.y, a.radius)) {

                        b.alive = false;

                        if (a.size != 1) {
                            auto splits = forecast_instantaneous_asteroid_bullet_splits_from_velocity(a, b.vx, b.vy, game_state);
                            std::apply([&](const auto&... sp) {
                                (new_asteroids.push_back(sp), ...);
                            }, splits);
                        }

                        a.alive = false;
                        break;  // Only one asteroid per bullet
                    }
                }
            }
        }

        // Add new asteroids after all iterations are complete
        game_state.asteroids.insert(
            game_state.asteroids.end(),
            new_asteroids.begin(),
            new_asteroids.end()
        );

        // Ship action record
        if (!wait_out_mines)
            ship_move_sequence.push_back(Action(thrust, turn_rate, fire_this_timestep, drop_mine_this_timestep, initial_timestep + future_timesteps));

        // --- Mine/Asteroid and Mine/Ship collisions ---
        
        new_asteroids.clear();
        bool mine_got_destroyed = false;
        for (Mine& mine : game_state.mines) {
            if (mine.alive && mine.remaining_time < EPS) {
                mine.alive = false; mine_got_destroyed = true;
                for (Asteroid& asteroid : game_state.asteroids) {
                    if (asteroid.alive) {
                        double delta_x = asteroid.x - mine.x;
                        double delta_y = asteroid.y - mine.y;
                        double separation = asteroid.radius + MINE_BLAST_RADIUS;
                        if (std::abs(delta_x) <= separation && std::abs(delta_y) <= separation && delta_x*delta_x + delta_y*delta_y <= separation*separation) {
                            if (asteroid.size != 1) {
                                std::apply([&](const auto&... sp) {
                                    (new_asteroids.push_back(sp), ...);
                                }, forecast_asteroid_mine_instantaneous_splits(asteroid, mine, game_state));
                            }
                            asteroid.alive = false;
                        }
                    }
                }
                if (!wait_out_mines) {
                    if (!ship_state.is_respawning) {
                        double delta_x = ship_state.x - mine.x;
                        double delta_y = ship_state.y - mine.y;
                        double separation = SHIP_RADIUS + MINE_BLAST_RADIUS;
                        if (std::abs(delta_x) <= separation && std::abs(delta_y) <= separation && delta_x*delta_x + delta_y*delta_y <= separation*separation) {
                            return_value = false;
                            ship_crashed = true;
                            --ship_state.lives_remaining;
                            ship_state.is_respawning = true;
                            ship_state.speed = 0.0;
                            ship_state.vx = 0.0; ship_state.vy = 0.0;
                            respawn_timer = 3.0;
                        }
                    }
                    else if (respawn_maneuver_pass_number == 1) {
                        double delta_x = ship_state.x - mine.x;
                        double delta_y = ship_state.y - mine.y;
                        double separation = SHIP_RADIUS + MINE_BLAST_RADIUS;
                        if (std::abs(delta_x) <= separation && std::abs(delta_y) <= separation && delta_x*delta_x + delta_y*delta_y <= separation*separation) {
                            last_timestep_colliding = initial_timestep + future_timesteps;
                        }
                    }
                }
            }
        }
        if (mine_got_destroyed && wait_out_mines) {
            std::vector<Mine> new_mines;
            for (const auto& m : game_state.mines)
                if (m.alive) new_mines.push_back(m);
            game_state.mines = new_mines;
        }
        game_state.asteroids.insert(
            game_state.asteroids.end(),
            new_asteroids.begin(),
            new_asteroids.end()
        );
        if (!wait_out_mines) {
            // --- Ship/Asteroid collisions ---
            if (!ship_state.is_respawning) {
                if constexpr (ENABLE_SANITY_CHECKS) {
                    if (respawn_maneuver_pass_number == 0) {
                        assert(!return_value.has_value());
                    }
                }
                new_asteroids.clear();
                for (Asteroid& asteroid : game_state.asteroids) {
                    if (asteroid.alive) {
                        double delta_x = ship_state.x - asteroid.x;
                        double delta_y = ship_state.y - asteroid.y;
                        double separation = SHIP_RADIUS + asteroid.radius;

                        if (std::abs(delta_x) <= separation &&
                            std::abs(delta_y) <= separation &&
                            delta_x * delta_x + delta_y * delta_y <= separation * separation) {

                            if (asteroid.size != 1) {
                                auto splits = forecast_asteroid_ship_splits(asteroid, 0, ship_state.vx, ship_state.vy, game_state);
                                std::apply([&](const auto&... sp) {
                                    (new_asteroids.push_back(sp), ...);
                                }, splits);
                            }

                            asteroid.alive = false;
                            return_value = false;
                            ship_crashed = true;
                            --ship_state.lives_remaining;
                            ship_state.is_respawning = true;
                            ship_state.speed = 0.0;
                            ship_state.vx = 0.0;
                            ship_state.vy = 0.0;
                            respawn_timer = 3.0;
                            break;
                        }
                    }
                }

                // Append any new asteroids after the loop
                game_state.asteroids.insert(
                    game_state.asteroids.end(),
                    new_asteroids.begin(),
                    new_asteroids.end()
                );
            }
            else if (respawn_maneuver_pass_number == 1) {
                assert(halt_shooting);
                for (const auto& asteroid : game_state.asteroids) {
                    if (asteroid.alive) {
                        double delta_x = ship_state.x - asteroid.x;
                        double delta_y = ship_state.y - asteroid.y;
                        double separation = SHIP_RADIUS + asteroid.radius;
                        if (std::abs(delta_x) <= separation && std::abs(delta_y) <= separation && delta_x*delta_x + delta_y*delta_y <= separation*separation) {
                            last_timestep_colliding = initial_timestep + future_timesteps;
                            break;
                        }
                    }
                }
            }
        }

        // Timers and simulation frame increment/postprocessing
        if (!wait_out_mines) {
            future_timesteps += 1;
            game_state.sim_frame += 1;
        }
        respawn_timer_history.push_back(respawn_timer);

        // Final return
        if (!return_value.has_value()) return true;
        return return_value.value();
    }

    bool rotate_heading(double heading_difference_deg, bool shoot_on_first_timestep = false) {
        double target_heading = std::fmod(ship_state.heading + heading_difference_deg + 360.0, 360.0);
        double still_need_to_turn = heading_difference_deg;

        while (std::abs(still_need_to_turn) > SHIP_MAX_TURN_RATE * DELTA_TIME + EPS) {
            assert(-SHIP_MAX_TURN_RATE <= SHIP_MAX_TURN_RATE * sign(heading_difference_deg) &&
                SHIP_MAX_TURN_RATE * sign(heading_difference_deg) <= SHIP_MAX_TURN_RATE);

            if (!update(0.0, SHIP_MAX_TURN_RATE * sign(heading_difference_deg), shoot_on_first_timestep, false)) {
                return false;
            }
            shoot_on_first_timestep = false;
            still_need_to_turn -= SHIP_MAX_TURN_RATE * sign(heading_difference_deg) * DELTA_TIME;
        }

        assert(-SHIP_MAX_TURN_RATE <= still_need_to_turn * FPS && still_need_to_turn * FPS <= SHIP_MAX_TURN_RATE);

        if (!update(0.0, still_need_to_turn * FPS, shoot_on_first_timestep, false)) {
            return false;
        }

        if constexpr (ENABLE_SANITY_CHECKS) {
            assert(std::abs(angle_difference_deg(target_heading, ship_state.heading)) <= GRAIN);
        }
        return true;
    }

    std::vector<Action> get_rotate_heading_move_sequence(double heading_difference_deg, bool shoot_on_first_timestep = false) const {
        std::vector<Action> move_sequence;
        if (std::abs(heading_difference_deg) < GRAIN) {
            // We still need a null sequence here, so that we don't end up with a 0 frame maneuver!
            move_sequence.push_back(Action{0.0, 0.0, shoot_on_first_timestep, false, 0});
            return move_sequence;
        }
        double still_need_to_turn = heading_difference_deg;
        while (std::abs(still_need_to_turn) > SHIP_MAX_TURN_RATE * DELTA_TIME) {
            assert(-SHIP_MAX_TURN_RATE <= SHIP_MAX_TURN_RATE * sign(heading_difference_deg)
                && SHIP_MAX_TURN_RATE * sign(heading_difference_deg) <= SHIP_MAX_TURN_RATE);
            move_sequence.push_back(Action{0.0, SHIP_MAX_TURN_RATE * sign(heading_difference_deg), shoot_on_first_timestep, false, 0});
            shoot_on_first_timestep = false;
            still_need_to_turn -= SHIP_MAX_TURN_RATE * sign(heading_difference_deg) * DELTA_TIME;
        }
        if (std::abs(still_need_to_turn) > EPS) {
            assert(-SHIP_MAX_TURN_RATE <= still_need_to_turn * FPS && still_need_to_turn * FPS <= SHIP_MAX_TURN_RATE);
            move_sequence.push_back(Action{0.0, still_need_to_turn * FPS, shoot_on_first_timestep, false, 0});
        }
        return move_sequence;
    }

    bool accelerate(double target_speed, double turn_rate = 0.0) {
        if (sim_id == 4271) {
            std::cout << "Accelerating to speed " << target_speed << " while our speed is already " << ship_state.speed << std::endl;
        }
        // Keep in mind speed can be negative
        // Drag will always slow down the ship
        while (std::abs(ship_state.speed - target_speed) > EPS) {
            double drag = -SHIP_DRAG * sign(ship_state.speed);
            double drag_amount = SHIP_DRAG * DELTA_TIME;
            if (drag_amount > std::abs(ship_state.speed)) {
                // The drag amount is reduced if it would make the ship cross 0 speed on its own
                double adjust_drag_by = std::abs((drag_amount - std::abs(ship_state.speed)) * FPS);
                drag -= adjust_drag_by * sign(drag);
            }
            double delta_speed_to_target = target_speed - ship_state.speed;
            double thrust_amount = delta_speed_to_target * FPS - drag;
            // Clamp thrust
            thrust_amount = std::min(std::max(-SHIP_MAX_THRUST, thrust_amount), SHIP_MAX_THRUST);

            if (!update(thrust_amount, turn_rate)) {
                if (sim_id == 4271) {
                    std::cout << "AHA!" << std::endl;
                }
                return false;
            }
        }
        return true;
    }

    bool cruise(int64_t cruise_time, double cruise_turn_rate = 0.0) {
        // Maintain current speed
        for (int64_t i = 0; i < cruise_time; ++i) {
            if (sim_id == 4271) {
                std::cout << "In respawn sim that'll crash, future_timesteps=" << future_timesteps
                        << " respawn_timer=" << respawn_timer
                        << " ship_state=" << ship_state.str() << std::endl;
            }
            if (!update(sign(ship_state.speed) * SHIP_DRAG, cruise_turn_rate)) {
                if (sim_id == 4271) {
                    std::cout << "AHA CRUISE FAILED SOMEHOW?!!!! " << ship_state.str() << " ";
                    // Assuming respawn_timer_history is a vector<double>
                    std::cout << "[";
                    for (size_t k = 0; k < respawn_timer_history.size(); ++k) {
                        if (k > 0) std::cout << ", ";
                        std::cout << respawn_timer_history[k];
                    }
                    std::cout << "]" << std::endl;
                }
                return false;
            }
        }
        return true;
    }

    std::vector<Action> get_move_sequence() const {
        if constexpr (ENABLE_SANITY_CHECKS) {
            int64_t last_ts_shot = -10;
            for (const Action& move : ship_move_sequence) {
                if (move.fire) {
                    assert(move.timestep > last_ts_shot);
                    if (move.timestep - last_ts_shot < 3) {
                        // Printing the whole sequence for debugging
                        std::cout << "ship_move_sequence = [";
                        for (const auto& act : ship_move_sequence) {
                            std::cout << "(t=" << act.timestep << ",f=" << act.fire << "), ";
                        }
                        std::cout << "]\n";
                        throw std::runtime_error("Uhh wth");
                    }
                    last_ts_shot = move.timestep;
                }
            }
        }
        return ship_move_sequence;
    }

    std::vector<Action> get_intended_move_sequence() const {
        if (!intended_move_sequence.empty()) {
            return intended_move_sequence;
        } else {
            return ship_move_sequence;
        }
    }

    std::vector<SimState> get_state_sequence() {
        if (!state_sequence.empty() && state_sequence.back().timestep != initial_timestep + future_timesteps) {
            assert(state_sequence.back().timestep + 1 == initial_timestep + future_timesteps);
            // Build deep copies
            std::vector<Asteroid> forecasted_splits_copy;
            for (const Asteroid& a : forecasted_asteroid_splits) {
                forecasted_splits_copy.push_back(a);
            }

            SimState new_state(
                initial_timestep + future_timesteps,
                ship_state,
                get_game_state(),
                asteroids_pending_death,
                forecasted_splits_copy
            );
            state_sequence.push_back(new_state);
            if constexpr (ENABLE_SANITY_CHECKS) {
                for (const Asteroid& a : forecasted_asteroid_splits) {
                    assert(a.alive);
                }
            }
        }
        return state_sequence;
    }

    int64_t get_sequence_length() const {
        // debug_print
        if constexpr (ENABLE_SANITY_CHECKS) {
            size_t ship_moves = ship_move_sequence.size();
            size_t states = state_sequence.size();
            if (!(ship_moves + 1 == states || ship_moves == states)) {
                std::cout << "len(ship_move_sequence): " << ship_moves << ", len(state_sequence): " << states << std::endl;
            }
            assert(ship_moves + 1 == states || ship_moves == states);
        }
        return static_cast<int64_t>(state_sequence.size());
    }

    int64_t get_future_timesteps() const {
        return future_timesteps;
    }

    std::pair<double, double> get_position() const {
        return std::make_pair(ship_state.x, ship_state.y);
    }

    int64_t get_last_timestep_fired() const {
        return this->last_timestep_fired;
    }

    int64_t get_last_timestep_mined() const {
        return last_timestep_mined;
    }

    std::pair<double, double> get_velocity() const {
        return std::make_pair(ship_state.vx, ship_state.vy);
    }

    double get_heading() const {
        return static_cast<double>(ship_state.heading);
    }
};

struct CompletedSimulation {
    Matrix sim;
    double fitness;
    std::array<double, 9> fitness_breakdown;
    std::string action_type;
    std::string state_type;
    std::tuple<double, double, double, int64_t, double> maneuver_tuple;
};










class NeoController {
private:
    int _ship_id = 0;

public:
    // --- Properties ---
    //std::string name() const { return "Neo"; }
    std::string custom_sprite_path() const {
        return "Neo.png";
    }
    
    std::string name() const {
        return "Neo++";
    }

    int ship_id() const {
        return _ship_id;
    }

    void set_ship_id(int value) {
        _ship_id = value;
    }

    // --- Public Variables (Python: made them all instance, here public/protected for simplicity) ---

    bool init_done = false;
    int64_t ship_id_internal = -1;
    int64_t current_timestep = -1;
    std::deque<std::tuple<int64_t, double, double, bool, bool>> action_queue; // (timestep, thrust, turn_rate, fire, drop_mine)
    //std::optional<GameStatePlotter> game_state_plotter;
    std::unordered_set<int64_t> actioned_timesteps;
    std::vector<CompletedSimulation> sims_this_planning_period; // The first is stationary targeting, rest are maneuvers
    double best_fitness_this_planning_period = -inf;
    int64_t best_fitness_this_planning_period_index = INT_NEG_INF;
    double second_best_fitness_this_planning_period = -inf;
    int64_t second_best_fitness_this_planning_period_index = INT_NEG_INF;
    int64_t stationary_targetting_sim_index = INT_NEG_INF;
    double current_sequence_fitness = -inf;
    std::unordered_map<int64_t, double> respawn_timer_history;
    std::unordered_map<int64_t, int64_t> last_timestep_fired_schedule = {{0, INT_NEG_INF}};
    std::unordered_map<int64_t, int64_t> last_timestep_mined_schedule = {{0, INT_NEG_INF}};
    std::unordered_set<int64_t> fire_next_timestep_schedule;
    std::unordered_map<int64_t, std::unordered_map<int64_t, std::vector<Asteroid>>> asteroids_pending_death_schedule;
    std::unordered_map<int64_t, std::vector<Asteroid>> forecasted_asteroid_splits_schedule;
    std::unordered_map<int64_t, std::set<std::pair<double, double>>> mine_positions_placed_schedule;

    std::optional<BasePlanningGameState> game_state_to_base_planning;
    std::optional<std::tuple<double, double, double, double, int64_t, double, int64_t, int64_t>> base_gamestate_analysis;
    std::unordered_set<int64_t> set_of_base_gamestate_timesteps;
    std::unordered_map<int64_t, BasePlanningGameState> base_gamestates; // Key is timestep, value is the state

    bool other_ships_exist = false;
    //std::vector<std::unordered_map<std::string, BasePlanningGameState>> reality_move_sequence;
    std::unordered_map<int64_t, SimState> simulated_gamestate_history;
    std::unordered_set<int64_t> lives_remaining_that_we_did_respawn_maneuver_for;
    bool last_timestep_ship_is_respawning = false;
    bool fire_next_timestep_flag = false;

    // For performance controller
    std::vector<double> outside_controller_time_intervals;
    std::vector<double> inside_controller_iteration_time_intervals;
    double last_entrance_time = std::numeric_limits<double>::quiet_NaN();
    double last_exit_time = std::numeric_limits<double>::quiet_NaN();
    double last_iteration_start_time = std::numeric_limits<double>::quiet_NaN();
    double average_iteration_time = DELTA_TIME*0.1;

    // --- Ctor ---
    NeoController(const std::optional<std::array<double, 9>> chromosome = std::nullopt)
    {
        std::cout << BUILD_NUMBER << std::endl;
        // Could add __FILE__ or __func__ here, but omitted.
        reset(chromosome);
    }

    // --- Reset function ---
    void reset(const std::optional<std::array<double, 9>> chromosome = std::nullopt)
    {
        init_done = false;
        // DO NOT overwrite _ship_id
        ship_id_internal = -1;
        current_timestep = -1;
        action_queue.clear();
        //game_state_plotter.reset();
        actioned_timesteps.clear();
        sims_this_planning_period.clear();
        best_fitness_this_planning_period = -inf;
        best_fitness_this_planning_period_index = INT_NEG_INF;
        second_best_fitness_this_planning_period = -inf;
        second_best_fitness_this_planning_period_index = INT_NEG_INF;
        stationary_targetting_sim_index = INT_NEG_INF;
        current_sequence_fitness = -inf;
        respawn_timer_history.clear();
        last_timestep_fired_schedule = {{0, INT_NEG_INF}};
        last_timestep_mined_schedule = {{0, INT_NEG_INF}};
        fire_next_timestep_schedule.clear();
        asteroids_pending_death_schedule.clear();
        forecasted_asteroid_splits_schedule.clear();
        mine_positions_placed_schedule.clear();
        game_state_to_base_planning.reset();
        base_gamestate_analysis.reset();
        set_of_base_gamestate_timesteps.clear();
        base_gamestates.clear();
        other_ships_exist = false;
        simulated_gamestate_history.clear();
        lives_remaining_that_we_did_respawn_maneuver_for.clear();
        last_timestep_ship_is_respawning = false;
        // For chromosomes
        if (chromosome.has_value()) fitness_function_weights = chromosome;
        // For performance controller
        outside_controller_time_intervals.clear();
        inside_controller_iteration_time_intervals.clear();
        last_entrance_time = std::numeric_limits<double>::quiet_NaN();
        last_exit_time = std::numeric_limits<double>::quiet_NaN();
        last_iteration_start_time = std::numeric_limits<double>::quiet_NaN();
        average_iteration_time = DELTA_TIME*0.1;
        // Clear "global" variables
        explanation_messages_with_timestamps.clear();
        abs_cruise_speeds = {SHIP_MAX_SPEED/2};
        cruise_timesteps_global_history = {static_cast<int64_t>(std::round(MAX_CRUISE_TIMESTEPS/2))};
        overall_fitness_record.clear();
        unwrap_cache.clear();
        total_sim_timesteps = 0;
    }

    // --- Init helper ---
    void finish_init(GameState& game_state, Ship& ship_state)
    {
        if (ship_id_internal == -1)
            ship_id_internal = ship_state.id;
        // (Optional plotting etc.)
        // if (GAMESTATE_PLOTTING)
        //     game_state_plotter.emplace(game_state);
        if (!get_other_ships(game_state, ship_id_internal).empty()) {
            other_ships_exist = true;
            print_explanation("I've got another ship friend here with me..."
                , current_timestep);
        } else {
            other_ships_exist = false;
            print_explanation("I'm alone. I can see into the future perfectly!", current_timestep);
        }
    }

    // --- Queue actions ---
    void enqueue_action(int64_t timestep, double thrust=0.0, double turn_rate=0.0, bool fire=false, bool drop_mine=false)
    {
        action_queue.push_back(std::make_tuple(timestep, thrust, turn_rate, fire, drop_mine));
    }

    bool decide_next_action_continuous(const GameState &game_state, const Ship &ship_state, bool force_decision) {
        // Extern global, as in Python
        extern std::unordered_map<int64_t, std::vector<Asteroid>> unwrap_cache;

        debug_print("Calling decide next action continuous on timestep " + std::to_string(game_state.sim_frame) + ", and force_decision=" + std::string(force_decision ? "true" : "false"));
        assert(game_state_to_base_planning.has_value());
        assert(best_fitness_this_planning_period_index != INT_NEG_INF);

        debug_print("\nDeciding next action! We're picking out of " + std::to_string(sims_this_planning_period.size()) + " total sims");
        //std::cout << "Deciding next action! We're picking out of " << std::to_string(sims_this_planning_period.size()) << " total sims on ts" << std::to_string(this->current_timestep) << std::endl;
        // Assume a helper function to pretty-print fitnesses if desired.

        // (Optional plotting omitted, see Python for matplotlib calls.)

        assert(sims_this_planning_period.at(best_fitness_this_planning_period_index).state_type == "exact");

        // Setup placeholders for the variables we pick out below.
        Matrix best_action_sim;
        double best_action_fitness;
        std::array<double, 9> best_action_fitness_breakdown;
        std::optional<std::tuple<double, double, double, int64_t, double>> best_action_maneuver_tuple;

        /*for (int64_t val : this->lives_remaining_that_we_did_respawn_maneuver_for) {
            std::cout << val << " ";
        }
        std::cout << std::endl;*/
        const CompletedSimulation &sim = sims_this_planning_period.at(best_fitness_this_planning_period_index);
        // --- Multi-pass respawn handling ---
        if (game_state_to_base_planning->respawning && sim.sim.get_respawn_maneuver_pass_number() == 1) {
            // Make super sure this really is the second pass of a respawn maneuver! Just because we're currently invincible doesn't mean we're doing a respawn maneuver!!!!
            const Matrix &first_pass_sim = sim.sim;
            double first_pass_fitness = sims_this_planning_period.at(best_fitness_this_planning_period_index).fitness;
            bool first_pass_sim_fire_next_timestep_flag = first_pass_sim.get_fire_next_timestep_flag();
            assert(first_pass_sim.get_respawn_maneuver_pass_number() == 1);
            // Construct second-pass respawn maneuver simulation
            best_action_sim = Matrix(
                game_state,
                ship_state,
                current_timestep,
                game_state_to_base_planning->ship_respawn_timer,
                &game_state_to_base_planning->asteroids_pending_death,
                &game_state_to_base_planning->forecasted_asteroid_splits,
                game_state_to_base_planning->last_timestep_fired,
                game_state_to_base_planning->last_timestep_mined,
                &game_state_to_base_planning->mine_positions_placed,
                game_state_to_base_planning->respawning,
                game_state_to_base_planning->fire_next_timestep_flag,
                /*verify_first_shot=*/true, /*verify_maneuver_shots=*/true,
                first_pass_sim.get_last_timestep_colliding(), // pass down
                2 // Respawn maneuver pass 2
                //game_state_plotter
            );

            auto first_pass_move_sequence = first_pass_sim.get_intended_move_sequence();
            best_action_sim.apply_move_sequence(first_pass_move_sequence, true);
            best_action_sim.set_fire_next_timestep_flag(first_pass_sim_fire_next_timestep_flag);

            best_action_fitness = best_action_sim.get_fitness();
            best_action_fitness_breakdown = best_action_sim.get_fitness_breakdown();
            best_action_maneuver_tuple = sims_this_planning_period.at(best_fitness_this_planning_period_index).maneuver_tuple;

            // If second pass was significantly worse for some reason, revert to first pass.
            if (first_pass_fitness > best_action_fitness + 0.015) {
                best_action_sim = sims_this_planning_period.at(best_fitness_this_planning_period_index).sim;
                best_action_fitness = first_pass_fitness;
                best_action_fitness_breakdown = sims_this_planning_period.at(best_fitness_this_planning_period_index).fitness_breakdown;
                best_action_maneuver_tuple = sims_this_planning_period.at(best_fitness_this_planning_period_index).maneuver_tuple;
            }
        } else {
            // Exact one-pass
            assert(sim.sim.get_respawn_maneuver_pass_number() == 0);
            best_action_sim = sim.sim;
            best_action_fitness = sim.fitness;
            best_action_fitness_breakdown = sim.fitness_breakdown;
            best_action_maneuver_tuple = sim.maneuver_tuple;
        }

        // Only switch to this sequence if its fitness is better.
        if (!force_decision) {
            if (best_action_fitness > current_sequence_fitness) {
                debug_print("Wipe the current move sequence and switch to the new better sequence! Current action seq fitness is " +
                            std::to_string(current_sequence_fitness) + " but we can do " + std::to_string(best_action_fitness));
                action_queue.clear();
                actioned_timesteps.clear();
                fire_next_timestep_schedule.clear();
            } else {
                this->sims_this_planning_period.clear();
                this->best_fitness_this_planning_period = -inf;
                this->best_fitness_this_planning_period_index = INT_NEG_INF;
                this->second_best_fitness_this_planning_period = -inf;
                this->second_best_fitness_this_planning_period_index = INT_NEG_INF;
                this->stationary_targetting_sim_index = INT_NEG_INF;
                this->base_gamestate_analysis.reset();
                unwrap_cache.clear();
                return false;
            }
        } else {
            assert(action_queue.empty());
        }

        // LEARNING statistics (rolling averages for maneuver learning)
        if (best_action_maneuver_tuple.has_value() && !game_state_to_base_planning->respawning && best_action_fitness_breakdown.at(5) != 0.0) {
            abs_cruise_speeds.push_back(std::abs(std::get<1>(best_action_maneuver_tuple.value())));
            cruise_timesteps_global_history.push_back(std::get<3>(best_action_maneuver_tuple.value()));
            if (abs_cruise_speeds.size() > MANEUVER_TUPLE_LEARNING_ROLLING_AVG_PERIOD)
                abs_cruise_speeds.erase(abs_cruise_speeds.begin(), abs_cruise_speeds.end() - MANEUVER_TUPLE_LEARNING_ROLLING_AVG_PERIOD);
            if (cruise_timesteps_global_history.size() > MANEUVER_TUPLE_LEARNING_ROLLING_AVG_PERIOD)
                cruise_timesteps_global_history.erase(cruise_timesteps_global_history.begin(), cruise_timesteps_global_history.end() - MANEUVER_TUPLE_LEARNING_ROLLING_AVG_PERIOD);
        }
        overall_fitness_record.push_back(best_action_fitness);
        if (overall_fitness_record.size() > OVERALL_FITNESS_ROLLING_AVERAGE_PERIOD)
            overall_fitness_record.erase(overall_fitness_record.begin(), overall_fitness_record.end() - OVERALL_FITNESS_ROLLING_AVERAGE_PERIOD);

        // -- Explanations and status dump --
        if (PRINT_EXPLANATIONS) {
            if (stationary_targetting_sim_index != INT_NEG_INF) {
                auto stationary_safety_messages = sims_this_planning_period.at(stationary_targetting_sim_index).sim.get_safety_messages();
                for (const auto& msg : stationary_safety_messages)
                    print_explanation(msg, current_timestep);
            }
            if (best_action_fitness_breakdown[5] == 0.0) {
                print_explanation("RIP, I'm gonna die", current_timestep);
            }
            if (game_state_to_base_planning->respawning) {
                print_explanation("Doing a respawn maneuver to get to a safe spot using my respawn invincibility. This maneuver was the best one picked out of "
                    + std::to_string(sims_this_planning_period.size()) + " randomly chosen maneuvers!", current_timestep);
            }
            const auto& best_sim = sims_this_planning_period.at(best_fitness_this_planning_period_index);
            if (best_sim.action_type == "random_maneuver" || best_sim.action_type == "heuristic_maneuver") {
                if (stationary_targetting_sim_index != INT_NEG_INF) {
                    const auto& stationary_fitness_breakdown = sims_this_planning_period.at(stationary_targetting_sim_index).fitness_breakdown;
                    if (best_action_fitness_breakdown[1] == 1.0 && stationary_fitness_breakdown[1] == 1.0) {
                        if (best_action_fitness_breakdown[0] > stationary_fitness_breakdown[0])
                            print_explanation("Doing a maneuver to dodge asteroids! This maneuver was the best one picked out of "
                                + std::to_string(sims_this_planning_period.size()) + " randomly chosen maneuvers!", current_timestep);
                    } else if (best_action_fitness_breakdown[1] > stationary_fitness_breakdown[1]) {
                        print_explanation("Doing a maneuver to dodge a mine! This maneuver was the best one picked out of "
                            + std::to_string(sims_this_planning_period.size()) + " randomly chosen maneuvers!", current_timestep);
                    }
                    if (best_action_fitness_breakdown[4] > stationary_fitness_breakdown[4] + 0.05)
                        print_explanation("Doing a maneuver to get away from the other ship! This maneuver was the best one picked out of "
                            + std::to_string(sims_this_planning_period.size()) + " randomly chosen maneuvers!", current_timestep);
                }
            }
        }

        // Compose the best move sequence, enqueue it, and update planning state
        std::vector<Action> best_move_sequence = best_action_sim.get_move_sequence();
        debug_print("Best sim ID: " + std::to_string(best_action_sim.get_sim_id()) + ", with index " + std::to_string(best_fitness_this_planning_period_index) + " and fitness " + std::to_string(best_action_fitness));
        std::vector<SimState> best_action_sim_state_sequence = best_action_sim.get_state_sequence();
        if constexpr (VALIDATE_ALL_SIMULATED_STATES && !PRUNE_SIM_STATE_SEQUENCE) {
            for (const SimState& state : best_action_sim_state_sequence) {
                simulated_gamestate_history[state.timestep] = state;
            }
        }
        if (PRINT_EXPLANATIONS) {
            auto explanations = best_action_sim.get_explanations();
            for (const std::string& exp : explanations) {
                print_explanation(exp, current_timestep);
            }
            if (random_double() < 0.1) {
                print_explanation("I currently feel " + std::to_string(weighted_average(overall_fitness_record) * 100.0)
                    + "% safe, considering how long I can stay here without being hit by asteroids or mines, and my proximity to the other ship.",
                    current_timestep);
            }
        }
        if (best_action_sim_state_sequence.empty()) {
            throw std::runtime_error("Why in the world is this state sequence empty?");
        }

        auto best_action_sim_last_state = best_action_sim_state_sequence.back();
        auto asteroids_pending_death = best_action_sim.get_asteroids_pending_death();

        for (int64_t timestep = current_timestep; timestep < best_action_sim_last_state.timestep; ++timestep) {
            asteroids_pending_death.erase(timestep);
        }

        auto forecasted_asteroid_splits = best_action_sim.get_forecasted_asteroid_splits();
        auto next_base_game_state = best_action_sim.get_game_state();

        // Made this change, because if we're waiting out mines, that'll mess up the game state. But the state sequence still has the last actual game state, so we'll use that!
        // Pretty sure this is obsolete with continuous planning though!
        this->set_of_base_gamestate_timesteps.insert(best_action_sim_last_state.timestep);
        auto new_ship_state = best_action_sim.get_ship_state();
        bool new_fire_next_timestep_flag = best_action_sim.get_fire_next_timestep_flag();

        if (new_fire_next_timestep_flag && new_ship_state.is_respawning && lives_remaining_that_we_did_respawn_maneuver_for.count(new_ship_state.lives_remaining) == 0) {
            // Forcing off the fire next timestep, because we just took damage and we're going into respawn maneuver mode! Don't want to get wombo combo'd
            new_fire_next_timestep_flag = false;
        }

        if constexpr (ENABLE_SANITY_CHECKS) {
            if (lives_remaining_that_we_did_respawn_maneuver_for.count(new_ship_state.lives_remaining) == 0 && new_ship_state.is_respawning) {
            // If our ship is hurt in our next next action and I haven't done a respawn maneuver yet,
            // Then assert our next action is not a respawning action (REMOVED: Python's commented-out assertion).
            //if (game_state_to_base_planning->respawning || new_fire_next_timestep_flag) {
            //    std::cerr << "We haven't done a respawn maneuver for having " << new_ship_state.lives_remaining << " lives left\n";
            //    std::cerr << "game_state_to_base_planning->respawning: " << game_state_to_base_planning->respawning << ", new_fire_next_timestep_flag: " << new_fire_next_timestep_flag << ", respawn_timer=" << best_action_sim.get_respawn_timer() << "\n";
            //}
            // assert(!game_state_to_base_planning->respawning && !new_fire_next_timestep_flag);
            }
        }

        // Update planning state for next tick
        /*
        game_state_to_base_planning = BasePlanningGameState{
            best_action_sim_last_state.timestep,
            lives_remaining_that_we_did_respawn_maneuver_for.find(new_ship_state.lives_remaining) == lives_remaining_that_we_did_respawn_maneuver_for.end() && new_ship_state.is_respawning,
            new_ship_state,
            next_base_game_state,
            best_action_sim.get_respawn_timer(),
            asteroids_pending_death,
            forecasted_asteroid_splits,
            best_action_sim.get_last_timestep_fired(),
            best_action_sim.get_last_timestep_mined(),
            best_action_sim.get_mine_positions_placed(),
            new_fire_next_timestep_flag
        };*/


        // Histories
        respawn_timer_history = best_action_sim.get_respawn_timer_history();
        asteroids_pending_death_schedule = best_action_sim.get_asteroids_pending_death_history();
        forecasted_asteroid_splits_schedule = best_action_sim.get_forecasted_asteroid_splits_history();
        mine_positions_placed_schedule = best_action_sim.get_mine_positions_placed_history();
        int64_t last_timestep_fired = best_action_sim.get_last_timestep_fired();
        //std::cout << last_timestep_fired << std::endl;
        if (new_fire_next_timestep_flag) {
            // Remember that we want to shoot the next frame!
            fire_next_timestep_schedule.insert(best_move_sequence.back().timestep + 1);
            debug_print("Just added " + std::to_string(best_move_sequence.back().timestep + 1) + " to fire_next_timestep_schedule.");
        }

        if constexpr (ENABLE_SANITY_CHECKS) {
            assert((bool)game_state_to_base_planning->ship_respawn_timer == game_state_to_base_planning->ship_state.is_respawning);
        }
        if (game_state_to_base_planning->respawning) {
            // The action we're doing now is a respawn maneuver, so mark the number of lives we now have after losing a life
            lives_remaining_that_we_did_respawn_maneuver_for.insert(new_ship_state.lives_remaining);
        }

        base_gamestates[best_action_sim_last_state.timestep] = game_state_to_base_planning.value(); // Save state for validation/debug

        // Optionally dump state to file
        // if (KEY_STATE_DUMP)
        // if (SIMULATION_STATE_DUMP)
        // if (game_state_plotter.has_value() && ...)

        assert(action_queue.empty());
        if (CONTINUOUS_LOOKAHEAD_PLANNING) {
            if (best_move_sequence.front().timestep != game_state.sim_frame) {
                std::cerr << "Assertion failed: best_move_sequence.front().timestep == game_state.sim_frame\n";
                std::cerr << "best_move_sequence.front().timestep = " << best_move_sequence.front().timestep << "\n";
                std::cerr << "game_state.sim_frame = " << game_state.sim_frame << "\n";
                for (const auto& m : best_move_sequence) {
                    std::cerr << m << std::endl;
                }
                assert(false);
            }
        }
        for (const Action& move : best_move_sequence) {
            if constexpr (ENABLE_SANITY_CHECKS) {
                assert(actioned_timesteps.count(move.timestep) == 0 && "DUPLICATE TIMESTEPS IN ENQUEUED MOVES");
                actioned_timesteps.insert(move.timestep);
                assert(move.timestep >= game_state.sim_frame);
            }
            enqueue_action(move.timestep, move.thrust, move.turn_rate, move.fire, move.drop_mine);

            if (CONTINUOUS_LOOKAHEAD_PLANNING) {
                if (move.fire)
                    last_timestep_fired_schedule[move.timestep + 1] = move.timestep;
                else
                    last_timestep_fired_schedule[move.timestep + 1] = last_timestep_fired_schedule[move.timestep];
                if (move.drop_mine)
                    last_timestep_mined_schedule[move.timestep + 1] = move.timestep;
                else
                    last_timestep_mined_schedule[move.timestep + 1] = last_timestep_mined_schedule[move.timestep];
            }
        }
        if (last_timestep_fired_schedule[best_move_sequence.back().timestep + 1] != last_timestep_fired) {
            print_sorted_dict(last_timestep_fired_schedule);
            std::cout << last_timestep_fired << std::endl;
        }
        if (CONTINUOUS_LOOKAHEAD_PLANNING)
            assert(last_timestep_fired_schedule[best_move_sequence.back().timestep + 1] == last_timestep_fired);

        current_sequence_fitness = best_action_fitness;

        // Reset planning bookkeeping
        this->sims_this_planning_period.clear();
        this->best_fitness_this_planning_period = -inf;
        this->best_fitness_this_planning_period_index = INT_NEG_INF;
        this->second_best_fitness_this_planning_period = -inf;
        this->second_best_fitness_this_planning_period_index = INT_NEG_INF;
        this->stationary_targetting_sim_index = INT_NEG_INF;
        this->base_gamestate_analysis.reset();

        unwrap_cache.clear();
        return true;
    }

    void plan_action_continuous(bool other_ships_exist, bool base_state_is_exact, bool iterations_boost = false, bool plan_stationary = false) {
        // other_ships_exist: True means it's multiagent, False means single agent
        // base_state_is_exact: Whether the base state is the current exact state or a future deterministic state, or a future predicted state which could be invalid due to the other ship

        // Simulate and look for a good move
        // We have two options. Stay put and focus on targetting asteroids, or we can come up with an avoidance maneuver and target asteroids along the way if convenient
        // We simulate both options, and take the one with the higher fitness score
        // If we stay still, we can potentially keep shooting asteroids that are on collision course with us without having to move
        // But if we're overwhelmed, it may be a lot better to move to a safer spot
        // The third scenario is that even if we're safe where we are, we may be able to be on the offensive and seek out asteroids to lay mines, so that can also increase the fitness function of moving, making it better than staying still
        // Our number one priority is to stay alive. Second priority is to shoot as much as possible. And if we can, lay mines without putting ourselves in danger.
        // assert self.game_state_to_base_planning is not None
        assert(this->game_state_to_base_planning.has_value());
        auto& planning_state = this->game_state_to_base_planning.value();

        // assert base_state_is_exact
        assert(base_state_is_exact);
        std::string state_type = base_state_is_exact ? "exact" : "predicted";
        // We only plan for respawn maneuvers if we're currently in our respawn invincibility (duh), AND we aren't at the very tail end of a respawn maneuver where we just came to a complete stop, and we should be ditching our invincibility and starting the next non-respawn action!
        // Another criteria for deciding the tail end of the maneuver, is if the ship's respawn invincibility time we have left is below a threshold. Probably want to be "conservative" here and add some buffer to this threshold.
        assert((planning_state.respawning && planning_state.ship_respawn_timer != 0.0) || !planning_state.respawning);
        if (planning_state.respawning && !(planning_state.ship_respawn_timer < 3.0 - (1.0 + TIMESTEPS_IT_TAKES_SHIP_TO_ACCELERATE_TO_FULL_SPEED_FROM_DEAD_STOP) * DELTA_TIME && is_kinda_close_to_zero(planning_state.ship_state.speed) && this->lives_remaining_that_we_did_respawn_maneuver_for.count(planning_state.ship_state.lives_remaining))) {
            // --- Respawn branch ---
            // Simulate and look for a good move
            //std::cout << "Planning a respawn maneuver" << std::endl;
            double MAX_CRUISE_SECONDS = 1.0 + 26.0 * DELTA_TIME;
            int search_iterations_count = 0;

            // assert not game_state_to_base_planning['fire_next_timestep_flag']
            assert(!planning_state.fire_next_timestep_flag);

            while ( // TODO: Fix this lmao
                (search_iterations_count < get_min_respawn_per_timestep_search_iterations(
                    planning_state.ship_state.lives_remaining,
                    weighted_average(overall_fitness_record)
                ) || true) &&
                search_iterations_count < MAX_RESPAWN_PER_TIMESTEP_SEARCH_ITERATIONS)
            {
                //this->performance_controller_start_iteration();
                search_iterations_count++;
                int num_sims_this_planning_period = int(this->sims_this_planning_period.size());

                double random_ship_heading_angle = 0.0;
                double ship_accel_turn_rate = 0.0;
                double ship_cruise_speed = 0.0;
                double ship_cruise_turn_rate = 0.0;
                int64_t ship_cruise_timesteps = 0;

                if (num_sims_this_planning_period == 0) {
                    // On the first iteration, try the null action. For ring scenarios, it may be best to stay at the center of the ring.
                    // TODO: RESTORE NULL ACTION
                    random_ship_heading_angle = 0.0;
                    ship_accel_turn_rate = 0.0;
                    ship_cruise_speed = 0.0;
                    ship_cruise_turn_rate = 0.0;
                    ship_cruise_timesteps = 0;
                } else if (num_sims_this_planning_period == 1) {
                    // On the second iteration, try staying still for 1 second (just turn a little bit so we can use the same framework)
                    random_ship_heading_angle = 180.0;
                    ship_accel_turn_rate = 180.0;
                    ship_cruise_speed = 0.0;
                    ship_cruise_turn_rate = 0.0;
                    ship_cruise_timesteps = 0;
                } else if (num_sims_this_planning_period == 2) {
                    // On the third iteration, try staying still for 2 seconds (just turn a little bit etc.)
                    random_ship_heading_angle = 180.0;
                    ship_accel_turn_rate = 90.0;
                    ship_cruise_speed = 0.0;
                    ship_cruise_turn_rate = 0.0;
                    ship_cruise_timesteps = 0;
                } else {
                    random_ship_heading_angle = rand_uniform(-20.0, 20.0);
                    ship_accel_turn_rate = rand_uniform(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE);
                    if (random_double() < 0.5)
                        ship_cruise_speed = SHIP_MAX_SPEED;
                    else
                        ship_cruise_speed = -SHIP_MAX_SPEED;
                    ship_cruise_turn_rate = 0.0;
                    ship_cruise_timesteps = randint(0, int64_t(round(MAX_CRUISE_SECONDS * FPS)));
                }
                if constexpr (ENABLE_SANITY_CHECKS) {
                    if (!(bool(planning_state.ship_respawn_timer) == planning_state.ship_state.is_respawning)) {
                        std::cout << "BAD, game_state_to_base_planning->ship_respawn_timer: "
                                << planning_state.ship_respawn_timer
                                << ", game_state_to_base_planning->ship_state.is_respawning: "
                                << planning_state.ship_state.is_respawning
                                << std::endl;
                    }
                }

                // TODO: There's a hardcoded false in the arguments to the following sim. Investigate!!!

                //std::cout << "Making a matrix in respawn starting on ts " << planning_state.timestep << std::endl;
                Matrix maneuver_sim(
                    planning_state.game_state,
                    planning_state.ship_state,
                    planning_state.timestep,
                    planning_state.ship_respawn_timer,
                    &planning_state.asteroids_pending_death,
                    &planning_state.forecasted_asteroid_splits,
                    planning_state.last_timestep_fired,
                    planning_state.last_timestep_mined,
                    &planning_state.mine_positions_placed,
                    /* halt_shooting */ true,
                    /* fire_first_timestep */ false && planning_state.fire_next_timestep_flag,
                    /* verify_first_shot */ false,
                    /* verify_maneuver_shots */ false,
                    -1, // Last timestep colliding, dunno
                    1 // Respawn maneuver pass 1
                    //this->game_state_plotter
                );

                double current_ship_speed = planning_state.ship_state.speed;
                double timesteps_it_takes_to_stop_from_current_speed = std::ceil(current_ship_speed / (SHIP_DRAG + SHIP_MAX_THRUST) * FPS);
                double timesteps_we_have_for_respawn_maneuver = std::round(planning_state.ship_respawn_timer * FPS);
                double timesteps_we_have_for_middle_of_maneuver = std::round(timesteps_we_have_for_respawn_maneuver - timesteps_it_takes_to_stop_from_current_speed);
                assert(timesteps_we_have_for_middle_of_maneuver > 0.0 && "We don't have a positive number of timesteps for a maneuver!");
                bool start_of_respawn_maneuver = planning_state.ship_respawn_timer == 3.0;
                // If this is true, then that means the turning and acceleration phase of the existing respawn maneuver is done
                bool respawn_maneuver_at_max_speed = is_close(std::abs(current_ship_speed), SHIP_MAX_SPEED);
                assert(!(start_of_respawn_maneuver && respawn_maneuver_at_max_speed)); // We can't be at the start of a maneuver, and still already be at max speed! Respawn maneuvers start from 0 speed!
                // We have to make sure the respawn maneuver finishes before the ship's respawn invincibility is up.
                // Because this is continuously planned, at any point during the maneuver sequence, it will try to start a new sequence.
                // So we have to use this preview move sequence to get an idea of the length of the maneuver.
                // It's possible to calculate this mathematically, but this might be more accurate.
                bool respawn_maneuver_without_crash;
                if (planning_state.ship_respawn_timer <= (3.0 + TIMESTEPS_IT_TAKES_SHIP_TO_ACCELERATE_TO_FULL_SPEED_FROM_DEAD_STOP + TIMESTEPS_IT_TAKES_SHIP_TO_COME_TO_DEAD_STOP_FROM_FULL_SPEED) * DELTA_TIME + GRAIN) { // Honestly it's kinda sketch cutting off the respawn maneuver after 3-2=1 seconds have passed, but we can come up with a better way to do it later shall I wish to
                    // We don't really have time, so just come to a stop and call the respawn maneuver right there
                    assert(!is_kinda_close_to_zero(current_ship_speed) && "Uhh the ship shouldn't be stationary here!");
                    respawn_maneuver_without_crash = maneuver_sim.accelerate(0.0, rand_uniform(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE));
                } else {
                    // We have sufficient time left.
                    // Do a respawn maneuver, but keep to the constraint that the time it takes has to be at most the invincibility period we have left, so as to finish the maneuver while invincible
                    ship_accel_turn_rate = rand_uniform(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE);
                    if (start_of_respawn_maneuver) {
                        // Decide the direction to go since we're at the start of a respawn maneuver
                        if (random_double() < 0.5)
                            ship_cruise_speed = SHIP_MAX_SPEED;
                        else
                            ship_cruise_speed = -SHIP_MAX_SPEED;
                    } else {
                        // We're in the middle of a respawn maneuver, so just keep the same direction and don't try to go the other way!
                        ship_cruise_speed = SHIP_MAX_SPEED * sign(current_ship_speed);
                    }
                    //ship_cruise_turn_rate = 0.0;
                    random_ship_heading_angle = (respawn_maneuver_at_max_speed) ? 0.0 : rand_uniform(-20.0, 20.0);
                    double turning_time = (respawn_maneuver_at_max_speed) ? 0.0 : std::ceil(std::abs(random_ship_heading_angle) / (SHIP_MAX_TURN_RATE * DELTA_TIME));
                    assert(sign(ship_cruise_speed) == sign(current_ship_speed) || is_close_to_zero(current_ship_speed));
                    assert(std::abs(ship_cruise_speed) >= std::abs(current_ship_speed));
                    double acceleration_time = (respawn_maneuver_at_max_speed) ? 0.0 : std::ceil(std::abs(ship_cruise_speed - current_ship_speed) / (SHIP_MAX_THRUST - SHIP_DRAG) * FPS);
                    ship_cruise_timesteps = randint(0, int64_t(round(MAX_CRUISE_SECONDS * FPS)));
                    double deceleration_time = std::ceil(std::abs(ship_cruise_speed) / (SHIP_DRAG + SHIP_MAX_THRUST) * FPS);
                    double timesteps_this_respawn_maneuver_would_take = turning_time + acceleration_time + static_cast<double>(ship_cruise_timesteps) + deceleration_time;
                    int64_t rejection_sample_count = 0;
                    while (timesteps_this_respawn_maneuver_would_take >= timesteps_we_have_for_respawn_maneuver) {
                        ++rejection_sample_count;
                        random_ship_heading_angle = (respawn_maneuver_at_max_speed) ? 0.0 : rand_uniform(-20.0, 20.0);
                        turning_time = (respawn_maneuver_at_max_speed) ? 0.0 : std::ceil(std::abs(random_ship_heading_angle) / (SHIP_MAX_TURN_RATE * DELTA_TIME));
                        if (start_of_respawn_maneuver) {
                            // Decide the direction to go since we're at the start of a respawn maneuver
                            if (random_double() < 0.5)
                                ship_cruise_speed = SHIP_MAX_SPEED;
                            else
                                ship_cruise_speed = -SHIP_MAX_SPEED;
                        } // Else we already have this set correctly
                        acceleration_time = (respawn_maneuver_at_max_speed) ? 0.0 : std::ceil(std::abs(ship_cruise_speed - current_ship_speed) / (SHIP_MAX_THRUST - SHIP_DRAG) * FPS);
                        ship_cruise_timesteps = randint(0, int64_t(round(MAX_CRUISE_SECONDS * FPS)));
                        deceleration_time = std::ceil(std::abs(ship_cruise_speed) / (SHIP_DRAG + SHIP_MAX_THRUST) * FPS);
                        timesteps_this_respawn_maneuver_would_take = turning_time + acceleration_time + static_cast<double>(ship_cruise_timesteps) + deceleration_time;
                        assert(sign(ship_cruise_speed) == sign(current_ship_speed) || is_close_to_zero(current_ship_speed));
                        assert(std::abs(ship_cruise_speed) >= std::abs(current_ship_speed));
                        if (rejection_sample_count > 1000) {
                            std::cout << "Rejections: " << rejection_sample_count << ", Time we have: " << timesteps_we_have_for_respawn_maneuver << std::endl;
                            std::cout << "Expected total time: " << timesteps_this_respawn_maneuver_would_take << ", Turning time: " << turning_time << ", Accel time: " << acceleration_time << ", Cruise time: " << ship_cruise_timesteps << ", Decel time: " << deceleration_time << std::endl;
                        }
                    }
                    if (rejection_sample_count > 20) {
                        //std::cout << "Rejection sample count: " << std::to_string(rejection_sample_count) << std::endl;
                    }
                    if constexpr (ENABLE_SANITY_CHECKS) {
                        auto move_seq_preview = get_ship_maneuver_move_sequence(
                            random_ship_heading_angle, ship_cruise_speed, ship_accel_turn_rate,
                            ship_cruise_timesteps, ship_cruise_turn_rate, planning_state.ship_state.speed);
                        if (!(move_seq_preview.size() <= timesteps_we_have_for_respawn_maneuver)) {
                            std::cout << "Move seq length is " << move_seq_preview.size() << " and the ts we have for respawn is " << timesteps_we_have_for_respawn_maneuver << std::endl;
                            print_vector(move_seq_preview);
                            std::cout << "Expected total time: " << timesteps_this_respawn_maneuver_would_take << ", Turning time: " << turning_time << ", Accel time: " << acceleration_time << ", Cruise time: " << ship_cruise_timesteps << ", Decel time: " << deceleration_time << std::endl;
                            std::cout << "Current ship speed: " << current_ship_speed << ", Ship cruise speed: " << ship_cruise_speed << std::endl;
                        }
                        assert(move_seq_preview.size() <= timesteps_we_have_for_respawn_maneuver);
                    }

                    respawn_maneuver_without_crash = (respawn_maneuver_at_max_speed || ((maneuver_sim.rotate_heading(random_ship_heading_angle)) && (maneuver_sim.accelerate(ship_cruise_speed, ship_accel_turn_rate)))) && maneuver_sim.cruise(ship_cruise_timesteps, ship_cruise_turn_rate) && maneuver_sim.accelerate(0.0, 0.0);
                }

                assert(respawn_maneuver_without_crash && "The respawn maneuver somehow crashed. Maybe it's too long! The respawn timer was game_state_to_base_planning->ship_respawn_timer and the maneuver length was maneuver_sim.get_sequence_length()");

                double maneuver_fitness = maneuver_sim.get_fitness();

                this->sims_this_planning_period.push_back({
                    maneuver_sim,
                    maneuver_fitness,
                    maneuver_sim.get_fitness_breakdown(),
                    "respawn",
                    state_type,
                    std::make_tuple(random_ship_heading_angle, ship_cruise_speed, ship_accel_turn_rate, ship_cruise_timesteps, ship_cruise_turn_rate)
                });

                if (maneuver_fitness > this->best_fitness_this_planning_period) {
                    this->second_best_fitness_this_planning_period = this->best_fitness_this_planning_period;
                    this->second_best_fitness_this_planning_period_index = this->best_fitness_this_planning_period_index;

                    this->best_fitness_this_planning_period = maneuver_fitness;
                    this->best_fitness_this_planning_period_index = (int)this->sims_this_planning_period.size() - 1;
                }
            }
        } else {
            //std::cout << "Planning a regular maneuver" << std::endl;
            // --- Non-respawn move ---
            if (this->base_gamestate_analysis == std::nullopt) {
                debug_print("Analyzing heuristic maneuver");
                //std::cout << planning_state.game_state << std::endl;
                this->base_gamestate_analysis = analyze_gamestate_for_heuristic_maneuver(
                    planning_state.game_state,
                    planning_state.ship_state
                );
            }

            bool ship_is_stationary = is_close_to_zero(planning_state.ship_state.speed);

            if (plan_stationary && planning_state.ship_state.bullets_remaining != 0 && ship_is_stationary) {
                // No need to check whether this is allowed in our time/iterations budget, because we need to do this iteration at minimum
                // The first list element is the stationary targetting
                //this->performance_controller_start_iteration();
                //std::cout << "Making a stationary targ matrix starting on ts " << planning_state.timestep << std::endl;
                Matrix stationary_targetting_sim(
                    planning_state.game_state,
                    planning_state.ship_state,
                    planning_state.timestep,
                    planning_state.ship_respawn_timer,
                    &planning_state.asteroids_pending_death,
                    &planning_state.forecasted_asteroid_splits,
                    planning_state.last_timestep_fired,
                    planning_state.last_timestep_mined,
                    &planning_state.mine_positions_placed,
                    /* halt_shooting */ false,
                    /* fire_first_timestep */ planning_state.fire_next_timestep_flag,
                    /* verify_first_shot */ (this->sims_this_planning_period.size() == 0 && other_ships_exist),
                    /* verify_maneuver_shots */ false,
                    -1, // Last timestep colliding, dunno
                    0 // Respawn maneuver pass
                    //this->game_state_plotter
                );
                stationary_targetting_sim.target_selection();

                double best_stationary_targetting_fitness = stationary_targetting_sim.get_fitness();

                if (this->sims_this_planning_period.size() == 0) {
                    if (stationary_targetting_sim.get_cancel_firing_first_timestep()) {
                        // The plan was to fire at the first timestep this planning period. However, due to non-determinism caused by the existence of another ship, this shot would actually miss. We checked and caught this, so we're going to just nix the idea of shooting on the first timestep.
                        assert(planning_state.fire_next_timestep_flag);
                        planning_state.fire_next_timestep_flag = false;
                    }
                }

                this->sims_this_planning_period.emplace_back(
                    CompletedSimulation{
                        stationary_targetting_sim, // Sim
                        best_stationary_targetting_fitness, // Fitness
                        stationary_targetting_sim.get_fitness_breakdown(), // Fitness breakdown
                        "targetting", // Action type
                        state_type, // State type
                        std::make_tuple(0.0, 0.0, 0.0, 0, 0.0)  // Maneuver tuple TODO: Check that this makes sense!
                    }
                );

                this->stationary_targetting_sim_index = (int)this->sims_this_planning_period.size() - 1;

                if (best_stationary_targetting_fitness > this->best_fitness_this_planning_period) {
                    this->second_best_fitness_this_planning_period = this->best_fitness_this_planning_period;
                    this->second_best_fitness_this_planning_period_index = this->best_fitness_this_planning_period_index;

                    this->best_fitness_this_planning_period = best_stationary_targetting_fitness;
                    this->best_fitness_this_planning_period_index = this->stationary_targetting_sim_index;
                }
            }

            if constexpr (ENABLE_SANITY_CHECKS) {
                if (plan_stationary && !ship_is_stationary) {
                printf("\nWARNING: The ship wasn't stationary after the last maneuver, so we're skipping stationary targeting! Our planning period starts on ts %d\n", (int)planning_state.timestep);
                }
            }

            bool heuristic_maneuver;
            // Try moving! Run a simulation and find a course of action to put me to safety
            if ((this->sims_this_planning_period.size() == 0
                || (this->sims_this_planning_period.size() == 1
                    && this->sims_this_planning_period.at(0).action_type != "heuristic_maneuver"))
                && ship_is_stationary)
            {
                heuristic_maneuver = USE_HEURISTIC_MANEUVER;
            }
            else {
                heuristic_maneuver = false;
            }

            // Unpack tuple
            double imminent_asteroid_speed,
                imminent_asteroid_relative_heading,
                largest_gap_relative_heading,
                nearby_asteroid_average_speed,
                average_directional_speed;

            // Temporaries for int64_t values
            int64_t nearby_asteroid_count,
                    total_asteroids_count,
                    current_asteroids_count;

            std::tie(imminent_asteroid_speed, imminent_asteroid_relative_heading, largest_gap_relative_heading,
                    nearby_asteroid_average_speed, nearby_asteroid_count,
                    average_directional_speed, total_asteroids_count, current_asteroids_count) =
                    *this->base_gamestate_analysis; // assuming this is std::optional<std::tuple...>

            double ship_cruise_speed_mode, ship_cruise_timesteps_mode, max_pre_maneuver_turn_timesteps;
            // Let's just pretend the following is a fuzzy system lol
            // For performance and simplicity, I'll just use a bunch of if statements
            if (average_directional_speed > 80.0 && current_asteroids_count > 5 && total_asteroids_count >= 100) {
                print_explanation("Wall scenario detected! Preferring trying longer cruise lengths", this->current_timestep);
                ship_cruise_speed_mode = SHIP_MAX_SPEED;
                ship_cruise_timesteps_mode = MAX_CRUISE_TIMESTEPS;
                max_pre_maneuver_turn_timesteps = 6.0;
            } else if (std::any_of(
                            planning_state.game_state.mines.begin(),
                            planning_state.game_state.mines.end(),
                            [this, &planning_state](const Mine& m) {
                                return planning_state.mine_positions_placed.count({m.x, m.y}) > 0; })
                    )
            {
                print_explanation("We're probably within the radius of a mine we placed! Biasing faster/longer moves to be more likely to escape the mine.",
                    this->current_timestep);
                ship_cruise_speed_mode = SHIP_MAX_SPEED;
                ship_cruise_timesteps_mode = MAX_CRUISE_TIMESTEPS * 0.75;
                max_pre_maneuver_turn_timesteps = 10.0;
            } else {
                max_pre_maneuver_turn_timesteps = 15.0;
                ship_cruise_speed_mode = weighted_average(abs_cruise_speeds);
                ship_cruise_timesteps_mode = weighted_average(cruise_timesteps_global_history);
            }

            int search_iterations_count = 0;
            while (
                (search_iterations_count < get_min_maneuver_per_timestep_search_iterations(
                    planning_state.ship_state.lives_remaining, weighted_average(overall_fitness_record))
                || true)
                //this->performance_controller_check_whether_i_can_do_another_iteration())
                && search_iterations_count < MAX_MANEUVER_PER_TIMESTEP_SEARCH_ITERATIONS)
            {
                //this->performance_controller_start_iteration();
                search_iterations_count++;

                double random_ship_heading_angle, ship_accel_turn_rate, ship_cruise_speed, ship_cruise_turn_rate;
                int64_t ship_cruise_timesteps;
                /*
                double thrust_direction = 0.0;
                if (USE_HEURISTIC_MANEUVER && heuristic_maneuver) {
                    random_ship_heading_angle = 0.0;
                    double ship_cruise_timesteps_float;
                    std::tie(ship_accel_turn_rate, ship_cruise_speed, ship_cruise_turn_rate, ship_cruise_timesteps_float, thrust_direction) =
                        maneuver_heuristic_fis(imminent_asteroid_speed, imminent_asteroid_relative_heading,
                                largest_gap_relative_heading, nearby_asteroid_average_speed, nearby_asteroid_count);
                    ship_cruise_timesteps = int(round(ship_cruise_timesteps_float));
                    if (thrust_direction < -GRAIN)
                        ship_cruise_speed = -ship_cruise_speed;
                    else if (fabs(thrust_direction) < GRAIN)
                        heuristic_maneuver = false;
                }*/
                if (!heuristic_maneuver || !USE_HEURISTIC_MANEUVER) {
                    random_ship_heading_angle = rand_triangular(-DEGREES_TURNED_PER_TIMESTEP * max_pre_maneuver_turn_timesteps,
                                                            DEGREES_TURNED_PER_TIMESTEP * max_pre_maneuver_turn_timesteps, 0);
                    ship_accel_turn_rate = rand_triangular(0, SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE)
                                        * (2.0 * double(rand() %2) - 1.0);

                    if (std::isnan(ship_cruise_speed_mode))
                        ship_cruise_speed = rand_uniform(-SHIP_MAX_SPEED, SHIP_MAX_SPEED);
                    else
                        ship_cruise_speed = rand_triangular(0, SHIP_MAX_SPEED, ship_cruise_speed_mode)
                                            * (2.0 * double(rand()%2) - 1.0);
                    ship_cruise_turn_rate = rand_triangular(0, SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE)
                                            * (2.0*double(rand()%2)-1.0);
                    if (std::isnan(ship_cruise_timesteps_mode))
                        ship_cruise_timesteps = randint(0, int64_t(round(MAX_CRUISE_TIMESTEPS)));
                    else
                        ship_cruise_timesteps = int64_t(floor(rand_triangular(0.0, MAX_CRUISE_TIMESTEPS, ship_cruise_timesteps_mode)));
                }

                auto preview_move_sequence = get_ship_maneuver_move_sequence(
                    random_ship_heading_angle, ship_cruise_speed, ship_accel_turn_rate,
                    ship_cruise_timesteps, ship_cruise_turn_rate, planning_state.ship_state.speed
                );
                //std::cout << "Making a matrix starting on ts " << planning_state.timestep << std::endl;
                Matrix maneuver_sim(
                    planning_state.game_state,
                    planning_state.ship_state,
                    planning_state.timestep,
                    planning_state.ship_respawn_timer,
                    &planning_state.asteroids_pending_death,
                    &planning_state.forecasted_asteroid_splits,
                    planning_state.last_timestep_fired,
                    planning_state.last_timestep_mined,
                    &planning_state.mine_positions_placed,
                    /* halt_shooting */ false,
                    /* fire_first_timestep */ planning_state.fire_next_timestep_flag,
                    /* verify_first_shot */ (this->sims_this_planning_period.size() == 0 && other_ships_exist),
                    /* verify_maneuver_shots */ false,
                    -1, // Last timestep colliding, dunno
                    0 // Respawn maneuver pass 0, AKA not doing a respawn maneuver!
                    //this->game_state_plotter
                );

                maneuver_sim.simulate_maneuver(preview_move_sequence, true, true);

                double maneuver_fitness = maneuver_sim.get_fitness();
                std::array<double, 9> maneuver_fitness_breakdown = maneuver_sim.get_fitness_breakdown();

                if ((int)this->sims_this_planning_period.size() == 0) {
                    if (maneuver_sim.get_cancel_firing_first_timestep()) {
                        assert(planning_state.fire_next_timestep_flag);
                        planning_state.fire_next_timestep_flag = false;
                    }
                }
                this->sims_this_planning_period.emplace_back(
                    maneuver_sim,
                    maneuver_fitness,
                    maneuver_fitness_breakdown,
                    heuristic_maneuver ? "heuristic_maneuver" : "random_maneuver",
                    state_type,
                    std::make_tuple(
                        random_ship_heading_angle,
                        ship_cruise_speed,
                        ship_accel_turn_rate,
                        ship_cruise_timesteps,
                        ship_cruise_turn_rate
                    )
                );


                if (maneuver_fitness > this->best_fitness_this_planning_period) {
                    this->second_best_fitness_this_planning_period = this->best_fitness_this_planning_period;
                    this->second_best_fitness_this_planning_period_index = this->best_fitness_this_planning_period_index;

                    this->best_fitness_this_planning_period = maneuver_fitness;
                    this->best_fitness_this_planning_period_index = (int)this->sims_this_planning_period.size() - 1;
                }

                if (heuristic_maneuver)
                    heuristic_maneuver = false;
            }
        }
    }

    std::tuple<double, double, bool, bool>
    actions(const py::dict& ship_state_dict, const py::dict& game_state_dict)
    {
        // Optionally reseed RNG if flag enabled
        if (RESEED_RNG) {
            std::srand(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
        }
        if (current_timestep == -1) {
            reseed_rng(0);
        }

        ++this->current_timestep;
        bool recovering_from_crash = false;
        //std::cout << "Calling actions on timestep " << this->current_timestep << std::endl;
        Ship ship_state = create_ship_from_dict(ship_state_dict);
        GameState game_state = create_game_state_from_dict(game_state_dict);

        // Check for simulator/controller desync and perform state recovery/reset as in Python
        if constexpr (CLEAN_UP_STATE_FOR_SUBSEQUENT_SCENARIO_RUNS || STATE_CONSISTENCY_CHECK_AND_RECOVERY) {
            bool timestep_mismatch = !(game_state.sim_frame == this->current_timestep);
            bool action_queue_desync = !action_queue.empty() && std::get<0>(action_queue.front()) != this->current_timestep;
            bool planning_base_state_outdated = (game_state_to_base_planning.has_value() && game_state_to_base_planning->timestep < this->current_timestep);

            if (timestep_mismatch || (STATE_CONSISTENCY_CHECK_AND_RECOVERY &&
                (action_queue_desync || (planning_base_state_outdated && !CONTINUOUS_LOOKAHEAD_PLANNING))))
            {
                if (timestep_mismatch && !(action_queue_desync || (planning_base_state_outdated && !CONTINUOUS_LOOKAHEAD_PLANNING))) {
                    debug_print("This was not a fresh run of the controller! I'll try cleaning up the previous run and reset the state.");
                } else if (timestep_mismatch) {
                    debug_print("Neo didn't start from time 0. Was there a controller exception? Setting timestep to match the passed-in game state's nonzero starting timestep of: " + std::to_string(game_state.sim_frame));
                }
                this->reset();
                ++this->current_timestep;
                if (STATE_CONSISTENCY_CHECK_AND_RECOVERY &&
                    (action_queue_desync || (planning_base_state_outdated && !CONTINUOUS_LOOKAHEAD_PLANNING))) {
                    debug_print("Neo probably crashed or something because the internal state is all messed up. Welp, let's try this again.");
                    recovering_from_crash = true;
                }
                if (timestep_mismatch)
                    this->current_timestep = game_state.sim_frame;
            }
        }

        if (this->current_timestep == 0) {
            inspect_scenario(game_state, ship_state);
        }

        if (!this->init_done) {
            this->finish_init(game_state, ship_state);
            this->init_done = true;
        }
        //this->performance_controller_enter();

        bool iterations_boost = false;
        if (this->current_timestep == 0) iterations_boost = true;

        // ------------------------------- PLANNING LOGIC ------------------------

        if (CONTINUOUS_LOOKAHEAD_PLANNING) {
            // == CONTINUOUS PLANNING ==
            if (this->other_ships_exist) {
                // == Other ships exist (non-deterministic mode)!
                // We cannot use deterministic mode to plan ahead
                // We can still try to plan ahead, but we need to compare the predicted state with the actual state
                // Note that if the other ship dies, then we will switch from this case to the case where other ships don't exist

                // Since other ships exist right now and the game isn't deterministic, we can die at any time even during the middle of a planned maneuver where we SHOULD survive.
                // Or maybe we planned to die at the end of the maneuver, but we died in the middle instead. That's a sneaky case that's possible too. Handle all of these!
                // Check for that case:
                bool unexpected_death = false;
                // If we're dead/respawning but we didn't plan a respawn maneuver for it, OR if we do expect to die at the end of the maneuver, however we actually died mid-maneuver
                // Originally I thought it'd be a necessary condition to check (not self.last_timestep_ship_is_respawning and ship_state.is_respawning and ship_state.lives_remaining not in self.lives_remaining_that_we_did_respawn_maneuver_for) however WE DO NOT want to check that the last timestep we weren't respawning!
                // Because a sneaky edge case is, what if we did a respawn maneuver, and then we began to shoot in the middle of the respawn maneuver RIGHT AS the other ship is inside of us? Then we stay in the respawning state without ever getting out of it, but we just lose a life. Losing a life is the main thing we need to check for! And yes, this is an edge case I experienced and spent an hour tracking down.
                if ((ship_state.is_respawning &&
                    this->lives_remaining_that_we_did_respawn_maneuver_for.count(ship_state.lives_remaining) == 0) ||
                    (!this->action_queue.empty() && !this->last_timestep_ship_is_respawning && ship_state.is_respawning &&
                    this->lives_remaining_that_we_did_respawn_maneuver_for.count(ship_state.lives_remaining)))
                {
                    print_explanation("Ouch, I died in the middle of a maneuver where I expected to survive, due to other ships being present!", this->current_timestep);
                    // Clear the move queue, since previous moves have been invalidated by us taking damage
                    std::cerr << "CLEAARING ACTION QUEUE\n";
                    this->action_queue.clear();
                    this->actioned_timesteps.clear(); // If we don't clear it, we'll have duplicated moves since we have to overwrite our planned moves to get to safety, which means enqueuing moves on timesteps we already enqueued moves for.
                    this->fire_next_timestep_flag = false; // If we were planning on shooting this timestep but we unexpectedly got hit, DO NOT SHOOT! Actually even if we didn't reset this variable here, we'd only shoot after the respawn maneuver is done and then we'd miss a shot. And yes that was a bug that I fixed lmao
                    this->sims_this_planning_period.clear();
                    this->best_fitness_this_planning_period_index = INT_NEG_INF;
                    this->best_fitness_this_planning_period = -inf;
                    this->second_best_fitness_this_planning_period_index = INT_NEG_INF;
                    this->second_best_fitness_this_planning_period = -inf;
                    this->base_gamestate_analysis = std::nullopt;
                    unexpected_death = true;
                    iterations_boost = true;
                    if (this->lives_remaining_that_we_did_respawn_maneuver_for.count(ship_state.lives_remaining)) {
                        // We expected to die at the end of the maneuver, however we actually died mid-maneuver, so we have to revoke the respawn maneuver we had planned, and plan a new one.
                        // Removing the life remaining number from this set will allow us to plan a new maneuver for this number of lives remaining
                        this->lives_remaining_that_we_did_respawn_maneuver_for.erase(ship_state.lives_remaining);
                    }
                }
                bool unexpected_survival = false;
                // If we're alive at the end of a maneuver but we're expecting to be dead at the end of the maneuver and we've planned a respawn maneuver
                if (this->action_queue.empty() && game_state_to_base_planning.has_value() && !ship_state.is_respawning && game_state_to_base_planning->ship_state.is_respawning && game_state_to_base_planning->respawning) {
                    // We thought this maneuver would end in us dying, with the next move being a respawn maneuver. However this is not the case. We're alive at the end of the maneuver! This must be because the other ship saved us by shooting an asteroid that was going to hit us, or something.
                    // This assertion isn't true because we could be doing a respawn maneuver, dying, and doing another respawn maneuver!
                    print_explanation("\nI thought I would die, but the other ship saved me!!!", this->current_timestep);
                    // Clear the move queue, since previous moves have been invalidated by us taking damage
                    std::cerr << "CLEAARING ACTION QUEUE\n";
                    this->action_queue.clear();
                    this->actioned_timesteps.clear(); // If we don't clear it, we'll have duplicated moves since we have to overwrite our planned moves to get to safety, which means enqueuing moves on timesteps we already enqueued moves for.
                    this->fire_next_timestep_flag = false; // This should be false anyway!
                    this->sims_this_planning_period.clear();
                    this->best_fitness_this_planning_period_index = INT_NEG_INF;
                    this->best_fitness_this_planning_period = -inf;
                    this->second_best_fitness_this_planning_period_index = INT_NEG_INF;
                    this->second_best_fitness_this_planning_period = -inf;
                    this->base_gamestate_analysis = std::nullopt;
                    iterations_boost = true;
                    unexpected_survival = true;
                    // Yoink this life remaining from the respawn maneuvers, since we no longer are doing one
                    if (this->lives_remaining_that_we_did_respawn_maneuver_for.count(ship_state.lives_remaining - 1)) {
                        // We need to subtract one from the lives remaining, because when we added it, it was from a simulated ship that had one fewer life. In reality we never lost that life, so we subtract one from our actual lives.
                        this->lives_remaining_that_we_did_respawn_maneuver_for.erase(ship_state.lives_remaining - 1);
                    }
                }
                // Set up the actions planning
                if (unexpected_death) {
                    // We need to refresh the state if we died unexpectedly
                    print_explanation("Ouch! Due to the other ship, I unexpectedly died!", this->current_timestep);
                    //game_state_to_base_planning = create_base_planning_state(this->current_timestep, ship_state, game_state, 3.0, recovering_from_crash);
                    this->game_state_to_base_planning = {
                        this->current_timestep,
                        // respawning: ship_state.is_respawning AND ship_state.lives_remaining not in lives_remaining_that_we_did_respawn_maneuver_for
                        ship_state.is_respawning && (std::find(
                            lives_remaining_that_we_did_respawn_maneuver_for.begin(),
                            lives_remaining_that_we_did_respawn_maneuver_for.end(),
                            ship_state.lives_remaining) == lives_remaining_that_we_did_respawn_maneuver_for.end()),
                        ship_state,
                        game_state,
                        3.0, // ship_respawn_timer
                        // asteroids_pending_death: empty if not found, else value for this timestep
                        asteroids_pending_death_schedule.count(current_timestep) ? asteroids_pending_death_schedule[current_timestep] : std::unordered_map<int64_t, std::vector<Asteroid>>{},
                        // forecasted_asteroid_splits: empty if not found, else value for this timestep
                        forecasted_asteroid_splits_schedule.count(current_timestep) ? forecasted_asteroid_splits_schedule[current_timestep] : std::vector<Asteroid>{},
                        // last_timestep_fired: as per Python logic
                        recovering_from_crash ? (current_timestep-1) :
                            (current_timestep==0 ?
                                INT_NEG_INF :
                                (last_timestep_fired_schedule.count(current_timestep) ? last_timestep_fired_schedule[current_timestep] : INT_NEG_INF)
                            ),
                        // last_timestep_mined: as per Python logic
                        recovering_from_crash ? (current_timestep-1) :
                            (current_timestep==0 ?
                                INT_NEG_INF :
                                (last_timestep_mined_schedule.count(current_timestep) ? last_timestep_mined_schedule[current_timestep] : INT_NEG_INF)
                            ),
                        mine_positions_placed_schedule.count(current_timestep) ? mine_positions_placed_schedule[current_timestep] : std::set<std::pair<double, double>>{},
                        fire_next_timestep_schedule.count(current_timestep) > 0
                    };
                    if (game_state_to_base_planning->respawning)
                        this->lives_remaining_that_we_did_respawn_maneuver_for.insert(ship_state.lives_remaining);
                } else if (unexpected_survival) {
                    // We need to refresh the state if we survived unexpectedly. Technically if we still had the remainder of the maneuver from before we could use that, but it's easier to just make a new maneuver from this starting point.
                    debug_print("Unexpected survival, the ship state is "+ship_state.str());
                    //game_state_to_base_planning = create_base_planning_state(this->current_timestep, ship_state, game_state, 0.0, false);
                    this->game_state_to_base_planning = {
                        this->current_timestep,
                        // respawning: ship_state.is_respawning AND ship_state.lives_remaining not in lives_remaining_that_we_did_respawn_maneuver_for
                        ship_state.is_respawning && (std::find(
                            lives_remaining_that_we_did_respawn_maneuver_for.begin(),
                            lives_remaining_that_we_did_respawn_maneuver_for.end(),
                            ship_state.lives_remaining) == lives_remaining_that_we_did_respawn_maneuver_for.end()),
                        ship_state,
                        game_state,
                        0.0, // ship_respawn_timer
                        // asteroids_pending_death: empty if not found, else value for this timestep
                        asteroids_pending_death_schedule.count(current_timestep) ? asteroids_pending_death_schedule[current_timestep] : std::unordered_map<int64_t, std::vector<Asteroid>>{},
                        // forecasted_asteroid_splits: empty if not found, else value for this timestep
                        forecasted_asteroid_splits_schedule.count(current_timestep) ? forecasted_asteroid_splits_schedule[current_timestep] : std::vector<Asteroid>{},
                        // last_timestep_fired: as per Python logic
                        recovering_from_crash ? (current_timestep-1) :
                            (current_timestep==0 ?
                                INT_NEG_INF :
                                (last_timestep_fired_schedule.count(current_timestep) ? last_timestep_fired_schedule[current_timestep] : INT_NEG_INF)
                            ),
                        // last_timestep_mined: as per Python logic
                        recovering_from_crash ? (current_timestep-1) :
                            (current_timestep==0 ?
                                INT_NEG_INF :
                                (last_timestep_mined_schedule.count(current_timestep) ? last_timestep_mined_schedule[current_timestep] : INT_NEG_INF)
                            ),
                        mine_positions_placed_schedule.count(current_timestep) ? mine_positions_placed_schedule[current_timestep] : std::set<std::pair<double, double>>{},
                        fire_next_timestep_schedule.count(current_timestep) > 0
                    };
                } else if (!game_state_to_base_planning.has_value()) {
                    this->game_state_to_base_planning = {
                        this->current_timestep,
                        // respawning
                        ship_state.is_respawning && (
                            std::find(
                                lives_remaining_that_we_did_respawn_maneuver_for.begin(),
                                lives_remaining_that_we_did_respawn_maneuver_for.end(),
                                ship_state.lives_remaining
                            ) == lives_remaining_that_we_did_respawn_maneuver_for.end()
                        ),
                        ship_state,
                        game_state,
                        // ship_respawn_timer: 0.0 if timestep == 0 else respawn_timer_history[cur_timestep]
                        this->current_timestep == 0 ? 0.0 :
                            (respawn_timer_history.count(this->current_timestep) ?
                                respawn_timer_history[this->current_timestep] : 3.0),  // fallback to 3.0 if key missing (optional)

                        // asteroids_pending_death
                        asteroids_pending_death_schedule.count(this->current_timestep) ?
                            asteroids_pending_death_schedule[this->current_timestep] :
                            std::unordered_map<int64_t, std::vector<Asteroid>>{},

                        // forecasted_asteroid_splits
                        forecasted_asteroid_splits_schedule.count(this->current_timestep) ?
                            forecasted_asteroid_splits_schedule[this->current_timestep] :
                            std::vector<Asteroid>{},

                        // last_timestep_fired
                        recovering_from_crash ? (this->current_timestep - 1) :
                            (this->current_timestep == 0 ? INT_NEG_INF :
                                (last_timestep_fired_schedule.count(this->current_timestep) ?
                                    last_timestep_fired_schedule[this->current_timestep] : INT_NEG_INF
                                )
                            ),

                        // last_timestep_mined
                        recovering_from_crash ? (this->current_timestep - 1) :
                            (this->current_timestep == 0 ? INT_NEG_INF :
                                (last_timestep_mined_schedule.count(this->current_timestep) ?
                                    last_timestep_mined_schedule[this->current_timestep] : INT_NEG_INF
                                )
                            ),

                        // mine_positions_placed
                        mine_positions_placed_schedule.count(this->current_timestep) ?
                            mine_positions_placed_schedule[this->current_timestep] :
                            std::set<std::pair<double, double>>{},

                        // fire_next_timestep_flag
                        fire_next_timestep_schedule.count(this->current_timestep) > 0
                    };
                    if (game_state_to_base_planning->respawning)
                        this->lives_remaining_that_we_did_respawn_maneuver_for.insert(ship_state.lives_remaining);
                    assert((bool)game_state_to_base_planning->ship_respawn_timer == game_state_to_base_planning->ship_state.is_respawning);
                } else {
                    // Refresh the state anyway to the latest state:
                    this->game_state_to_base_planning = {
                        this->current_timestep,
                        // respawning
                        ship_state.is_respawning && (
                            std::find(
                                lives_remaining_that_we_did_respawn_maneuver_for.begin(),
                                lives_remaining_that_we_did_respawn_maneuver_for.end(),
                                ship_state.lives_remaining
                            ) == lives_remaining_that_we_did_respawn_maneuver_for.end()
                        ),
                        ship_state,
                        game_state,
                        // ship_respawn_timer: 0.0 if timestep == 0 else respawn_timer_history[cur_timestep]
                        this->current_timestep == 0 ? 0.0 :
                            (respawn_timer_history.count(this->current_timestep) ?
                                respawn_timer_history[this->current_timestep] : 3.0),  // fallback to 3.0 if key missing (optional)

                        // asteroids_pending_death
                        asteroids_pending_death_schedule.count(this->current_timestep) ?
                            asteroids_pending_death_schedule[this->current_timestep] :
                            std::unordered_map<int64_t, std::vector<Asteroid>>{},

                        // forecasted_asteroid_splits
                        forecasted_asteroid_splits_schedule.count(this->current_timestep) ?
                            forecasted_asteroid_splits_schedule[this->current_timestep] :
                            std::vector<Asteroid>{},

                        // last_timestep_fired
                        recovering_from_crash ? (this->current_timestep - 1) :
                            (this->current_timestep == 0 ? INT_NEG_INF :
                                (last_timestep_fired_schedule.count(this->current_timestep) ?
                                    last_timestep_fired_schedule[this->current_timestep] : INT_NEG_INF
                                )
                            ),

                        // last_timestep_mined
                        recovering_from_crash ? (this->current_timestep - 1) :
                            (this->current_timestep == 0 ? INT_NEG_INF :
                                (last_timestep_mined_schedule.count(this->current_timestep) ?
                                    last_timestep_mined_schedule[this->current_timestep] : INT_NEG_INF
                                )
                            ),

                        // mine_positions_placed
                        mine_positions_placed_schedule.count(this->current_timestep) ?
                            mine_positions_placed_schedule[this->current_timestep] :
                            std::set<std::pair<double, double>>{},

                        // fire_next_timestep_flag
                        fire_next_timestep_schedule.count(this->current_timestep) > 0
                    };
                }

                if (action_queue.empty()) {
                    // Only when we're at the end of our sequence, do we run the stationary targeting sim once. Basically we just keep doing stationary targeting unless we have a better maneuver found
                    plan_action_continuous(false, true, iterations_boost, true);
                    plan_action_continuous(false, true, iterations_boost, false);
                    bool success = decide_next_action_continuous(game_state, ship_state, true);
                    assert(success);
                } else {
                    // We're still in the middle of a maneuver sequence. Run some planning iterations, and switch over to the new sequence if it's better than our fitness
                    plan_action_continuous(false, true, iterations_boost, false);
                    bool success = decide_next_action_continuous(game_state, ship_state, false);
                    debug_print(success ? "Switched to a better maneuver" : "Didn't find better maneuvers");
                }

                if (get_other_ships(game_state, this->ship_id_internal).empty()) {
                    print_explanation("I'm alone. I can see into the future perfectly now!", this->current_timestep);
                    this->simulated_gamestate_history.clear();
                    this->set_of_base_gamestate_timesteps.clear();
                    this->other_ships_exist = false;
                }
            } else {
                // == CONTINUOUS, NO OTHER SHIPS EXIST (fully deterministic) ==
                // No other ships exist, we're deterministically planning the future
                // Always set the latest state to the base state!
                // TODO: Use more accurate stuff for the carryover info!
                if (recovering_from_crash) {
                    std::cerr << "RECOVERING FROM A CRASH!!!\n";
                }
                debug_print("Asteroid scheds: " + std::to_string(asteroids_pending_death_schedule.size()) + "," +
                            std::to_string(forecasted_asteroid_splits_schedule.size()) + "," +
                            std::to_string(last_timestep_fired_schedule.size()) + "," +
                            std::to_string(last_timestep_mined_schedule.size()) + "," +
                            std::to_string(mine_positions_placed_schedule.size()));

                this->game_state_to_base_planning = {
                    this->current_timestep,
                    // respawning: just ship_state.is_respawning (no lives_remaining check in this version)
                    ship_state.is_respawning,
                    ship_state,
                    game_state,
                    // ship_respawn_timer
                    this->current_timestep == 0 ? 0.0 :
                        (respawn_timer_history.count(this->current_timestep) ?
                            respawn_timer_history[this->current_timestep] : 0.0),  // fallback to 0.0 if key missing (optional)

                    // asteroids_pending_death
                    asteroids_pending_death_schedule.count(this->current_timestep) ?
                        asteroids_pending_death_schedule[this->current_timestep] :
                        std::unordered_map<int64_t, std::vector<Asteroid>>{},

                    // forecasted_asteroid_splits
                    forecasted_asteroid_splits_schedule.count(this->current_timestep) ?
                        forecasted_asteroid_splits_schedule[this->current_timestep] :
                        std::vector<Asteroid>{},

                    // last_timestep_fired
                    recovering_from_crash ? (this->current_timestep - 1) :
                        (this->current_timestep == 0 ? INT_NEG_INF :
                            (last_timestep_fired_schedule.count(this->current_timestep) ?
                                last_timestep_fired_schedule[this->current_timestep] : INT_NEG_INF
                            )
                        ),

                    // last_timestep_mined
                    recovering_from_crash ? (this->current_timestep - 1) :
                        (this->current_timestep == 0 ? INT_NEG_INF :
                            (last_timestep_mined_schedule.count(this->current_timestep) ?
                                last_timestep_mined_schedule[this->current_timestep] : INT_NEG_INF
                            )
                        ),

                    // mine_positions_placed
                    mine_positions_placed_schedule.count(this->current_timestep) ?
                        mine_positions_placed_schedule[this->current_timestep] :
                        std::set<std::pair<double, double>>{},

                    // fire_next_timestep_flag
                    fire_next_timestep_schedule.count(this->current_timestep) > 0
                };
                if (action_queue.empty()) {
                    // Only when we're at the end of our sequence, do we run the stationary targeting sim once. Basically we just keep doing stationary targeting unless we have a better maneuver found
                    plan_action_continuous(false, true, iterations_boost, true);
                    plan_action_continuous(false, true, iterations_boost, false);
                    bool success = decide_next_action_continuous(game_state, ship_state, true);
                    assert(success);
                } else {
                    // We're still in the middle of a maneuver sequence. Run some planning iterations, and switch over to the new sequence if it's better than our fitness
                    plan_action_continuous(false, true, iterations_boost, false);
                    bool success = decide_next_action_continuous(game_state, ship_state, false);
                    debug_print(success ? "Switched to a better maneuver" : "Didn't find better maneuvers");
                }
            }
        }/* else { DEPRECATED:
            // == NON-CONTINUOUS LOOKAHEAD BLOCKS ==
            if (this->other_ships_exist) {
                // == NON-CONTINUOUS, other ships exist! ==
                bool unexpected_death = false;
                if ((ship_state.is_respawning &&
                    this->lives_remaining_that_we_did_respawn_maneuver_for.count(ship_state.lives_remaining) == 0) ||
                    (!this->action_queue.empty() && !this->last_timestep_ship_is_respawning && ship_state.is_respawning &&
                    this->lives_remaining_that_we_did_respawn_maneuver_for.count(ship_state.lives_remaining)))
                {
                    print_explanation("Ouch, I died in the middle of a maneuver where I expected to survive, due to other ships being present!", this->current_timestep);
                    std::cerr << "CLEAARING ACTION QUEUE\n";
                    this->action_queue.clear();
                    this->actioned_timesteps.clear();
                    this->fire_next_timestep_flag = false;
                    this->sims_this_planning_period.clear();
                    this->best_fitness_this_planning_period_index = INT_NEG_INF;
                    this->best_fitness_this_planning_period = -inf;
                    this->second_best_fitness_this_planning_period_index = INT_NEG_INF;
                    this->second_best_fitness_this_planning_period = -inf;
                    this->base_gamestate_analysis = std::nullopt;
                    unexpected_death = true;
                    iterations_boost = true;
                    if (this->lives_remaining_that_we_did_respawn_maneuver_for.count(ship_state.lives_remaining))
                        this->lives_remaining_that_we_did_respawn_maneuver_for.erase(ship_state.lives_remaining);
                }
                bool unexpected_survival = false;
                if (this->action_queue.empty() && game_state_to_base_planning.has_value() && !ship_state.is_respawning &&
                    game_state_to_base_planning->ship_state.is_respawning && game_state_to_base_planning->respawning)
                {
                    print_explanation("\nI thought I would die, but the other ship saved me!!!", this->current_timestep);
                    std::cerr << "CLEAARING ACTION QUEUE\n";
                    this->action_queue.clear();
                    this->actioned_timesteps.clear();
                    this->fire_next_timestep_flag = false;
                    this->sims_this_planning_period.clear();
                    this->best_fitness_this_planning_period_index = INT_NEG_INF;
                    this->best_fitness_this_planning_period = -inf;
                    this->second_best_fitness_this_planning_period_index = INT_NEG_INF;
                    this->second_best_fitness_this_planning_period = -inf;
                    this->base_gamestate_analysis = std::nullopt;
                    iterations_boost = true;
                    unexpected_survival = true;
                    if (this->lives_remaining_that_we_did_respawn_maneuver_for.count(ship_state.lives_remaining-1))
                        this->lives_remaining_that_we_did_respawn_maneuver_for.erase(ship_state.lives_remaining-1);
                }
                if (unexpected_death) {
                    print_explanation("Ouch! Due to the other ship, I unexpectedly died!", this->current_timestep);
                    if (!game_state_to_base_planning.has_value()) {
                        debug_print("WARNING: The game state to base planning was none. This better be because I'm recovering from a controller exception!");
                        this->game_state_to_base_planning = {
                            this->current_timestep,
                            recovering_from_crash, // respawning
                            ship_state,
                            game_state,
                            3.0,   // ship_respawn_timer
                            {},    // asteroids_pending_death
                            {},    // forecasted_asteroid_splits
                            -1,    // last_timestep_fired
                            -1,    // last_timestep_mined
                            {},    // mine_positions_placed
                            false  // fire_next_timestep_flag
                        };
                    }
                    this->game_state_to_base_planning = {
                        this->current_timestep,
                        recovering_from_crash, // respawning
                        ship_state,
                        game_state,
                        3.0,   // ship_respawn_timer
                        {},    // asteroids_pending_death
                        {},    // forecasted_asteroid_splits
                        -1,    // last_timestep_fired
                        -1,    // last_timestep_mined
                        {},    // mine_positions_placed
                        false  // fire_next_timestep_flag
                    };
                    if (game_state_to_base_planning->respawning)
                        this->lives_remaining_that_we_did_respawn_maneuver_for.insert(ship_state.lives_remaining);
                } else if (unexpected_survival) {
                    debug_print("Unexpected survival, the ship state is "+ship_state.str());
                    this->game_state_to_base_planning = {
                        this->current_timestep,
                        false,  // respawning
                        ship_state,
                        game_state,
                        0.0,    // ship_respawn_timer
                        {},     // asteroids_pending_death
                        {},     // forecasted_asteroid_splits
                        -1,     // last_timestep_fired
                        -1,     // last_timestep_mined
                        {},     // mine_positions_placed
                        false   // fire_next_timestep_flag
                    };
                } else if (!game_state_to_base_planning.has_value()) {
                    this->game_state_to_base_planning = {
                        this->current_timestep,
                        false,  // respawning
                        ship_state,
                        game_state,
                        0.0,    // ship_respawn_timer
                        {},     // asteroids_pending_death
                        {},     // forecasted_asteroid_splits
                        -1,     // last_timestep_fired
                        -1,     // last_timestep_mined
                        {},     // mine_positions_placed
                        false   // fire_next_timestep_flag
                    };
                    if (game_state_to_base_planning->respawning)
                        this->lives_remaining_that_we_did_respawn_maneuver_for.insert(ship_state.lives_remaining);
                    assert((bool)game_state_to_base_planning->ship_respawn_timer == game_state_to_base_planning->ship_state.is_respawning);
                }
                if (!this->action_queue.empty()) {
                    plan_action(this->other_ships_exist, false, iterations_boost, false);
                } else {
                    game_state_to_base_planning->ship_state = ship_state;
                    game_state_to_base_planning->game_state = game_state;
                    if (!game_state_to_base_planning->ship_state.is_respawning && bool(game_state_to_base_planning->ship_respawn_timer))
                        game_state_to_base_planning->ship_respawn_timer = 0.0;
                    assert((bool)game_state_to_base_planning->ship_respawn_timer == game_state_to_base_planning->ship_state.is_respawning);
                    plan_action(this->other_ships_exist, true, iterations_boost, true);
                    assert(this->best_fitness_this_planning_period_index != INT_NEG_INF);
                    while (this->sims_this_planning_period.size() < (get_planning_min_iterations())) {
                        plan_action(this->other_ships_exist, true, false, false);
                    }
                    assert(this->current_timestep == game_state_to_base_planning->timestep);
                    bool ok = decide_next_action(game_state, ship_state);
                    if (!ok) {
                        for (int j = 0; j < 60; ++j) {
                            plan_action(this->other_ships_exist, true, false, false);
                            if (this->second_best_fitness_this_planning_period > 0.93)
                                break;
                        }
                        bool success = decide_next_action(game_state, ship_state);
                        assert(success);
                    }
                }
                if (get_other_ships(game_state, this->ship_id_internal).empty()) {
                    print_explanation("I'm alone. I can see into the future perfectly now!", this->current_timestep);
                    this->simulated_gamestate_history.clear();
                    this->set_of_base_gamestate_timesteps.clear();
                    this->other_ships_exist = false;
                }
            } else {
                // == NON-CONTINUOUS, full deterministic plan ==
                if (!game_state_to_base_planning.has_value()) {
                    if (ENABLE_SANITY_CHECKS && !recovering_from_crash) {
                        std::cerr << "WARNING, Why is the game state to plan empty when we're not on timestep 0?! Maybe we're recovering from a controller exception\n";
                    }
                    if (this->current_timestep == 0 || recovering_from_crash)
                        iterations_boost = true;
                    game_state_to_base_planning = create_base_planning_state(this->current_timestep, ship_state, game_state, 0.0, recovering_from_crash);
                    if (recovering_from_crash)
                        print_explanation("Recovering from crash! Setting the base gamestate. The timestep is "+std::to_string(this->current_timestep), this->current_timestep);
                    assert((bool)game_state_to_base_planning->ship_respawn_timer == game_state_to_base_planning->ship_state.is_respawning);
                }
                if (this->sims_this_planning_period.empty()) {
                    plan_action(this->other_ships_exist, true, iterations_boost, true);
                } else {
                    plan_action(this->other_ships_exist, true, iterations_boost, false);
                }
                if (this->action_queue.empty()) {
                    assert(this->best_fitness_this_planning_period_index != INT_NEG_INF);
                    while (this->sims_this_planning_period.size() < (get_planning_min_iterations())) {
                        plan_action(this->other_ships_exist, true, false, false);
                    }
                    if (!(this->current_timestep == game_state_to_base_planning->timestep && !recovering_from_crash))
                        throw std::runtime_error("The actions queue is empty, however the base state's timestep doesn't match!");
                    bool ok = decide_next_action(game_state, ship_state);
                    assert(ok);
                }
            }
        }*/

        // -- EXECUTE PLANNED ACTION FOR THIS TIMESTEP --
        double thrust = 0.0, turn_rate = 0.0;
        bool fire = false, drop_mine = false;
        if (!action_queue.empty() && std::get<0>(action_queue.front()) == this->current_timestep) {
            auto [_, t, tr, f, m] = action_queue.front();
            thrust = t; turn_rate = tr; fire = f; drop_mine = m;
            action_queue.pop_front();
        } else {
            throw std::runtime_error("Sequence error on timestep "+std::to_string(this->current_timestep)+"!");
            thrust = 0.0; turn_rate = 0.0; fire = false; drop_mine = false;
        }

        if constexpr (ENABLE_SANITY_CHECKS) {
            if (thrust < -SHIP_MAX_THRUST || thrust > SHIP_MAX_THRUST) {
                thrust = std::clamp(thrust, -SHIP_MAX_THRUST, SHIP_MAX_THRUST);
                throw std::runtime_error("Dude the thrust is too high, go fix your code >:(");
            }
            if (turn_rate < -SHIP_MAX_TURN_RATE || turn_rate > SHIP_MAX_TURN_RATE) {
                turn_rate = std::clamp(turn_rate, -SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE);
                throw std::runtime_error("Dude the turn rate is too high, go fix your code >:(");
            }
            if (fire && !ship_state.can_fire) {
                throw std::runtime_error("Why are you trying to fire when you haven't waited out the cooldown yet?");
            }
        }

        // Optional sleep for visualization
        if (double(this->current_timestep) > SLOW_DOWN_GAME_AFTER_SECOND*FPS) {
            std::this_thread::sleep_for(std::chrono::duration<double>(SLOW_DOWN_GAME_PAUSE_TIME));
        }

        // Optional plotting, state validation, and so forth would go here, if ported

        this->last_timestep_ship_is_respawning = ship_state.is_respawning;

        //std::cout << "Thrust: " << thrust << ", Turn rate: " << turn_rate << ", Fire: " << fire << ", Drop mine: " << drop_mine << std::endl;
        return std::make_tuple(thrust, turn_rate, fire, drop_mine);
    }
};

PYBIND11_MODULE(neo_controller, m) {
    py::class_<NeoController>(m, "NeoController")
        .def(py::init<>())
        .def("actions", &NeoController::actions)
        .def_property_readonly("name", &NeoController::name)
        .def_property("ship_id", &NeoController::ship_id, &NeoController::set_ship_id)
        .def_property_readonly("custom_sprite_path", &NeoController::custom_sprite_path);
}
