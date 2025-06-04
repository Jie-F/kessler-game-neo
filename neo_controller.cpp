// neo_controller.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <cmath>
#include <vector>
#include <tuple>
#include <map>
#include <string>
#include <limits>
#include <algorithm>
#include <optional>
#include <random>
#include <chrono>
#include <deque>
#include <utility> // for std::pair
#include <iostream>

namespace py = pybind11;

using i64 = int64_t;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const double inf = std::numeric_limits<double>::infinity();
//const double nan = std::numeric_limits<double>::quiet_NaN();

// Build Info
constexpr const char* BUILD_NUMBER = "2025-06-04 Neo - Jie Fan (jie.f@pm.me)";

// Output Config
constexpr bool DEBUG_MODE = false;
constexpr bool PRINT_EXPLANATIONS = false;
constexpr double EXPLANATION_MESSAGE_SILENCE_INTERVAL_S = 2.0;

// Safety&Performance Flags
inline bool STATE_CONSISTENCY_CHECK_AND_RECOVERY = true;
inline bool CLEAN_UP_STATE_FOR_SUBSEQUENT_SCENARIO_RUNS = true;
constexpr bool ENABLE_SANITY_CHECKS = false;
constexpr bool PRUNE_SIM_STATE_SEQUENCE = true;
constexpr bool VALIDATE_SIMULATED_KEY_STATES = false;
constexpr bool VALIDATE_ALL_SIMULATED_STATES = false;
constexpr bool VERIFY_AST_TRACKING = false;
constexpr bool RESEED_RNG = false;

// Strategic/algorithm switches
constexpr bool CONTINUOUS_LOOKAHEAD_PLANNING = true;
constexpr bool USE_HEURISTIC_MANEUVER = false;
constexpr i64 END_OF_SCENARIO_DONT_CARE_TIMESTEPS = 8;
constexpr i64 ADVERSARY_ROTATION_TIMESTEP_FUDGE = 20;
constexpr double UNWRAP_ASTEROID_COLLISION_FORECAST_TIME_HORIZON = 8.0;
constexpr double UNWRAP_ASTEROID_TARGET_SELECTION_TIME_HORIZON = 2.3;

// Asteroid priorities
const std::vector<double> ASTEROID_SIZE_SHOT_PRIORITY = {std::numeric_limits<double>::quiet_NaN(), 1, 2, 3, 4};

// Optional weights for fitness function
std::optional<std::tuple<double, double, double, double, double, double, double, double, double>> fitness_function_weights = std::nullopt;

// Mine settings
constexpr i64 MINE_DROP_COOLDOWN_FUDGE_TS = 61;
constexpr double MINE_ASTEROID_COUNT_FUDGE_DISTANCE = 50.0;
constexpr i64 MINE_OPPORTUNITY_CHECK_INTERVAL_TS = 10;
constexpr double MINE_OTHER_SHIP_RADIUS_FUDGE = 40.0;
constexpr i64 MINE_OTHER_SHIP_ASTEROID_COUNT_EQUIVALENT = 10;
constexpr double TARGETING_AIMING_UNDERTURN_ALLOWANCE_DEG = 6.0;

// Fitness Weights (default)
const std::tuple<double,double,double,double,double,double,double,double,double> DEFAULT_FITNESS_WEIGHTS =
    {0.0, 0.13359801675028146, 0.1488417344765523, 0.0, 0.06974293843076491, 0.20559835937182916, 0.12775194210275548, 0.14357775694291458, 0.17088925192490204};

// Angle cone/culling parameters
const double MANEUVER_CONVENIENT_SHOT_CHECKER_CONE_WIDTH_ANGLE_HALF = 45.0;
const double MANEUVER_CONVENIENT_SHOT_CHECKER_CONE_WIDTH_ANGLE_HALF_COSINE = std::cos(MANEUVER_CONVENIENT_SHOT_CHECKER_CONE_WIDTH_ANGLE_HALF * M_PI / 180.0);

const double MANEUVER_BULLET_SIM_CULLING_CONE_WIDTH_ANGLE_HALF = 60.0;
const double MANEUVER_BULLET_SIM_CULLING_CONE_WIDTH_ANGLE_HALF_COSINE = std::cos(MANEUVER_BULLET_SIM_CULLING_CONE_WIDTH_ANGLE_HALF * M_PI / 180.0);

const double MAX_CRUISE_TIMESTEPS = 30.0;
constexpr i64 MANEUVER_TUPLE_LEARNING_ROLLING_AVG_PERIOD = 10;
constexpr i64 OVERALL_FITNESS_ROLLING_AVERAGE_PERIOD = 5;
const double AIMING_CONE_FITNESS_CONE_WIDTH_HALF = 18.0;
const double AIMING_CONE_FITNESS_CONE_WIDTH_HALF_COSINE = std::cos(AIMING_CONE_FITNESS_CONE_WIDTH_HALF * M_PI / 180.0);

constexpr i64 MANEUVER_SIM_DISALLOW_TARGETING_FOR_START_TIMESTEPS_AMOUNT = 10;
constexpr double ASTEROID_AIM_BUFFER_PIXELS = 1.0;
constexpr double COORDINATE_BOUND_CHECK_PADDING = 1.0;
constexpr int SHIP_AVOIDANCE_PADDING = 25;
constexpr double SHIP_AVOIDANCE_SPEED_PADDING_RATIO = 1.0/100.0;
constexpr i64 PERFORMANCE_CONTROLLER_ROLLING_AVERAGE_FRAME_INTERVAL = 10;
constexpr i64 RANDOM_WALK_SCHEDULE_LENGTH = 3;
constexpr double PERFORMANCE_CONTROLLER_PUSHING_THE_ENVELOPE_FUDGE_MULTIPLIER = 0.55;
constexpr double MINIMUM_DELTA_TIME_FRACTION_BUDGET = 0.55;
constexpr bool ENABLE_PERFORMANCE_CONTROLLER = false;

// Per-lives/per-fitness LUTs (represent as vector of vectors)
const std::vector<std::vector<i64>> MIN_RESPAWN_PER_TIMESTEP_SEARCH_ITERATIONS_LUT = {
    {80, 55, 14},
    {70, 40, 13},
    {60, 28, 12},
    {50, 26, 11},
    {45, 14, 10},
    {16, 12, 9},
    {15, 11, 8},
    {14, 10, 7},
    {13, 9, 6},
    {12, 8, 5}
};
const std::vector<std::vector<i64>> MIN_RESPAWN_PER_PERIOD_SEARCH_ITERATIONS_LUT = {
    {1000, 900, 440},
    {950, 810, 430},
    {925, 780, 420},
    {900, 730, 410},
    {850, 715, 400},
    {815, 680, 390},
    {790, 660, 380},
    {760, 640, 370},
    {730, 620, 360},
    {700, 600, 350}
};
const std::vector<std::vector<i64>> MIN_MANEUVER_PER_TIMESTEP_SEARCH_ITERATIONS_LUT = {
    {85, 65, 30},
    {65, 52, 25},
    {55, 40, 20},
    {45, 25, 15},
    {25, 12, 9},
    {20, 9, 6},
    {14, 7, 5},
    {8, 5, 4},
    {7, 4, 3},
    {6, 3, 2}
};
const std::vector<std::vector<i64>> MIN_MANEUVER_PER_PERIOD_SEARCH_ITERATIONS_LUT = {
    {300, 230, 105},
    {230, 185, 88},
    {193, 140, 70},
    {160, 88, 55},
    {88, 42, 32},
    {56, 28, 21},
    {35, 25, 18},
    {25, 18, 14},
    {14, 11, 10},
    {11, 11, 7}
};
const std::vector<std::vector<i64>> MIN_MANEUVER_PER_PERIOD_SEARCH_ITERATIONS_IF_WILL_DIE_LUT = {
    {860, 680, 340},
    {830, 660, 330},
    {800, 640, 320},
    {770, 620, 310},
    {740, 600, 300},
    {710, 580, 290},
    {690, 560, 280},
    {660, 540, 270},
    {630, 520, 260},
    {600, 500, 250}
};

// State dumping for debug
constexpr bool PLOT_MANEUVER_TRACES = false;
constexpr i64 PLOT_MANEUVER_MIN_TRACE_FOR_PLOT = 30;
constexpr bool REALITY_STATE_DUMP = false;
constexpr bool SIMULATION_STATE_DUMP = false;
constexpr bool KEY_STATE_DUMP = false;
inline bool GAMESTATE_PLOTTING = false;
constexpr bool BULLET_SIM_PLOTTING = false;
inline bool NEXT_TARGET_PLOTTING = false;
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
constexpr i64 INT_NEG_INF = -1000000;
constexpr i64 INT_INF = 1000000;
const double RAD_TO_DEG = 180.0/M_PI;
const double DEG_TO_RAD = M_PI/180.0;
const double TAU = 2.0*M_PI;

// Kessler game constants
constexpr i64 FIRE_COOLDOWN_TS = 3;
constexpr i64 MINE_COOLDOWN_TS = 30;
const double FPS = 30.0;
const double DELTA_TIME = 1.0/FPS;
const double SHIP_FIRE_TIME = 1.0/10.0; // seconds
const double BULLET_SPEED = 800.0;
constexpr double BULLET_MASS = 1.0;
const double BULLET_LENGTH = 12.0;
const double BULLET_LENGTH_RECIPROCAL = 1.0/BULLET_LENGTH;
const double TWICE_BULLET_LENGTH_RECIPROCAL = 2.0/BULLET_LENGTH;
const double SHIP_MAX_TURN_RATE = 180.0;
const double SHIP_MAX_TURN_RATE_RAD = DEG_TO_RAD*SHIP_MAX_TURN_RATE;
const double SHIP_MAX_TURN_RATE_RAD_RECIPROCAL = 1.0/SHIP_MAX_TURN_RATE_RAD;
const double SHIP_MAX_TURN_RATE_DEG_TS = DELTA_TIME*SHIP_MAX_TURN_RATE;
const double SHIP_MAX_TURN_RATE_RAD_TS = DEG_TO_RAD*SHIP_MAX_TURN_RATE_DEG_TS;
const double SHIP_MAX_THRUST = 480.0;
const double SHIP_DRAG = 80.0;
const double SHIP_MAX_SPEED = 240.0;
const double SHIP_RADIUS = 20.0;
const double SHIP_MASS = 300.0;
const i64 TIMESTEPS_UNTIL_SHIP_ACHIEVES_MAX_SPEED =
    static_cast<i64>(std::ceil(SHIP_MAX_SPEED/(SHIP_MAX_THRUST - SHIP_DRAG)*FPS));
const double MINE_BLAST_RADIUS = 150.0;
const double MINE_RADIUS = 12.0;
const double MINE_BLAST_PRESSURE = 2000.0;
const double MINE_FUSE_TIME = 3.0;
const double MINE_MASS = 25.0;

const std::vector<double> ASTEROID_RADII_LOOKUP = {0, 8, 16, 24, 32};
const std::vector<double> ASTEROID_AREA_LOOKUP = {
    M_PI*ASTEROID_RADII_LOOKUP[0]*ASTEROID_RADII_LOOKUP[0],
    M_PI*ASTEROID_RADII_LOOKUP[1]*ASTEROID_RADII_LOOKUP[1],
    M_PI*ASTEROID_RADII_LOOKUP[2]*ASTEROID_RADII_LOOKUP[2],
    M_PI*ASTEROID_RADII_LOOKUP[3]*ASTEROID_RADII_LOOKUP[3],
    M_PI*ASTEROID_RADII_LOOKUP[4]*ASTEROID_RADII_LOOKUP[4]};
const std::vector<double> ASTEROID_MASS_LOOKUP = {
    0.25*M_PI*std::pow(8*0,2),
    0.25*M_PI*std::pow(8*1,2),
    0.25*M_PI*std::pow(8*2,2),
    0.25*M_PI*std::pow(8*3,2),
    0.25*M_PI*std::pow(8*4,2)
};
constexpr double RESPAWN_INVINCIBILITY_TIME_S = 3.0;
const std::vector<int> ASTEROID_COUNT_LOOKUP = {0, 1, 4, 13, 40};
const double DEGREES_BETWEEN_SHOTS = double(FIRE_COOLDOWN_TS)*SHIP_MAX_TURN_RATE*DELTA_TIME;
const double DEGREES_TURNED_PER_TIMESTEP = SHIP_MAX_TURN_RATE*DELTA_TIME;
const double SHIP_RADIUS_PLUS_SIZE_4_ASTEROID_RADIUS = SHIP_RADIUS + ASTEROID_RADII_LOOKUP[4];

// FIS Settings
constexpr i64 ASTEROIDS_HIT_VERY_GOOD = 65;
constexpr int ASTEROIDS_HIT_OKAY_CENTER = 23;

// Dirty globals - reset these if sim re-initialized
inline std::map<std::string, i64> explanation_messages_with_timestamps;
inline std::vector<double> abs_cruise_speeds = {SHIP_MAX_SPEED/2.0};
inline std::vector<i64> cruise_timesteps = {static_cast<i64>(std::round(MAX_CRUISE_TIMESTEPS/2))};
inline std::vector<double> overall_fitness_record;
inline i64 total_sim_timesteps = 0;

// Unwrap cache
inline std::map<i64, std::vector<py::object>> unwrap_cache; // use py::object for Asteroid stub

// ---------------------------------- CLASSES ----------------------------------
struct Asteroid {
    double x = 0, y = 0, vx = 0, vy = 0;
    i64 size = 0;
    double mass = 0, radius = 0;
    i64 timesteps_until_appearance = 0;
    bool alive = true;

    Asteroid() = default;
    Asteroid(double x, double y, double vx, double vy, i64 size, double mass, double radius, i64 t = 0)
        : x(x), y(y), vx(vx), vy(vy), size(size), mass(mass), radius(radius), timesteps_until_appearance(t), alive(true) {}

    std::string str() const {
        return "Asteroid(position=(" + std::to_string(x) + ", " + std::to_string(y)
            + "), velocity=(" + std::to_string(vx) + ", " + std::to_string(vy) + "), size=" + std::to_string(size)
            + ", mass=" + std::to_string(mass) + ", radius=" + std::to_string(radius)
            + ", timesteps_until_appearance=" + std::to_string(timesteps_until_appearance) + ")";
    }
    std::string repr() const { return str(); }
    bool operator==(const Asteroid& other) const {
        return x == other.x && y == other.y && vx == other.vx && vy == other.vy &&
               size == other.size && mass == other.mass && radius == other.radius &&
               timesteps_until_appearance == other.timesteps_until_appearance;
    }
    std::size_t hash() const {
        double combined = x + 0.4266548291679171*y + 0.8164926348982552*vx + 0.8397584399461026*vy;
        double scaled = combined * 1'000'000'000.0;
        return static_cast<std::size_t>(scaled) + static_cast<std::size_t>(size);
    }
    double float_hash() const {
        return x + 0.4266548291679171*y + 0.8164926348982552*vx + 0.8397584399461026*vy;
    }
    i64 int_hash() const {
        return static_cast<i64>(1'000'000'000.0*float_hash());
    }
    Asteroid copy() const {
        if(!alive) throw std::runtime_error("Trying to copy unalive object");
        return *this;
    }
};

struct Ship {
    bool is_respawning = false;
    double x = 0, y = 0, vx = 0, vy = 0;
    double speed = 0, heading = 0, mass = 0, radius = 0;
    i64 id = 0;
    std::string team;
    i64 lives_remaining = 0, bullets_remaining = 0, mines_remaining = 0;
    bool can_fire = true, can_deploy_mine = true;
    double fire_rate = 0.0, mine_deploy_rate = 0.0;
    std::pair<double,double> thrust_range = {-SHIP_MAX_THRUST, SHIP_MAX_THRUST};
    std::pair<double,double> turn_rate_range = {-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE};
    double max_speed = SHIP_MAX_SPEED, drag = SHIP_DRAG;

    Ship() = default;
    Ship(bool is_respawning, double x, double y, double vx, double vy, double speed, double heading, double mass, double radius,
         i64 id, std::string team, i64 lives_remaining, i64 bullets_remaining, i64 mines_remaining, bool can_fire, double fire_rate,
         bool can_deploy_mine, double mine_deploy_rate, std::pair<double,double> thrust_range, std::pair<double,double> turn_rate_range,
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
    Ship copy() const { return *this; }
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
    Mine copy() const {
        if (!alive) throw std::runtime_error("Trying to copy unalive object");
        return *this;
    }
    bool operator==(const Mine& other) const {
        return x == other.x && y == other.y && mass == other.mass &&
            fuse_time == other.fuse_time && remaining_time == other.remaining_time;
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
    Bullet copy() const {
        if (!alive) throw std::runtime_error("Trying to copy unalive object");
        return *this;
    }
    bool operator==(const Bullet& other) const {
        return x == other.x && y == other.y && vx == other.vx && vy == other.vy &&
            heading == other.heading && mass == other.mass && tail_delta_x == other.tail_delta_x &&
            tail_delta_y == other.tail_delta_y;
    }
};

struct GameState {
    std::vector<Asteroid> asteroids;
    std::vector<Ship> ships;
    std::vector<Bullet> bullets;
    std::vector<Mine> mines;

    double map_size_x = 0, map_size_y = 0;
    double time = 0, delta_time = 0;
    i64 sim_frame = 0;
    double time_limit = 0;

    GameState() = default;
    GameState(const std::vector<Asteroid>& asteroids, const std::vector<Ship>& ships,
              const std::vector<Bullet>& bullets, const std::vector<Mine>& mines,
              double map_size_x, double map_size_y, double time, double delta_time,
              i64 sim_frame, double time_limit)
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
        for(const auto& a : asteroids) if(a.alive) alive_asteroids.push_back(a.copy());
        std::vector<Bullet> alive_bullets;
        for(const auto& b : bullets) if(b.alive) alive_bullets.push_back(b.copy());
        std::vector<Mine> alive_mines;
        for(const auto& m : mines) if(m.alive) alive_mines.push_back(m.copy());
        std::vector<Ship> ships_copy;
        for(const auto& s : ships) ships_copy.push_back(s.copy());
        return GameState(alive_asteroids, ships_copy, alive_bullets, alive_mines,
            map_size_x, map_size_y, time, delta_time, sim_frame, time_limit);
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
    i64 aiming_timesteps_required = 0;
    double interception_time_s = 0.0;
    double intercept_x = 0.0, intercept_y = 0.0;
    double asteroid_dist_during_interception = 0.0;
    double imminent_collision_time_s = 0.0;
    bool asteroid_will_get_hit_by_my_mine = false, asteroid_will_get_hit_by_their_mine = false;

    Target() = default;
    Target(const Asteroid& asteroid, bool feasible = false, double shooting_angle_error_deg = 0.0, i64 aiming_timesteps_required = 0, double interception_time_s = 0.0, double intercept_x = 0.0, double intercept_y = 0.0, double asteroid_dist_during_interception = 0.0, double imminent_collision_time_s = 0.0, bool asteroid_will_get_hit_by_my_mine = false, bool asteroid_will_get_hit_by_their_mine = false)
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
    Target copy() const { return *this; }
};

struct Action {
    double thrust = 0.0;
    double turn_rate = 0.0;
    bool fire = false;
    bool drop_mine = false;
    i64 timestep = 0;
    Action() = default;
    Action(double thrust, double turn_rate, bool fire, bool drop_mine, i64 timestep)
        : thrust(thrust), turn_rate(turn_rate), fire(fire), drop_mine(drop_mine), timestep(timestep) {}
    std::string str() const {
        return "Action(thrust=" + std::to_string(thrust)
            + ", turn_rate=" + std::to_string(turn_rate)
            + ", fire=" + std::to_string(fire)
            + ", drop_mine=" + std::to_string(drop_mine)
            + ", timestep=" + std::to_string(timestep) + ")";
    }
    std::string repr() const { return str(); }
    Action copy() const { return *this; }
};

struct SimState {
    i64 timestep = 0;
    Ship ship_state;
    std::optional<GameState> game_state;
    std::optional<std::map<i64, std::vector<Asteroid>>> asteroids_pending_death;
    std::optional<std::vector<Asteroid>> forecasted_asteroid_splits;

    SimState() = default;
    SimState(i64 timestep, Ship ship_state, std::optional<GameState> game_state = std::nullopt, std::optional<std::map<i64, std::vector<Asteroid>>> asteroids_pending_death = std::nullopt, std::optional<std::vector<Asteroid>> forecasted_asteroid_splits = std::nullopt)
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
            ship_state.copy(),
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
    auto pos = d["position"].cast<std::pair<double,double>>();
    auto vel = d["velocity"].cast<std::pair<double,double>>();
    return Asteroid(
        pos.first, pos.second,
        vel.first, vel.second,
        d["size"].cast<i64>(),
        d["mass"].cast<double>(),
        d["radius"].cast<double>());
}

Ship create_ship_from_dict(py::dict d) {
    auto pos = d.contains("position") ? d["position"].cast<std::pair<double,double>>() : std::make_pair(0.0, 0.0);
    auto vel = d.contains("velocity") ? d["velocity"].cast<std::pair<double,double>>() : std::make_pair(0.0, 0.0);
    auto thrust_range = d.contains("thrust_range") ? d["thrust_range"].cast<std::pair<double,double>>() : std::make_pair(-SHIP_MAX_THRUST, SHIP_MAX_THRUST);
    auto turn_range = d.contains("turn_rate_range") ? d["turn_rate_range"].cast<std::pair<double,double>>() : std::make_pair(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE);
    return Ship(
        d.contains("is_respawning") ? d["is_respawning"].cast<bool>() : false,
        pos.first, pos.second, vel.first, vel.second,
        d.contains("speed") ? d["speed"].cast<double>() : 0.0,
        d.contains("heading") ? d["heading"].cast<double>() : 0.0,
        d.contains("mass") ? d["mass"].cast<double>() : 0.0,
        d.contains("radius") ? d["radius"].cast<double>() : 0.0,
        d.contains("id") ? d["id"].cast<i64>() : 0,
        d.contains("team") ? d["team"].cast<std::string>() : "",
        d.contains("lives_remaining") ? d["lives_remaining"].cast<i64>() : 0,
        d.contains("bullets_remaining") ? d["bullets_remaining"].cast<i64>() : 0,
        d.contains("mines_remaining") ? d["mines_remaining"].cast<i64>() : 0,
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
    auto pos = d["position"].cast<std::pair<double,double>>();
    return Mine(pos.first, pos.second,
                d["mass"].cast<double>(),
                d["fuse_time"].cast<double>(),
                d["remaining_time"].cast<double>());
}
Bullet create_bullet_from_dict(py::dict d) {
    auto pos = d["position"].cast<std::pair<double,double>>();
    auto vel = d["velocity"].cast<std::pair<double,double>>();
    double heading = d["heading"].cast<double>();
    double heading_rad = heading*DEG_TO_RAD;
    return Bullet(
        pos.first, pos.second, vel.first, vel.second, heading, d["mass"].cast<double>(),
        -BULLET_LENGTH*cos(heading_rad), -BULLET_LENGTH*sin(heading_rad));
}
GameState create_game_state_from_dict(py::dict game_state_dict) {
    std::vector<Asteroid> asteroids;
    for(auto a : game_state_dict["asteroids"].cast<py::list>()) 
        asteroids.push_back(create_asteroid_from_dict(a.cast<py::dict>()));
    std::vector<Ship> ships;
    for(auto s : game_state_dict["ships"].cast<py::list>())
        ships.push_back(create_ship_from_dict(s.cast<py::dict>()));
    std::vector<Bullet> bullets;
    for(auto b : game_state_dict["bullets"].cast<py::list>())
        bullets.push_back(create_bullet_from_dict(b.cast<py::dict>()));
    std::vector<Mine> mines;
    for(auto m : game_state_dict["mines"].cast<py::list>())
        mines.push_back(create_mine_from_dict(m.cast<py::dict>()));
    auto map_size = game_state_dict["map_size"].cast<std::pair<double,double>>();
    return GameState(
        asteroids, ships, bullets, mines,
        map_size.first, map_size.second,
        game_state_dict["time"].cast<double>(),
        game_state_dict["delta_time"].cast<double>(),
        game_state_dict["sim_frame"].cast<i64>(),
        game_state_dict["time_limit"].cast<double>());
}






// Thread-safe random (can adjust as needed for your codebase)
inline static thread_local std::mt19937 rng(std::random_device{}());
inline static thread_local std::uniform_real_distribution<> std_uniform(0.0, 1.0);

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

inline int64_t fast_randint(int64_t a, int64_t b) {
    // Generate uniform random in [a, b]
    return a + static_cast<int64_t>(std::floor((b - a + 1) * random_double()));
}

inline double fast_uniform(double a, double b) {
    return a + (b - a) * random_double();
}

inline double fast_triangular(double low, double high, double mode) {
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

inline bool is_close_to_zero(double x) {
    return std::abs(x) <= EPS;
}

// ------ Fast and SuperFast Trig Functions ------

inline double super_fast_acos(double x) {
    return (-0.69813170079773212*x*x - 0.87266462599716477)*x + 1.5707963267948966;
}
inline double fast_acos(double x) {
    double negate = static_cast<double>(x < 0);
    x = std::abs(x);
    double ret = (((-0.0187293*x + 0.0742610)*x - 0.2121144)*x + 1.5707288)*std::sqrt(1.0 - x);
    return negate*pi + ret - 2.0*negate*ret;
}

inline double super_fast_asin(double x) {
    double x_square = x * x;
    return x * (0.9678828 + x_square * (0.8698691 - x_square * (2.166373 - x_square * 1.848968)));
}
inline double fast_asin(double x) {
    double negate = static_cast<double>(x < 0);
    x = std::abs(x);
    double ret = (((-0.0187293*x + 0.0742610)*x - 0.2121144)*x + 1.5707288);
    ret = 0.5*pi - std::sqrt(1.0 - x)*ret;
    return ret - 2.0*negate*ret;
}

inline double super_fast_atan2(double y, double x) {
    // Handle edge cases for 0 inputs
    if (x == 0.0) {
        if (y == 0.0) {
            return 0.0; // atan2(0, 0) is undefined, return 0 for simplicity
        } else {
            return (y > 0.0 ? 0.5*pi : -0.5*pi);
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
            atan_result = 0.5*pi - atan_result;
        } else {
            atan_result = -0.5*pi - atan_result;
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
            return (y > 0.0 ? 0.5*pi : -0.5*pi);
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
            atan_result = 0.5*pi - atan_result;
        } else {
            atan_result = -0.5*pi - atan_result;
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
        return "Neo C++";
    }

    int ship_id() const {
        return _ship_id;
    }

    void set_ship_id(int value) {
        _ship_id = value;
    }

    // --- Public Variables (Python: made them all instance, here public/protected for simplicity) ---

    bool init_done = false;
    i64 ship_id_internal = -1;
    i64 current_timestep = -1;
    std::deque<std::tuple<i64,double,double,bool,bool>> action_queue; // (timestep, thrust, turn_rate, fire, drop_mine)
    std::optional<GameStatePlotter> game_state_plotter;
    std::set<i64> actioned_timesteps;
    std::vector<SimState> sims_this_planning_period; // The first is stationary targeting, rest are maneuvers
    double best_fitness_this_planning_period = -inf;
    i64 best_fitness_this_planning_period_index = INT_NEG_INF;
    double second_best_fitness_this_planning_period = -inf;
    i64 second_best_fitness_this_planning_period_index = INT_NEG_INF;
    i64 stationary_targetting_sim_index = INT_NEG_INF;
    double current_sequence_fitness = -inf;
    std::map<i64, double> respawn_timer_history;
    std::map<i64, i64> last_timestep_fired_schedule = {{0, INT_NEG_INF}};
    std::map<i64, i64> last_timestep_mined_schedule = {{0, INT_NEG_INF}};
    std::set<i64> fire_next_timestep_schedule;
    std::map<i64, std::map<i64, std::vector<Asteroid>>> asteroids_pending_death_schedule;
    std::map<i64, std::vector<Asteroid>> forecasted_asteroid_splits_schedule;
    std::map<i64, std::set<std::pair<double,double>>> mine_positions_placed_schedule;

    std::optional<BasePlanningState> game_state_to_base_planning;
    std::optional<std::tuple<double,double,double,double,i64,double,i64,i64>> base_gamestate_analysis;
    std::set<i64> set_of_base_gamestate_timesteps;
    std::map<i64, std::map<std::string, pybind11::object>> base_gamestates; // You can replace value type with a suitable C++ struct

    bool other_ships_exist = false;
    //std::vector<std::map<std::string, pybind11::object>> reality_move_sequence;
    std::map<i64, SimState> simulated_gamestate_history;
    std::set<i64> lives_remaining_that_we_did_respawn_maneuver_for;
    bool last_timestep_ship_is_respawning = false;

    // For performance controller
    std::vector<double> outside_controller_time_intervals;
    std::vector<double> inside_controller_iteration_time_intervals;
    double last_entrance_time = nan;
    double last_exit_time = nan;
    double last_iteration_start_time = nan;
    double average_iteration_time = DELTA_TIME*0.1;

    // --- Ctor ---
    NeoController(const std::optional<std::tuple<double,double,double,double,double,double,double,double,double>>& chromosome = std::nullopt)
    {
        std::cout << BUILD_NUMBER << std::endl;
        // Could add __FILE__ or __func__ here, but omitted.
        reset(chromosome);
    }

    // --- Reset function ---
    void reset(const std::optional<std::tuple<double,double,double,double,double,double,double,double,double>>& chromosome = std::nullopt)
    {
        init_done = false;
        // DO NOT overwrite _ship_id
        ship_id_internal = -1;
        current_timestep = -1;
        action_queue.clear();
        game_state_plotter.reset();
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
        last_entrance_time = nan;
        last_exit_time = nan;
        last_iteration_start_time = nan;
        average_iteration_time = DELTA_TIME*0.1;
        // Clear "global" variables
        explanation_messages_with_timestamps.clear();
        abs_cruise_speeds = {SHIP_MAX_SPEED/2};
        cruise_timesteps = {static_cast<i64>(std::round(MAX_CRUISE_TIMESTEPS/2))};
        overall_fitness_record.clear();
        unwrap_cache.clear();
        total_sim_timesteps = 0; // REMOVE_FOR_COMPETITION
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
    void enqueue_action(i64 timestep, double thrust=0.0, double turn_rate=0.0, bool fire=false, bool drop_mine=false)
    {
        action_queue.push_back(std::make_tuple(timestep, thrust, turn_rate, fire, drop_mine));
    }

    bool NeoController::decide_next_action_continuous(const GameState &game_state, const Ship &ship_state, bool force_decision) {
        // Extern global, as in Python
        extern std::map<i64, std::vector<pybind11::object>> unwrap_cache;

        debug_print("Calling decide next action continuous on timestep " + std::to_string(game_state.sim_frame) + ", and force_decision=" + std::string(force_decision ? "true" : "false"));
        assert(game_state_to_base_planning.has_value());
        assert(best_fitness_this_planning_period_index != INT_NEG_INF); // REMOVE_FOR_COMPETITION

        debug_print("\nDeciding next action! We're picking out of " + std::to_string(sims_this_planning_period.size()) + " total sims");

        // Assume a helper function to pretty-print fitnesses if desired.

        // (Optional plotting omitted, see Python for matplotlib calls.)

        // REMOVE_FOR_COMPETITION: this assertion says that the "best" sim is always 'exact'
        assert(sims_this_planning_period.at(best_fitness_this_planning_period_index).state_type == "exact");

        // Setup placeholders for the variables we pick out below.
        Matrix best_action_sim;
        double best_action_fitness;
        std::vector<double> best_action_fitness_breakdown;
        std::optional<std::vector<double>> best_action_maneuver_tuple;

        // --- Multi-pass respawn handling ---
        if (game_state_to_base_planning->respawning) {
            Matrix &first_pass_sim = sims_this_planning_period.at(best_fitness_this_planning_period_index).sim;
            double first_pass_fitness = sims_this_planning_period.at(best_fitness_this_planning_period_index).fitness;
            bool first_pass_sim_fire_next_timestep_flag = first_pass_sim.get_fire_next_timestep_flag();

            // Construct second-pass simulation. The Matrix constructor signatures and methods would be 
            // as per your implementation -- adjust as needed.
            best_action_sim = Matrix(
                game_state,
                ship_state,
                current_timestep,
                game_state_to_base_planning->ship_respawn_timer,
                game_state_to_base_planning->asteroids_pending_death,
                game_state_to_base_planning->forecasted_asteroid_splits,
                game_state_to_base_planning->last_timestep_fired,
                game_state_to_base_planning->last_timestep_mined,
                game_state_to_base_planning->mine_positions_placed,
                game_state_to_base_planning->respawning,
                game_state_to_base_planning->fire_next_timestep_flag,
                /*verify_first_shot=*/true, /*verify_maneuver_shots=*/true,
                first_pass_sim.get_last_timestep_colliding(), // pass down
                game_state_plotter
            );

            auto first_pass_move_sequence = first_pass_sim.get_intended_move_sequence();
            best_action_sim.apply_move_sequence(first_pass_move_sequence, true);
            best_action_sim.set_fire_next_timestep_flag(first_pass_sim_fire_next_timestep_flag);

            best_action_fitness = best_action_sim.get_fitness();
            best_action_fitness_breakdown = best_action_sim.get_fitness_breakdown();
            best_action_maneuver_tuple = sims_this_planning_period.at(best_fitness_this_planning_period_index).maneuver_tuple;

            // If second pass was worse for some reason, revert to first pass.
            if (first_pass_fitness > best_action_fitness + 0.015) {
                best_action_sim = sims_this_planning_period.at(best_fitness_this_planning_period_index).sim;
                best_action_fitness = first_pass_fitness;
                best_action_fitness_breakdown = sims_this_planning_period.at(best_fitness_this_planning_period_index).fitness_breakdown;
                best_action_maneuver_tuple = sims_this_planning_period.at(best_fitness_this_planning_period_index).maneuver_tuple;
            }
        } else {
            // Exact one-pass
            const auto &sim = sims_this_planning_period.at(best_fitness_this_planning_period_index);
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
                actioned_timesteps.clear(); // REMOVE_FOR_COMPETITION
                fire_next_timestep_schedule.clear();
            } else {
                sims_this_planning_period.clear();
                best_fitness_this_planning_period = -inf;
                best_fitness_this_planning_period_index = INT_NEG_INF;
                second_best_fitness_this_planning_period = -inf;
                second_best_fitness_this_planning_period_index = INT_NEG_INF;
                stationary_targetting_sim_index = INT_NEG_INF;
                base_gamestate_analysis.reset();
                unwrap_cache.clear();
                return false;
            }
        } else {
            assert(action_queue.empty()); // REMOVE_FOR_COMPETITION
        }

        // LEARNING statistics (rolling averages for maneuver learning)
        if (best_action_maneuver_tuple.has_value() && !game_state_to_base_planning->respawning && best_action_fitness_breakdown.at(5) != 0.0) {
            abs_cruise_speeds.push_back(std::abs(best_action_maneuver_tuple.value().at(1)));
            cruise_timesteps.push_back(best_action_maneuver_tuple.value().at(3));
            if (abs_cruise_speeds.size() > MANEUVER_TUPLE_LEARNING_ROLLING_AVG_PERIOD)
                abs_cruise_speeds.erase(abs_cruise_speeds.begin(), abs_cruise_speeds.end() - MANEUVER_TUPLE_LEARNING_ROLLING_AVG_PERIOD);
            if (cruise_timesteps.size() > MANEUVER_TUPLE_LEARNING_ROLLING_AVG_PERIOD)
                cruise_timesteps.erase(cruise_timesteps.begin(), cruise_timesteps.end() - MANEUVER_TUPLE_LEARNING_ROLLING_AVG_PERIOD);
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
        auto best_move_sequence = best_action_sim.get_move_sequence();
        debug_print("Best sim ID: " + std::to_string(best_action_sim.get_sim_id()) + ", with index "
            + std::to_string(best_fitness_this_planning_period_index) + " and fitness " + std::to_string(best_action_fitness));
        auto best_action_sim_state_sequence = best_action_sim.get_state_sequence();
        if (VALIDATE_ALL_SIMULATED_STATES && !PRUNE_SIM_STATE_SEQUENCE) { // REMOVE_FOR_COMPETITION
            for (const auto& state : best_action_sim_state_sequence) {
                simulated_gamestate_history[state.timestep] = state;
            }
        }
        if (PRINT_EXPLANATIONS) {
            auto explanations = best_action_sim.get_explanations();
            for (const auto& exp : explanations)
                print_explanation(exp, current_timestep);
            if (random_double() < 0.1) {
                print_explanation("I currently feel " + std::to_string(weighted_average(overall_fitness_record) * 100.0)
                    + "% safe, considering how long I can stay here without being hit by asteroids or mines, and my proximity to the other ship.",
                    current_timestep);
            }
        }
        if (best_action_sim_state_sequence.empty())
            throw std::runtime_error("Why in the world is this state sequence empty?");

        auto best_action_sim_last_state = best_action_sim_state_sequence.back();
        auto asteroids_pending_death = best_action_sim.get_asteroids_pending_death();

        for (i64 timestep = current_timestep; timestep < best_action_sim_last_state.timestep; ++timestep)
            asteroids_pending_death.erase(timestep);

        auto forecasted_asteroid_splits = best_action_sim.get_forecasted_asteroid_splits();
        auto next_base_game_state = best_action_sim.get_game_state();

        set_of_base_gamestate_timesteps.insert(best_action_sim_last_state.timestep);
        auto new_ship_state = best_action_sim.get_ship_state();
        bool new_fire_next_timestep_flag = best_action_sim.get_fire_next_timestep_flag();

        if (new_ship_state.is_respawning && new_fire_next_timestep_flag &&
            lives_remaining_that_we_did_respawn_maneuver_for.count(new_ship_state.lives_remaining) == 0)
            new_fire_next_timestep_flag = false;

        if (ENABLE_SANITY_CHECKS && lives_remaining_that_we_did_respawn_maneuver_for.count(new_ship_state.lives_remaining) == 0
            && new_ship_state.is_respawning)
        {
            // If our ship is hurt in our next next action and I haven't done a respawn maneuver yet,
            // Then assert our next action is not a respawning action (REMOVED: Python's commented-out assertion).
            if (game_state_to_base_planning->respawning || new_fire_next_timestep_flag) {
                std::cerr << "We haven't done a respawn maneuver for having " << new_ship_state.lives_remaining << " lives left\n";
                std::cerr << "game_state_to_base_planning->respawning: " << game_state_to_base_planning->respawning
                    << ", new_fire_next_timestep_flag: " << new_fire_next_timestep_flag << ", respawn_timer=" << best_action_sim.get_respawn_timer() << "\n";
            }
            // assert(!game_state_to_base_planning->respawning && !new_fire_next_timestep_flag);
        }

        // Update planning state for next tick
        game_state_to_base_planning = BasePlanningState{
            best_action_sim_last_state.timestep,
            new_ship_state.lives_remaining not in lives_remaining_that_we_did_respawn_maneuver_for && new_ship_state.is_respawning,
            new_ship_state,
            next_base_game_state,
            best_action_sim.get_respawn_timer(),
            asteroids_pending_death,
            forecasted_asteroid_splits,
            best_action_sim.get_last_timestep_fired(),
            best_action_sim.get_last_timestep_mined(),
            best_action_sim.get_mine_positions_placed(),
            new_fire_next_timestep_flag
        };

        // Histories
        respawn_timer_history = best_action_sim.get_respawn_timer_history();
        asteroids_pending_death_schedule = best_action_sim.get_asteroids_pending_death_history();
        forecasted_asteroid_splits_schedule = best_action_sim.get_forecasted_asteroid_splits_history();
        mine_positions_placed_schedule = best_action_sim.get_mine_positions_placed_history();
        i64 last_timestep_fired = best_action_sim.get_last_timestep_fired();

        if (new_fire_next_timestep_flag) {
            fire_next_timestep_schedule.insert(best_move_sequence.back().timestep + 1);
            debug_print("Just added " + std::to_string(best_move_sequence.back().timestep + 1) + " to fire_next_timestep_schedule.");
        }

        if (ENABLE_SANITY_CHECKS) {
            assert((bool)game_state_to_base_planning->ship_respawn_timer == game_state_to_base_planning->ship_state.is_respawning);
        }
        if (game_state_to_base_planning->respawning)
            lives_remaining_that_we_did_respawn_maneuver_for.insert(new_ship_state.lives_remaining);

        base_gamestates[best_action_sim_last_state.timestep] = *game_state_to_base_planning; // Save state for validation/debug

        // Optionally dump state to file (REMOVE_FOR_COMPETITION)
        // if (KEY_STATE_DUMP)
        // if (SIMULATION_STATE_DUMP)
        // if (game_state_plotter.has_value() && ...)

        assert(action_queue.empty());
        if (CONTINUOUS_LOOKAHEAD_PLANNING) {
            assert(best_move_sequence.front().timestep == game_state.sim_frame);
        }
        for (const auto& move : best_move_sequence) {
            if (ENABLE_SANITY_CHECKS) {
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
        if (CONTINUOUS_LOOKAHEAD_PLANNING)
            assert(last_timestep_fired_schedule[best_move_sequence.back().timestep + 1] == last_timestep_fired);

        current_sequence_fitness = best_action_fitness;

        // Reset planning bookkeeping
        sims_this_planning_period.clear();
        best_fitness_this_planning_period = -inf;
        best_fitness_this_planning_period_index = INT_NEG_INF;
        second_best_fitness_this_planning_period = -inf;
        second_best_fitness_this_planning_period_index = INT_NEG_INF;
        stationary_targetting_sim_index = INT_NEG_INF;
        base_gamestate_analysis.reset();

        unwrap_cache.clear();
        return true;
    }

    void NeoController::plan_maneuver_iteration(bool plan_stationary, const std::string& state_type, bool other_ships_exist)
    {
        // --- Respawn maneuver branch ---
        if (game_state_to_base_planning->respawning) {
            double random_ship_heading_angle, ship_accel_turn_rate, ship_cruise_speed, ship_cruise_turn_rate;
            int ship_cruise_timesteps;

            random_ship_heading_angle = fast_uniform(-20.0, 20.0);
            ship_accel_turn_rate = fast_uniform(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE);
            if (random_double() < 0.5) {
                ship_cruise_speed = SHIP_MAX_SPEED;
            } else {
                ship_cruise_speed = -SHIP_MAX_SPEED;
            }
            ship_cruise_turn_rate = 0.0;
            ship_cruise_timesteps = fast_randint(0, static_cast<int>(std::round(MAX_CRUISE_SECONDS*FPS)));

            if (ENABLE_SANITY_CHECKS && !(bool(game_state_to_base_planning->ship_respawn_timer) == game_state_to_base_planning->ship_state.is_respawning)) {
                std::cerr << "BAD, game_state_to_base_planning->ship_respawn_timer: " << game_state_to_base_planning->ship_respawn_timer
                        << ", game_state_to_base_planning->ship_state.is_respawning: " << game_state_to_base_planning->ship_state.is_respawning << std::endl;
            }
            // TODO: There's a hardcoded false in args below. Investigate!
            //assert(!game_state_to_base_planning->fire_next_timestep_flag); // REMOVE_FOR_COMPETITION
            // Set up simulation with maneuver parameters
            Matrix maneuver_sim(
                game_state_to_base_planning->game_state,
                game_state_to_base_planning->ship_state,
                game_state_to_base_planning->timestep,
                game_state_to_base_planning->ship_respawn_timer,
                game_state_to_base_planning->asteroids_pending_death,
                game_state_to_base_planning->forecasted_asteroid_splits,
                game_state_to_base_planning->last_timestep_fired,
                game_state_to_base_planning->last_timestep_mined,
                game_state_to_base_planning->mine_positions_placed,
                /* halt_shooting */ true,
                /* fire_first_timestep */ false && game_state_to_base_planning->fire_next_timestep_flag,
                /* verify_first_shot */ false,
                /* verify_maneuver_shots */ false,
                game_state_plotter // may be std::nullopt
            );

            // Reject overly-long moves to stay within respawn invuln
            auto move_seq_preview = get_ship_maneuver_move_sequence(
                random_ship_heading_angle, ship_cruise_speed, ship_accel_turn_rate,
                ship_cruise_timesteps, ship_cruise_turn_rate, game_state_to_base_planning->ship_state.speed
            );
            while (move_seq_preview.size() >= game_state_to_base_planning->ship_respawn_timer*FPS) {
                random_ship_heading_angle = fast_uniform(-20.0, 20.0);
                ship_accel_turn_rate = fast_uniform(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE);
                if (random_double() < 0.5)
                    ship_cruise_speed = SHIP_MAX_SPEED;
                else
                    ship_cruise_speed = -SHIP_MAX_SPEED;
                ship_cruise_turn_rate = 0.0;
                ship_cruise_timesteps = fast_randint(0, static_cast<int>(std::round(MAX_CRUISE_SECONDS*FPS)));
                move_seq_preview = get_ship_maneuver_move_sequence(
                    random_ship_heading_angle, ship_cruise_speed, ship_accel_turn_rate,
                    ship_cruise_timesteps, ship_cruise_turn_rate, game_state_to_base_planning->ship_state.speed
                );
            }

            bool respawn_maneuver_without_crash =
                maneuver_sim.rotate_heading(random_ship_heading_angle) &&
                maneuver_sim.accelerate(ship_cruise_speed, ship_accel_turn_rate) &&
                maneuver_sim.cruise(ship_cruise_timesteps, ship_cruise_turn_rate) &&
                maneuver_sim.accelerate(0.0, 0.0);

            assert(respawn_maneuver_without_crash && "The respawn maneuver somehow crashed. Maybe it's too long!");

            double maneuver_fitness = maneuver_sim.get_fitness();

            // Record to planning period list
            CompletedSimulation sim_rec;
            sim_rec.sim = maneuver_sim;
            sim_rec.fitness = maneuver_fitness;
            sim_rec.fitness_breakdown = maneuver_sim.get_fitness_breakdown();
            sim_rec.action_type = "respawn";
            sim_rec.state_type = state_type;
            sim_rec.maneuver_tuple = {random_ship_heading_angle, ship_cruise_speed, ship_accel_turn_rate, static_cast<double>(ship_cruise_timesteps), ship_cruise_turn_rate};
            sims_this_planning_period.push_back(sim_rec);
            if (maneuver_fitness > best_fitness_this_planning_period) {
                second_best_fitness_this_planning_period = best_fitness_this_planning_period;
                second_best_fitness_this_planning_period_index = best_fitness_this_planning_period_index;
                best_fitness_this_planning_period = maneuver_fitness;
                best_fitness_this_planning_period_index = sims_this_planning_period.size() - 1;
            }
            return;
        }

        // --- Non-respawn branch (normal planning) ---
        if (!game_state_to_base_planning.has_value()) return;

        if (!base_gamestate_analysis.has_value()) {
            debug_print("Analyzing heuristic maneuver");
            base_gamestate_analysis = analyze_gamestate_for_heuristic_maneuver(
                game_state_to_base_planning->game_state, game_state_to_base_planning->ship_state);
        }
        bool ship_is_stationary = is_close_to_zero(game_state_to_base_planning->ship_state.speed);

        // --- Stationary targeting sim branch ---
        if (plan_stationary &&
            game_state_to_base_planning->ship_state.bullets_remaining != 0 && ship_is_stationary)
        {
            this->performance_controller_start_iteration();
            Matrix stationary_targetting_sim(
                game_state_to_base_planning->game_state,
                game_state_to_base_planning->ship_state,
                game_state_to_base_planning->timestep,
                game_state_to_base_planning->ship_respawn_timer,
                game_state_to_base_planning->asteroids_pending_death,
                game_state_to_base_planning->forecasted_asteroid_splits,
                game_state_to_base_planning->last_timestep_fired,
                game_state_to_base_planning->last_timestep_mined,
                game_state_to_base_planning->mine_positions_placed,
                /* halt_shooting */ false,
                /* fire_first_timestep */ game_state_to_base_planning->fire_next_timestep_flag,
                /* verify_first_shot */ (sims_this_planning_period.size() == 0 && other_ships_exist),
                /* verify_maneuver_shots */ false,
                game_state_plotter
            );
            stationary_targetting_sim.target_selection();

            double best_stationary_targetting_fitness = stationary_targetting_sim.get_fitness();

            if (sims_this_planning_period.size() == 0) {
                if (stationary_targetting_sim.get_cancel_firing_first_timestep()) {
                    assert(game_state_to_base_planning->fire_next_timestep_flag);
                    game_state_to_base_planning->fire_next_timestep_flag = false;
                }
            }
            CompletedSimulation sim_rec;
            sim_rec.sim = stationary_targetting_sim;
            sim_rec.fitness = best_stationary_targetting_fitness;
            sim_rec.fitness_breakdown = stationary_targetting_sim.get_fitness_breakdown();
            sim_rec.action_type = "targetting";
            sim_rec.state_type = state_type;
            sim_rec.maneuver_tuple = std::nullopt; // stationary, so no maneuver tuple
            sims_this_planning_period.push_back(sim_rec);
            stationary_targetting_sim_index = sims_this_planning_period.size() - 1;
            if (best_stationary_targetting_fitness > best_fitness_this_planning_period) {
                second_best_fitness_this_planning_period = best_fitness_this_planning_period;
                second_best_fitness_this_planning_period_index = best_fitness_this_planning_period_index;
                best_fitness_this_planning_period = best_stationary_targetting_fitness;
                best_fitness_this_planning_period_index = stationary_targetting_sim_index;
            }
        }
        if (plan_stationary && !ship_is_stationary) {
            std::cerr << "\nWARNING: The ship wasn't stationary after the last maneuver, so we're skipping stationary targeting! Our planning period starts on ts "
                << game_state_to_base_planning->timestep << std::endl;
        }

        // Heuristic maneuver setup
        bool heuristic_maneuver;
        if ((sims_this_planning_period.size() == 0 ||
            (sims_this_planning_period.size() == 1 && sims_this_planning_period[0].action_type != "heuristic_maneuver"))
            && ship_is_stationary)
            heuristic_maneuver = USE_HEURISTIC_MANEUVER;
        else
            heuristic_maneuver = false;

        // Unpack analysis stat tuple (see your Python, order must match)
        auto [imminent_asteroid_speed, imminent_asteroid_relative_heading, largest_gap_relative_heading,
            nearby_asteroid_average_speed, nearby_asteroid_count, average_directional_speed,
            total_asteroids_count, current_asteroids_count] = base_gamestate_analysis.value();

        // --- Adaptive cruise parameter selection ---
        double ship_cruise_speed_mode, ship_cruise_timesteps_mode, max_pre_maneuver_turn_timesteps;
        if (average_directional_speed > 80.0 && current_asteroids_count > 5 && total_asteroids_count >= 100) {
            print_explanation("Wall scenario detected! Preferring trying longer cruise lengths", current_timestep);
            ship_cruise_speed_mode = SHIP_MAX_SPEED;
            ship_cruise_timesteps_mode = MAX_CRUISE_TIMESTEPS;
            max_pre_maneuver_turn_timesteps = 6.0;
        } else if (std::any_of(game_state_to_base_planning->game_state.mines.begin(),
                            game_state_to_base_planning->game_state.mines.end(),
                            [&](const Mine& m) {
                                return game_state_to_base_planning->mine_positions_placed.count({m.x, m.y}) != 0;
                            })) {
            print_explanation("We're probably within the radius of a mine we placed! Biasing faster/longer moves to be more likely to escape the mine.", current_timestep);
            ship_cruise_speed_mode = SHIP_MAX_SPEED;
            ship_cruise_timesteps_mode = MAX_CRUISE_TIMESTEPS*0.75;
            max_pre_maneuver_turn_timesteps = 10.0;
        } else {
            max_pre_maneuver_turn_timesteps = 15.0;
            ship_cruise_speed_mode = weighted_average(abs_cruise_speeds); // see global
            ship_cruise_timesteps_mode = weighted_average(cruise_timesteps);
        }

        int search_iterations_count = 0;
        int iteration_limit = MAX_MANEUVER_PER_TIMESTEP_SEARCH_ITERATIONS;
        double fitness_avg = weighted_average(overall_fitness_record); // for get_min_maneuver_per_timestep_search_iterations

        while ((search_iterations_count < get_min_maneuver_per_timestep_search_iterations(
                    game_state_to_base_planning->ship_state.lives_remaining, fitness_avg)
            || performance_controller_check_whether_i_can_do_another_iteration()) &&
            search_iterations_count < iteration_limit)
        {
            performance_controller_start_iteration();
            ++search_iterations_count;
            double random_ship_heading_angle = 0, ship_accel_turn_rate = 0, ship_cruise_speed = 0, ship_cruise_turn_rate = 0;
            int ship_cruise_timesteps = 0, thrust_direction = 0;
            if (USE_HEURISTIC_MANEUVER && heuristic_maneuver) {
                random_ship_heading_angle = 0.0;
                double ship_cruise_timesteps_float = 0;
                std::tie(ship_accel_turn_rate, ship_cruise_speed, ship_cruise_turn_rate, ship_cruise_timesteps_float, thrust_direction) =
                    maneuver_heuristic_fis(imminent_asteroid_speed, imminent_asteroid_relative_heading, largest_gap_relative_heading,
                                        nearby_asteroid_average_speed, nearby_asteroid_count);
                ship_cruise_timesteps = static_cast<int>(std::round(ship_cruise_timesteps_float));
                if (thrust_direction < -GRAIN)
                    ship_cruise_speed = -ship_cruise_speed;
                else if (std::abs(thrust_direction) < GRAIN)
                    heuristic_maneuver = false; // FIS can't decide, skip this heuristic
            }
            if (!heuristic_maneuver || !USE_HEURISTIC_MANEUVER) {
                double a = -DEGREES_TURNED_PER_TIMESTEP*max_pre_maneuver_turn_timesteps;
                double b = +DEGREES_TURNED_PER_TIMESTEP*max_pre_maneuver_turn_timesteps;
                random_ship_heading_angle = fast_triangular(a, b, 0);
                ship_accel_turn_rate = fast_triangular(0, SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE)
                                    * (2.0*double(rand()%2)-1.0);
                if (std::isnan(ship_cruise_speed_mode)) {
                    ship_cruise_speed = fast_uniform(-SHIP_MAX_SPEED, SHIP_MAX_SPEED);
                } else {
                    ship_cruise_speed = fast_triangular(0, SHIP_MAX_SPEED, ship_cruise_speed_mode)
                                    * (2.0*double(rand()%2)-1.0);
                }
                ship_cruise_turn_rate = fast_triangular(0, SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE)
                                    * (2.0*double(rand()%2)-1.0);
                if (std::isnan(ship_cruise_timesteps_mode))
                    ship_cruise_timesteps = fast_randint(0, static_cast<int>(std::round(MAX_CRUISE_TIMESTEPS)));
                else
                    ship_cruise_timesteps = static_cast<int>(std::floor(fast_triangular(0.0, MAX_CRUISE_TIMESTEPS, ship_cruise_timesteps_mode)));
            }

            auto preview_move_sequence = get_ship_maneuver_move_sequence(
                random_ship_heading_angle, ship_cruise_speed, ship_accel_turn_rate,
                ship_cruise_timesteps, ship_cruise_turn_rate, game_state_to_base_planning->ship_state.speed
            );
            Matrix maneuver_sim(
                game_state_to_base_planning->game_state,
                game_state_to_base_planning->ship_state,
                game_state_to_base_planning->timestep,
                game_state_to_base_planning->ship_respawn_timer,
                game_state_to_base_planning->asteroids_pending_death,
                game_state_to_base_planning->forecasted_asteroid_splits,
                game_state_to_base_planning->last_timestep_fired,
                game_state_to_base_planning->last_timestep_mined,
                game_state_to_base_planning->mine_positions_placed,
                /* halt_shooting */ false,
                /* fire_first_timestep */ game_state_to_base_planning->fire_next_timestep_flag,
                /* verify_first_shot */ (sims_this_planning_period.size() == 0 && other_ships_exist),
                /* verify_maneuver_shots */ false,
                game_state_plotter
            );
            // If maneuver sim crashes, it returns false
            maneuver_sim.simulate_maneuver(preview_move_sequence, true, true);

            double maneuver_fitness = maneuver_sim.get_fitness();
            auto maneuver_fitness_breakdown = maneuver_sim.get_fitness_breakdown();

            if (sims_this_planning_period.size() == 0) {
                if (maneuver_sim.get_cancel_firing_first_timestep()) {
                    assert(game_state_to_base_planning->fire_next_timestep_flag);
                    game_state_to_base_planning->fire_next_timestep_flag = false;
                }
            }
            CompletedSimulation sim_rec;
            sim_rec.sim = maneuver_sim;
            sim_rec.fitness = maneuver_fitness;
            sim_rec.fitness_breakdown = maneuver_fitness_breakdown;
            sim_rec.action_type = heuristic_maneuver ? "heuristic_maneuver" : "random_maneuver";
            sim_rec.state_type = state_type;
            sim_rec.maneuver_tuple = {random_ship_heading_angle, ship_cruise_speed, ship_accel_turn_rate, static_cast<double>(ship_cruise_timesteps), ship_cruise_turn_rate};
            sims_this_planning_period.push_back(sim_rec);

            if (maneuver_fitness > best_fitness_this_planning_period) {
                second_best_fitness_this_planning_period = best_fitness_this_planning_period;
                second_best_fitness_this_planning_period_index = best_fitness_this_planning_period_index;
                best_fitness_this_planning_period = maneuver_fitness;
                best_fitness_this_planning_period_index = sims_this_planning_period.size()-1;
            }
            // Only try one heuristic maneuver per planning period!
            if (heuristic_maneuver)
                heuristic_maneuver = false;
        }
    }

    std::tuple<double, double, bool, bool>
    actions(const py::dict& ship_state, const py::dict& game_state)
    {
        // Optionally reseed RNG if flag enabled
        if (RESEED_RNG) {
            std::srand(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
        }

        ++this->current_timestep;
        bool recovering_from_crash = false;

        Ship ship_state = create_ship_from_dict(ship_state_dict);
        GameState game_state = create_game_state_from_dict(game_state_dict);

        // Check for simulator/controller desync and perform state recovery/reset as in Python
        if (CLEAN_UP_STATE_FOR_SUBSEQUENT_SCENARIO_RUNS || STATE_CONSISTENCY_CHECK_AND_RECOVERY) {
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
        this->performance_controller_enter();

        bool iterations_boost = false;
        if (this->current_timestep == 0) iterations_boost = true;

        // ------------------------------- PLANNING LOGIC ------------------------

        if (CONTINUOUS_LOOKAHEAD_PLANNING) {
            // == CONTINUOUS PLANNING ==
            if (this->other_ships_exist) {
                // == Other ships exist (non-deterministic mode)!
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
                    game_state_to_base_planning = create_base_planning_state(this->current_timestep, ship_state, game_state, 3.0, recovering_from_crash);
                    if (game_state_to_base_planning->respawning)
                        this->lives_remaining_that_we_did_respawn_maneuver_for.insert(ship_state.lives_remaining);
                } else if (unexpected_survival) {
                    debug_print("Unexpected survival, the ship state is "+ship_state.str());
                    game_state_to_base_planning = create_base_planning_state(this->current_timestep, ship_state, game_state, 0.0, false);
                } else if (!game_state_to_base_planning.has_value()) {
                    game_state_to_base_planning = create_base_planning_state(
                        this->current_timestep, ship_state, game_state,
                        (this->current_timestep == 0 ? 0.0 : (respawn_timer_history.count(this->current_timestep) ? respawn_timer_history[this->current_timestep] : 0.0)),
                        false);
                    if (game_state_to_base_planning->respawning)
                        this->lives_remaining_that_we_did_respawn_maneuver_for.insert(ship_state.lives_remaining);
                    assert((bool)game_state_to_base_planning->ship_respawn_timer == game_state_to_base_planning->ship_state.is_respawning);
                } else {
                    game_state_to_base_planning = create_base_planning_state(
                        this->current_timestep, ship_state, game_state,
                        (this->current_timestep == 0 ? 0.0 : (respawn_timer_history.count(this->current_timestep) ? respawn_timer_history[this->current_timestep] : 0.0)),
                        false);
                }

                if (action_queue.empty()) {
                    plan_action_continuous(false, true, iterations_boost, true);
                    plan_action_continuous(false, true, iterations_boost, false);
                    bool success = decide_next_action_continuous(game_state, ship_state, true);
                    assert(success);
                } else {
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
                if (recovering_from_crash) {
                    std::cerr << "RECOVERING FROM A CRASH!!!\n";
                }
                debug_print("Asteroid scheds: " + std::to_string(asteroids_pending_death_schedule.size()) + "," +
                            std::to_string(forecasted_asteroid_splits_schedule.size()) + "," +
                            std::to_string(last_timestep_fired_schedule.size()) + "," +
                            std::to_string(last_timestep_mined_schedule.size()) + "," +
                            std::to_string(mine_positions_placed_schedule.size()));

                game_state_to_base_planning = create_base_planning_state(
                    this->current_timestep, ship_state, game_state,
                    (this->current_timestep == 0 ? 0.0 : (respawn_timer_history.count(this->current_timestep) ? respawn_timer_history[this->current_timestep] : 0.0)),
                    false
                );
                if (action_queue.empty()) {
                    plan_action_continuous(false, true, iterations_boost, true);
                    plan_action_continuous(false, true, iterations_boost, false);
                    bool success = decide_next_action_continuous(game_state, ship_state, true);
                    assert(success);
                } else {
                    plan_action_continuous(false, true, iterations_boost, false);
                    bool success = decide_next_action_continuous(game_state, ship_state, false);
                    debug_print(success ? "Switched to a better maneuver" : "Didn't find better maneuvers");
                }
            }
        } else {
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
                        game_state_to_base_planning = create_base_planning_state(this->current_timestep, ship_state, game_state, 3.0, recovering_from_crash);
                    }
                    game_state_to_base_planning = create_base_planning_state(this->current_timestep, ship_state, game_state, 3.0, recovering_from_crash);
                    if (game_state_to_base_planning->respawning)
                        this->lives_remaining_that_we_did_respawn_maneuver_for.insert(ship_state.lives_remaining);
                } else if (unexpected_survival) {
                    debug_print("Unexpected survival, the ship state is "+ship_state.str());
                    game_state_to_base_planning = create_base_planning_state(this->current_timestep, ship_state, game_state, 0.0, false);
                } else if (!game_state_to_base_planning.has_value()) {
                    game_state_to_base_planning = create_base_planning_state(this->current_timestep, ship_state, game_state, 0.0, false);
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
        }

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

        // == SANITY CHECKS! (REMOVE_FOR_COMPETITION) ==
        if (ENABLE_SANITY_CHECKS) {
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

        // Optional sleep for visualization (REMOVE_FOR_COMPETITION)
        if (float(this->current_timestep) > SLOW_DOWN_GAME_AFTER_SECOND*FPS)
            std::this_thread::sleep_for(std::chrono::duration<double>(SLOW_DOWN_GAME_PAUSE_TIME));

        // Optional plotting, state validation, and so forth would go here, if ported

        this->last_timestep_ship_is_respawning = ship_state.is_respawning;
        return std::make_tuple(thrust, turn_rate, fire, drop_mine);
    }
};

PYBIND11_MODULE(neo_controller, m) {
    py::class_<NeoController>(m, "NeoController")
        .def(py::init<>())
        .def("actions", &NeoController::actions)
        .def_property_readonly("name", &NeoController::name)
        .def_property("ship_id", &NeoController::ship_id, &NeoController::set_ship_id);
}
