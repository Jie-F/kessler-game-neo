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

inline i64 get_min_respawn_per_timestep_search_iterations(i64 lives, double average_fitness) {
    assert(0.0 <= average_fitness && average_fitness < 1.0);
    int lives_lookup_index = static_cast<int>(std::min<i64>(3, lives));
    int fitness_lookup_index = static_cast<int>(std::floor(average_fitness * 10.0));
    return MIN_RESPAWN_PER_TIMESTEP_SEARCH_ITERATIONS_LUT.at(fitness_lookup_index).at(lives_lookup_index - 1);
}

inline i64 get_min_respawn_per_period_search_iterations(i64 lives, double average_fitness) {
    assert(0.0 <= average_fitness && average_fitness < 1.0);
    int lives_lookup_index = static_cast<int>(std::min<i64>(3, lives));
    int fitness_lookup_index = static_cast<int>(std::floor(average_fitness * 10.0));
    return MIN_RESPAWN_PER_PERIOD_SEARCH_ITERATIONS_LUT.at(fitness_lookup_index).at(lives_lookup_index - 1);
}

inline i64 get_min_maneuver_per_timestep_search_iterations(i64 lives, double average_fitness) {
    assert(0.0 <= average_fitness && average_fitness < 1.0);
    int lives_lookup_index = static_cast<int>(std::min<i64>(3, lives));
    int fitness_lookup_index = static_cast<int>(std::floor(average_fitness * 10.0));
    return MIN_MANEUVER_PER_TIMESTEP_SEARCH_ITERATIONS_LUT.at(fitness_lookup_index).at(lives_lookup_index - 1);
}

inline i64 get_min_maneuver_per_period_search_iterations(i64 lives, double average_fitness) {
    assert(0.0 <= average_fitness && average_fitness < 1.0);
    int lives_lookup_index = static_cast<int>(std::min<i64>(3, lives));
    int fitness_lookup_index = static_cast<int>(std::floor(average_fitness * 10.0));
    return MIN_MANEUVER_PER_PERIOD_SEARCH_ITERATIONS_LUT.at(fitness_lookup_index).at(lives_lookup_index - 1);
}

inline i64 get_min_maneuver_per_period_search_iterations_if_will_die(i64 lives, double average_fitness) {
    assert(0.0 <= average_fitness && average_fitness < 1.0);
    int lives_lookup_index = static_cast<int>(std::min<i64>(3, lives));
    int fitness_lookup_index = static_cast<int>(std::floor(average_fitness * 10.0));
    return MIN_MANEUVER_PER_PERIOD_SEARCH_ITERATIONS_IF_WILL_DIE_LUT.at(fitness_lookup_index).at(lives_lookup_index - 1);
}

// Mine FIS stuff is not included

// Forward declarations for helpers/constants assumed as globals or methods somewhere:
// int count_asteroids_in_mine_blast_radius(const GameState&, double x, double y, int timesteps);
// bool mine_fis(i64 mines_remaining, i64 lives_remaining, i64 mine_ast_count);
// const double MINE_BLAST_RADIUS, MINE_OTHER_SHIP_RADIUS_FUDGE;
// const i64 MINE_OTHER_SHIP_ASTEROID_COUNT_EQUIVALENT;
// const double MINE_FUSE_TIME, FPS;

inline bool check_mine_opportunity(const Ship& ship_state, const GameState& game_state, const std::vector<Ship>& other_ships)
{
    // If there's already more than one mine on the field, don't consider laying another
    if (game_state.mines.size() > 1)
        return false;

    int mine_ast_count = count_asteroids_in_mine_blast_radius(
        game_state, ship_state.x, ship_state.y, static_cast<int>(std::round(MINE_FUSE_TIME * FPS))
    );
    i64 lives_fudge = 0;

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
inline double linear(double x, std::pair<double,double> point1, std::pair<double,double> point2) {
    double x1 = point1.first, y1 = point1.second;
    double x2 = point2.first, y2 = point2.second;
    assert(x1 < x2); // REMOVE_FOR_COMPETITION
    if (x <= x1) return y1;
    else if (x >= x2) return y2;
    else return y1 + (x-x1)*(y2-y1)/(x2-x1);
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
inline void log_explanation(const std::string& message, int64_t current_timestep, const std::string& log_file = "Neo Explanations.txt")
{
    try {
        std::ofstream file(log_file, std::ios::app);
        if (!file) throw std::runtime_error("Could not open log file.");
        file << "Timestep " << current_timestep << " - " << message << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred when trying to log explanation: " << e.what() << std::endl;
    }
}

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

void inspect_scenario(const GameState& game_state, const Ship& ship_state) {
    const auto& asteroids = game_state.asteroids;
    double width = game_state.map_size_x;
    double height = game_state.map_size_y;
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
        double percent = static_cast<double>(ship_state.bullets_remaining)/std::max(1,asteroids_count);
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
        for (const auto& a : asteroids) {
            total_x_velocity += a.vx;
            total_y_velocity += a.vy;
        }
        int num_asteroids = static_cast<int>(asteroids.size());
        if (num_asteroids == 0)
            return {0.0, 0.0};
        else
            return { total_x_velocity/num_asteroids, total_y_velocity/num_asteroids };
    };

    auto average_speed = [&]() -> double {
        double total_speed = 0.0;
        for (const auto& a : asteroids) {
            total_speed += std::sqrt(a.vx*a.vx + a.vy*a.vy);
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

inline std::pair<int64_t, int64_t> asteroid_counter(const std::vector<Asteroid>& asteroids) {
    int64_t current_count = static_cast<int64_t>(asteroids.size());
    int64_t total_count = 0;
    for (const auto& a : asteroids) {
        total_count += ASTEROID_COUNT_LOOKUP.at(a.size);
    }
    return {total_count, current_count};
}

// Get all ships except self
inline std::vector<Ship> get_other_ships(const GameState& game_state, int64_t self_ship_id) {
    std::vector<Ship> result;
    result.reserve(game_state.ships.size());
    for (const auto& ship : game_state.ships) {
        if (ship.id != self_ship_id)
            result.push_back(ship);
    }
    return result;
}

// Angle difference (radians): wraps to [-pi, +pi]
inline double angle_difference_rad(double angle1, double angle2) {
    constexpr double pi = M_PI;
    constexpr double tau = 2.0 * M_PI;
    double diff = std::fmod(angle1 - angle2 + pi, tau);
    // fmod may return negative, wrap up by tau if so
    if (diff < 0) diff += tau;
    return diff - pi;
}

// Angle difference (degrees): wraps to [-180, +180]
inline double angle_difference_deg(double angle1, double angle2) {
    double diff = std::fmod(angle1 - angle2 + 180.0, 360.0);
    if (diff < 0) diff += 360.0;
    return diff - 180.0;
}

inline std::vector<Action>
get_ship_maneuver_move_sequence(double ship_heading_angle, double ship_cruise_speed, double ship_accel_turn_rate, int64_t ship_cruise_timesteps, double ship_cruise_turn_rate, double ship_starting_speed = 0.0)
{
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
            update(0.0, still_need_to_turn * (1.0/DELTA_TIME));
        }
    };

    // --- accelerate helper ---
    auto accelerate = [&](double target_speed, double turn_rate) {
        while (std::abs(target_speed - ship_speed) > EPS) {
            double drag = -SHIP_DRAG * sign(ship_speed);
            double drag_amount = SHIP_DRAG * DELTA_TIME;
            if (drag_amount > std::abs(ship_speed)) {
                double adjust_drag_by = std::abs((drag_amount - std::abs(ship_speed)) * (1.0/DELTA_TIME));
                drag -= adjust_drag_by * sign(drag);
            }
            double delta_speed_to_target = target_speed - ship_speed;
            double thrust_amount = delta_speed_to_target * (1.0/DELTA_TIME) - drag;
            thrust_amount = std::clamp(thrust_amount, -SHIP_MAX_THRUST, SHIP_MAX_THRUST);
            update(thrust_amount, turn_rate);
        }
    };

    // --- cruise helper ---
    auto cruise = [&](int64_t cruise_timesteps, double cruise_turn_rate) {
        for (int64_t i=0; i < cruise_timesteps; ++i) {
            update(sign(ship_speed)*SHIP_DRAG, cruise_turn_rate);
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

// Returns: {t_enter, t_exit} if potentially colliding, or {nan, nan} if no collision in future. 
inline std::pair<double, double> collision_prediction(
    double Oax, double Oay, double Dax, double Day, double ra,
    double Obx, double Oby, double Dbx, double Dby, double rb)
{
    double separation = ra + rb;
    double delta_x = Oax - Obx;
    double delta_y = Oay - Oby;

    if (is_close_to_zero(Dax) && is_close_to_zero(Day) &&
        is_close_to_zero(Dbx) && is_close_to_zero(Dby))
    {
        // Both stationary – just check overlap now:
        if (std::abs(delta_x) <= separation &&
            std::abs(delta_y) <= separation &&
            (delta_x*delta_x + delta_y*delta_y <= separation*separation))
        {
            // "collide now and forever"
            return std::make_pair(-inf, inf);
        } else {
            // Never collide
            return std::make_pair(nan, nan);
        }
    } else {
        // Relative velocity
        double vel_delta_x = Dax - Dbx;
        double vel_delta_y = Day - Dby;
        double a = vel_delta_x * vel_delta_x + vel_delta_y * vel_delta_y;
        double b = 2.0 * (delta_x * vel_delta_x + delta_y * vel_delta_y);
        double c = delta_x * delta_x + delta_y * delta_y - separation * separation;
        return solve_quadratic(a, b, c);
    }
}

inline double predict_next_imminent_collision_time_with_asteroid(
    double ship_pos_x, double ship_pos_y, double ship_vel_x, double ship_vel_y, double ship_r,
    double ast_pos_x, double ast_pos_y, double ast_vel_x, double ast_vel_y, double ast_radius,
    const GameState& game_state)
{
    // REMOVE_FOR_COMPETITION assertions:
    assert(is_close_to_zero(ship_vel_x) && is_close_to_zero(ship_vel_y));
    assert(check_coordinate_bounds(game_state, ship_pos_x, ship_pos_y));

    auto [start_collision_time, end_collision_time] = collision_prediction(
        ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_r,
        ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius
    );
    if (std::isnan(start_collision_time) || std::isnan(end_collision_time)) {
        return inf;
    } else {
        // REMOVE_FOR_COMPETITION
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

// === 1. find_time_interval_in_which_unwrapped_asteroid_is_within_main_wrap ===
inline std::pair<double, double> find_time_interval_in_which_unwrapped_asteroid_is_within_main_wrap(
    double ast_pos_x, double ast_pos_y, double ast_vel_x, double ast_vel_y,
    const GameState& game_state)
{
    std::pair<double, double> x_interval, y_interval;

    if (is_close_to_zero(ast_vel_x)) {
        if (check_coordinate_bounds(game_state, ast_pos_x, 0.0)) {
            x_interval = { -inf, inf };
        } else {
            return { nan, nan };
        }
    } else {
        if (ast_vel_x > 0.0) {
            x_interval = { -ast_pos_x/ast_vel_x, (game_state.map_size_x - ast_pos_x) / ast_vel_x };
        } else {
            x_interval = { (game_state.map_size_x - ast_pos_x) / ast_vel_x, -ast_pos_x / ast_vel_x };
        }
    }

    if (is_close_to_zero(ast_vel_y)) {
        if (check_coordinate_bounds(game_state, 0.0, ast_pos_y)) {
            y_interval = { -inf, inf };
        } else {
            return { nan, nan };
        }
    } else {
        if (ast_vel_y > 0.0) {
            y_interval = { -ast_pos_y/ast_vel_y, (game_state.map_size_y - ast_pos_y) / ast_vel_y };
        } else {
            y_interval = { (game_state.map_size_y - ast_pos_y) / ast_vel_y, -ast_pos_y / ast_vel_y };
        }
    }

    assert(x_interval.first <= x_interval.second);
    assert(y_interval.first <= y_interval.second);

    // Take the intersection of intervals
    if (x_interval.second < y_interval.first || y_interval.second < x_interval.first) {
        return { nan, nan };
    } else {
        double start = std::max(x_interval.first, y_interval.first);
        double end = std::min(x_interval.second, y_interval.second);
        return { start, end };
    }
}


// =============== 2. calculate_border_crossings ========================
// Returns a vector of (universe_x, universe_y) (int,int) pairs in order
inline std::vector<std::pair<int64_t,int64_t>> calculate_border_crossings(
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
    std::vector<std::pair<int64_t,int64_t>> universes;
    for (bool crossing : border_crossing_sequence) {
        if (crossing) current_universe_x += universe_increment_direction_x;
        else          current_universe_y += universe_increment_direction_y;
        universes.emplace_back(current_universe_x, current_universe_y);
    }
    return universes;
}


// ============= 3. unwrap_asteroid =====================
// Use copy constructor and int_hash (see Asteroid definition).
inline std::vector<Asteroid> unwrap_asteroid(
    const Asteroid& asteroid, double max_x, double max_y,
    double time_horizon_s = 10.0, bool use_cache = true)
{
    // Compute hash
    int64_t ast_hash = asteroid.int_hash();
    if (use_cache) {
        auto it = unwrap_cache.find(ast_hash);
        if (it != unwrap_cache.end())
            return it->second;
    }
    std::vector<Asteroid> unwrapped_asteroids;
    unwrapped_asteroids.push_back(asteroid.copy());
    if (std::abs(asteroid.vx) < EPS && std::abs(asteroid.vy) < EPS) {
        if (use_cache) unwrap_cache[ast_hash] = unwrapped_asteroids;
        return unwrapped_asteroids;
    }
    for (const auto& universe : calculate_border_crossings(asteroid.x, asteroid.y, asteroid.vx, asteroid.vy, max_x, max_y, time_horizon_s)) {
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
    if (use_cache) unwrap_cache[ast_hash] = unwrapped_asteroids;
    return unwrapped_asteroids;
}

// --- check_coordinate_bounds ---
inline bool check_coordinate_bounds(const GameState& game_state, double x, double y) {
    // Python: 0.0 <= x <= max_x and 0.0 <= y <= max_y
    return (0.0 <= x && x <= game_state.map_size_x &&
            0.0 <= y && y <= game_state.map_size_y);
}

// --- check_coordinate_bounds_exact ---
inline bool check_coordinate_bounds_exact(const GameState& game_state, double x, double y) {
    double x_wrapped = std::fmod(x, game_state.map_size_x);
    if (x_wrapped < 0) x_wrapped += game_state.map_size_x; // fmod can be negative
    double y_wrapped = std::fmod(y, game_state.map_size_y);
    if (y_wrapped < 0) y_wrapped += game_state.map_size_y;
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
                return {nan, nan};
        } else {
            double x = -c / b;
            return {x, x};
        }
    }
    double discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) {
        // No real solutions
        return {nan, nan};
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
        if (std::isnan(t) || t < 0.0)
            continue;
        double x = ax_delayed + t * avx;
        double y = ay_delayed + t * avy;
        double theta = fast_atan2(y, x);

        // Interception position in world/absolute coordinates
        double intercept_x = x + origin_x;
        double intercept_y = y + origin_y;
        bool feasible = check_coordinate_bounds(game_state, intercept_x, intercept_y);
        if (!feasible) continue;

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
    return std::make_tuple(false, nan, nan, nan, nan, nan, nan);
}

// === 1. Heading-based bullet splits
inline std::tuple<Asteroid, Asteroid, Asteroid> forecast_asteroid_bullet_splits_from_heading(
    const Asteroid& a,
    int64_t timesteps_until_appearance,
    double bullet_heading_deg,
    const GameState& game_state)
{
    double bullet_heading_rad = bullet_heading_deg * DEG_TO_RAD;
    double bullet_vel_x = std::cos(bullet_heading_rad) * BULLET_SPEED;
    double bullet_vel_y = std::sin(bullet_heading_rad) * BULLET_SPEED;
    double vfx = (1.0 / (BULLET_MASS + a.mass)) * (BULLET_MASS * bullet_vel_x + a.mass * a.vx);
    double vfy = (1.0 / (BULLET_MASS + a.mass)) * (BULLET_MASS * bullet_vel_y + a.mass * a.vy);
    double v = std::sqrt(vfx * vfx + vfy * vfy);
    return forecast_asteroid_splits(a, timesteps_until_appearance, vfx, vfy, v, 15.0, game_state);
}

// === 2. Instantaneous (velocity) bullet splits
inline std::tuple<Asteroid, Asteroid, Asteroid> forecast_instantaneous_asteroid_bullet_splits_from_velocity(
    const Asteroid& a, double bullet_vx, double bullet_vy, const GameState& game_state)
{
    double vfx = (1.0 / (BULLET_MASS + a.mass)) * (BULLET_MASS * bullet_vx + a.mass * a.vx);
    double vfy = (1.0 / (BULLET_MASS + a.mass)) * (BULLET_MASS * bullet_vy + a.mass * a.vy);
    double v = std::sqrt(vfx * vfx + vfy * vfy);
    return forecast_asteroid_splits(a, 0, vfx, vfy, v, 15.0, game_state);
}

// === 3. Mine splits
inline std::tuple<Asteroid, Asteroid, Asteroid> forecast_asteroid_mine_instantaneous_splits(
    const Asteroid& asteroid, const Mine& mine, const GameState& game_state)
{
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

// === 4. Ship splits
inline std::tuple<Asteroid, Asteroid, Asteroid> forecast_asteroid_ship_splits(
    const Asteroid& asteroid, int64_t timesteps_until_appearance, double ship_vx, double ship_vy, const GameState& game_state)
{
    double vfx = (1.0 / (SHIP_MASS + asteroid.mass)) * (SHIP_MASS * ship_vx + asteroid.mass * asteroid.vx);
    double vfy = (1.0 / (SHIP_MASS + asteroid.mass)) * (SHIP_MASS * ship_vy + asteroid.mass * asteroid.vy);
    double v = std::sqrt(vfx * vfx + vfy * vfy);
    return forecast_asteroid_splits(asteroid, timesteps_until_appearance, vfx, vfy, v, 15.0, game_state);
}

// === 5. Actual splitting logic
inline std::tuple<Asteroid, Asteroid, Asteroid> forecast_asteroid_splits(
    const Asteroid& a, int64_t timesteps_until_appearance, double vfx, double vfy, double v, double split_angle, const GameState& game_state)
{
    double theta = std::atan2(vfy, vfx) * RAD_TO_DEG; // DO NOT use an approximation!
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
            Asteroid(
                a.x, a.y, v * cos_angle_left, v * sin_angle_left,
                new_size, new_mass, new_radius, 0),
            Asteroid(
                a.x, a.y, v * cos_angle_center, v * sin_angle_center,
                new_size, new_mass, new_radius, 0),
            Asteroid(
                a.x, a.y, v * cos_angle_right, v * sin_angle_right,
                new_size, new_mass, new_radius, 0)
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

// --- Maintain split asteroids' forecast ---
inline std::vector<Asteroid> maintain_forecasted_asteroids(const std::vector<Asteroid>& forecasted_asteroid_splits, const GameState& game_state) {
    std::vector<Asteroid> updated_asteroids;
    for (const auto& forecasted_asteroid : forecasted_asteroid_splits) {
        if (forecasted_asteroid.timesteps_until_appearance > 1) {
            // Python's % matches always-positive modulus - adjust for negatives in C++
            auto wrap = [](double coord, double mod) {
                double r = std::fmod(coord, mod);
                return r < 0 ? r + mod : r;
            };
            updated_asteroids.emplace_back(
                wrap(forecasted_asteroid.x + forecasted_asteroid.vx * DELTA_TIME, game_state.map_size_x),
                wrap(forecasted_asteroid.y + forecasted_asteroid.vy * DELTA_TIME, game_state.map_size_y),
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
    for (const auto& a : list_of_asteroids) {
        // Compare with fuzzy position (including wrap-around), velocity, and size
        bool x_match = is_close(a.x, asteroid.x) ||
                       is_close_to_zero(game_state.map_size_x - std::abs(a.x - asteroid.x));
        bool y_match = is_close(a.y, asteroid.y) ||
                       is_close_to_zero(game_state.map_size_y - std::abs(a.y - asteroid.y));
        if (x_match && y_match &&
            is_close(a.vx, asteroid.vx) &&
            is_close(a.vy, asteroid.vy) &&
            a.size == asteroid.size) {
            return true;
        }
    }
    return false;
}

inline int64_t count_asteroids_in_mine_blast_radius(
    const GameState& game_state, double mine_x, double mine_y, int64_t future_check_timesteps)
{
    int64_t count = 0;
    for (const auto& a : game_state.asteroids) {
        if (a.alive) {
            // Project asteroid position into future (with correct wrapping)
            auto wrap = [](double coord, double mod) {
                double r = std::fmod(coord, mod);
                return r < 0 ? r + mod : r;
            };
            double asteroid_future_pos_x =
                wrap(a.x + static_cast<double>(future_check_timesteps) * a.vx * DELTA_TIME, game_state.map_size_x);
            double asteroid_future_pos_y =
                wrap(a.y + static_cast<double>(future_check_timesteps) * a.vy * DELTA_TIME, game_state.map_size_y);
            // Fast bounding check (no function call)
            double delta_x = asteroid_future_pos_x - mine_x;
            double delta_y = asteroid_future_pos_y - mine_y;
            double separation = a.radius + (MINE_BLAST_RADIUS - MINE_ASTEROID_COUNT_FUDGE_DISTANCE);
            if (std::abs(delta_x) <= separation &&
                std::abs(delta_y) <= separation &&
                delta_x * delta_x + delta_y * delta_y <= separation * separation)
            {
                ++count;
            }
        }
    }
    return count;
}

inline double predict_ship_mine_collision(
    double ship_pos_x, double ship_pos_y,
    const Mine& mine,
    int64_t future_timesteps = 0)
{
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

    // Compute distances from asteroid center to bullet head and tail
    double bhdx = bullet_head_x - asteroid_x;
    double bhdy = bullet_head_y - asteroid_y;
    double a = std::sqrt(bhdx*bhdx + bhdy*bhdy);

    double btdx = bullet_tail_x - asteroid_x;
    double btdy = bullet_tail_y - asteroid_y;
    double b = std::sqrt(btdx*btdx + btdy*btdy);

    double s = 0.5 * (a + b + BULLET_LENGTH);
    double squared_area = s * (s - a) * (s - b) * (s - BULLET_LENGTH);

    // Heron's height of triangle from asteroid center to bullet line
    double triangle_height = TWICE_BULLET_LENGTH_RECIPROCAL * std::sqrt(std::max(0.0, squared_area));

    return triangle_height < asteroid_radius;
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
        return std::make_tuple(nan, nan, 0, nan, nan, nan);
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
        double theta_old = initial_guess, theta_new, func_value, initial_func_value = nan;
        for (int64_t j = 0; j < max_iterations; ++j) {
            func_value = root_function(theta_old);
            if (std::abs(func_value) < TAD)
                return theta_old;
            if (std::isnan(initial_func_value))
                initial_func_value = func_value;
            double derivative_value = root_function_derivative(theta_old);
            double second_derivative_value = root_function_second_derivative(theta_old);
            double denominator = 2.0 * derivative_value * derivative_value - func_value * second_derivative_value;
            if (denominator == 0.0) return nan;
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
        return nan;
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
        double delta_theta_solution = nan;
        if (std::abs(avx) < GRAIN && std::abs(avy) < GRAIN) {
            delta_theta_solution = std::get<1>(naive_solution);
        } else {
            delta_theta_solution = turbo_rootinator_5000(std::get<1>(naive_solution), TAD, 4);
        }

        if (std::isnan(delta_theta_solution)) {
            return std::make_tuple(false, nan, -1, nan, nan, nan, nan);
        }
        double absolute_theta_solution = delta_theta_solution + theta_0;
        assert(-pi <= delta_theta_solution && delta_theta_solution <= pi); // REMOVE_FOR_COMPETITION

        double delta_theta_solution_deg = delta_theta_solution * RAD_TO_DEG;
        double t_rot = rotation_time(delta_theta_solution);

        assert(is_close(t_rot, std::abs(delta_theta_solution_deg) / SHIP_MAX_TURN_RATE)); // REMOVE_FOR_COMPETITION

        double t_bullet = bullet_travel_time(delta_theta_solution, t_rot);
        if (t_bullet < 0)
            return std::make_tuple(false, nan, -1, nan, nan, nan, nan);

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
                    return std::make_tuple(false, nan, -1, nan, nan, nan, nan);
                assert(t_rot_ts == std::get<2>(discrete_solution)); // REMOVE_FOR_COMPETITION
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
    return std::make_tuple(false, nan, -1, nan, nan, nan, nan);
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
