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

class NeoController {
private:
    int _ship_id = 0;

public:
    NeoController() = default;

    std::tuple<double, double, bool, bool>
    actions(const py::dict& ship_state,
            const py::dict& game_state)
    {
        // ---- Convert incoming Python state dicts to your structs ----
        Ship ship = create_ship_from_dict(ship_state);
        GameState game = create_game_state_from_dict(game_state);

        //std::cout << game << std::endl;

        double thrust = 0.0; // full thrust

        // Find nearest asteroid
        const Asteroid* nearest = nullptr;
        double min_dist2 = std::numeric_limits<double>::infinity();
        for (const auto& ast : game.asteroids) {
            if (!ast.alive) continue;  // only alive
            double dx = ast.x - ship.x;
            double dy = ast.y - ship.y;
            double dist2 = dx*dx + dy*dy;
            if (dist2 < min_dist2) {
                min_dist2 = dist2;
                nearest = &ast;
            }
        }

        double turn_rate = 0.0f;
        if (nearest) {
            double dx = nearest->x - ship.x;
            double dy = nearest->y - ship.y;
            double angle_to_asteroid = std::atan2(dy, dx) * 180.0 / M_PI;  // degrees

            // Compute smallest signed angle difference to current heading (also in degrees)
            double angle_diff = angle_to_asteroid - ship.heading;
            // Normalize to [-180, 180):
            while (angle_diff < -180.0) angle_diff += 360.0;
            while (angle_diff >= 180.0) angle_diff -= 360.0;

            // Clamp to ship's max turn per step, if needed
            // But you may choose to set full turn_rate and let game handle per-timestep clip
            if (angle_diff > SHIP_MAX_TURN_RATE/30.0)
                turn_rate = (double)SHIP_MAX_TURN_RATE;
            else if (angle_diff < -SHIP_MAX_TURN_RATE/30.0)
                turn_rate = (double)-SHIP_MAX_TURN_RATE;
            else
                turn_rate = (double)angle_diff*30.0; // direct, but may want to clamp to max turn per timestep
        }

        bool fire = true;
        bool drop_mine = false;

        //double thrust = 480.0f;
        //double turn_rate = -9.0f;
        //bool fire = true;
        //bool drop_mine = false;

        return std::make_tuple(thrust, turn_rate, fire, drop_mine);
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
};

PYBIND11_MODULE(neo_controller, m) {
    py::class_<NeoController>(m, "NeoController")
        .def(py::init<>())
        .def("actions", &NeoController::actions)
        .def_property_readonly("name", &NeoController::name)
        .def_property("ship_id", &NeoController::ship_id, &NeoController::set_ship_id);
}
