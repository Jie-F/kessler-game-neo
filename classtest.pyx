# distutils: language = c++
# cython: language_level=3
# Save as asteroid.pyx

from libc.math cimport pi, ceil, cos, sin, pi

cdef double FIRE_COOLDOWN_TS = 3
cdef double MINE_COOLDOWN_TS = 30
cdef double FPS = 30.0
cdef double DELTA_TIME = 1.0 / 30.0
cdef double SHIP_FIRE_TIME = 0.1
cdef double BULLET_SPEED = 800.0
cdef double BULLET_MASS = 1.0
cdef double BULLET_LENGTH = 12.0
cdef double BULLET_LENGTH_RECIPROCAL = 1.0 / 12.0
cdef double TWICE_BULLET_LENGTH_RECIPROCAL = 2.0 / 12.0
cdef double SHIP_MAX_TURN_RATE = 180.0
cdef double SHIP_MAX_TURN_RATE_RAD = 180.0 * (pi / 180.0)
cdef double SHIP_MAX_TURN_RATE_RAD_RECIPROCAL = 1.0 / (180.0 * (pi / 180.0))
cdef double SHIP_MAX_TURN_RATE_DEG_TS = (1.0/30.0) * 180.0
cdef double SHIP_MAX_TURN_RATE_RAD_TS = ((1.0/30.0) * 180.0) * (pi / 180.0)
cdef double SHIP_MAX_THRUST = 480.0
cdef double SHIP_DRAG = 80.0
cdef double SHIP_MAX_SPEED = 240.0
cdef double SHIP_RADIUS = 20.0
cdef double SHIP_MASS = 300.0
cdef int TIMESTEPS_UNTIL_SHIP_ACHIEVES_MAX_SPEED = <int>ceil(240.0 / (480.0 - 80.0) * 30.0)
cdef double MINE_BLAST_RADIUS = 150.0
cdef double MINE_RADIUS = 12.0
cdef double MINE_BLAST_PRESSURE = 2000.0
cdef double MINE_FUSE_TIME = 3.0
cdef double MINE_MASS = 25.0
cdef double RESPAWN_INVINCIBILITY_TIME_S = 3.0

ASTEROID_RADII_LOOKUP = (0.0, 8.0, 16.0, 24.0, 32.0)
ASTEROID_AREA_LOOKUP = (pi*0.0**2, pi*8.0**2, pi*16.0**2, pi*24.0**2, pi*32.0**2)
ASTEROID_MASS_LOOKUP = (0.0, 0.25*pi*(8.0*1)**2, 0.25*pi*(8.0*2)**2, 0.25*pi*(8.0*3)**2, 0.25*pi*(8.0*4)**2)
ASTEROID_COUNT_LOOKUP = (0, 1, 4, 13, 40)

DEGREES_BETWEEN_SHOTS = float(FIRE_COOLDOWN_TS) * SHIP_MAX_TURN_RATE * DELTA_TIME
DEGREES_TURNED_PER_TIMESTEP = SHIP_MAX_TURN_RATE * DELTA_TIME
SHIP_RADIUS_PLUS_SIZE_4_ASTEROID_RADIUS = SHIP_RADIUS + ASTEROID_RADII_LOOKUP[4]

cdef class Asteroid:
    cdef public double x
    cdef public double y
    cdef public double vx
    cdef public double vy
    cdef public int    size
    cdef public double mass
    cdef public double radius
    cdef public int    timesteps_until_appearance

    def __cinit__(
        self,
        double x=0.0, double y=0.0,
        double vx=0.0, double vy=0.0,
        int size=0, double mass=0.0,
        double radius=0.0, int timesteps_until_appearance=0,
    ):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.mass = mass
        self.radius = radius
        self.timesteps_until_appearance = timesteps_until_appearance

    # Ultra-fast cdef-level copy
    cdef Asteroid fastcopy(self):
        cdef Asteroid newobj = Asteroid.__new__(Asteroid)
        newobj.x = self.x
        newobj.y = self.y
        newobj.vx = self.vx
        newobj.vy = self.vy
        newobj.size = self.size
        newobj.mass = self.mass
        newobj.radius = self.radius
        newobj.timesteps_until_appearance = self.timesteps_until_appearance
        return newobj

    # Optional: Python-friendly method
    def copy(self):
        """Python-visible copy method, just calls the fast cdef one."""
        return self.fastcopy()

    def __str__(self):
        return (f'Asteroid(x={self.x}, y={self.y}, vx={self.vx}, vy={self.vy}, '
                f'size={self.size}, mass={self.mass}, radius={self.radius}, '
                f'timesteps_until_appearance={self.timesteps_until_appearance})')

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Asteroid):
            return NotImplemented
        return (
            self.x == other.x and
            self.y == other.y and
            self.vx == other.vx and
            self.vy == other.vy and
            self.size == other.size and
            self.mass == other.mass and
            self.radius == other.radius and
            self.timesteps_until_appearance == other.timesteps_until_appearance
        )

    def __hash__(self):
        cdef double combined = self.x + 0.4266548291679171 * self.y + 0.8164926348982552 * self.vx + 0.8397584399461026 * self.vy
        return <int>(combined * 1_000_000_000) + self.size

    # If you want pure C-API versions of the hash helpers:
    cdef double float_hash(self):
        return self.x + 0.4266548291679171 * self.y + 0.8164926348982552 * self.vx + 0.8397584399461026 * self.vy

    cdef int int_hash(self):
        cdef double combined = self.x + 0.4266548291679171 * self.y + 0.8164926348982552 * self.vx + 0.8397584399461026 * self.vy
        return <int>(1_000_000_000 * combined)

# cython: language_level=3

cdef class Ship:
    cdef public bint is_respawning
    cdef public double position0, position1
    cdef public double velocity0, velocity1
    cdef public double speed
    cdef public double heading
    cdef public double mass
    cdef public double radius
    cdef public int id
    cdef public object team # can be str or int for team; use str if needed
    cdef public int lives_remaining
    cdef public int bullets_remaining
    cdef public int mines_remaining
    cdef public bint can_fire
    cdef public double fire_rate
    cdef public bint can_deploy_mine
    cdef public double mine_deploy_rate
    cdef public double thrust_range0, thrust_range1
    cdef public double turn_rate_range0, turn_rate_range1
    cdef public double max_speed
    cdef public double drag

    def __cinit__(self, bint is_respawning=False, tuple position=(0.0,0.0), tuple velocity=(0.0,0.0), double speed=0.0, double heading=0.0, double mass=0.0, double radius=0.0, int id=0, team='', int lives_remaining=0, int bullets_remaining=0, int mines_remaining=0, bint can_fire=True, double fire_rate=0.0, bint can_deploy_mine=True, double mine_deploy_rate=0.0, tuple thrust_range=(-SHIP_MAX_THRUST, SHIP_MAX_THRUST), tuple turn_rate_range=(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE), double max_speed=SHIP_MAX_SPEED, double drag=SHIP_DRAG):
        self.is_respawning = is_respawning; self.position0 = position[0]; self.position1 = position[1]; self.velocity0 = velocity[0]; self.velocity1 = velocity[1]; self.speed = speed; self.heading = heading; self.mass = mass; self.radius = radius; self.id = id; self.team = team; self.lives_remaining = lives_remaining; self.bullets_remaining = bullets_remaining; self.mines_remaining = mines_remaining; self.can_fire = can_fire; self.fire_rate = fire_rate; self.can_deploy_mine = can_deploy_mine; self.mine_deploy_rate = mine_deploy_rate; self.thrust_range0 = thrust_range[0]; self.thrust_range1 = thrust_range[1]; self.turn_rate_range0 = turn_rate_range[0]; self.turn_rate_range1 = turn_rate_range[1]; self.max_speed = max_speed; self.drag = drag

    cdef Ship fastcopy(self):
        cdef Ship s = Ship.__new__(Ship)
        s.is_respawning = self.is_respawning; s.position0 = self.position0; s.position1 = self.position1; s.velocity0 = self.velocity0; s.velocity1 = self.velocity1; s.speed = self.speed; s.heading = self.heading; s.mass = self.mass; s.radius = self.radius; s.id = self.id; s.team = self.team; s.lives_remaining = self.lives_remaining; s.bullets_remaining = self.bullets_remaining; s.mines_remaining = self.mines_remaining; s.can_fire = self.can_fire; s.fire_rate = self.fire_rate; s.can_deploy_mine = self.can_deploy_mine; s.mine_deploy_rate = self.mine_deploy_rate; s.thrust_range0 = self.thrust_range0; s.thrust_range1 = self.thrust_range1; s.turn_rate_range0 = self.turn_rate_range0; s.turn_rate_range1 = self.turn_rate_range1; s.max_speed = self.max_speed; s.drag = self.drag
        return s

    def copy(self):
        return self.fastcopy()

    def __str__(self):
        return f'Ship(is_respawning={self.is_respawning}, position=({self.position0}, {self.position1}), velocity=({self.velocity0}, {self.velocity1}), speed={self.speed}, heading={self.heading}, mass={self.mass}, radius={self.radius}, id={self.id}, team="{self.team}", lives_remaining={self.lives_remaining}, bullets_remaining={self.bullets_remaining}, mines_remaining={self.mines_remaining}, can_fire={self.can_fire}, fire_rate={self.fire_rate}, can_deploy_mine={self.can_deploy_mine}, mine_deploy_rate={self.mine_deploy_rate}, thrust_range=({self.thrust_range0},{self.thrust_range1}), turn_rate_range=({self.turn_rate_range0},{self.turn_rate_range1}), max_speed={self.max_speed}, drag={self.drag})'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Ship): return NotImplemented
        return self.is_respawning == other.is_respawning and self.position0 == other.position0 and self.position1 == other.position1 and self.velocity0 == other.velocity0 and self.velocity1 == other.velocity1 and self.speed == other.speed and self.heading == other.heading and self.mass == other.mass and self.radius == other.radius and self.id == other.id and self.team == other.team and self.lives_remaining == other.lives_remaining and self.bullets_remaining == other.bullets_remaining and self.mines_remaining == other.mines_remaining and self.fire_rate == other.fire_rate and self.mine_deploy_rate == other.mine_deploy_rate and self.thrust_range0 == other.thrust_range0 and self.thrust_range1 == other.thrust_range1 and self.turn_rate_range0 == other.turn_rate_range0 and self.turn_rate_range1 == other.turn_rate_range1 and self.max_speed == other.max_speed and self.drag == other.drag


cdef class Mine:
    cdef public double pos_x, pos_y, mass, fuse_time, remaining_time

    def __cinit__(self, position=(0.0, 0.0), double mass=0.0, double fuse_time=0.0, double remaining_time=0.0):
        self.pos_x = position[0]; self.pos_y = position[1]; self.mass = mass; self.fuse_time = fuse_time; self.remaining_time = remaining_time

    property position:
        def __get__(self): return (self.pos_x, self.pos_y)
        def __set__(self, v): self.pos_x = v[0]; self.pos_y = v[1]

    def __str__(self): return "Mine(position=(%f, %f), mass=%f, fuse_time=%f, remaining_time=%f)" % (self.pos_x, self.pos_y, self.mass, self.fuse_time, self.remaining_time)
    def __repr__(self): return self.__str__()

    cdef Mine fastcopy(self):
        cdef Mine m = Mine.__new__(Mine)
        m.pos_x = self.pos_x; m.pos_y = self.pos_y; m.mass = self.mass; m.fuse_time = self.fuse_time; m.remaining_time = self.remaining_time
        return m

    def copy(self): return self.fastcopy()

    def __eq__(self, other):
        if not isinstance(other, Mine): return NotImplemented
        return (self.pos_x, self.pos_y) == other.position and self.mass == other.mass and self.fuse_time == other.fuse_time and self.remaining_time == other.remaining_time


cdef class Bullet:
    cdef public double pos_x, pos_y, vel_x, vel_y, heading, mass, tail_dx, tail_dy

    def __cinit__(self, position=(0.0, 0.0), velocity=(0.0,0.0), double heading=0.0, double mass=BULLET_MASS, tail_delta=None):
        self.pos_x = position[0]; self.pos_y = position[1]; self.vel_x = velocity[0]; self.vel_y = velocity[1]; self.heading = heading; self.mass = mass
        if tail_delta is not None:
            self.tail_dx = tail_delta[0]; self.tail_dy = tail_delta[1]
        else:
            angle_rad = heading * (pi / 180.0)
            self.tail_dx = -BULLET_LENGTH * cos(angle_rad); self.tail_dy = -BULLET_LENGTH * sin(angle_rad)

    property position:
        def __get__(self): return (self.pos_x, self.pos_y)
        def __set__(self, v): self.pos_x = v[0]; self.pos_y = v[1]

    property velocity:
        def __get__(self): return (self.vel_x, self.vel_y)
        def __set__(self, v): self.vel_x = v[0]; self.vel_y = v[1]

    property tail_delta:
        def __get__(self): return (self.tail_dx, self.tail_dy)
        def __set__(self, v): self.tail_dx = v[0]; self.tail_dy = v[1]

    def __str__(self): return f'Bullet(position=({self.pos_x}, {self.pos_y}), velocity=({self.vel_x}, {self.vel_y}), heading={self.heading}, mass={self.mass}, tail_delta=({self.tail_dx}, {self.tail_dy}))'
    def __repr__(self): return self.__str__()

    cdef Bullet fastcopy(self):
        cdef Bullet b = Bullet.__new__(Bullet)
        b.pos_x = self.pos_x; b.pos_y = self.pos_y; b.vel_x = self.vel_x; b.vel_y = self.vel_y; b.heading = self.heading; b.mass = self.mass; b.tail_dx = self.tail_dx; b.tail_dy = self.tail_dy
        return b

    def copy(self): return self.fastcopy()

    def __eq__(self, other):
        if not isinstance(other, Bullet): return NotImplemented
        return (self.pos_x, self.pos_y) == other.position and (self.vel_x, self.vel_y) == other.velocity and self.heading == other.heading and self.mass == other.mass and (self.tail_dx, self.tail_dy) == other.tail_delta

cdef class GameState:
    cdef public object asteroids, ships, bullets, mines
    cdef public double map_size_0, map_size_1, time, delta_time, time_limit
    cdef public int sim_frame

    def __cinit__(self, asteroids, ships, bullets, mines, map_size=(0.0, 0.0), double time=0.0, double delta_time=0.0, int sim_frame=0, double time_limit=0.0):
        self.asteroids = asteroids; self.ships = ships; self.bullets = bullets; self.mines = mines; self.map_size_0 = map_size[0]; self.map_size_1 = map_size[1]; self.time = time; self.delta_time = delta_time; self.sim_frame = sim_frame; self.time_limit = time_limit

    property map_size:
        def __get__(self): return (self.map_size_0, self.map_size_1)
        def __set__(self, v): self.map_size_0 = v[0]; self.map_size_1 = v[1]

    def __str__(self): return f'GameState(asteroids={self.asteroids}, ships={self.ships}, bullets={self.bullets}, mines={self.mines}, map_size=({self.map_size_0}, {self.map_size_1}), time={self.time}, delta_time={self.delta_time}, sim_frame={self.sim_frame}, time_limit={self.time_limit})'
    def __repr__(self): return self.__str__()

    def copy(self):
        return GameState([asteroid.copy() for asteroid in self.asteroids], [ship.copy() for ship in self.ships], [bullet.copy() for bullet in self.bullets], [mine.copy() for mine in self.mines], (self.map_size_0, self.map_size_1), self.time, self.delta_time, self.sim_frame, self.time_limit)

    def __eq__(self, other):
        if not isinstance(other, GameState): return NotImplemented
        if len(self.asteroids) != len(other.asteroids): print("Asteroids lists are different lengths!"); return False
        for i, (ast_a, ast_b) in enumerate(zip(self.asteroids, other.asteroids)):
            if ast_a != ast_b: print(f"Asteroids don't match at index {i}: {ast_a} vs {ast_b}"); return False
        if len(self.bullets) != len(other.bullets): print("Bullet lists are different lengths!"); return False
        for i, (bul_a, bul_b) in enumerate(zip(self.bullets, other.bullets)):
            if bul_a != bul_b: print(f"Bullets don't match at index {i}: {bul_a} vs {bul_b}"); return False
        if len(self.mines) != len(other.mines): print("Mine lists are different lengths!"); return False
        for i, (mine_a, mine_b) in enumerate(zip(self.mines, other.mines)):
            if mine_a != mine_b: print(f"Mines don't match at index {i}: {mine_a} vs {mine_b}"); return False
        # Ships comparison rewritten but commented, as requested:
        #if len(self.ships) != len(other.ships): print("Ships lists are different lengths!"); return False
        #for i, (ship_a, ship_b) in enumerate(zip(self.ships, other.ships)):
        #    if ship_a != ship_b: print(f"Ships don't match at index {i}: {ship_a} vs {ship_b}"); return False
        return True
cdef class Target:
    cdef public Asteroid asteroid
    cdef public bint feasible
    cdef public double shooting_angle_error_deg
    cdef public int aiming_timesteps_required
    cdef public double interception_time_s
    cdef public double intercept_x
    cdef public double intercept_y
    cdef public double asteroid_dist_during_interception
    cdef public double imminent_collision_time_s
    cdef public bint asteroid_will_get_hit_by_my_mine
    cdef public bint asteroid_will_get_hit_by_their_mine

    def __cinit__(self, Asteroid asteroid, bint feasible=False, double shooting_angle_error_deg=0.0, int aiming_timesteps_required=0, double interception_time_s=0.0, double intercept_x=0.0, double intercept_y=0.0, double asteroid_dist_during_interception=0.0, double imminent_collision_time_s=0.0, bint asteroid_will_get_hit_by_my_mine=False, bint asteroid_will_get_hit_by_their_mine=False):
        self.asteroid=asteroid; self.feasible=feasible; self.shooting_angle_error_deg=shooting_angle_error_deg; self.aiming_timesteps_required=aiming_timesteps_required; self.interception_time_s=interception_time_s; self.intercept_x=intercept_x; self.intercept_y=intercept_y; self.asteroid_dist_during_interception=asteroid_dist_during_interception; self.imminent_collision_time_s=imminent_collision_time_s; self.asteroid_will_get_hit_by_my_mine=asteroid_will_get_hit_by_my_mine; self.asteroid_will_get_hit_by_their_mine=asteroid_will_get_hit_by_their_mine

    def __str__(self): return f'Target(asteroid={self.asteroid}, feasible={self.feasible}, shooting_angle_error_deg={self.shooting_angle_error_deg}, aiming_timesteps_required={self.aiming_timesteps_required}, interception_time_s={self.interception_time_s}, intercept_x={self.intercept_x}, intercept_y={self.intercept_y}, asteroid_dist_during_interception={self.asteroid_dist_during_interception}, imminent_collision_time_s={self.imminent_collision_time_s}, asteroid_will_get_hit_by_my_mine={self.asteroid_will_get_hit_by_my_mine}, asteroid_will_get_hit_by_their_mine={self.asteroid_will_get_hit_by_their_mine})'
    def __repr__(self): return self.__str__()

    cdef Target fastcopy(self):
        cdef Target t = Target.__new__(Target)
        t.asteroid=self.asteroid.copy(); t.feasible=self.feasible; t.shooting_angle_error_deg=self.shooting_angle_error_deg; t.aiming_timesteps_required=self.aiming_timesteps_required; t.interception_time_s=self.interception_time_s; t.intercept_x=self.intercept_x; t.intercept_y=self.intercept_y; t.asteroid_dist_during_interception=self.asteroid_dist_during_interception; t.imminent_collision_time_s=self.imminent_collision_time_s; t.asteroid_will_get_hit_by_my_mine=self.asteroid_will_get_hit_by_my_mine; t.asteroid_will_get_hit_by_their_mine=self.asteroid_will_get_hit_by_their_mine
        return t

    def copy(self): return self.fastcopy()
