import functools
import random
from copy import copy
import colorsys
import time
from typing import Optional
from importlib import resources as importlib_resources
import math
import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
import pygame
from Box2D import b2World, b2PolygonShape
import Box2D
import os


class BoxAgent:
    def __init__(self, body: Box2D.b2Body, h: float = 0.666, v: float = 1, s: float = 1):
        self.body = body
        self.yvels = []
        self.n = 10
        self.fallen = False
        self.last_stood_on = 'floor'  # index of supporting box or 'floor'

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        self.colour = (int(r*255), int(g*255), int(b*255))

    def step(self):
        self.yvels.append(abs(self.body.linearVelocity.y))
        if len(self.yvels) > self.n:
            self.yvels.pop(0)

    def can_jump(self):
        return len(self.yvels) >= self.n and max(self.yvels) < 0.5


# class ContactDetector(contactListener):
#     def __init__(self, env):
#         contactListener.__init__(self)
#         self.env = env
#
#     def BeginContact(self, contact):
#         boxes = [b.body for b in self.env.boxes]
#         if contact.fixtureA.body in boxes and contact.fixtureB.body.is_grounded:
#             contact.fixtureA.body.is_grounded = True
#         elif contact.fixtureB.body in boxes:
#             contact.fixtureA.body.is_grounded = True
#             self.env.game_over = True
#         for i in range(2):
#             if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
#                 self.env.legs[i].ground_contact = True
#
#     def EndContact(self, contact):
#         for i in range(2):
#             if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
#                 self.env.legs[i].ground_contact = False


VIEWPORT_W = 800
VIEWPORT_H = 600
SCALE = 30 # Scale from pygame units to pixels
FPS = 40
MAX_HOR_SPD = 2
NEGATIVE_THRESHOLD = -0.001

# H, W are in (Py)game units
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE

FLOOR_Y = 15


class BoxJumpEnvironment(ParallelEnv):
    """
    Reward modes:
        highest        - reward at time t = max(new best height - old best height, 0)
                         meaning rewards are given when a new best height is achieved in the current frame
        highest_stable - same as highest but only considers boxes which are stable i.e. not mid-jump
        height_sq      - each agent rewarded by its current height squared at each step
        stable_sum     - same as height_sq but reward is 0 for agents which are currently falling. multiplied by 50 to
                          achieve a similar order of magnitude to height_sq.
        dense_height   - each agent rewarded by the maximum height of all boxes at each step
        dense_height_stable - same as dense_height but only considers boxes which are stable i.e. not mid-jump
        dense_height_sq   - each agent rewarded by the maximum height of all boxes at each step squared
        dense_height_stable_sq - same as dense_height_sq but only considers boxes which are stable i.e. not mid-jump
        dense_height_cube - each agent rewarded by the maximum height of all boxes at each step cubed
        dense_height_stable_cube - same as dense_height_cube but only considers boxes which are stable i.e. not mid-jump

        A total reward (with highest mode) of N means that a height of N boxes above the floor was achieved.
    """

    metadata = {
        "name": "boxjump_v0",
    }

    def __init__(self, num_boxes=4, world_width=10, world_height=6, box_width=1, box_height=1, render_mode=None,
                 gravity=10, friction=0.8, spacing=1.5, random_spacing=0.5, angular_damping=1, agent_one_hot=False,
                 max_timestep=400, fixed_rotation=True, reward_mode: str = "highest", include_time: bool = False,
                 include_highest: bool = False, penalty_fall: float = 20, physics_steps_per_action=6, physics_timestep_multiplier=2.0,
                 include_num_boxes: bool = True, termination_max_height: Optional[float] = None, termination_reward_coef: float = 0.0,
                 termination_on_fall: bool = True, penalty_fall_termination: float = -100.0,
                 reward_only_biggest_tower: bool = True):
        """
        :param num_boxes: Number of agents.
        :param world_width:Environment width.
        :param world_height: Environment height.
        :param box_width: Width of each agent.
        :param box_height: Height of each agent.
        :param render_mode: None or "human" - None does not render, while "human" uses PyGame to render.
        :param gravity: Gravity force.
        :param friction: Friction force.
        :param spacing: Horizontal spacing at environment start.
        :param random_spacing: Randomness in horizontal spacing at environment start (shouldn't be over half spacing, or overlaps will occur).
        :param angular_damping: Angular damping to prevent too much spinning.
        :param agent_one_hot: Whether or not to include one-hot agent representations in the observations.
        :param max_timestep: Length of an episode.
        :param fixed_rotation: If true, the boxes cannot rotate. This makes the environment a lot easier.
        :param reward_mode: highest / highest_stable / height_sq / stable_sum. See BoxJumpEnvironment docstring.
        :param include_time: Whether to include the elapsed time (normalised 0 to 1) in the observations.
        :param include_highest: Whether to include the best height achieved so far in the observations.
        :param penalty_fall: How much of a reward penalty should be given for falling off the edge of the map. Applied every step (divided by max_timestep) to agents off the edge.
        :param physics_steps_per_action: Number of physics steps to run per environment step (action repetition). Higher values make each action have more impact.
        :param physics_timestep_multiplier: Multiplier for physics timestep. Higher values make time pass faster per physics step.
        :param include_num_boxes: Whether to append the number of boxes to obs/state.
        :param termination_max_height: Early terminate when max height is reached (None disables).
        :param termination_reward_coef: Multiplier applied to current step rewards at termination_max_height.
        :param termination_on_fall: If True, terminate when any agent leaves the horizontal area [0, W].
        :param penalty_fall_termination: Penalty to the fallen agent when termination_on_fall is True.
        :param reward_only_biggest_tower: If True, only agents in the tallest tower (by last_stood_on chain) receive rewards each step.
        """
        self.num_boxes = num_boxes
        self.width = world_width
        self.height = world_height
        self.box_width = box_width
        self.box_height = box_height
        self.gravity = gravity
        self.friction = friction
        self.spacing = spacing
        self.random_spacing = random_spacing
        self.angular_damping = angular_damping
        self.agent_one_hot = agent_one_hot
        self.max_timestep = max_timestep
        self.fixed_rotation = fixed_rotation
        self.penalty_fall = penalty_fall
        self.physics_steps_per_action = physics_steps_per_action
        self.physics_timestep_multiplier = physics_timestep_multiplier
        self.include_num_boxes = include_num_boxes
        self.termination_max_height = termination_max_height
        self.termination_reward_coef = termination_reward_coef
        self.termination_on_fall = termination_on_fall
        self.penalty_fall_termination = penalty_fall_termination
        self.reward_only_biggest_tower = reward_only_biggest_tower

        self.include_time = include_time
        self.include_highest = include_highest

        assert reward_mode in [
            "highest",
            "highest_stable",
            "height_sq",
            "stable_sum",
            "dense_height",
            "dense_height_stable",
            "dense_height_sq",
            "dense_height_stable_sq",
        ]
        self.reward_mode = reward_mode

        low = [-1, -1, -2, -2, -0.5, -2, 0, 0, 0, 0, 0]
        high = [1, self.num_boxes, 2, 2, 0.5, 2, 1, 1, 1, 1, 1]
        if self.include_highest:
            # current best height: from 0..1
            low.append(0)
            high.append(self.num_boxes)
        if self.include_time:
            # time elapsed: from 0..1
            low.append(0)
            high.append(1)
        if self.agent_one_hot:
            low += [0] * num_boxes
            high += [1] * num_boxes
        if self.include_num_boxes:
            low.append(1)
            high.append(self.num_boxes)
        size = len(low)
        low, high = np.array(low), np.array(high)
        self.obs_space = Box(low, high, shape=[size])

        # Construct state_space as a concatenation of subsets of all obs_spaces
        #  (basically just discards per-agent max ob, and per-agent one-hot ob)
        self.keep_inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        low_state = np.repeat(low[self.keep_inds], num_boxes)
        high_state  = np.repeat(high[self.keep_inds], num_boxes)

        if self.include_highest:
            low_state = np.concatenate((low_state, [0]))
            high_state = np.concatenate((high_state, [self.num_boxes]))

        if self.include_time:
            low_state = np.concatenate((low_state, [0]))
            high_state = np.concatenate((high_state, [1]))
        if self.agent_one_hot:
            # Reserve space in state for one-hot features if enabled
            low_state = np.concatenate((low_state, [0] * num_boxes))
            high_state = np.concatenate((high_state, [1] * num_boxes))
        if self.include_num_boxes:
            low_state = np.concatenate((low_state, [self.num_boxes]))
            high_state = np.concatenate((high_state, [self.num_boxes]))

        size = low_state.shape[0]
        self.state_space = Box(low=low_state, high=high_state, shape=[size])

        self.boxes = None
        self.world = None
        self._state = None
        self.possible_agents = [f"box-{i}" for i in range(1, num_boxes+1)]

        self.render_mode = render_mode
        self.screen = None
        self.highest_y = 0
        
        # Load emotion images
        self.emotion_images = {}
        # We'll load images later when pygame is initialized

    def _load_emotion_images(self):
        """Load emotion PNG images packaged with the library, independent of CWD."""
        emotion_files = {
            'happy': 'happy.png',
            'squished': 'squished.png', 
            'jumping': 'jumping.png',
            'super_happy': 'super_happy.png'
        }

        for emotion, filename in emotion_files.items():
            image = self._load_image_resource(filename)
            if image is None:
                print(f"Warning: Could not load emotion image '{filename}' from packaged assets")
                # Create a simple transparent fallback surface
                image = pygame.Surface((int(self.box_width * SCALE), int(self.box_height * SCALE)), pygame.SRCALPHA)
                image.fill((255, 255, 255, 0))

            # Scale image to match box size using high-quality scaling
            scaled_image = pygame.transform.smoothscale(
                image,
                (int(self.box_width * SCALE), int(self.box_height * SCALE))
            )
            self.emotion_images[emotion] = scaled_image

    def _load_image_resource(self, name: str):
        """Try to load an image from package resources, fallback to module dir and CWD.
        Expects assets to be installed under the package path: boxjump/assets/<name>.
        """
        # 1) importlib.resources from this package
        try:
            res = importlib_resources.files(__package__).joinpath('assets', name)
            with importlib_resources.as_file(res) as p:
                if os.path.exists(p):
                    return pygame.image.load(str(p)).convert_alpha()
        except Exception:
            pass

        # 2) Relative to this module file (useful for editable installs)
        try:
            module_dir = os.path.dirname(__file__)
            fs_path = os.path.abspath(os.path.join(module_dir, '..', 'assets', name))
            if os.path.exists(fs_path):
                return pygame.image.load(fs_path).convert_alpha()
        except Exception:
            pass

        # 3) CWD fallback (last resort)
        try:
            fs2 = os.path.join('assets', name)
            if os.path.exists(fs2):
                return pygame.image.load(fs2).convert_alpha()
        except Exception:
            pass

        return None

    @functools.lru_cache(maxsize=None)
    def action_space(self, _):
        return Discrete(4)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, _):
        return self.obs_space

    def state(self):
        return self._state

    def get_all_obs(self):
        # obs =
        #  position x
        #  position y
        #  velocity x
        #  velocity y
        #  angle
        #  angular velocity
        #  is_standing_on_surface
        #  dist left (float)
        #  dist right (float)
        #  dist below (float)
        #  dist above (float)

        xs = np.array([box.body.transform.position.x for box in self.boxes])
        ys = np.array([box.body.transform.position.y for box in self.boxes])

        # xs_i = np.argsort(xs)
        # xs_s = xs[xs_i] / W
        # left = np.concatenate(([1], xs_s[1:] - xs_s[:-1]))[xs_i]
        # right = np.concatenate((xs_s[1:] - xs_s[:-1], [1]))[xs_i]

        # Todo: O(N^2) algo below could be replaced with a more efficient left-to-right sweep algorithm
        above = []
        below = []

        left = []
        right = []

        h = self.box_height / 2
        w = self.box_width / 2
        threshold = 0.1  # how much they must overlap by to register

        for i in range(self.num_boxes):
            # Extract boxes which overlap horizontally with this box
            x = xs[i]
            y = ys[i]
            relevant = np.logical_and(xs - w < x + w - threshold, xs + w > x - w + threshold)
            if np.sum(relevant) == 1:
                above.append(1)
                below.append((FLOOR_Y - y) / H)
            else:
                # above means SMALLER y value
                abv = np.logical_and(relevant, ys < y)
                blw = np.logical_and(relevant, ys > y)

                abv = ((y - np.max(ys[abv])) / H) if np.sum(abv)>0 else 1
                blw = ((np.min(ys[blw]) - y) / H) if np.sum(blw) > 0 else (FLOOR_Y - y) / H

                above.append(abv)
                below.append(blw)

            # Repeat for left/right
            relevant = np.logical_and(ys - h < y + h - threshold, ys + h > y - h + threshold)
            if np.sum(relevant) == 1:
                left.append(1)
                right.append(1)
            else:
                lft = np.logical_and(relevant, xs < x)
                rgt = np.logical_and(relevant, xs > x)

                lft = ((x - np.max(xs[lft])) / W) if np.sum(lft) > 0 else 1
                rgt = ((np.min(xs[rgt]) - x) / W) if np.sum(rgt) > 0 else 1

                left.append(lft)
                right.append(rgt)

        # Update last_stood_on only when truly touching a support or floor; otherwise keep previous
        for i in range(self.num_boxes):
            support_idx = self._get_support_below_touching(i)
            if support_idx is not None:
                self.boxes[i].last_stood_on = support_idx
            else:
                # Check floor touching
                eps = 0.05 * self.box_height
                if abs(ys[i] - (FLOOR_Y - self.box_height / 2)) <= eps:
                    self.boxes[i].last_stood_on = 'floor'

        obs = {}
        state = []
        for i in range(self.num_boxes):
            b: BoxAgent = self.boxes[i]

            # A rotation of 90 degrees is equivalent to a rotation of 0 degrees for these boxes
            #  so just report the number of quarter turns from -1/2 to 1/2, with 0=no rotation
            quarter_turns = (b.body.angle / (math.pi / 2))  # number of quarter turns
            quarter_turns = ((quarter_turns + 0.5) % 1.0) - 0.5

            # (red if can jump, blue otherwise)
            # self.boxes[i].colour = (255, 0, 0) if int(b.can_jump()) else (0, 0, 255)

            # 0 = you are on the floor
            # n = you are n boxes in height (so one box standing on another gets a score of 1)
            height_above_floor = (FLOOR_Y - b.body.position.y - self.box_height / 2) / self.box_height
            if height_above_floor < NEGATIVE_THRESHOLD:
                height_above_floor = -1

            # -1 = left, 0 = middle, 1 = right
            xpos = (b.body.position.x / W) * 2 - 1

            ob = [xpos,
                  height_above_floor,  # y position, -1 to n
                  b.body.linearVelocity.x / FPS * 20,  # constants chosen to scale approxmiately to [-1, 1]
                  b.body.linearVelocity.y / FPS * 5,
                  quarter_turns,
                  b.body.angularVelocity / FPS * 20,
                  left[i],   # distance to closest box on the left
                  right[i],  # distance to closest box on the right
                  above[i],  # distance to closest box above which overlaps horizontally
                  below[i],  # distance to closest box below which overlaps horizontally
                  int(b.can_jump())
            ]

            if self.include_highest:
                ob.append(self.highest_y)
            if self.include_time:
                ob.append(self.timestep / self.max_timestep)

            if self.agent_one_hot:
                ob += [0] * i + [1] + [0] * (self.num_boxes - i - 1)
            if self.include_num_boxes:
                ob.append(self.num_boxes)

            ob = np.array(ob, dtype=np.float32)
            obs[self.agents[i]] = ob
            state.append(ob[self.keep_inds])

        if self.include_highest:
            state.append(np.array([self.highest_y]))  # only include once
        if self.include_time:
            state.append(np.array([self.timestep / self.max_timestep]))  # only include once
        if self.include_num_boxes:
            state.append(np.array([self.num_boxes]))  # only include once
        self._state = np.concatenate(state)

        return obs

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.np_random = np.random.default_rng(seed)
        self._state = None
        self.world = b2World(gravity=(0, self.gravity), doSleep=True)

        self.highest_y = 0

        if self.render_mode == "human" and self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            # Load emotion images if not already loaded
            if not hasattr(self, 'emotion_images') or not self.emotion_images:
                self._load_emotion_images()

        # Make boxes
        self.boxes = []
        total_x = self.spacing * (self.num_boxes - 1)
        x = (W / 2) - (total_x / 2)
        start_y = FLOOR_Y - self.box_height / 2
        for i in range(self.num_boxes):
            xpos = x + self.np_random.uniform(-self.random_spacing, self.random_spacing)

            body = self.world.CreateDynamicBody(position=(xpos, start_y))
            shape = b2PolygonShape(box=(self.box_width / 2, self.box_height / 2))
            body.CreateFixture(shape=shape, density=1, friction=self.friction)
            x += self.spacing
            body.angularDamping = self.angular_damping
            body.fixedRotation = self.fixed_rotation

            # (a nice blue gradient I made up)
            hue = (i / self.num_boxes) * 0.2 + 0.5
            val = (i / self.num_boxes) * 0.2 + 0.8
            self.boxes.append(BoxAgent(body, hue, val))

        # Make floor
        floor_body = self.world.CreateStaticBody(position=(0, H))
        floor_shape = b2PolygonShape(box=(W, (H - FLOOR_Y)))
        floor_body.CreateFixture(shape=floor_shape)

        obs = self.get_all_obs()
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        # Initialize accumulated rewards
        accumulated_rewards = {agent: 0.0 for agent in self.agents}

        # Run multiple physics steps per action
        for physics_step in range(self.physics_steps_per_action):
            step_rewards, done = self._single_physics_step(actions)

            # Accumulate rewards
            for agent in self.agents:
                if agent in step_rewards:
                    accumulated_rewards[agent] += step_rewards[agent]

            if done:
                break

        # Only increment timestep once per environment step
        self.timestep += 1

        # Check termination/truncation conditions
        terminations = {a: done for a in self.agents}
        truncations = {a: False for a in self.agents}
        if self.timestep > self.max_timestep:
            accumulated_rewards = {a: 0 for a in self.agents}
            truncations = {a: True for a in self.agents}

        # Get final observation after all physics steps
        obs = self.get_all_obs()
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return obs, accumulated_rewards, terminations, truncations, infos

    def _single_physics_step(self, actions):
        """Run a single physics step with the given actions.
        Returns: (rewards_dict, done_bool)
        """
        step_start_time = time.time()
        
        rewards = {}
        done = False

        prev_best = self.highest_y
        new_best = self.highest_y
        max_height_above_floor = -1

        # Apply actions to all agents
        for idx, i in enumerate(self.agents):
            action = actions[i]
            self.boxes[idx].step()
            body = self.boxes[idx].body
            if action == 0:
                pass
            elif action in [1, 2]:
                mul = -1 if action == 1 else 1
                if not ((mul > 0 and body.linearVelocity.x > MAX_HOR_SPD) or (mul < 0 and body.linearVelocity.x < -MAX_HOR_SPD)):
                    body.ApplyForceToCenter((mul * 10 / self.physics_timestep_multiplier, 0), True)
            elif action == 3:
                if self.boxes[idx].can_jump():
                    body.ApplyForceToCenter((0, -250 / self.physics_timestep_multiplier), True)

            # Calculate height for reward calculations
            height_above_floor = (FLOOR_Y - self.boxes[idx].body.position.y - self.box_height / 2) / self.box_height

            if self.reward_mode == "highest_stable":
                valid = self.boxes[idx].can_jump()
            elif self.reward_mode == "highest":
                valid = True
            else:
                valid = True

            if height_above_floor > new_best and valid:
                new_best = height_above_floor

            max_height_above_floor = max(max_height_above_floor, height_above_floor)

        # Calculate rewards for all agents
        for ind, i in enumerate(self.agents):
            height_above_floor = (FLOOR_Y - self.boxes[ind].body.position.y - self.box_height / 2) / self.box_height
            
            # Terminate or penalize when leaving horizontal bounds (fall off area on x-axis)
            pos_x = self.boxes[ind].body.position.x
            offscreen_horizontal = (pos_x < 0) or (pos_x > W)
            if offscreen_horizontal:
                if self.termination_on_fall:
                    rewards[i] = self.penalty_fall_termination
                    self.boxes[ind].fallen = True
                    done = True
                    break
                else:
                    # Continuous penalty mode
                    rewards[i] = -self.penalty_fall / self.max_timestep / self.physics_steps_per_action
                self.boxes[ind].fallen = True
                continue

            if self.reward_mode == "highest" or self.reward_mode == "highest_stable":
                rewards[i] = (new_best - prev_best) / self.num_boxes / self.physics_steps_per_action
            elif self.reward_mode == "height_sq":
                rewards[i] = (height_above_floor ** 2) / self.num_boxes / self.max_timestep / self.physics_steps_per_action
            elif self.reward_mode == "stable_sum":
                if self.boxes[ind].can_jump():
                    rewards[i] = 50 * (height_above_floor ** 2) / self.num_boxes / self.max_timestep / self.physics_steps_per_action
                else:
                    rewards[i] = 0
            elif self.reward_mode == "dense_height":
                rewards[i] = max_height_above_floor / self.num_boxes / self.max_timestep * 50 / self.physics_steps_per_action
            elif self.reward_mode == "dense_height_stable":
                max_stable_height = 0
                for box_idx, box in enumerate(self.boxes):
                    if box.can_jump():
                        box_height = (FLOOR_Y - box.body.position.y - self.box_height / 2) / self.box_height
                        max_stable_height = max(max_stable_height, box_height)
                rewards[i] = max_stable_height / self.num_boxes / self.max_timestep * 50 / self.physics_steps_per_action
            elif self.reward_mode == "dense_height_sq":
                exp_reward = (max_height_above_floor ** 2) / self.num_boxes / self.max_timestep * 10
                rewards[i] = min(exp_reward, 1000.0) / self.physics_steps_per_action
            elif self.reward_mode == "dense_height_stable_sq":
                max_stable_height = 0
                for box_idx, box in enumerate(self.boxes):
                    if box.can_jump():
                        box_height = (FLOOR_Y - box.body.position.y - self.box_height / 2) / self.box_height
                        max_stable_height = max(max_stable_height, box_height)
                exp_reward = (max_stable_height ** 2) / self.num_boxes / self.max_timestep * 10
                rewards[i] = min(exp_reward, 1000.0) / self.physics_steps_per_action
            elif self.reward_mode == "dense_height_cube":
                exp_reward = (max_height_above_floor ** 3) / self.num_boxes / self.max_timestep * 10
                rewards[i] = min(exp_reward, 1000.0) / self.physics_steps_per_action
            elif self.reward_mode == "dense_height_stable_cube":
                max_stable_height = 0
                for box_idx, box in enumerate(self.boxes):
                    if box.can_jump():
                        box_height = (FLOOR_Y - box.body.position.y - self.box_height / 2) / self.box_height
                        max_stable_height = max(max_stable_height, box_height)
                exp_reward = (max_stable_height ** 3) / self.num_boxes / self.max_timestep * 10
                rewards[i] = min(exp_reward, 1000.0) / self.physics_steps_per_action

        # If reward_only_biggest_tower: zero rewards for agents not in tallest tower chain
        if getattr(self, 'reward_only_biggest_tower', False):
            tower_indices = self._get_tower_indices()
            for idx, a in enumerate(self.agents):
                if idx not in tower_indices:
                    rewards[a] = 0.0

        # Step the physics simulation with modified timestep
        timestep = (1 / FPS) * self.physics_timestep_multiplier
        self.world.Step(timestep, 30, 30)

        # Update highest achieved
        self.highest_y = new_best

        # Terminate early on reaching max height, apply termination reward multiplier
        if (self.termination_max_height is not None) and (new_best >= self.termination_max_height):
            # Multiply current step rewards by termination_reward_coef and mark done
            rewards = {agent: rewards.get(agent, 0.0) * self.termination_reward_coef for agent in self.agents}
            done = True

        # Render after each physics step to maintain smooth animation
        if self.render_mode == "human":
            self.render()

            # Sleep to maintain target FPS timing for each physics step
            step_duration = time.time() - step_start_time
            target_step_time = 1 / FPS
            if step_duration < target_step_time:
                time.sleep(target_step_time - step_duration)

        return rewards, done

    def _get_box_emotion(self, box_idx):
        """Determine the emotion for a given box based on its state"""
        box = self.boxes[box_idx]
        
        # Get height above floor
        height_above_floor = (FLOOR_Y - box.body.position.y - self.box_height / 2) / self.box_height
        
        # Check if this box is the highest
        max_height = -1
        for b in self.boxes:
            b_height = (FLOOR_Y - b.body.position.y - self.box_height / 2) / self.box_height
            max_height = max(max_height, b_height)
        
        is_highest = (abs(height_above_floor - max_height) < 0.1 and height_above_floor > 0.1)
        
        # Check if jumping (in the air - can't jump means not stable/grounded)
        is_jumping = not box.can_jump()
        
        # Check if squished - there's a box above pressing down
        is_squished = self._has_box_above(box_idx)
        
        # Determine emotion based on priority
        if is_highest and height_above_floor > 0.9:  # Only super happy if significantly high
            return 'super_happy'
        elif is_squished:  # Squished has higher priority than jumping
            return 'squished'
        elif is_jumping:  # In air and not squished
            return 'jumping'
        else:  # on floor or default state
            return 'happy'
    
    def _has_box_above(self, box_idx):
        """Check if there's a touching box above this one (using same touching criteria)."""
        current_box = self.boxes[box_idx]
        current_x = current_box.body.position.x
        current_y = current_box.body.position.y
        h = self.box_height / 2
        w = self.box_width / 2

        eps = 0.05 * self.box_height  # touching tolerance
        for i, other_box in enumerate(self.boxes):
            if i == box_idx:
                continue
            other_x = other_box.body.position.x
            other_y = other_box.body.position.y

            horizontal_overlap = (abs(other_x - current_x) <= (self.box_width - eps))
            is_above = other_y < current_y
            vertical_touch = abs((current_y - h) - (other_y + h)) <= eps

            if horizontal_overlap and is_above and vertical_touch:
                return True
        return False

    def _get_support_below_touching(self, idx: int) -> Optional[int]:
        """Return index of the box directly touching and supporting idx (same frame), or None.
        Touching = horizontal overlap within width-eps AND vertical faces are within eps.
        Does NOT override if nothing touches this frame.
        """
        x = self.boxes[idx].body.position.x
        y = self.boxes[idx].body.position.y
        h = self.box_height / 2
        eps = 0.05 * self.box_height

        best_j = None
        best_gap = float('inf')
        for j, other in enumerate(self.boxes):
            if j == idx:
                continue
            ox = other.body.position.x
            oy = other.body.position.y

            if oy <= y:
                continue

            # Vertical faces: bottom of idx (y + h) vs top of other (oy - h)
            vertical_gap = abs((y + h) - (oy - h))

            # Horizontal overlap by center distance within width-eps
            horizontal_overlap = abs(ox - x) <= (self.box_width - eps)
            if horizontal_overlap and vertical_gap <= eps:
                if vertical_gap < best_gap:
                    best_gap = vertical_gap
                    best_j = j

        return best_j

    def _get_tower_indices(self) -> set:
        """Compute indices of the tallest tower by following last_stood_on chain from the tallest box."""
        tallest_idx = None
        tallest_height = -1
        for idx, _a in enumerate(self.agents):
            h = (FLOOR_Y - self.boxes[idx].body.position.y - self.box_height / 2) / self.box_height
            if h > tallest_height:
                tallest_height = h
                tallest_idx = idx

        contributors = set()
        cursor = tallest_idx
        visited = set()
        while cursor is not None and cursor not in visited:
            visited.add(cursor)
            contributors.add(cursor)
            support = self.boxes[cursor].last_stood_on
            if support == 'floor':
                break
            if isinstance(support, int) and 0 <= support < self.num_boxes:
                cursor = support
            else:
                break
        return contributors

    def render(self, save_path=None):
        if self.render_mode is None:
            return

        self.screen.fill((255, 255, 255))

        square_image = pygame.Surface((self.box_width * SCALE, self.box_height * SCALE), pygame.SRCALPHA)
        square_image.fill((0, 0, 255))

        for idx, b in enumerate(self.boxes):
            square_image.fill(b.colour)

            # Visuallize tallest tower
            #if idx in self._get_tower_indices():
                #square_image.fill((20, 255, 20))

            position = b.body.position * SCALE
            angle = np.degrees(b.body.angle)

            # Draw the box
            rotated_square = pygame.transform.rotate(square_image, -angle)
            rotated_rect = rotated_square.get_rect(center=position)
            self.screen.blit(rotated_square, rotated_rect)
            
            # Draw the emotion sprite on top
            if hasattr(self, 'emotion_images') and self.emotion_images:
                emotion = self._get_box_emotion(idx)
                if emotion in self.emotion_images:
                    emotion_sprite = self.emotion_images[emotion]
                    # Rotate the emotion sprite to match the box rotation
                    rotated_emotion = pygame.transform.rotate(emotion_sprite, -angle)
                    emotion_rect = rotated_emotion.get_rect(center=position)
                    self.screen.blit(rotated_emotion, emotion_rect)

        # Render floor
        pygame.draw.rect(self.screen, (0, 0, 0), (0, FLOOR_Y * SCALE, W * SCALE, (H - FLOOR_Y) * SCALE))

        # Draw line for highest y coordinate
        y = FLOOR_Y - self.highest_y * self.box_height 
        y -= self.box_height
        y *= SCALE
        pygame.draw.rect(self.screen, (255, 0, 0), (0, y, W * SCALE, 2))

        pygame.display.flip()

        if save_path is not None:
            print("Saving render to", save_path)
            pygame.image.save(self.screen, save_path)


if __name__ == "__main__":
    import time

    print("Running Box Jump environment in visualisation mode...")

    env = BoxJumpEnvironment(render_mode="human", num_boxes=16, spacing=1.1, 
                           physics_steps_per_action=8, physics_timestep_multiplier=2)

    n = 0
    obs, _ = env.reset(seed=n)

    while True:
        t = time.time()
        actions = {i: env.action_space(i).sample() for i in env.possible_agents}
        obs, rewards, term, trunc, info = env.step(actions)
        env.render()

        sum_rewards = sum(rewards.values())
        print(f"Step {env.timestep}, sum rewards: {sum_rewards}")

        if not env.agents:
            n += 1
            obs, _ = env.reset(seed=n)

        # Sleep until next frame if we are above the desired FPS
        length = time.time() - t
        if length < 1 / FPS:
            time.sleep(1 / FPS - length)
            pass
