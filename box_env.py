import functools
import random
from copy import copy
import colorsys

import math
import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv

import pygame
from Box2D import b2World, b2PolygonShape
import Box2D


class BoxAgent:
    def __init__(self, body: Box2D.b2Body, h: float = 0.666, v: float = 1, s: float = 1):
        self.body = body
        self.yvels = []
        self.n = 10
        self.fallen = False

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


VIEWPORT_W = 600
VIEWPORT_H = 400
SCALE = 20  # Scale from pygame units to pixels
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

        A total reward (with highest mode) of N means that a height of N boxes above the floor was achieved.
    """

    metadata = {
        "name": "boxjump_v0",
    }

    def __init__(self, num_boxes=4, world_width=10, world_height=6, box_width=1, box_height=1, render_mode=None,
                 gravity=10, friction=0.8, spacing=1.5, random_spacing=0.5, angular_damping=1, agent_one_hot=False,
                 max_timestep=400, fixed_rotation=False, reward_mode: str = "highest", include_time: bool = True,
                 include_highest: bool = True, penalty_fall: float = 20):
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

        self.include_time = include_time
        self.include_highest = include_highest

        assert reward_mode in ["highest", "highest_stable", "height_sq", "stable_sum"]
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

        size = low_state.shape[0]
        self.state_space = Box(low=low_state, high=high_state, shape=[size])

        self.boxes = None
        self.world = None
        self._state = None
        self.possible_agents = [f"box-{i}" for i in range(1, num_boxes+1)]

        self.render_mode = render_mode
        self.screen = None
        self.highest_y = 0

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

            ob = np.array(ob, dtype=np.float32)
            obs[self.agents[i]] = ob
            state.append(ob[self.keep_inds])

        if self.include_highest:
            state.append(np.array([self.highest_y]))  # only include once
        if self.include_time:
            state.append(np.array([self.timestep / self.max_timestep]))  # only include once
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

            # body.ApplyForceToCenter((self.np_random.uniform(-150, 150), 0), True)

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
        rewards = {}

        prev_best = self.highest_y
        new_best = self.highest_y
        max_height_above_floor = -1

        final_step = (self.timestep + 1) > self.max_timestep

        for idx, i in enumerate(self.agents):
            action = actions[i]
            self.boxes[idx].step()
            body = self.boxes[idx].body
            if action == 0:
                pass
            elif action in [1, 2]:
                mul = -1 if action == 1 else 1
                if not ((mul > 0 and body.linearVelocity.x > MAX_HOR_SPD) or (mul < 0 and body.linearVelocity.x < -MAX_HOR_SPD)):
                    body.ApplyForceToCenter((mul * 10, 0), True)
            elif action == 3:
                if self.boxes[idx].can_jump():
                    body.ApplyForceToCenter((0, -250), True)

            # height_above_floor = 0 -> you are on the floor
            # height_above_floor = 1 -> the bottom of your square has just exited the screen off the top
            # height_above_floor = (FLOOR_Y - self.boxes[idx].body.position.y - self.box_height / 2) / FLOOR_Y

            # 0 = you are on the floor
            # n = you are n boxes in height (so one box standing on another gets a score of 1)
            height_above_floor = (FLOOR_Y - self.boxes[idx].body.position.y - self.box_height / 2) / self.box_height

            if self.reward_mode == "highest_stable":
                valid = self.boxes[idx].can_jump()
            elif self.reward_mode == "highest":
                valid = True
            else:
                # Doesn't matter
                valid = True

            if height_above_floor > new_best and valid:
                new_best = height_above_floor

            max_height_above_floor = max(max_height_above_floor, height_above_floor)

        for ind, i in enumerate(self.agents):
            # Apply fall penalty: single penalty mode
            # if not self.boxes[ind].fallen:
            #     height_above_floor = (FLOOR_Y - self.boxes[ind].body.position.y - self.box_height / 2) / self.box_height
            #     if height_above_floor < NEGATIVE_THRESHOLD:
            #         rewards[i] = -self.penalty_fall
            #         self.boxes[ind].fallen = True
            #         print("A box fell")
            #         continue

            # Apply fall penalty: every step penalty mode
            height_above_floor = (FLOOR_Y - self.boxes[ind].body.position.y - self.box_height / 2) / self.box_height
            if height_above_floor < NEGATIVE_THRESHOLD:
                rewards[i] = -self.penalty_fall / self.max_timestep
                self.boxes[ind].fallen = True
                continue

            if self.reward_mode == "highest" or self.reward_mode == "highest_stable":
                rewards[i] = (new_best - prev_best) / self.num_boxes
            elif self.reward_mode == "height_sq":
                height_above_floor = (FLOOR_Y - self.boxes[ind].body.position.y - self.box_height / 2) / self.box_height
                rewards[i] = (height_above_floor ** 2) / self.num_boxes / self.max_timestep
            elif self.reward_mode == "stable_sum":
                height_above_floor = (FLOOR_Y - self.boxes[ind].body.position.y - self.box_height / 2) / self.box_height
                if self.boxes[ind].can_jump():
                    rewards[i] = 50 * (height_above_floor ** 2) / self.num_boxes / self.max_timestep
                else:
                    rewards[i] = 0

                # if final_step:
                #     rewards[i] += self.highest_y / self.num_boxes

            # if self.reward_scheme == 1:
            #     rewards[i] = max_height_above_floor / self.num_boxes
            # elif self.reward_scheme == 2:
            #     rewards[i] = (new_best - prev_best) / self.num_boxes
            # elif self.reward_scheme == 3:
            #     time = self.timestep / self.max_timestep
            #     rewards[i] = time * max_height_above_floor / self.num_boxes
            # elif self.reward_scheme == 4:
            #     height_above_floor = (FLOOR_Y - self.boxes[ind].body.position.y - self.box_height / 2) / self.box_height
            #     rewards[i] = height_above_floor / self.num_boxes

                # xs = np.array([b.body.position.x / W for b in self.boxes])
                # rewards[i] = -np.sum((xs - 0.5) ** 2) / (self.num_boxes ** 2)

        self.world.Step(1 / FPS, 30, 30)
        self.timestep += 1

        terminations = {a: False for a in self.agents}

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > self.max_timestep:
            rewards = {a: 0 for a in self.agents}
            truncations = {a: True for a in self.agents}

        self.render()

        obs = self.get_all_obs()
        infos = {a: {} for a in self.agents}

        self.highest_y = new_best

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def render(self, save_path=None):
        if self.render_mode is None:
            return

        self.screen.fill((255, 255, 255))

        square_image = pygame.Surface((self.box_width * SCALE, self.box_height * SCALE), pygame.SRCALPHA)
        square_image.fill((0, 0, 255))

        for b in self.boxes:
            square_image.fill(b.colour)

            position = b.body.position * SCALE
            angle = np.degrees(b.body.angle)

            rotated_square = pygame.transform.rotate(square_image, -angle)
            rotated_rect = rotated_square.get_rect(center=position)
            self.screen.blit(rotated_square, rotated_rect)

        # Render floor
        pygame.draw.rect(self.screen, (0, 0, 0), (0, FLOOR_Y * SCALE, W * SCALE, (H - FLOOR_Y) * SCALE))

        # Draw line for highest y coordinate
        # body position y
        y = FLOOR_Y - self.highest_y * self.box_height # + self.box_height / 2
        y -= self.box_height
        y *= SCALE
        pygame.draw.rect(self.screen, (255, 0, 0), (0, y, W * SCALE, 2))
        # height_above_floor = (FLOOR_Y - self.boxes[idx].body.position.y - self.box_height / 2) / self.box_height

        pygame.display.flip()

        if save_path is not None:
            print("Saving render to", save_path)
            pygame.image.save(self.screen, save_path)


if __name__ == "__main__":
    import time

    print("Running Box Jump environment in visualisation mode...")

    env = BoxJumpEnvironment(render_mode="human", num_boxes=16, spacing=1.1)

    n = 0
    obs, _ = env.reset(seed=n)

    while True:
        t = time.time()
        actions = {i: env.action_space(i).sample() for i in env.possible_agents}
        obs, rewards, term, trunc, info = env.step(actions)
        env.render()

        if not env.agents:
            n += 1
            obs, _ = env.reset(seed=n)

        # Sleep until next frame if we are above the desired FPS
        length = time.time() - t
        if length < 1 / FPS:
            time.sleep(1 / FPS - length)

