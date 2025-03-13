# Box Jump
A fully co-operative MARL environment, compatible with PettingZoo.

<p align="center"">
    <img src="assets/preview_gif.gif" width="60%"/>
</p>

### Description
There are _n_ boxes, each controlled by an agent.
There is a discrete action space with four possible actions at each time step: left, right, no-op or jump.
Jumping is only allowed if a box is stable (i.e. not in the air).

The goal is to build the highest tower possible in the time permitted.
Whenever a new maximum y-value is achieved (by any agent), all agents receive a reward
equal to the difference between the new and the old maximum - meaning the overall reward for the
episode is the maximum y-value achieved over the whole episode.
The red line shows the current best y-value.

Box rotation may be enabled or disabled - the task is a lot easier with it disabled (as in the GIF above).

<details>
<summary>Full environment details</summary>


### Observation Space
Each agent has an observation of dimension 13 at each step (regardless
of the number of agents), with the following values (in order):
 - Horizontal position (0 to 1).
 - Height above floor (0 = on the floor, n = you are n boxes high).
 - Velocity (vx, vy).
 - Angle (measured in _quarter turns_, so a box rotated 90 degrees is indistinguishable from the original). Stays at 0 when rotation disabled.
 - Angular velocity. Stays at 0 when rotation disabled.
 - Left/right/up/down raycast distances. Imagine shooting a ray 
in each direction from each box, and measuring the distance til it hits the floor or another box.
E.g. distance of 0.1 in the left direction means another box is very close on the left, and a distance of 1 in the down direction means the box is mid-jump.
 - Whether the box can currently jump (0/1).
 - Highest y-coordinate this episode (the height of the red line).
 - Time remaining this episode (0 to 1).

The observation is 'local', and has constant size regardless of the number of
agents, so the environment can scale to many agents.
It should be possible to build really good policies from this though,
even in a decentralized way: for example, if a box sees an agent on its left, but no agent above it,
it should probably try jumping on top of that box.

### Global State
The global state (`env.state()`) is the concatenation of all the per-agent observations,
but only including highest y-coordinate and time remaining (the last two bullet-points) once as these
are global properties.

### Action Space
A discrete action space of size 4:
 - 0 = do nothing
 - 1 = apply force to the left
 - 2 = apply force to the right
 - 3 = jump (does nothing if the agent cannot currently jump)

### Rewards
There are a few different reward schemes, which are explained in the docstring.
The default gives rewards whenever the red line moves up (the highest y-coordinate
during the episode), and are shared between all agents, making the task fully co-operative.

</details>


## Installation
```
git clone https://github.com/zzbuzzard/boxjump
cd boxjump
pip install swig
pip install -e .
```
Be aware `box2d-py` can be annoying to install sometimes.

## Usage
We conform to [PettingZoo](https://pettingzoo.farama.org/index.html)'s API,
so the environment may be used like any other PettingZoo environment.
Here is a complete example:
```python
from boxjump.box_env import BoxJumpEnvironment

# render_mode="human" creates a PyGame window to visualise the environment.
#  set render_mode=None to run without visualisation (during training).
env = BoxJumpEnvironment(render_mode="human", num_boxes=16, fixed_rotation=True)

seed = 0

obs, _ = env.reset(seed=seed)
global_state = env.state()

agent_names = env.possible_agents

while True:
    actions = {}
    for agent in agent_names:
        observation = obs[agent]
        action_space = env.action_space(agent)
        
        # TODO: Add decision logic
        action = action_space.sample()
        
        actions[agent] = action
    
    obs, rewards, term, trunc, info = env.step(actions)
    env.render()  # (does nothing if render_mode="human")
    
    if not env.agents:  # episode has finished
        seed += 1
        obs, _ = env.reset(seed=seed)
```


The environment code is all in one file, `box_env.py`.
If you run this file, it will visualise the environment with random
agent behaviour for testing.

## Citation
If you use boxjump, please cite the original paper (paper [here](https://arxiv.org/pdf/2503.09521), code [here](https://github.com/zzbuzzard/PairVDN)):
```
@article{buzzard2025pairvdn,
      title={PairVDN - Pair-wise Decomposed Value Functions}, 
      author={Zak Buzzard},
      year={2025},
      url={https://arxiv.org/abs/2503.09521}, 
}
```
(and if you get any cool results, send me an email!)
