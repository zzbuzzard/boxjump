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

For full details, see the docstring.

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

