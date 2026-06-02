"""MiniGrid LoCA environment for the JAX DreamerV3 (embodied.Env).

Native port of r2dreamer/envs/minigrid_loca.py. Implements the 2-phase LoCA
curriculum via `set_phase_and_eval(phase, is_eval)`:
  * phase 1: agent spawns anywhere in the grid; goal (1,1) gives reward 4,
    goal (n-2,n-2) gives 2.
  * phase 2: agent confined to a small one-way passage (top-left corner); goal
    (1,1) reward drops to 1. Leaving the passage is reverted.
  * eval: full-grid spawn (tests retention/generalization).

The gym_minigrid import is deferred to construction time so parallel Driver
workers can pickle the constructor.
"""

import itertools

import elements
import embodied
import numpy as np


def _build_loca_env(grid_size, one_way_passage_size, seed):
  from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv
  from gym import spaces

  class MiniGridLoCA(MiniGridEnv):
    def __init__(self, grid_size=10, one_way_passage_size=2, seed=None):
      if grid_size < 2 * one_way_passage_size + 4 + (grid_size % 2):
        raise ValueError(
            'Grid and one_way_passage sizes do not meet required conditions')
      terminal_set = {(1, 1), (grid_size - 2, grid_size - 2)}
      full_grid = list(
          set(itertools.product(
              np.arange(grid_size - 2) + 1, np.arange(grid_size - 2) + 1))
          - terminal_set)
      passage = list(
          set(itertools.product(
              np.arange(one_way_passage_size) + 1,
              np.arange(one_way_passage_size) + 1)) - terminal_set)
      self.init_state_phase1 = full_grid
      self.init_state_phase2 = passage
      self.init_state_eval = full_grid
      self.one_way_passage_area = list(itertools.product(
          np.arange(one_way_passage_size) + 1,
          np.arange(one_way_passage_size) + 1))
      self.init_state = None
      self.reward_terminal1 = None
      self.reward_terminal2 = None
      self.set_phase_and_eval(1, False)

      self._manual_agent_pos = None
      self._manual_agent_dir = None
      self.mission_space = spaces.Box(
          low=0, high=100, shape=(1,), dtype=np.float32)
      super().__init__(
          mission_space=self.mission_space, grid_size=grid_size,
          max_steps=4 * grid_size * grid_size, see_through_walls=True)
      if seed is not None:
        try:
          self.seed(seed)
        except (AttributeError, TypeError):
          self._pending_seed = int(seed)
      self.action_space = spaces.Discrete(3)

    def set_phase_and_eval(self, phase, is_eval):
      if is_eval:
        self.init_state = self.init_state_eval
        self.reward_terminal1 = 4 if phase == 1 else 1
        self.reward_terminal2 = 2
      else:
        if phase == 1:
          self.init_state = self.init_state_phase1
          self.reward_terminal1 = 4
          self.reward_terminal2 = 2
        else:
          self.init_state = self.init_state_phase2
          self.reward_terminal1 = 1
          self.reward_terminal2 = 2

    def _gen_grid(self, width, height):
      self.grid = Grid(width, height)
      self.grid.wall_rect(0, 0, width, height)
      self.put_obj(Goal(), width - 2, height - 2)
      self.put_obj(Goal(), 1, 1)
      if self._manual_agent_pos is not None:
        self.agent_pos = self._manual_agent_pos
      else:
        self.agent_pos = self.init_state[self._rand_int(0, len(self.init_state))]
      if self._manual_agent_dir is not None:
        self.agent_dir = self._manual_agent_dir
      else:
        self.agent_dir = self._rand_int(0, 4)
      self.mission = 'get to the highest reward green goal square'

    def _check_inside_one_way_passage(self, pos=None):
      if pos is None:
        pos = self.agent_pos
      for inside in self.one_way_passage_area:
        if pos[0] == inside[0] and pos[1] == inside[1]:
          return True
      return False

    def _reward(self):
      if self.agent_pos[0] == 1 and self.agent_pos[1] == 1:
        return self.reward_terminal1
      return self.reward_terminal2

    def step(self, action):
      prev_pos = self.agent_pos
      prev_inside = self._check_inside_one_way_passage()
      result = super().step(action)
      if len(result) == 4:
        obs, reward, done, info = result
        terminated, truncated = done, False
      else:
        obs, reward, terminated, truncated, info = result
      if prev_inside and not self._check_inside_one_way_passage():
        self.agent_pos = prev_pos
        obs = self.gen_obs()
      return obs, reward, terminated, truncated, info

  return MiniGridLoCA(
      grid_size=grid_size, one_way_passage_size=one_way_passage_size, seed=seed)


def _resize(image, size):
  if image.shape[:2] == tuple(size):
    return image
  try:
    import cv2
    return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
  except ImportError:
    from PIL import Image
    return np.asarray(
        Image.fromarray(image).resize((size[1], size[0]), Image.BILINEAR))


class MinigridLoca(embodied.Env):

  def __init__(self, task='default', grid_size=10, one_way_passage_size=2,
               size=(64, 64), seed=0):
    del task
    from gym_minigrid.wrappers import (
        FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper)
    self._size = tuple(size)
    self._loca_env = _build_loca_env(grid_size, one_way_passage_size, seed)
    self._env = ImgObsWrapper(RGBImgObsWrapper(FullyObsWrapper(self._loca_env)))
    self._action_n = int(self._loca_env.action_space.n)
    self._initial_seed = int(seed)
    self._needs_seed = True
    self._done = True

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, self._size + (3,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        'action': elements.Space(np.int32, (), 0, self._action_n),
    }

  def set_phase_and_eval(self, phase, is_eval):
    self._loca_env.set_phase_and_eval(phase, is_eval)

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      if self._needs_seed:
        try:
          result = self._env.reset(seed=self._initial_seed)
        except TypeError:
          result = self._env.reset()
        self._needs_seed = False
      else:
        result = self._env.reset()
      image = result[0] if isinstance(result, tuple) else result
      return self._obs(image, 0.0, is_first=True)
    result = self._env.step(int(action['action']))
    if len(result) == 4:
      image, reward, done, info = result
      terminated, truncated = done, False
    else:
      image, reward, terminated, truncated, info = result
      done = terminated or truncated
    self._done = bool(done)
    return self._obs(
        image, float(reward), is_last=bool(done),
        is_terminal=bool(terminated and not truncated))

  def _obs(self, image, reward, is_first=False, is_last=False, is_terminal=False):
    return dict(
        image=_resize(np.asarray(image, dtype=np.uint8), self._size),
        reward=np.float32(reward),
        is_first=is_first, is_last=is_last, is_terminal=is_terminal)
