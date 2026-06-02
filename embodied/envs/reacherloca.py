"""Reacher LoCA environment for the JAX DreamerV3 (embodied.Env).

Native port of r2dreamer/envs/dmc_loca.py. The underlying dm_control task
"reacherloca" must be registered by the host environment (the patched
dm_control used by the LoCA_v3 / Dreamer_v1_Ali setup).

LoCA via `set_phase_and_eval(phase, is_eval)`:
  * phase 1: standard reacher (reach target_1).
  * phase 2 (train): agent spawned inside a one-way wall around target_1 and must
    escape and reach target_2; leaving the wall is reverted (physics rollback).
  * phase 2 (eval): no constraint; agent resampled outside the wall.
"""

import elements
import embodied
import numpy as np


class ReacherLoca(embodied.Env):

  def __init__(self, task='easy', size=(64, 64), camera=0, loca_phase='phase_1',
               loca_mode='train', one_way_wall_radius=0.1, seed=0):
    from dm_control import suite
    self._size = tuple(size)
    self._camera = camera
    self._loca_phase = loca_phase
    self._loca_mode = loca_mode
    self._one_way_wall_radius = one_way_wall_radius
    self._done = True

    self._env = suite.load('reacherloca', task, task_kwargs={'random': seed})
    if self._loca_phase != 'phase_1':
      self._env._task.switch_loca_task()

    self._actuators_length = np.array([
        self._env._physics.named.model.body_pos['hand', 'x'],
        self._env._physics.named.model.body_pos['finger', 'x'],
    ])
    spec = self._env.action_spec()
    self._act_low = np.asarray(spec.minimum, np.float32)
    self._act_high = np.asarray(spec.maximum, np.float32)
    self._adim = int(np.prod(spec.shape))

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
        'action': elements.Space(
            np.float32, (self._adim,), self._act_low, self._act_high),
    }

  # --- LoCA helpers (ported verbatim) ---
  def get_target_1_pos(self):
    return self._env._physics.named.data.geom_xpos['target_1', :2]

  def get_finger_pos(self):
    return self._env._physics.named.data.geom_xpos['finger', :2]

  def check_inside_one_way_wall(self):
    return (np.linalg.norm(self.get_finger_pos() - self.get_target_1_pos())
            <= self._one_way_wall_radius)

  def is_phase_2(self):
    return self._loca_phase == 'phase_2'

  def sample_in_one_way_wall(self):
    def sample_xy():
      x = np.random.uniform(low=0, high=self._one_way_wall_radius)
      y = np.sqrt(np.random.uniform(
          low=0, high=(self._one_way_wall_radius ** 2 - x ** 2)))
      return np.array([x, y])

    def inverse_kinematics(pos):
      x, y = pos[0], pos[1]
      a1, a2 = self._actuators_length[0], self._actuators_length[1]
      d = (x ** 2 + y ** 2 - a1 ** 2 - a2 ** 2) / (2 * a1 * a2)
      d = np.clip(d, -1.0, 1.0)
      theta2 = np.arccos(d) if np.random.rand() < 0.5 else -np.arccos(d)
      theta1 = np.arctan2(y, x) - np.arctan2(
          a2 * np.sin(theta2), a1 + a2 * np.cos(theta2))
      return theta1, theta2

    def forward_kinematics(theta1, theta2):
      a1, a2 = self._actuators_length[0], self._actuators_length[1]
      x = a1 * np.cos(theta1) + a2 * np.cos(theta1 + theta2)
      y = a1 * np.sin(theta1) + a2 * np.sin(theta1 + theta2)
      return np.array([x, y])

    target_1_pos = self.get_target_1_pos()
    while True:
      final_finger_pos = sample_xy() + target_1_pos
      if np.linalg.norm(final_finger_pos) > self._actuators_length.sum():
        continue
      if final_finger_pos[0] == 0.0:
        continue
      theta1, theta2 = inverse_kinematics(final_finger_pos)
      if np.isnan(theta1) or np.isnan(theta2):
        continue
      if np.linalg.norm(
          final_finger_pos - forward_kinematics(theta1, theta2)) > 1e-5:
        theta1 += np.pi
      return theta1, theta2

  def set_phase_and_eval(self, phase, is_eval):
    phase_names = {1: 'phase_1', 2: 'phase_2', 3: 'phase_3'}
    new_phase = phase_names.get(phase, 'phase_1')
    if new_phase != 'phase_1' and self._loca_phase == 'phase_1':
      try:
        self._env._task.switch_loca_task()
      except AttributeError:
        pass
    self._loca_phase = new_phase
    self._loca_mode = 'eval' if is_eval else 'train'

  # --- embodied.Env interface ---
  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      return self._reset()
    act = np.asarray(action['action'], np.float32)
    prev_inside = self.check_inside_one_way_wall()
    prev_physics_state = self._env._physics.get_state()
    time_step = self._env.step(act)
    reward = time_step.reward or 0.0
    done = time_step.last()
    if prev_inside and not self.check_inside_one_way_wall():
      with self._env._physics.reset_context():
        self._env._physics.set_state(prev_physics_state)
    is_terminal = False if time_step.first() else time_step.discount == 0
    self._done = bool(done)
    return self._obs(float(reward), is_last=bool(done),
                     is_terminal=bool(is_terminal))

  def _reset(self):
    if self._loca_mode == 'eval':
      self._env.reset()
      while self.check_inside_one_way_wall():
        self._env.reset()
      return self._obs(0.0, is_first=True)
    if self.is_phase_2():
      self._env.reset()
      theta1, theta2 = self.sample_in_one_way_wall()
      with self._env._physics.reset_context():
        self._env._physics.named.data.qpos['shoulder'] = theta1
        self._env._physics.named.data.qpos['wrist'] = theta2
      return self._obs(0.0, is_first=True)
    self._env.reset()
    return self._obs(0.0, is_first=True)

  def _render(self):
    return self._env._physics.render(*self._size, camera_id=self._camera)

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    return dict(
        image=np.asarray(self._render(), np.uint8),
        reward=np.float32(reward),
        is_first=is_first, is_last=is_last, is_terminal=is_terminal)
