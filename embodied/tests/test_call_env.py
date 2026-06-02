import elements
import embodied
import numpy as np
import pytest


class PhaseEnv(embodied.Env):

  def __init__(self):
    self.phase = None
    self.is_eval = None

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, (4, 4, 3)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool)}

  @property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        'action': elements.Space(np.int32, (), 0, 3)}

  def step(self, action):
    return dict(
        image=np.zeros((4, 4, 3), np.uint8), reward=np.float32(0.0),
        is_first=bool(action['reset']), is_last=False, is_terminal=False)

  def set_phase_and_eval(self, phase, is_eval):
    self.phase, self.is_eval = phase, is_eval
    return (phase, is_eval)


class TestCallEnv:

  @pytest.mark.parametrize('parallel', [False, True])
  def test_call_env_reaches_all_workers(self, parallel):
    driver = embodied.Driver([PhaseEnv for _ in range(3)], parallel=parallel)
    try:
      res = driver.call_env('set_phase_and_eval', 2, True)
      assert res == [(2, True), (2, True), (2, True)], res
      # Env still steps after the call.
      policy = lambda c, o: (c, {'action': np.zeros(3, np.int32)}, {})
      driver.reset()
      driver(policy, steps=3)
    finally:
      driver.close()
