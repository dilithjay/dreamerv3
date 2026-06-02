import embodied
import numpy as np
import pytest


def _step(value, is_first=False, is_last=False):
  return {
      'image': np.full((4, 4, 3), value, np.uint8),
      'reward': np.float32(0.0),
      'is_first': bool(is_first),
      'is_last': bool(is_last),
      'is_terminal': False,
  }


def _drive(replay, values):
  n = len(values)
  for i, v in enumerate(values):
    replay.add(_step(v, is_first=(i == 0), is_last=(i == n - 1)), worker=0)


def _repr_from_value(img):
  # repr = the (constant) pixel value -> identical images at distance 0.
  return np.array([float(img[0, 0, 0])], np.float32)


class TestLoFoReplay:

  def test_v1_evicts_oldest_neighbor(self):
    r = embodied.LoFoReplay(
        length=1, capacity=1000, variant='lofo_v1',
        repr_fn=_repr_from_value, radius=0.5, count_thresh=3)
    _drive(r, [7] * 8)
    assert len(r.items) == r.count_thresh
    # Only the most recent count_thresh items survive (oldest evicted).
    assert sorted(r.items) == sorted(r.items)[-r.count_thresh:]

  def test_v1_keeps_diverse(self):
    r = embodied.LoFoReplay(
        length=1, capacity=1000, variant='lofo_v1',
        repr_fn=_repr_from_value, radius=0.5, count_thresh=3)
    _drive(r, [1, 2, 3, 4, 5, 6])
    assert len(r.items) == 6

  def test_v2_caps_bucket(self):
    r = embodied.LoFoReplay(
        length=1, capacity=1000, variant='lofo_v2',
        repr_fn=_repr_from_value, hash_dim=8, per_bucket_capacity=2)
    _drive(r, [9] * 5)
    assert len(r.items) == 2
    assert all(i in r.item_bucket for i in r.items)
    assert sum(len(d) for d in r.buckets.values()) == len(r.items)

  def test_capacity_cap_bounds_fifo_fallback(self):
    r = embodied.LoFoReplay(
        length=1, capacity=3, variant='lofo_v1', repr_fn=None)
    _drive(r, list(range(10)))
    assert len(r.items) <= 3

  def test_sample_returns_active_items(self):
    r = embodied.LoFoReplay(
        length=1, capacity=1000, variant='lofo_v1',
        repr_fn=_repr_from_value, radius=0.5, count_thresh=3)
    _drive(r, [1, 2, 3, 4, 5])
    batch = r.sample(4, mode='train')
    assert batch['image'].shape[0] == 4

  def test_never_underflows_active_set(self):
    # Aggressive eviction (thresh=1) must never drop below 1 item.
    r = embodied.LoFoReplay(
        length=1, capacity=1000, variant='lofo_v1',
        repr_fn=_repr_from_value, radius=10.0, count_thresh=1)
    _drive(r, [5] * 6)
    assert len(r.items) >= 1
