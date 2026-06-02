import threading
from collections import defaultdict, deque

import numpy as np

from .replay import Replay


class LoFoReplay(Replay):
  """Local-Forgetting replay buffer (LoFoV1 / LoFoV2).

  Extends the stock `Replay` (which is FIFO-eviction + uniform sampling) with
  *density-based* eviction over a per-item representation, ported from
  r2dreamer/buffers/lofo.py:

    * LoFoV1 ("lofo_v1"): when a newly inserted item has >= `count_thresh`
      neighbors within Euclidean `radius` (in representation space), the OLDEST
      neighbor is evicted. Keeps a sparse, well-separated set of items.
    * LoFoV2 ("lofo_v2"): locality-sensitive hashing. The representation is
      projected by a fixed random matrix to a `hash_dim`-bit code; each bucket
      holds at most `per_bucket_capacity` items (FIFO within the bucket).

  Sampling stays uniform over the surviving items (the "active set" is exactly
  `self.items`). The density rule runs on EVERY insert, not only when the buffer
  is full; the inherited capacity cap remains as a hard FIFO fallback.

  `repr_fn` maps a single observation image (HWC uint8) -> np.float32[repr_dim].
  It is the frozen, pretrained contrastive encoder built in main.py. If
  `repr_fn` is None (or the repr key is absent), the buffer degrades to plain
  FIFO behavior.
  """

  def __init__(
      self, *args, variant='lofo_v1', repr_fn=None, repr_key='image',
      radius=0.05, count_thresh=10, hash_dim=32, per_bucket_capacity=2000,
      **kwargs):
    super().__init__(*args, **kwargs)
    assert variant in ('lofo_v1', 'lofo_v2'), variant
    self.variant = variant
    self.repr_fn = repr_fn
    self.repr_key = repr_key
    # V1 params.
    self.radius = float(radius)
    self.count_thresh = int(count_thresh)
    # V2 params.
    self.hash_dim = int(hash_dim)
    self.per_bucket_capacity = int(per_bucket_capacity)
    self._A = None  # lazy random projection (seeded for reproducibility)
    # State (itemid is monotonic, so it doubles as an insertion timestamp).
    self.reprs = {}                      # itemid -> np.float32[repr_dim]
    self.buckets = defaultdict(deque)    # code   -> deque[itemid]   (V2)
    self.item_bucket = {}                # itemid -> code            (V2)
    self.lofo_lock = threading.RLock()   # guards reprs/buckets/item_bucket

  def _insert(self, chunkid, index):
    # Hard capacity cap (FIFO fallback) — identical to the parent.
    while self.capacity and len(self.items) >= self.capacity:
      self._remove()
    itemid = self.itemid
    self.itemid += 1
    self.items[itemid] = (chunkid, index)
    stepids = self._getseq(chunkid, index, ['stepid'])['stepid']
    self.sampler[itemid] = stepids
    self.fifo.append(itemid)
    # Density-based (LoFo) eviction over the item representation.
    if self.repr_fn is None:
      return
    try:
      img = self._getseq(chunkid, index, [self.repr_key])[self.repr_key][0]
    except KeyError:
      return  # No repr key in this env's observations -> behave like FIFO.
    repr_vec = np.asarray(self.repr_fn(img), np.float32).reshape(-1)
    with self.lofo_lock:
      self.reprs[itemid] = repr_vec
      if self.variant == 'lofo_v1':
        self._density_evict_v1(itemid, repr_vec)
      else:
        self._density_evict_v2(itemid, repr_vec)

  def _density_evict_v1(self, itemid, repr_vec):
    # Compare against existing items only (the new item is always kept).
    others = [i for i in self.reprs if i != itemid]
    if len(others) < 1 or len(self.items) < 2:
      return
    mat = np.stack([self.reprs[i] for i in others])
    dists = np.sqrt(((mat - repr_vec[None, :]) ** 2).sum(1))
    neighbors = [others[k] for k in range(len(others)) if dists[k] < self.radius]
    if len(neighbors) >= self.count_thresh:
      self._evict(min(neighbors))  # oldest = smallest itemid

  def _density_evict_v2(self, itemid, repr_vec):
    if self._A is None:
      rng = np.random.default_rng(0)
      self._A = rng.standard_normal(
          (self.hash_dim, repr_vec.shape[0])).astype(np.float32)
    bits = (self._A @ repr_vec) > 0
    code = 0
    for b in bits:
      code = (code << 1) | int(b)
    bucket = self.buckets[code]
    if len(bucket) >= self.per_bucket_capacity and len(self.items) >= 2:
      self._evict(bucket.popleft())
    bucket.append(itemid)
    self.item_bucket[itemid] = code

  def _remove(self):
    # Capacity-cap removal: evict the globally oldest item (FIFO fallback).
    self._evict(self.fifo[0])

  def _evict(self, itemid):
    # Single eviction primitive used by both the density rule and capacity cap.
    del self.sampler[itemid]
    chunkid, index = self.items.pop(itemid)
    with self.lofo_lock:
      self.reprs.pop(itemid, None)
      code = self.item_bucket.pop(itemid, None)
      if code is not None:
        try:
          self.buckets[code].remove(itemid)
        except ValueError:
          pass
    try:
      self.fifo.remove(itemid)
    except ValueError:
      pass
    with self.refs_lock:
      self.refs[chunkid] -= 1
      if self.refs[chunkid] < 1:
        del self.refs[chunkid]
        chunk = self.chunks.pop(chunkid)
        if chunk.succ in self.refs:
          self.refs[chunk.succ] -= 1
