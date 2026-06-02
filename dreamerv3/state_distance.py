"""Contrastive state-distance model for LoFo replay buffers (JAX port).

Faithful port of r2dreamer/state_distance.py::SimpleContrastiveStateDistanceModel
to raw JAX + optax (no flax/ninjax dependency). The model is an action-agnostic
conv encoder trained on consecutive-frame positive pairs + random negatives.
Once trained it is FROZEN and `get_representation(image)` produces the embedding
the LoFo buffer uses for similarity/eviction.

It runs on CPU by default so it does not contend with the agent for the
accelerator (the encoder is tiny and is called per inserted item on the host).

Image convention: takes (..., H, W, 3) uint8 and normalizes to float [0, 1].
Conv layers use kernel 4 / stride 2 / VALID padding (NHWC) matching the PyTorch
original, then an MLP (512, 64, repr_dim).
"""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax


def _cpu_device():
  try:
    return jax.devices('cpu')[0]
  except RuntimeError:
    return jax.devices()[0]


def _init_params(rng, in_channels, image_hw, hidden_channels, mlp):
  """He-initialized conv + MLP params; flat dim is computed by a dry forward."""
  params = {'conv': [], 'mlp': []}
  chans = [in_channels, *hidden_channels]
  for i in range(len(hidden_channels)):
    rng, k = jax.random.split(rng)
    fan_in = 4 * 4 * chans[i]
    w = jax.random.normal(k, (4, 4, chans[i], chans[i + 1]), jnp.float32)
    w = w * np.sqrt(2.0 / fan_in)
    b = jnp.zeros((chans[i + 1],), jnp.float32)
    params['conv'].append((w, b))
  # Dry forward to get the flattened conv output size.
  dummy = jnp.zeros((1, image_hw[0], image_hw[1], in_channels), jnp.float32)
  flat = _conv_forward(params, dummy).reshape(1, -1).shape[1]
  dims = [flat, *mlp]
  for i in range(len(mlp)):
    rng, k = jax.random.split(rng)
    fan_in = dims[i]
    w = jax.random.normal(k, (dims[i], dims[i + 1]), jnp.float32)
    w = w * np.sqrt(2.0 / fan_in)
    b = jnp.zeros((dims[i + 1],), jnp.float32)
    params['mlp'].append((w, b))
  return params


def _conv_forward(params, x):
  h = x
  for w, b in params['conv']:
    h = jax.lax.conv_general_dilated(
        h, w, window_strides=(2, 2), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')) + b
    h = jax.nn.relu(h)
  return h


def _encode(params, x):
  """x: (B, H, W, 3) float in [0, 1] -> (B, repr_dim)."""
  h = _conv_forward(params, x).reshape(x.shape[0], -1)
  *hidden, last = params['mlp']
  for w, b in hidden:
    h = jax.nn.relu(h @ w + b)
  w, b = last
  return h @ w + b


def _to_nhwc_float(image):
  """uint8/float (H,W,3) or (B,H,W,3) -> float32 (B,H,W,3) in [0,1]."""
  x = jnp.asarray(image)
  if x.ndim == 3:
    x = x[None]
  if x.dtype == jnp.uint8 or x.dtype == np.uint8:
    x = x.astype(jnp.float32) / 255.0
  else:
    x = x.astype(jnp.float32)
  return x


class ContrastiveStateDistance:

  def __init__(
      self, repr_dim=32, lr=1e-4, negative_distance_target=50.0,
      negative_loss_ratio=0.1, num_negative_samples=128, num_training_epochs=5,
      batch_size=32, image_hw=(64, 64), seed=0, device='cpu'):
    self.repr_dim = int(repr_dim)
    self.lr = float(lr)
    self.neg_target = float(negative_distance_target)
    self.neg_ratio = float(negative_loss_ratio)
    self.num_neg = int(num_negative_samples)
    self.epochs = int(num_training_epochs)
    self.batch_size = int(batch_size)
    self.image_hw = tuple(image_hw)
    self.out_dim = self.repr_dim
    self._device = _cpu_device() if device == 'cpu' else jax.devices()[0]
    rng = jax.random.PRNGKey(seed)
    mlp = (512, 64, self.repr_dim)
    params = _init_params(rng, 3, self.image_hw, (32, 64, 128, 256), mlp)
    self.params = jax.device_put(params, self._device)
    self._apply = jax.jit(_encode, device=self._device)

  @staticmethod
  def _make_pairs(images, episode_ids):
    images = np.asarray(images)
    episode_ids = np.asarray(episode_ids).reshape(-1)
    keep = (episode_ids[:-1] == episode_ids[1:])
    idx = np.where(keep)[0]
    return images[idx], images[idx + 1]

  def train(self, images, episode_ids, log_fn=print):
    anchors, positives = self._make_pairs(images, episode_ids)
    if len(anchors) == 0:
      raise ValueError('No positive pairs available - check episode_ids')
    n = anchors.shape[0]
    log_fn(f'[distance_model] training on {n} pairs for {self.epochs} epochs')

    target, ratio, num_neg = self.neg_target, self.neg_ratio, self.num_neg

    def loss_fn(params, a, p, neg):
      ar = _encode(params, a)
      pr = _encode(params, p)
      b, k = neg.shape[0], neg.shape[1]
      nr = _encode(params, neg.reshape((b * k, *neg.shape[2:]))).reshape(b, k, -1)
      pos_loss = ((ar - pr) ** 2).sum()
      neg_dists = ((ar[:, None, :] - nr) ** 2).sum(-1).mean()
      neg_loss = ratio * (target - neg_dists) ** 2
      return pos_loss + neg_loss

    opt = optax.adam(self.lr)
    opt_state = opt.init(self.params)
    params = self.params

    @jax.jit
    def step(params, opt_state, a, p, neg):
      loss, grads = jax.value_and_grad(loss_fn)(params, a, p, neg)
      updates, opt_state = opt.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss

    rng = np.random.default_rng(0)
    for epoch in range(self.epochs):
      perm = rng.permutation(n)
      running, seen = 0.0, 0
      for start in range(0, n, self.batch_size):
        bidx = perm[start:start + self.batch_size]
        a = jax.device_put(_to_nhwc_float(anchors[bidx]), self._device)
        p = jax.device_put(_to_nhwc_float(positives[bidx]), self._device)
        neg_idx = rng.integers(0, n, size=(len(bidx), num_neg))
        neg = _to_nhwc_float(anchors[neg_idx.reshape(-1)])
        neg = jax.device_put(
            neg.reshape((len(bidx), num_neg, *neg.shape[1:])), self._device)
        params, opt_state, loss = step(params, opt_state, a, p, neg)
        running += float(loss) * len(bidx)
        seen += len(bidx)
      log_fn(f'[distance_model] epoch {epoch}: loss={running / max(seen, 1):.4f}')
    self.params = params

  def get_representation(self, obs):
    """obs: (H,W,3) or (B,H,W,3) uint8/float. Returns np.float32 (repr_dim,)
    for a single obs or (B, repr_dim) for a batch."""
    single = np.asarray(obs).ndim == 3
    x = jax.device_put(_to_nhwc_float(obs), self._device)
    out = np.asarray(self._apply(self.params, x))
    return out[0] if single else out

  def save(self, path):
    flat = jax.tree_util.tree_map(lambda x: np.asarray(x), self.params)
    with open(path, 'wb') as f:
      pickle.dump(flat, f)

  def load(self, path):
    with open(path, 'rb') as f:
      flat = pickle.load(f)
    self.params = jax.device_put(flat, self._device)
