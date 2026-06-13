import importlib
import os
import pathlib
import sys
from functools import partial as bind

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml


def main(argv=None):
  from .agent import Agent
  [elements.print(line) for line in Agent.banner]

  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))

  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
  print('Replica:', config.replica, '/', config.replicas)

  logdir = elements.Path(config.logdir)
  print('Logdir:', logdir)
  print('Run script:', config.script)
  if not config.script.endswith(('_env', '_replay')):
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer

  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
      loca=config.loca,
  )

  # For LoFo buffers, pretrain the frozen contrastive distance model once and
  # thread its representation function into the train replay buffer.
  repr_fn = None
  if config.script in ('train', 'train_eval', 'train_loca'):
    variant = str(config.replay.get('variant', 'fifo')).lower()
    if variant in ('lofo_v1', 'lofo_v2'):
      # make_repr_fn pretrains the distance model on the GPU, which initializes the XLA backend. XLA_FLAGS (set by internal.setup, including the command-buffer disable that prevents reacher's CUDA_ERROR_ILLEGAL_ADDRESS) are only read at backend init, so setup() must run BEFORE make_repr_fn touches the GPU. Agent.__new__ calls setup() again later; for a single
      # process it is idempotent.
      from embodied.jax.agent import Options
      from embodied.jax import internal
      # Mirror Agent.__new__: setup() gets the jax-config keys that are not
      # Options fields.
      internal.setup(**{
          k: v for k, v in config.jax.items()
          if k not in Options.__dataclass_fields__})
    repr_fn = make_repr_fn(config)

  if config.script == 'train':
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay', repr_fn=repr_fn),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'train_eval':
    embodied.run.train_eval(
        bind(make_agent, config),
        bind(make_replay, config, 'replay', repr_fn=repr_fn),
        bind(make_replay, config, 'eval_replay', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'train_loca':
    embodied.run.train_loca(
        bind(make_agent, config),
        bind(make_replay, config, 'replay', repr_fn=repr_fn),
        bind(make_replay, config, 'eval_replay', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'eval_only':
    embodied.run.eval_only(
        bind(make_agent, config),
        bind(make_env, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel':
    embodied.run.parallel.combined(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel_env':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_env(
        bind(make_env, config), config.replica, args, is_eval)

  elif config.script == 'parallel_envs':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_envs(
        bind(make_env, config), bind(make_env, config), args)

  elif config.script == 'parallel_replay':
    embodied.run.parallel.parallel_replay(
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_stream, config),
        args)

  else:
    raise NotImplementedError(config.script)


def make_agent(config):
  from .agent import Agent
  env = make_env(config, 0)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  if config.random_agent:
    return embodied.RandomAgent(obs_space, act_space)
  cpdir = elements.Path(config.logdir)
  cpdir = cpdir.parent if config.replicas > 1 else cpdir
  return Agent(obs_space, act_space, elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def make_logger(config):
  step = elements.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  outputs = []
  outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
  for output in config.logger.outputs:
    if output == 'jsonl':
      outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
      outputs.append(elements.logger.JSONLOutput(
          logdir, 'scores.jsonl', 'episode/score'))
    elif output == 'tensorboard':
      outputs.append(elements.logger.TensorBoardOutput(
          logdir, config.logger.fps))
    elif output == 'expa':
      exp = logdir.split('/')[-4]
      run = '/'.join(logdir.split('/')[-3:])
      proj = 'embodied' if logdir.startswith(('/cns/', 'gs://')) else 'debug'
      outputs.append(elements.logger.ExpaOutput(
          exp, run, proj, config.logger.user, config.flat))
    elif output == 'wandb':
      name = '/'.join(logdir.split('/')[-4:])
      outputs.append(elements.logger.WandBOutput(name))
    elif output == 'scope':
      outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
    else:
      raise NotImplementedError(output)
  logger = elements.Logger(step, outputs, multiplier)
  return logger


def make_replay(config, folder, mode='train', repr_fn=None):
  batlen = config.batch_length if mode == 'train' else config.report_length
  consec = config.consec_train if mode == 'train' else config.consec_report
  capacity = config.replay.size if mode == 'train' else config.replay.size / 10
  length = consec * batlen + config.replay_context
  assert config.batch_size * length <= capacity

  directory = elements.Path(config.logdir) / folder
  if config.replicas > 1:
    directory /= f'{config.replica:05}'

  variant = str(config.replay.get('variant', 'fifo')).lower()
  is_lofo = variant in ('lofo_v1', 'lofo_v2') and mode == 'train'

  kwargs = dict(
      length=length, capacity=int(capacity), online=config.replay.online,
      chunksize=config.replay.chunksize, directory=directory)

  if is_lofo:
    # LoFo wants pure uniform sampling over the surviving (active) set, and the
    # online queue would bypass that active set, so disable both extras.
    kwargs['online'] = False
    return embodied.LoFoReplay(
        variant=variant, repr_fn=repr_fn, repr_key=config.replay.lofo.repr_key,
        radius=config.replay.lofo.radius,
        count_thresh=config.replay.lofo.count_thresh,
        hash_dim=config.replay.lofo.hash_dim,
        per_bucket_capacity=config.replay.lofo.per_bucket_capacity,
        **kwargs)

  if config.replay.fracs.uniform < 1 and mode == 'train':
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')
    recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
    selectors = embodied.replay.selectors
    kwargs['selector'] = selectors.Mixture(dict(
        uniform=selectors.Uniform(),
        priority=selectors.Prioritized(**config.replay.prio),
        recency=selectors.Recency(recency),
    ), config.replay.fracs)

  return embodied.replay.Replay(**kwargs)


def make_repr_fn(config):
  """Pretrain the frozen contrastive distance model for LoFo buffers and return
  its `get_representation`. Returns None for non-LoFo variants."""
  variant = str(config.replay.get('variant', 'fifo')).lower()
  if variant not in ('lofo_v1', 'lofo_v2'):
    return None
  from .state_distance import ContrastiveStateDistance

  dm = config.distance_model
  repr_key = config.replay.lofo.repr_key

  # Spaces for the random agent (action sampling).
  env = make_env(config, 0)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()

  # Collect random-policy rollouts from a single (serial) env.
  driver = embodied.Driver([bind(make_env, config, 0)], parallel=False)
  images, episodes, ep = [], [], [0]

  def grab(tran, worker):
    images.append(np.asarray(tran[repr_key]))
    episodes.append(ep[0])
    if tran['is_last']:
      ep[0] += 1

  driver.on_step(grab)
  rand = embodied.RandomAgent(obs_space, act_space)
  driver.reset(rand.init_policy)
  print(f'[distance_model] collecting {int(dm.pretrain_steps)} random steps')
  driver(rand.policy, steps=int(dm.pretrain_steps))
  driver.close()

  size = config.env.get(config.task.split('_')[0], {}).get('size', [64, 64])
  # setup() ran before make_repr_fn, so the transfer guard is active. Run the
  # whole one-time pretraining under transfer_guard('allow'): construction
  # (PRNGKey + param init), train, and save all move data host<->device, and
  # this single scope covers every one of them. (get_representation is returned
  # and called later during the agent loop; it has its own allow-wrapper.)
  import jax
  with jax.transfer_guard('allow'):
    model = ContrastiveStateDistance(
        repr_dim=int(dm.repr_dim), lr=float(dm.lr),
        num_negative_samples=int(dm.num_negative_samples),
        num_training_epochs=int(dm.epochs), batch_size=int(dm.batch_size),
        image_hw=tuple(size), seed=config.seed)
    print(f'[distance_model] pretraining on {model._train_device}; '
          f'inference on {model._device}')
    model.train(np.stack(images), np.array(episodes))
    if not config.script.endswith(('_env', '_replay')):
      model.save(str(elements.Path(config.logdir) / 'distance_model.pkl'))
  return model.get_representation


def make_env(config, index, **overrides):
  suite, task = config.task.split('_', 1)
  if suite == 'memmaze':
    from embodied.envs import from_gym
    import memory_maze  # noqa
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'langroom': 'embodied.envs.langroom:LangRoom',
      'procgen': 'embodied.envs.procgen:ProcGen',
      'bsuite': 'embodied.envs.bsuite:BSuite',
      'minigridloca': 'embodied.envs.minigridloca:MinigridLoca',
      'reacherloca': 'embodied.envs.reacherloca:ReacherLoca',
      'memmaze': lambda task, **kw: from_gym.FromGym(
          f'MemoryMaze-{task}-v0', **kw),
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  if kwargs.pop('use_seed', False):
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  if kwargs.pop('use_logdir', False):
    kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'
  env = ctor(task, **kwargs)
  return wrap_env(env, config)


def wrap_env(env, config):
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.NormalizeAction(env, name)
  env = embodied.wrappers.UnifyDtypes(env)
  env = embodied.wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.ClipAction(env, name)
  return env


def make_stream(config, replay, mode):
  fn = bind(replay.sample, config.batch_size, mode)
  stream = embodied.streams.Stateless(fn)
  stream = embodied.streams.Consec(
      stream,
      length=config.batch_length if mode == 'train' else config.report_length,
      consec=config.consec_train if mode == 'train' else config.consec_report,
      prefix=config.replay_context,
      strict=(mode == 'train'),
      contiguous=True)

  return stream


if __name__ == '__main__':
  main()
