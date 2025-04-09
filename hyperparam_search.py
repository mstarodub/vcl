import experiments


def wrap_values(d):
  def wrap(v):
    if isinstance(v, list):
      return {'values': v}
    elif isinstance(v, dict):
      return v
    else:
      return {'value': v}

  return {k: wrap(v) for k, v in d.items()}


sweep = {
  # bayes
  'method': 'random',
  # 'parameters': {
  #   'epochs': {'min': 20, 'max': 400},
  #   'batch_size': {'values': [256, 512, None]},
  #   'learning_rate': {'values': [1e-3, 5e-4, 1e-4]},
  # },
  # 'early_terminate': {
  #   'type': 'hyperband',
  #   'min_iter': 2,
  #   'eta': 2,
  # },
}

sweep_discriminative = sweep | {
  'metric': {
    'name': 'test/test_acc',
    'goal': 'maximize',
  }
}

sweep_generative = sweep | {
  'metric': {
    'name': 'test/test_uncert',
    'goal': 'minimize',
  }
}

sweep_dvcl_pmnist_nocoreset = sweep_discriminative | {
  'parameters': wrap_values(
    experiments.disc_pmnist
    | {
      'model': 'vcl',
      'batch_size': 256,
      'epochs': 100,
      'learning_rate': 1e-3,
      'coreset_size': 0,
      # [True, False],
      'per_task_opt': False,
      # [1, 10, 100]
      'bayesian_test_samples': 100,
      # [1, 10]
      'bayesian_train_samples': 1,
      # [0, 10, 30, 100]
      'pretrain_epochs': [0, 10, 30, 100],
      # {'min': -32, 'max': 0}
      'layer_init_logstd_mean': {'min': -26.0, 'max': -24.0},
      # [1e-1, 1e-2, 1e-3, 1e-5]
      'layer_init_logstd_std': [1e-1, 1e-2, 1e-3, 1e-5],
    }
  )
}

sweep_dvcl_pmnist_coreset = sweep_discriminative | {
  'parameters': wrap_values(
    experiments.disc_pmnist
    | {
      'model': 'vcl',
      'batch_size': 256,
      'epochs': 100,
      'learning_rate': 1e-3,
      'coreset_size': {'min': 200, 'max': 50_000},
      # TODO, get best from ..._nocoreset run
      # 'per_task_opt': [True, False],
      # 'bayesian_test_samples': [1, 10, 100],
      # 'bayesian_train_samples': [1, 10, 100],
      # 'pretrain_epochs': [0, 10, 30, 100],
      # 'layer_init_logstd_mean, :layer_init_logstd_std...
    }
  )
}
