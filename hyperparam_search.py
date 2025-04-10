import numpy as np

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
      'coreset_size': [
        # 200,
        # 500,
        # 1_000,
        # 2_000,
        # 3_000,
        # 4_000,
        # 5_000,
        # 7_500,
        # 10_000,
        # 12_500,
        # 15_000,
        20_000,
        30_000,
        40_000,
        50_000,
      ],
      'per_task_opt': [True, False],
      'pretrain_epochs': 0,
      'layer_init_logstd_mean': {'min': -20.0, 'max': -10.0},
      'layer_init_logstd_std': 0.01,
    }
  )
}

sweep_dsi_pmnist = sweep_discriminative | {
  'parameters': wrap_values(
    experiments.disc_pmnist
    | {
      'model': 'si',
      'batch_size': 256,
      'learning_rate': 1e-3,
      'per_task_opt': False,
      'epochs': {'min': 20, 'max': 50},
      'c': {'min': 1e-4, 'max': 1.0},
      'xi': {'min': 5e-3, 'max': 0.5},
    }
  )
}

sweep_dvcl_smnist = sweep_discriminative | {
  'method': 'grid',
  'parameters': wrap_values(
    experiments.disc_smnist
    | {
      'model': 'vcl',
      'batch_size': None,
      'epochs': 120,
      'learning_rate': 1e-3,
      'coreset_size': 0,
      'per_task_opt': [True, False],
      'pretrain_epochs': 0,
      # {'min': -32, 'max': 0}
      'layer_init_logstd_mean': list(map(float, np.arange(-32, 0.5, 0.5))),
      'layer_init_logstd_std': 0.1,
    }
  ),
}

sweep_dsi_smnist = sweep_discriminative | {
  'parameters': wrap_values(
    experiments.disc_smnist
    | {
      'model': 'si',
      'batch_size': [None, 256],
      'learning_rate': 1e-3,
      'per_task_opt': [True, False],
      'epochs': {'min': 20, 'max': 60},
      'c': {'min': 1e-4, 'max': 1.0},
      'xi': {'min': 5e-3, 'max': 0.5},
    }
  )
}

sweep_dvcl_nmnist = sweep_discriminative | {
  'method': 'grid',
  'parameters': wrap_values(
    experiments.disc_nmnist
    | {
      'model': 'vcl',
      'batch_size': None,
      'epochs': 120,
      'learning_rate': 1e-3,
      'coreset_size': 0,
      'per_task_opt': [True, False],
      'pretrain_epochs': 0,
      # {'min': -32, 'max': 0}
      'layer_init_logstd_mean': list(map(float, np.arange(-32, 0.5, 0.5))),
      'layer_init_logstd_std': 0.1,
    }
  ),
}

sweep_dsi_nmnist = sweep_discriminative | {
  'parameters': wrap_values(
    experiments.disc_nmnist
    | {
      'model': 'si',
      'batch_size': [None, 256],
      'learning_rate': 1e-3,
      'per_task_opt': [True, False],
      'epochs': {'min': 20, 'max': 60},
      'c': {'min': 1e-4, 'max': 1.0},
      'xi': {'min': 5e-3, 'max': 0.5},
    }
  )
}

sweep_gvcl_mnist = sweep_generative | {
  'parameters': wrap_values(
    experiments.gen_mnist
    | {
      'model': 'vcl',
      'batch_size': 256,
      'epochs': 200,
      'learning_rate': 1e-3,
      'layer_init_logstd_mean': list(map(float, np.arange(-32, 0.5, 0.5))),
      'layer_init_logstd_std': 0.1,
    }
  )
}

sweep_gsi_mnist = sweep_generative | {
  'parameters': wrap_values(
    experiments.gen_mnist
    | {
      'model': 'si',
      'batch_size': 256,
      # XXX
      'epochs': {'min': 20, 'max': 50},
      'learning_rate': 1e-3,
      'c': {'min': 1e-4, 'max': 1.0},
      'xi': {'min': 5e-3, 'max': 0.5},
    }
  )
}

sweep_gvcl_nmnist = sweep_generative | {
  'parameters': wrap_values(
    experiments.gen_nmnist
    | {
      'model': 'vcl',
      'batch_size': 256,
      'epochs': 400,
      'learning_rate': 1e-3,
      'layer_init_logstd_mean': list(map(float, np.arange(-32, 0.5, 0.5))),
      'layer_init_logstd_std': 0.1,
    }
  )
}

sweep_gsi_nmnist = sweep_generative | {
  'parameters': wrap_values(
    experiments.gen_nmnist
    | {
      'model': 'si',
      'batch_size': 256,
      # XXX
      'epochs': {'min': 20, 'max': 50},
      'learning_rate': 1e-3,
      'c': {'min': 1e-4, 'max': 1.0},
      'xi': {'min': 5e-3, 'max': 0.5},
    }
  )
}

# gy/disc_pmnist_nocoreset:
# mean, std, pretrain
# -25, 0.01, 0
# -17, 0.0001, 100
# -25, 0.1, 30
# -19, 0.001, 100
# -17, 0.0001, 10
# -18, 0.0001, 10
# ua, ga/disc_pmnist_nocoreset:
# -23 bis -25, 0.01, 0
# np/disc_pmnist_nocoreset:
# -24.5 bis -26, std egal, pre egal
# => -15 - -26 gut
# disc_pmnist_coreset:
# kr: -20 bis -15 gut für large coresets
# 9x
#
# si disc pmnist:
# y3: epochs < 50
#   xi: 0.1, try even more
#   c: need more data
# m0:


# vcl disc split mnist:
# 4bq2a9ij
# XXX forgot per task opt

# si disc split mnist:
# 10nhr6rw
# XXX forgot per task opt
