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
  'method': 'random',
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

# the following configurations do not represent
# all values searched. i often did multiple runs, seeing which
# perform well to iteratively restrict the search space in new runs

sweep_dvcl_pmnist_nocoreset = sweep_discriminative | {
  'parameters': wrap_values(
    experiments.disc_pmnist
    | {
      'model': 'vcl',
      'batch_size': 256,
      'epochs': 100,
      'learning_rate': 1e-3,
      'coreset_size': 0,
      'per_task_opt': [True, False],
      'pretrain_epochs': [0, 10, 100],
      # {'min': -32, 'max': 0}
      'layer_init_logstd_mean': {'min': -26.0, 'max': -1.0},
      # [1e-1, 1e-2, 1e-3, 1e-5]
      'layer_init_logstd_std': 0.01,
      'gaussian': False,
      'heteroscedastic': False,
    }
  )
}

sweep_dvcl_pmnist_coreset = sweep_discriminative | {
  'method': 'grid',
  'parameters': wrap_values(
    experiments.disc_pmnist
    | {
      'model': 'vcl',
      'batch_size': 256,
      'epochs': 100,
      'learning_rate': 1e-3,
      'coreset_size': [
        200,
        400,
        1_000,
        2_000,
        2_500,
        4_000,
        5_000,
        10_000,
        25_000,
        50_000,
      ],
      'per_task_opt': True,
      'pretrain_epochs': 0,
      # {'min': -25.0, 'max': -3.0}
      'layer_init_logstd_mean': list(map(float, np.arange(-24, -3.5, 2.0))),
      'layer_init_logstd_std': 0.01,
      'gaussian': False,
      'heteroscedastic': False,
    }
  ),
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
      'per_task_opt': True,
      'pretrain_epochs': 0,
      'layer_init_logstd_mean': list(map(float, np.arange(-32, 0.5, 0.5))),
      'layer_init_logstd_std': 0.1,
      'gaussian': False,
      'heteroscedastic': False,
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
      'per_task_opt': True,
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
      'per_task_opt': True,
      'pretrain_epochs': 0,
      'layer_init_logstd_mean': list(map(float, np.arange(-32, 0.5, 0.5))),
      'layer_init_logstd_std': 0.1,
      'gaussian': False,
      'heteroscedastic': False,
    }
  ),
}

sweep_dsi_nmnist = sweep_discriminative | {
  'method': 'grid',
  'parameters': wrap_values(
    experiments.disc_nmnist
    | {
      'model': 'si',
      'batch_size': 256,
      'learning_rate': 1e-3,
      'per_task_opt': True,
      'epochs': 20,
      'c': list(map(float, np.arange(0, 1.05, 0.11))),
      'xi': [
        1e-5,
        5e-5,
        1e-4,
        5e-4,
        1e-3,
        2e-3,
        3e-3,
      ],
    }
  ),
}

sweep_gvcl_mnist = sweep_generative | {
  'method': 'grid',
  'parameters': wrap_values(
    experiments.gen_mnist
    | {
      'model': 'vcl',
      'batch_size': 256,
      'epochs': 20,
      'learning_rate': 1e-3,
      'layer_init_logstd_mean': [
        -25,
        -20,
        -17,
        -15,
        -12,
        -11,
        -10,
        -9,
        -8,
        -6,
        -5,
        -4,
        -3,
        -2,
      ],
      'layer_init_logstd_std': 0.1,
    }
  ),
}

sweep_gsi_mnist = sweep_generative | {
  'method': 'grid',
  'parameters': wrap_values(
    experiments.gen_mnist
    | {
      'model': 'si',
      'batch_size': 256,
      'epochs': 20,
      'learning_rate': 1e-3,
      'c': list(map(float, np.arange(0, 1.05, 0.11))),
      'xi': [
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        0.2,
        0.3,
        0.4,
        0.5,
      ],
    }
  ),
}

sweep_gvcl_nmnist = sweep_generative | {
  'method': 'grid',
  'parameters': wrap_values(
    experiments.gen_nmnist
    | {
      'model': 'vcl',
      'batch_size': 256,
      'epochs': 40,
      'learning_rate': 1e-3,
      'layer_init_logstd_mean': [
        -20,
        -15,
        -10,
        -9,
        -8,
        -7,
        -6,
        -5.5,
        -5,
        -4,
        -3,
        -2,
      ],
      'layer_init_logstd_std': 0.1,
    }
  ),
}

sweep_gsi_nmnist = sweep_generative | {
  'method': 'grid',
  'parameters': wrap_values(
    experiments.gen_nmnist
    | {
      'model': 'si',
      'batch_size': 256,
      'epochs': 40,
      'learning_rate': 1e-3,
      'c': [0] + list(map(float, np.arange(0.1, 1.05, 0.3))) + [0.99],
      'xi': [
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        0.3,
      ],
    }
  ),
}

sweep_dvcl_pmnist_gaussian = sweep_discriminative | {
  'method': 'grid',
  'parameters': wrap_values(
    experiments.disc_pmnist
    | {
      'model': 'vcl',
      'batch_size': 256,
      'epochs': 100,
      'learning_rate': 1e-3,
      'coreset_size': 0,
      'per_task_opt': True,
      'pretrain_epochs': 0,
      # list(map(float, np.arange(-40, 0.5, 0.5))),
      'layer_init_logstd_mean': list(map(float, np.arange(-28.5, -26, 0.25))),
      'layer_init_logstd_std': 0.01,
      'gaussian': True,
      'heteroscedastic': False,
    }
  ),
}

sweep_dvcl_pmnist_gaussian_hetero = sweep_discriminative | {
  'method': 'grid',
  'parameters': wrap_values(
    experiments.disc_pmnist
    | {
      'model': 'vcl',
      'batch_size': 256,
      'epochs': 100,
      'learning_rate': 1e-3,
      'coreset_size': 0,
      'per_task_opt': True,
      'pretrain_epochs': 0,
      # list(map(float, np.arange(-40, -1, 2))),
      'layer_init_logstd_mean': list(map(float, np.arange(-29.5, -22, 0.5))),
      'layer_init_logstd_std': 0.01,
      'gaussian': True,
      'heteroscedastic': True,
    }
  ),
}

sweep_dvcl_singlehead_smnist = sweep_discriminative | {
  'method': 'grid',
  'parameters': wrap_values(
    experiments.disc_singlehead_smnist
    | {
      'model': 'vcl',
      'batch_size': None,
      'epochs': 120,
      'learning_rate': 1e-3,
      'coreset_size': 0,
      'per_task_opt': True,
      'pretrain_epochs': 0,
      'layer_init_logstd_mean': list(map(float, np.arange(-7.75, -5, 0.25))),
      'layer_init_logstd_std': 0.01,
      'gaussian': False,
      'heteroscedastic': False,
    }
  ),
}
