import wandb

import util
import experiments
import hyperparam_search
from vae_gen import baseline_generative_model
from vcl_disc import discriminative_model_pipeline as vcl_discriminative_model_pipeline
from vcl_gen import generative_model_pipeline as vcl_generative_model_pipeline
from si_disc import discriminative_model_pipeline as si_discriminative_model_pipeline
from si_gen import generative_model_pipeline as si_generative_model_pipeline


def model_pipeline(params=None, wandb_log=True):
  wandb_mode = 'online' if wandb_log else 'disabled'
  with wandb.init(project='vcl', config=params, mode=wandb_mode):
    params = wandb.config
    util.wandb_setup_axes()

    model = None
    if params.model == 'vcl':
      if params.experiment == 'discriminative':
        model = vcl_discriminative_model_pipeline(params)
      if params.experiment == 'generative':
        model = vcl_generative_model_pipeline(params)
    if params.model == 'si':
      if params.experiment == 'discriminative':
        model = si_discriminative_model_pipeline(params)
      if params.experiment == 'generative':
        model = si_generative_model_pipeline(params)
  return model


if __name__ == '__main__':
  wandb.login()
  util.torch_version()
  util.seed(0)

  dvcl_pmnist = experiments.disc_pmnist | dict(
    model='vcl',
    epochs=100,
    batch_size=256,
    learning_rate=1e-3,
    coreset_size=0,
    per_task_opt=False,
    pretrain_epochs=30,
    layer_init_logstd_mean=-25.77,
    layer_init_logstd_std=0.01,
  )

  dvcl_smnist = experiments.disc_smnist | dict(
    model='vcl',
    epochs=120,
    batch_size=None,
    pretrain_epochs=0,
    # 200
    coreset_size=200,
    per_task_opt=False,
    layer_init_std=1e-3,
    learning_rate=1e-3,
  )

  dvcl_nmnist = experiments.disc_nmnist | dict(
    model='vcl',
    epochs=120,
    batch_size=None,
    pretrain_epochs=0,
    coreset_size=0,
    per_task_opt=False,
    layer_init_std=1e-3,
    learning_rate=1e-3,
  )

  gvcl_mnist = experiments.gen_mnist | dict(
    model='vcl',
    # 200
    epochs=5,
    # 50
    batch_size=256,
    layer_init_std=None,
    # 1e-4
    learning_rate=1e-3,
  )

  gvcl_nmnist = experiments.gen_nmnist | dict(
    model='vcl',
    # 400
    epochs=20,
    batch_size=256,
    layer_init_std=None,
    # 1e-4
    learning_rate=1e-3,
  )

  dsi_pmnist = experiments.disc_pmnist | dict(
    model='si',
    epochs=20,
    batch_size=256,
    learning_rate=1e-3,
    c=0.1,
    xi=0.1,
    per_task_opt=False,
  )

  dsi_smnist = experiments.disc_smnist | dict(
    model='si',
    epochs=20,
    batch_size=256,
    learning_rate=1e-3,
    c=0.1,
    xi=0.1,
    per_task_opt=True,
  )

  dsi_nmnist = experiments.disc_nmnist | dict(
    model='si',
    epochs=50,
    batch_size=256,
    learning_rate=1e-3,
    c=0.1,
    xi=0.1,
    per_task_opt=True,
  )

  gsi_mnist = experiments.gen_mnist | dict(
    model='si',
    epochs=20,
    batch_size=256,
    learning_rate=1e-3,
    c=0.1,
    xi=0.1,
  )

  gsi_nmnist = experiments.gen_nmnist | dict(
    model='si',
    epochs=20,
    batch_size=256,
    learning_rate=1e-3,
    c=0.1,
    xi=0.1,
  )

  # discriminative
  # vcl
  model = model_pipeline(dvcl_pmnist, wandb_log=True)
  # model = model_pipeline(dvcl_smnist, wandb_log=True)
  # model = model_pipeline(dvcl_nmnist, wandb_log=True)
  # si
  # model = model_pipeline(dsi_pmnist, wandb_log=True)
  # model = model_pipeline(dsi_smnist, wandb_log=True)
  # model = model_pipeline(dsi_nmnist, wandb_log=True)

  # generative
  # vae
  # model = baseline_generative_model(num_epochs=5, problem='mnist')
  # vcl
  # model = model_pipeline(gvcl_mnist, wandb_log=True)
  # model = model_pipeline(gvcl_nmnist, wandb_log=True)
  # si
  # model = model_pipeline(gsi_mnist, wandb_log=True)
  # model = model_pipeline(gsi_nmnist, wandb_log=True)

  # wandb bug: we cant properly join existing sweeps outside of __main__ with
  # > 1 dataloader num_worker - see https://github.com/wandb/wandb/issues/8953
  # so just run this inside __main__
  #
  sweep_params = hyperparam_search.sweep_dvcl_pmnist_nocoreset
  # sweep_id = 'gacjzl7i'
  # sweep_id = wandb.sweep(sweep_params, project='vcl', prior_runs=[])
  # wandb.agent(sweep_id, model_pipeline, project='vcl', count=200)
