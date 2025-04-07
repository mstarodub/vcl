import wandb

import util
from vae import baseline_generative_model
from vcl_gen import generative_model_pipeline as vcl_generative_model_pipeline
from vcl_disc import discriminative_model_pipeline as vcl_discriminative_model_pipeline
from si_disc import discriminative_model_pipeline as si_discriminative_model_pipeline


def model_pipeline(params, wandb_log=True):
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
  return model


if __name__ == '__main__':
  wandb.login()
  util.torch_version()
  util.seed(0)

  ddm_pmnist_run = dict(
    in_dim=28 * 28,
    hidden_dim=100,
    out_dim=10,
    hidden_layers=2,
    ntasks=10,
    epochs=100,
    batch_size=256,
    # 0
    pretrain_epochs=10,
    # 2000
    coreset_size=0,
    per_task_opt=False,
    layer_init_std=1e-10,
    bayesian_test_samples=100,
    bayesian_train_samples=10,
    learning_rate=1e-3,
    multihead=False,
    problem='pmnist',
    experiment='discriminative',
    model='vcl',
  )

  ddm_smnist_run = dict(
    in_dim=28 * 28,
    hidden_dim=256,
    out_dim=2,
    hidden_layers=2,
    ntasks=5,
    epochs=120,
    batch_size=None,
    pretrain_epochs=0,
    # 200
    coreset_size=200,
    per_task_opt=False,
    layer_init_std=1e-3,
    bayesian_test_samples=100,
    bayesian_train_samples=10,
    learning_rate=1e-3,
    multihead=True,
    problem='smnist',
    experiment='discriminative',
    model='vcl',
  )

  ddm_nmnist_run = dict(
    in_dim=28 * 28,
    hidden_dim=150,
    out_dim=2,
    hidden_layers=4,
    ntasks=5,
    epochs=120,
    batch_size=None,
    pretrain_epochs=0,
    coreset_size=0,
    per_task_opt=False,
    layer_init_std=1e-3,
    bayesian_test_samples=100,
    bayesian_train_samples=10,
    learning_rate=1e-3,
    multihead=True,
    problem='nmnist',
    experiment='discriminative',
    model='vcl',
  )

  dgm_mnist_run = dict(
    in_dim=28 * 28,
    hidden_dim=500,
    latent_dim=50,
    ntasks=10,
    # 200
    epochs=5,
    # 50
    batch_size=256,
    layer_init_std=None,
    # 1e-4
    learning_rate=1e-3,
    problem='mnist',
    experiment='generative',
    model='vcl',
  )

  dgm_nmnist_run = dict(
    in_dim=28 * 28,
    hidden_dim=500,
    latent_dim=50,
    ntasks=10,
    # 400
    epochs=20,
    batch_size=256,
    layer_init_std=None,
    # 1e-4
    learning_rate=1e-3,
    problem='nmnist',
    experiment='generative',
    model='vcl',
  )

  dsi_pmnist_run = dict(
    in_dim=28 * 28,
    hidden_dim=100,
    out_dim=10,
    hidden_layers=2,
    ntasks=10,
    epochs=20,
    batch_size=256,
    learning_rate=1e-3,
    c=0.1,
    xi=0.1,
    per_task_opt=False,
    multihead=False,
    problem='pmnist',
    experiment='discriminative',
    model='si',
  )

  dsi_smnist_run = dict(
    in_dim=28 * 28,
    hidden_dim=256,
    out_dim=10,
    hidden_layers=2,
    ntasks=5,
    epochs=20,
    batch_size=256,
    learning_rate=1e-3,
    c=0.1,
    xi=0.1,
    per_task_opt=True,
    multihead=True,
    problem='smnist',
    experiment='discriminative',
    model='si',
  )

  dsi_nmnist_run = dict(
    in_dim=28 * 28,
    hidden_dim=150,
    out_dim=10,
    hidden_layers=4,
    ntasks=5,
    epochs=50,
    batch_size=256,
    learning_rate=1e-3,
    c=0.1,
    xi=0.1,
    per_task_opt=True,
    multihead=True,
    problem='nmnist',
    experiment='discriminative',
    model='si',
  )

  # model = model_pipeline(ddm_pmnist_run, wandb_log=True)
  # model = model_pipeline(ddm_smnist_run, wandb_log=True)
  # model = model_pipeline(ddm_nmnist_run, wandb_log=True)

  # model = baseline_generative_model(num_epochs=5, problem='mnist')

  # model = model_pipeline(dgm_mnist_run, wandb_log=True)
  model = model_pipeline(dgm_nmnist_run, wandb_log=True)

  # model = model_pipeline(dsi_pmnist_run, wandb_log=True)
  # model = model_pipeline(dsi_smnist_run, wandb_log=True)
  # model = model_pipeline(dsi_nmnist_run, wandb_log=True)
