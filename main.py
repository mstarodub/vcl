import torch
import wandb

import vae
import util
import accuracy
import dataloaders
from vcl_gen import generative_model_pipeline
from vcl_disc import discriminative_model_pipeline


def model_pipeline(params, wandb_log=True):
  wandb_mode = 'online' if wandb_log else 'disabled'
  with wandb.init(project='vcl', config=params, mode=wandb_mode):
    params = wandb.config
    model = None
    if params.experiment == 'discriminative':
      model = discriminative_model_pipeline(params)
    if params.experiment == 'generative':
      model = generative_model_pipeline(params)
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
    epochs=100,
    # 50
    batch_size=128,
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
    epochs=400,
    batch_size=64,
    layer_init_std=None,
    learning_rate=1e-4,
    problem='nmnist',
    experiment='generative',
    model='vcl',
  )

  # model = model_pipeline(ddm_pmnist_run, wandb_log=True)
  # model = model_pipeline(ddm_smnist_run, wandb_log=True)
  # model = model_pipeline(ddm_nmnist_run, wandb_log=True)

  csf = accuracy.CNNEnsembleClassifier()
  csf.load_state_dict(torch.load('classifier.pt', map_location=util.torch_device()))
  csf.to(util.torch_device())
  # csf_train_loader, csf_test_loader = dataloaders.mnist_vanilla_task_loaders(256)
  # csf.train_run(csf_train_loader, csf_test_loader, num_epochs=30)
  # torch.save(csf.state_dict(), 'classifier.pt')

  model = vae.baseline_generative_model(num_epochs=20, classifier=csf)
  # model = model_pipeline(dgm_mnist_run, wandb_log=True)
  # model = model_pipeline(dgm_nmnist_run, wandb_log=False)
