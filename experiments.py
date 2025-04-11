disc_pmnist = dict(
  in_dim=28 * 28,
  hidden_dim=100,
  out_dim=10,
  hidden_layers=2,
  ntasks=10,
  multihead=False,
  problem='pmnist',
  experiment='discriminative',
)

disc_smnist = dict(
  in_dim=28 * 28,
  hidden_dim=256,
  out_dim=2,
  hidden_layers=2,
  ntasks=5,
  multihead=True,
  problem='smnist',
  experiment='discriminative',
)

disc_nmnist = dict(
  in_dim=28 * 28,
  hidden_dim=150,
  out_dim=2,
  hidden_layers=4,
  ntasks=5,
  multihead=True,
  problem='nmnist',
  experiment='discriminative',
)

gen_mnist = dict(
  in_dim=28 * 28,
  hidden_dim=500,
  latent_dim=50,
  ntasks=10,
  architecture=1,
  problem='mnist',
  experiment='generative',
)

gen_nmnist = dict(
  in_dim=28 * 28,
  hidden_dim=500,
  latent_dim=50,
  ntasks=10,
  architecture=1,
  problem='nmnist',
  experiment='generative',
)
