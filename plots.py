import wandb
import polars as pl
import plotly.express as px
from plotly.subplots import make_subplots
import os

import experiments


# disc_pmnist
if False:
  run_ids = {
    'r09vadzh': 'vcl',
    'dcy7qany': 'vcl + coreset (25k)',
    'z7b4hxg6': 'vcl + coreset (10k)',
    'gdo97xiw': 'vcl + coreset (5k)',
    'co0e8wpq': 'vcl + coreset (4k)',
    '5nwbbi8b': 'vcl + coreset (2.5k)',
    'jzc7wtcp': 'vcl + coreset (2k)',
    # '8nk4zrmw': 'si',
    # 'ora1ykic': 'baseline',
  }
  metrics = ['test/test_acc'] + [
    f'test/test_acc_task_{i}' for i in range(experiments.disc_pmnist['ntasks'])
  ]
  metric = 'test accuracy'
  # fname = 'disc_pmnist_per_task_acc'
  fname = 'disc_pmnist_per_task_acc_coreset'
# disc_smnist
if False:
  run_ids = {
    'kn20dwka': 'vcl',
    'eem35sel': 'si',
    '21kt316w': 'baseline',
  }
  metrics = ['test/test_acc'] + [
    f'test/test_acc_task_{i}' for i in range(experiments.disc_smnist['ntasks'])
  ]
  metric = 'test accuracy'
  fname = 'disc_smnist_per_task_acc'
# disc_nmnist
if False:
  run_ids = {
    '7fkimt7s': 'vcl',
    '67fmhyvf': 'si',
    'hn8gnzey': 'baseline',
  }
  metrics = ['test/test_acc'] + [
    f'test/test_acc_task_{i}' for i in range(experiments.disc_nmnist['ntasks'])
  ]
  metric = 'test accuracy'
  # fname = 'disc_nmnist_per_task_acc'
  fname = 'disc_nmnist_per_task_acc_withbaseline'
# gen_mnist
if False:
  run_ids = {
    '7ei1kvx5': 'vcl',
    'lfq5zvai': 'si',
    '6ty15p6s': 'baseline',
    'adz84vfv': 'arch2 baseline',
  }
  metrics = ['test/test_uncert'] + [
    f'test/test_uncert_task_{i}' for i in range(experiments.gen_mnist['ntasks'])
  ]
  # metrics = ['test/test_ll'] + [
  #   f'test/test_ll_task_{i}' for i in range(experiments.gen_mnist['ntasks'])
  # ]
  metric = 'classifier uncertainty'
  # metric = 'test log-likelihood'
  # fname = 'gen_mnist_per_task_uncert'
  fname = 'gen_mnist_per_task_uncert_withbaseline'
  # fname = 'gen_mnist_per_task_test_ll'
  # fname = 'gen_mnist_per_task_test_ll_withbaseline'
# gen_nmnist
if False:
  run_ids = {
    '6fb6ibf8': 'vcl',
    '0n1bb5zu': 'si',
    'ohbbj5vm': 'baseline',
    # 'gs8imknj': 'arch2 baseline',
  }
  # metrics = ['test/test_uncert'] + [
  #   f'test/test_uncert_task_{i}' for i in range(experiments.gen_nmnist['ntasks'])
  # ]
  metrics = ['test/test_ll'] + [
    f'test/test_ll_task_{i}' for i in range(experiments.gen_nmnist['ntasks'])
  ]
  # metric = 'classifier uncertainty'
  metric = 'test log-likelihood'
  # fname = 'gen_nmnist_per_task_uncert'
  # fname = 'gen_nmnist_per_task_uncert_withbaseline'
  # fname = 'gen_nmnist_per_task_test_ll'
  fname = 'gen_nmnist_per_task_test_ll_withbaseline'


def download_media(run, key):
  history = run.scan_history(keys=[key], page_size=1_000_000)
  for row in history:
    path = row[key]['path']
    run.file(path).download()


def fetch_key(run, key):
  # use '_step' for cont. logged keys
  history = run.scan_history(keys=['task', key], page_size=1_000_000)
  return list(history)


def fetch_run(api, run_id):
  run = api.run(f'mxst-university-of-oxford/vcl/{run_id}')
  xss = [fetch_key(run, m) for m in metrics]
  dfs = [pl.from_dicts(xs) for xs in xss if xs]
  dfs = [
    df.select(
      [
        pl.col(col).cast(pl.Float64) if df.schema[col].is_numeric() else pl.col(col)
        for col in df.columns
      ]
    )
    for df in dfs
  ]
  df = dfs[0]
  for d in dfs[1:]:
    df = df.join(d, on='task', how='full', coalesce=True)
  df = df.with_columns(pl.lit(run_ids[run_id]).alias('experiment'))
  return df


if __name__ == '__main__':
  api = wandb.Api()

  path = f'figures/data_raw/{fname}.csv'
  if os.path.exists(path):
    df = pl.read_csv(path)
  else:
    dfs = [fetch_run(api, run_id) for run_id in run_ids.keys()]
    df = pl.concat(dfs)
    df.write_csv(path)

  full_width = len(metrics) - 1
  subplot_widths = [full_width] + [(full_width - i) for i in range(len(metrics) - 1)]
  subplot_widths = [w / sum(subplot_widths) for w in subplot_widths]

  subplot_titles = ['total'] + [f'task {i}' for i in range(len(metrics) - 1)]
  fig = make_subplots(
    rows=1,
    cols=len(metrics),
    subplot_titles=subplot_titles,
    column_widths=subplot_widths,
    shared_yaxes=True,
    x_title='task step',
    y_title=metric,
  )

  for idx, key in enumerate(metrics):
    df_key = df.filter(pl.col(key).is_not_null())
    fig_px = px.line(df_key, x='task', y=key, color='experiment')
    for trace in fig_px.data:
      trace.showlegend = idx == 0
      fig.add_trace(trace, row=1, col=idx + 1)
    fig.update_traces(mode='lines+markers', row=1, col=idx + 1)

  fig.update_xaxes(dtick=1)
  # fig.update_yaxes(dtick=1)
  # fig.update_yaxes(dtick=0.1)
  # fig.update_yaxes(dtick=0.01)

  fig.write_image(f'figures/{fname}.png', width=1600, height=400, scale=2)
  fig.show()
