import wandb
import polars as pl
import plotly.express as px
from plotly.subplots import make_subplots
import os

import experiments


def download_media(run, key):
  history = run.scan_history(keys=[key], page_size=1_000_000)
  for row in history:
    path = row[key]['path']
    run.file(path).download()


def fetch_key(run, key):
  # use '_step' for cont. logged keys
  history = run.scan_history(keys=['task', key], page_size=1_000_000)
  return list(history)


def fetch_run(api, run_id, label, metrics):
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
  df = df.with_columns(pl.lit(label).alias('experiment'))
  return df


def plot_per_task(api, run_ids, metrics, metric_name, fname, y_dtick=None):
  path = f'figures/data_raw/{fname}.csv'
  if os.path.exists(path):
    df = pl.read_csv(path)
  else:
    dfs = [fetch_run(api, run_id, label, metrics) for run_id, label in run_ids.items()]
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
    y_title=metric_name,
  )

  for idx, key in enumerate(metrics):
    df_key = df.filter(pl.col(key).is_not_null())
    fig_px = px.line(df_key, x='task', y=key, color='experiment')
    for trace in fig_px.data:
      trace.showlegend = idx == 0
      fig.add_trace(trace, row=1, col=idx + 1)
    fig.update_traces(mode='lines+markers', row=1, col=idx + 1)

  fig.update_xaxes(dtick=1)
  if y_dtick:
    fig.update_yaxes(dtick=y_dtick)

  fig.write_image(f'figures/{fname}.png', width=1600, height=400, scale=2)
  fig.show()


def plot_disc_pmnist_per_task_acc(api, with_coreset):
  if with_coreset:
    run_ids = {
      'r09vadzh': 'vcl',
      'dcy7qany': 'vcl + coreset (25k)',
      'z7b4hxg6': 'vcl + coreset (10k)',
      'gdo97xiw': 'vcl + coreset (5k)',
      'co0e8wpq': 'vcl + coreset (4k)',
      '5nwbbi8b': 'vcl + coreset (2.5k)',
      'jzc7wtcp': 'vcl + coreset (2k)',
    }
    fname = 'disc_pmnist_per_task_acc_coreset'
    y_dtick = 0.01
  else:
    run_ids = {
      'r09vadzh': 'vcl',
      '8nk4zrmw': 'si',
      'ora1ykic': 'baseline',
    }
    fname = 'disc_pmnist_per_task_acc'
    y_dtick = 0.1
  metrics = ['test/test_acc'] + [
    f'test/test_acc_task_{i}' for i in range(experiments.disc_pmnist['ntasks'])
  ]
  plot_per_task(api, run_ids, metrics, 'test accuracy', fname, y_dtick=y_dtick)


def plot_disc_smnist_per_task_acc(api):
  run_ids = {
    'kn20dwka': 'vcl',
    'eem35sel': 'si',
    '21kt316w': 'baseline',
  }
  metrics = ['test/test_acc'] + [
    f'test/test_acc_task_{i}' for i in range(experiments.disc_smnist['ntasks'])
  ]
  fname = 'disc_smnist_per_task_acc'
  plot_per_task(api, run_ids, metrics, 'test accuracy', fname, y_dtick=0.01)


def plot_disc_nmnist_per_task_acc(api, with_baseline):
  if with_baseline:
    run_ids = {
      '7fkimt7s': 'vcl',
      '67fmhyvf': 'si',
      'hn8gnzey': 'baseline',
    }
    fname = 'disc_nmnist_per_task_acc_withbaseline'
    y_dtick = 0.1
  else:
    run_ids = {
      '7fkimt7s': 'vcl',
      '67fmhyvf': 'si',
    }
    fname = 'disc_nmnist_per_task_acc'
    y_dtick = 0.01
  metrics = ['test/test_acc'] + [
    f'test/test_acc_task_{i}' for i in range(experiments.disc_nmnist['ntasks'])
  ]
  plot_per_task(api, run_ids, metrics, 'test accuracy', fname, y_dtick=y_dtick)


def plot_gen_mnist(api, metric_type, with_baseline):
  if with_baseline and metric_type == 'll':
    run_ids = {
      '7ei1kvx5': 'vcl',
      'lfq5zvai': 'si',
      '6ty15p6s': 'baseline',
    }
  elif with_baseline and metric_type == 'uncert':
    run_ids = {
      '7ei1kvx5': 'vcl',
      'lfq5zvai': 'si',
      '6ty15p6s': 'baseline',
      'adz84vfv': 'arch2 baseline',
    }
  else:
    run_ids = {
      '7ei1kvx5': 'vcl',
      'lfq5zvai': 'si',
    }
  if metric_type == 'uncert':
    metrics = ['test/test_uncert'] + [
      f'test/test_uncert_task_{i}' for i in range(experiments.gen_mnist['ntasks'])
    ]
    metric_name = 'classifier uncertainty'
    fname_prefix = 'gen_mnist_per_task_uncert'
  elif metric_type == 'll':
    metrics = ['test/test_ll'] + [
      f'test/test_ll_task_{i}' for i in range(experiments.gen_mnist['ntasks'])
    ]
    metric_name = 'test log-likelihood'
    fname_prefix = 'gen_mnist_per_task_test_ll'
  else:
    return
  fname = fname_prefix + ('_withbaseline' * with_baseline)
  plot_per_task(api, run_ids, metrics, metric_name, fname, y_dtick=None)


def plot_gen_nmnist(api, metric_type, with_baseline):
  if with_baseline and metric_type == 'll':
    run_ids = {
      '6fb6ibf8': 'vcl',
      '0n1bb5zu': 'si',
      'ohbbj5vm': 'baseline',
    }
  elif with_baseline and metric_type == 'uncert':
    run_ids = {
      '6fb6ibf8': 'vcl',
      '0n1bb5zu': 'si',
      'ohbbj5vm': 'baseline',
      'gs8imknj': 'arch2 baseline',
    }
  else:
    run_ids = {
      '6fb6ibf8': 'vcl',
      '0n1bb5zu': 'si',
    }
  if metric_type == 'uncert':
    metrics = ['test/test_uncert'] + [
      f'test/test_uncert_task_{i}' for i in range(experiments.gen_nmnist['ntasks'])
    ]
    metric_name = 'classifier uncertainty'
    fname_prefix = 'gen_nmnist_per_task_uncert'
  elif metric_type == 'll':
    metrics = ['test/test_ll'] + [
      f'test/test_ll_task_{i}' for i in range(experiments.gen_nmnist['ntasks'])
    ]
    metric_name = 'test log-likelihood'
    fname_prefix = 'gen_nmnist_per_task_test_ll'
  else:
    return
  fname = fname_prefix + ('_withbaseline' * with_baseline)
  plot_per_task(api, run_ids, metrics, metric_name, fname, y_dtick=None)


def plot_total_acc_pmnist_smnist_nmnist(api):
  exp_run_ids = {
    'permuted MNIST': {
      '8nk4zrmw': 'si',
      'r09vadzh': 'vcl',
      'dcy7qany': 'vcl + coreset (25k)',
    },
    'split MNIST': {
      'eem35sel': 'si',
      'kn20dwka': 'vcl',
    },
    'notMNIST': {
      '67fmhyvf': 'si',
      '7fkimt7s': 'vcl',
    },
  }
  metrics = ['test/test_acc']
  metric_name = 'test accuracy (total)'
  fname = 'total_acc_pmnist_smnist_nmnist'
  plot_total_comparison(api, exp_run_ids, metrics, metric_name, fname)


def plot_total_comparison(api, exp_run_ids, metrics, metric_name, fname):
  path = f'figures/data_raw/{fname}.csv'
  if os.path.exists(path):
    df_all = pl.read_csv(path)
  else:
    all_dfs = []
    for exp in exp_run_ids.keys():
      for run_id, label in exp_run_ids[exp].items():
        df = fetch_run(api, run_id, label, metrics)
        df = df.with_columns(pl.lit(exp).alias('experiment_group'))
        all_dfs.append(df)
    df_all = pl.concat(all_dfs)
    df_all.write_csv(path)

  fig = make_subplots(
    rows=1,
    cols=len(exp_run_ids),
    subplot_titles=exp_run_ids.keys(),
    x_title='task step',
    y_title=metric_name,
  )

  for idx, exp in enumerate(exp_run_ids.keys()):
    df_exp = df_all.filter(pl.col('experiment_group') == exp)
    for metric in metrics:
      fig_px = px.line(df_exp, x='task', y=metric, color='experiment')
      for trace in fig_px.data:
        trace.showlegend = idx == 0
        fig.add_trace(trace, row=1, col=idx + 1)
    fig.update_traces(mode='lines+markers', row=1, col=idx + 1)

  fig.update_xaxes(dtick=1)
  fig.update_yaxes(dtick=0.02)

  fig.write_image(
    f'figures/{fname}.png', width=400 * len(exp_run_ids), height=400, scale=2
  )
  fig.show()


def plot_pmnist_vcl_regression_total_acc_rmse(api):
  run_ids = {
    'z1tu65iy': 'vcl regression homoscedastic',
    'bnbg5mo1': 'vcl regression heteroscedastic',
    'r09vadzh': 'vcl',
  }
  metrics = ['test/test_acc', 'test/test_rmse']
  fname = 'pmnist_vcl_regression_total_acc_rmse'
  path = f'figures/data_raw/{fname}.csv'
  if os.path.exists(path):
    df = pl.read_csv(path)
  else:
    dfs = [fetch_run(api, run_id, label, metrics) for run_id, label in run_ids.items()]
    df = pl.concat(dfs, how='diagonal')
    df.write_csv(path)

  fig = make_subplots(
    rows=1,
    cols=len(metrics),
    subplot_titles=['test accuracy', 'test RMSE'],
    x_title='task step',
  )
  # acc subplot
  df_acc = df.filter(pl.col(metrics[0]).is_not_null())
  fig_px_acc = px.line(df_acc, x='task', y=metrics[0], color='experiment')
  for trace in fig_px_acc.data:
    fig.add_trace(trace, row=1, col=1)
  # rmse subplot
  df_rmse = df.filter(
    (pl.col(metrics[1]).is_not_null())
    & (pl.col('experiment').str.contains('regression'))
  )
  fig_px_rmse = px.line(df_rmse, x='task', y=metrics[1], color='experiment')
  for trace in fig_px_rmse.data:
    trace.showlegend = False
    fig.add_trace(trace, row=1, col=2)

  fig.update_traces(mode='lines+markers')
  fig.update_xaxes(dtick=1)
  fig.update_yaxes(dtick=0.05, row=1, col=1)

  fig.write_image(f'figures/{fname}.png', width=2 * 400, height=400, scale=2)
  fig.show()


def plot_pmnist_vcl_regression_per_task(api, metric_type, reg_type):
  if reg_type == 'hetero':
    run_ids = {
      'bnbg5mo1': 'vcl regression heteroscedastic',
    }
  elif reg_type == 'homo':
    run_ids = {
      'z1tu65iy': 'vcl regression homoscedastic',
    }
  else:
    return

  if metric_type == 'acc':
    metrics = ['test/test_acc'] + [
      f'test/test_acc_task_{i}' for i in range(experiments.disc_pmnist['ntasks'])
    ]
    metric_name = 'test accuracy'
    fname = f'pmnist_vcl_regression_{reg_type}_per_task_acc'
    y_dtick = 0.01 if reg_type == 'homo' else 0.05
  elif metric_type == 'rmse':
    metrics = ['test/test_rmse'] + [
      f'test/test_rmse_task_{i}' for i in range(experiments.disc_pmnist['ntasks'])
    ]
    metric_name = 'test RMSE'
    fname = f'pmnist_vcl_regression_{reg_type}_per_task_rmse'
    y_dtick = None
  else:
    return
  plot_per_task(api, run_ids, metrics, metric_name, fname, y_dtick=y_dtick)


if __name__ == '__main__':
  api = wandb.Api()

  # plot_disc_pmnist_per_task_acc(api, with_coreset=False)
  # plot_disc_pmnist_per_task_acc(api, with_coreset=True)
  # plot_disc_smnist_per_task_acc(api)
  # plot_disc_nmnist_per_task_acc(api, with_baseline=False)
  # plot_disc_nmnist_per_task_acc(api, with_baseline=True)
  # plot_gen_mnist(api, 'uncert', with_baseline=False)
  # plot_gen_mnist(api, 'uncert', with_baseline=True)
  # plot_gen_mnist(api, 'll', with_baseline=False)
  # plot_gen_mnist(api, 'll', with_baseline=True)
  # plot_gen_nmnist(api, 'uncert', with_baseline=False)
  # plot_gen_nmnist(api, 'uncert', with_baseline=True)
  # plot_gen_nmnist(api, 'll', with_baseline=False)
  # plot_gen_nmnist(api, 'll', with_baseline=True)
  # plot_total_acc_pmnist_smnist_nmnist(api)
  # plot_pmnist_vcl_regression_total_acc_rmse(api)
  # plot_pmnist_vcl_regression_per_task(api, 'acc', 'homo')
  # plot_pmnist_vcl_regression_per_task(api, 'rmse', 'homo')
  # plot_pmnist_vcl_regression_per_task(api, 'acc', 'hetero')
  # plot_pmnist_vcl_regression_per_task(api, 'rmse', 'hetero')
