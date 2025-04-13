# pmnist
# vcl nocoreset
# r09vadzh
# vcl coreset
# si
# baseline cata forgetting

# smnist
# vcl
# si
# baseline
# baseline cata forgetting

# nmnist
# vcl
# si
# baseline
# baseline cata forgetting

### (generative)

# mnist
# vcl
# 7ei1kvx5
# si
# lfq5zvai
# arch2 perfect
# adz84vfv
# baseline cata forgetting
# TODO

# nmnist
# vcl
# 6fb6ibf8
# si
# 0n1bb5zu
# arch2 perfect
# TODO
# baseline cata forgetting
# ohbbj5vm

import wandb
import polars as pl
import plotly.express as px
from plotly.subplots import make_subplots

# metrics = [f'test/test_acc_task_{i}' for i in range(10)] + ['test/test_acc']
# metrics = [f'test/test_ll_task_{i}' for i in range(10)] + ['test/test_ll']
metrics = [f'test/test_uncert_task_{i}' for i in range(10)] + ['test/test_uncert']
metric = 'classifier uncertainty'
run_ids = {
  '7ei1kvx5': 'vcl',
  'lfq5zvai': 'si',
}
fname = 'gen_vcl_pmnist_per_task_acc'


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
  df = dfs[0]
  for d in dfs[1:]:
    df = df.join(d, on='task', how='full', coalesce=True)
  df = df.with_columns(pl.lit(run_ids[run_id]).alias('experiment'))
  return df


if __name__ == '__main__':
  api = wandb.Api()

  dfs = [fetch_run(api, run_id) for run_id in run_ids.keys()]
  df = pl.concat(dfs)

  full_width = len(metrics) - 1
  subplot_widths = [(full_width - i) for i in range(len(metrics) - 1)] + [full_width]
  subplot_widths = [w / sum(subplot_widths) for w in subplot_widths]

  subplot_titles = [f'task {i}' for i in range(len(metrics) - 1)] + ['total']
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

  # fig.write_image(f'figures/{fname}.png', width=1600, height=400, scale=2)
  fig.show()
