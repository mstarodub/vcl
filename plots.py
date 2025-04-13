import sys
import wandb
import polars as pl
import plotly.express as px

# e.g. 'test/test_uncert'
metrics = (
  [
    # 'test/test_acc',
  ]
  + [f'test/test_acc_task_{i}' for i in range(10)]
)


def download_media(run, metric):
  history = run.scan_history(keys=[metric], page_size=1_000_000)
  for row in history:
    path = row[metric]['path']
    run.file(path).download()


def fetch_metric(run, metric):
  # use '_step' for cont. logged metrics
  history = run.scan_history(keys=['task', metric], page_size=1_000_000)
  return list(history)


if __name__ == '__main__':
  id = sys.argv[1]
  fname = 'disc_vcl_pmnist_per_task_acc'

  api = wandb.Api()
  run = api.run(f'mxst-university-of-oxford/vcl/{id}')

  xss = [fetch_metric(run, m) for m in metrics]
  dfs = [pl.from_dicts(xs) for xs in xss if xs]
  df = dfs[0]
  for d in dfs[1:]:
    df = df.join(d, on='task', how='full', coalesce=True)

  variables = df.drop('task').columns
  fig = px.line(
    df,
    x='task',
    y=variables,
    labels={
      'value': 'test accuracy',
      'variable': 'task',
    },
  )
  fig.update_traces(mode='lines+markers')
  fig.update_layout(xaxis_dtick=1, showlegend=len(variables) > 1)
  rename = {col: col.split('_')[-1] for col in variables}
  fig.for_each_trace(lambda t: t.update(name=rename[t.name]))
  fig.write_image(f'figures/{fname}.png', width=800, height=600, scale=2)
  fig.show()
