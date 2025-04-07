import sys
import wandb
import polars as pl
import plotly.express as px

# e.g. 'test/test_uncert'
target = 'test/test_uncert_task_3'

if __name__ == '__main__':
  api = wandb.Api()
  run = api.run(f'mxst-university-of-oxford/vcl/{sys.argv[1]}')
  # use '_step' for cont. logged metrics
  history = run.scan_history(keys=['task', target], page_size=1_000_000)
  for row in history:
    print(row)

  # all files including stuff like the config
  # for file in run.files(per_page=1_000):
  #   print(file)

  # just the media
  # for row in run.scan_history(keys=['samples']):
  #   path = row['samples']['path']
  #   print(path)
  #   run.file(path).download()

  df = pl.from_dicts(list(history))
  # px.scatter(df, x='task', y=target).show()
