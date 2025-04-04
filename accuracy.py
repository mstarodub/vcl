import torch


# TODO
def classifier_certainty(classifier, gen, target):
  return torch.tensor(0)


def accuracy(pred, target):
  # undo one-hot encoding, if applicable
  target_idx = target.argmax(dim=1) if target.ndim == pred.ndim else target
  return (pred.argmax(dim=1) == target_idx).float().mean()
