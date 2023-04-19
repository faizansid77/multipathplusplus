import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from model.modules import MLP, CGBlock, MCGBlock, HistoryEncoder
from model.multipathpp import MultiPathPP
from model.data import get_dataloader, dict_to_cuda, normalize
from model.losses import pytorch_neg_multi_log_likelihood_batch, nll_with_covariances
from prerender.utils.utils import data_to_numpy, get_config
import subprocess
from matplotlib import pyplot as plt
import os
import glob
import sys
import random

from waymo_open_dataset.metrics.ops import py_metrics_ops

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class MotionMetrics():
  """Wrapper for motion metrics computation."""

  def __init__(self, config):
    super().__init__()
    self._prediction_trajectory = []
    self._prediction_score = []
    self._ground_truth_trajectory = []
    self._ground_truth_is_valid = []
    self._prediction_ground_truth_indices = []
    self._prediction_ground_truth_indices_mask = []
    self._object_type = []
    self._metrics_config = config

  def update_state(self, prediction_trajectory, prediction_score,
                   ground_truth_trajectory, ground_truth_is_valid,
                   prediction_ground_truth_indices,
                   prediction_ground_truth_indices_mask, object_type):
    self._prediction_trajectory.append(prediction_trajectory)
    self._prediction_score.append(prediction_score)
    self._ground_truth_trajectory.append(ground_truth_trajectory)
    self._ground_truth_is_valid.append(ground_truth_is_valid)
    self._prediction_ground_truth_indices.append(
        prediction_ground_truth_indices)
    self._prediction_ground_truth_indices_mask.append(
        prediction_ground_truth_indices_mask)
    self._object_type.append(object_type)

  def result(self):
    # [batch_size, num_preds, 1, 1, steps, 2].
    # The ones indicate top_k = 1, num_agents_per_joint_prediction = 1.
    prediction_trajectory = torch.concat(self._prediction_trajectory, 0)
    # [batch_size, num_preds, 1].
    prediction_score = torch.concat(self._prediction_score, 0)
    # [batch_size, num_agents, gt_steps, 7].
    ground_truth_trajectory = torch.concat(self._ground_truth_trajectory, 0)
    # [batch_size, num_agents, gt_steps].
    ground_truth_is_valid = torch.concat(self._ground_truth_is_valid, 0)
    # [batch_size, num_preds, 1].
    prediction_ground_truth_indices = torch.concat(
        self._prediction_ground_truth_indices, 0)
    # [batch_size, num_preds, 1].
    prediction_ground_truth_indices_mask = torch.concat(
        self._prediction_ground_truth_indices_mask, 0)
    # [batch_size, num_agents].
    object_type = torch.concat(self._object_type, 0).type(torch.int64)

    # We are predicting more steps than needed by the eval code. Subsample.
    interval = (
        self._metrics_config.track_steps_per_second //
        self._metrics_config.prediction_steps_per_second)
    prediction_trajectory = prediction_trajectory[...,
                                                  (interval - 1)::interval, :]

    return py_metrics_ops.motion_metrics(
        config=self._metrics_config.SerializeToString(),
        prediction_trajectory=prediction_trajectory,
        prediction_score=prediction_score,
        ground_truth_trajectory=ground_truth_trajectory,
        ground_truth_is_valid=ground_truth_is_valid,
        prediction_ground_truth_indices=prediction_ground_truth_indices,
        prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
        object_type=object_type)


def get_last_file(path):
    list_of_files = glob.glob(f'{path}/*')
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

config = get_config(sys.argv[1])
alias = sys.argv[1].split("/")[-1].split(".")[0]
try:
    models_path = "../models"
    # os.mkdir(tb_path)
    os.mkdir(models_path)
except:
    pass
last_checkpoint = get_last_file(models_path)
# dataloader = get_dataloader(config["train"]["data_config"])
test_dataloader = get_dataloader(config["test"]["data_config"])
model = MultiPathPP(config["model"])
model.cuda()

num_steps = 0
if last_checkpoint is not None:
    model.load_state_dict(torch.load(last_checkpoint)["model_state_dict"])
    
    num_steps = torch.load(last_checkpoint)["num_steps"]
    print("LOADED ", last_checkpoint)
this_num_steps = 0
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("N PARAMS=", params)

model.eval()
with torch.no_grad():
    losses = []
    min_ades = []
    first_batch = True
    for data in tqdm(test_dataloader):
        if config["test"]["normalize"]:
            data = normalize(data, config)
        dict_to_cuda(data)
        probas, coordinates, cov_mat, loss_coeff = model(data, num_steps)
        print(probas)
        print(coordinates)
        print(cov_mat)
        print(loss_coeff)
        print(data)
        batch_size = torch.shape(data['tracks_to_predict'])[0]
        num_samples = torch.shape(data['tracks_to_predict'])[1]

        pred_gt_indices = torch.arange(num_samples, dtype=torch.int64)
        # [batch_size, num_agents, 1].
        pred_gt_indices = torch.tile(pred_gt_indices[torch.newaxis, :, torch.newaxis],
                                    (batch_size, 1, 1))
        # [batch_size, num_agents, 1].
        pred_gt_indices_mask = inputs['tracks_to_predict'][..., torch.newaxis]
        if config["test"]["normalize_output"]:
            coordinates = coordinates * 10. + torch.Tensor([1.4715e+01, 4.3008e-03]).cuda()

        pred_score = torch.ones(shape=torch.shape(coordinates)[:3])
        # Compute ADE for each predicted trajectory
        ade_list = []
        for i in range(len(data["track_id"])):

            # pred_score = np.trace(cov_mat[:, i, :]) + loss_coeff[:, i] * np.linalg.det(cov_mat[:, i, :])
            pred_traj = coordinates[:, i, :2].cpu().numpy()
            if config["train"]["normalize_output"]:
                gt_traj = (data["target/future/xy"] - torch.Tensor([1.4715e+01, 4.3008e-03]).cuda()) / 10.
            gt_valid = data["target/future/valid"][:, i, :2].cpu().numpy()
            ade = metrics.motion_metrics.get_motion_metric_ops(pred_traj, pred_score, gt_traj, gt_valid, max_distance=2.0)
            ade_list.append(ade)
        # Compute minimum ADE across all predicted trajectories
        min_ade = min(ade_list)
        min_ades.append(min_ade)


