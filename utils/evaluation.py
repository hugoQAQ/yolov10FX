import torch
from .file_operations import load_pkl
# distance from point to an interval
def point_ival_dis(mins, maxs, X):
    lower_dis = mins - X
    upper_dis = X - maxs
    in_range = (mins <= X) & (X <= maxs)
    distance = torch.where(in_range, torch.zeros_like(X), torch.where(X < mins, lower_dis, upper_dis))
    return distance
# get min distance for a point to boxes
def get_min_distance(mins, maxs, X):
    point_box_dis = torch.sum(point_ival_dis(mins, maxs, X), dim=1)
    min_distance = torch.min(point_box_dis)
    return round(min_distance.item(), 4)


def get_distance_cls(feats_cls, monitor_cls):
    lb_monitors = torch.stack([torch.tensor(monitor.ivals)[:,0] for monitor in monitor_cls.good_ref])
    ub_monitors = torch.stack([torch.tensor(monitor.ivals)[:,1] for monitor in monitor_cls.good_ref])
    distances = []
    for feat in feats_cls:
        distances.append(get_min_distance(lb_monitors, ub_monitors, feat))
    return distances
def get_distance_dataset(monitors_dict, feats_dataset):
    distances_dict = {}
    for k, v in feats_dataset.items():
        v = torch.tensor(v)
        if k not in monitors_dict:
            continue
        monitor_cls = monitors_dict[k]
        distance_cls = get_distance_cls(v, monitor_cls)
        distances_dict[k] = distance_cls
    return distances_dict
def get_tpr(distances_cls, threshold):
    count = sum(1 for distance in distances_cls if distance < threshold)
    percentage = round((count / len(distances_cls)) * 100, 2)
    return percentage, count, len(distances_cls)
def get_fpr(distances_cls, threshold):
    count = sum(1 for distance in distances_cls if distance > threshold)
    percentage = round((count / len(distances_cls)) * 100, 2)
    return percentage, count, len(distances_cls)

def compute_fpr95(id, backbone, threshold, dataset_name, monitors_dict):
    feats_ood = load_pkl(f"feats/{id}/{backbone}/{dataset_name}.pkl")
    distances_dict = get_distance_dataset(monitors_dict, feats_ood)
    distances_ood = [distance for k, v in distances_dict.items() for distance in v]
    count = sum(1 for distance in distances_ood if distance < threshold)
    percentage = round((count / len(distances_ood)) * 100, 2)
    return percentage