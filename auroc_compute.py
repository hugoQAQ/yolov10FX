import torch 
import numpy as np
# build msp method (pass in pre-saved logits)
def msp_postprocess(logits):
    score = torch.softmax(logits, dim=1)
    conf, pred = torch.max(score, dim=1)
    return pred, conf
def ebo_postprocess(logits, temperature=1):
    score = torch.softmax(logits, dim=1)
    _, pred = torch.max(score, dim=1)
    conf = temperature * torch.logsumexp(logits / temperature,
                                                dim=1)
    return pred, conf
def maxlogits_postprocess(logits):
    conf, pred = torch.max(logits, dim=1)
    return pred, conf
import numpy as np
import torch
import sklearn.covariance

def mahalanobis_compute_mean(logits):
    num_classes = logits[0].shape[0]
    all_preds = np.array([logit.argmax(0) for logit in logits])
    all_preds = torch.from_numpy(all_preds)
    all_labels = all_preds
    all_feats = torch.from_numpy(logits)
    class_mean = []
    centered_data = []
    for c in range(num_classes):
        class_samples = all_feats[all_labels.eq(c)].data
        if class_samples.size(0) > 0:  # Check if there are samples for the class
            mean = class_samples.mean(0)
            centered = class_samples - mean.view(1, -1)
        else:  # If no samples, use a placeholder (e.g., zeros)
            feat_dim = all_feats.size(1)
            mean = torch.zeros(feat_dim)
            centered = torch.empty((0, feat_dim))  # Empty tensor with the correct second dimension
        class_mean.append(mean)
        centered_data.append(centered)

    # Ensure there's at least one class with samples to avoid errors in covariance calculation
    if len(torch.cat(centered_data)) == 0:
        raise ValueError("No samples available for any class to compute covariance.")
    
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    group_lasso.fit(torch.cat(centered_data).cpu().numpy().astype(np.float32))
    precision = torch.from_numpy(group_lasso.precision_).float()
    return class_mean, precision

def mahalanobis_postprocess(logits, class_mean, precision):
    num_classes = logits[0].shape[0]
    pred = logits.argmax(1)
    precision = precision.double()
    class_scores = torch.zeros((logits.shape[0], num_classes))
    for c in range(num_classes):
        if class_mean[c].numel() > 0:  # Check if the class mean was computed
            # Efficient computation of Mahalanobis distance for each sample
            for i in range(logits.shape[0]):
                tensor = logits[i] - class_mean[c].double()
                # Here, we avoid the large matrix multiplication and directly compute the distance
                score = torch.dot(tensor, torch.matmul(precision, tensor))
                class_scores[i, c] = -score  # Negative score to match the original logic

    conf = torch.max(class_scores, dim=1)[0]
    return pred, conf
import numpy as np
import json
import torch
import gc
from openood.evaluators.metrics import compute_all_metrics

def process_and_evaluate(id, postprocess_method, dataset_names, modes=['train', 'val']):
    results = dict()
    results['id'] = dict()

    # Load and process ID data
    for mode in modes:
        results['id'][mode] = dict()
        logits_path = f"/home/hugo/yolov10FX/feats/{model_type}_{id}{scratch}/{id}-{mode}/logits.npy"
        labels_path = f"/home/hugo/yolov10FX/feats/{model_type}_{id}{scratch}/{id}-{mode}/labels.npy"
        results['id'][mode]['logits'] = np.load(logits_path, mmap_mode='r')
        results['id'][mode]['labels'] = np.load(labels_path, mmap_mode='r')

    results['ood'] = dict()

    # Load and process OOD data
    for dataset_name in dataset_names:
        results['ood'][dataset_name] = dict()
        logits_path = f'/home/hugo/yolov10FX/feats/{model_type}_{id}{scratch}/{dataset_name}/logits.npy'
        results['ood'][dataset_name]['logits'] = np.load(logits_path, mmap_mode='r')
        results['ood'][dataset_name]['labels'] = np.full((results['ood'][dataset_name]['logits'].shape[0], ), -1)

    postprocess_results = dict()
    postprocess_results['id'] = dict()

    if postprocess_method == mahalanobis_postprocess:
        class_mean, precision = mahalanobis_compute_mean(results['id']['train']['logits'])
        del results['id']['train']
        torch.cuda.empty_cache()
        gc.collect()
        pred, conf = postprocess_method(torch.from_numpy(results['id']['val']['logits']), class_mean, precision)
    else:
        pred, conf = postprocess_method(torch.from_numpy(results['id']['val']['logits']))

    pred, conf = pred.numpy(), conf.numpy()
    pred = np.full((pred.shape[0], ), 1)
    gt = pred
    postprocess_results['id']['val'] = [pred, conf, gt]

    del results['id']
    torch.cuda.empty_cache()
    gc.collect()

    postprocess_results['ood'] = dict()

    for dataset_name in dataset_names:
        if postprocess_method == mahalanobis_postprocess:
            pred, conf = postprocess_method(torch.from_numpy(results['ood'][dataset_name]['logits']), class_mean, precision)
        else:
            pred, conf = postprocess_method(torch.from_numpy(results['ood'][dataset_name]['logits']))

        pred, conf = pred.numpy(), conf.numpy()
        pred = np.full((pred.shape[0], ), -1)
        gt = pred
        postprocess_results['ood'][dataset_name] = [pred, conf, gt]

        del results['ood'][dataset_name]
        torch.cuda.empty_cache()
        gc.collect()

    return eval_ood(postprocess_results, dataset_names)

def eval_ood(postprocess_results, dataset_names):
    [id_pred, id_conf, id_gt] = postprocess_results['id']['val']
    metrics_list = []
    fpr95 = []
    auroc = []

    for dataset_name in dataset_names:
        [ood_pred, ood_conf, ood_gt] = postprocess_results['ood'][dataset_name]

        pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_gt, ood_gt])

        ood_metrics = compute_all_metrics(conf, label, pred)
        fpr95.append(ood_metrics[0])
        auroc.append(ood_metrics[1])
        metrics_list.append(ood_metrics)

    metrics_list = np.array(metrics_list)
    metrics_mean = np.mean(metrics_list, axis=0)
    fpr95.append(metrics_mean[0])
    auroc.append(metrics_mean[1])

    # return [round(fpr, 4) for fpr in fpr95], [round(auc, 4) for auc in auroc]
    return [round(auc*100, 2) for auc in auroc]

# Evaluation with BM
import numpy as np
import pickle
import tqdm
from utils.monitor_construction import features_clustering_by_k_start, monitor_construction_from_features
from utils.evaluation import get_distance_dataset, compute_fpr95, get_distance_cls

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        pkl_file = pickle.load(f)
    return pkl_file

def save_pkl(file_path, pkl_file):
    with open(file_path, 'wb') as f:
        pickle.dump(pkl_file, f)

def npy2feats_dict(model_type, id, dataset_name):
    logits = np.load(f"/home/hugo/yolov10FX/feats/{model_type}_{id}{scratch}/{dataset_name}/logits.npy")
    labels = np.load(f"/home/hugo/yolov10FX/feats/{model_type}_{id}{scratch}/{dataset_name}/labels.npy")
    feats_dict = dict()
    for i in range(len(logits)):
        label = labels[i]
        if label not in feats_dict:
            feats_dict[label] = []
        feats_dict[label].append(logits[i])
    for k,v in feats_dict.items():
        feats_dict[k] = np.array(v)
    return feats_dict

import numpy as np
from sklearn.metrics import roc_curve, auc

def compute_ood_metrics(scores, labels):
    # Ensure labels are binary (0 for OOD, 1 for ID)
    binary_labels = (labels == 1).astype(int)
    
    # Compute ROC curve points
    fpr, tpr, thresholds = roc_curve(binary_labels, scores)
    
    # Compute AUROC
    auroc = auc(fpr, tpr)
    
    # If AUROC < 0.5, invert the scores and recompute
    if auroc < 0.5:
        fpr, tpr, thresholds = roc_curve(binary_labels, -scores)
        auroc = auc(fpr, tpr)
    
    # Compute FPR95
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx_tpr_95]
    
    return fpr95, auroc

def compute_fpr95(id, monitor_dict, model_type, distances_id, datasets):
    fpr95_list = []
    auroc_list = []
    ood_counts = []
    for dataset_name in datasets:
        feats_ood = npy2feats_dict(model_type, id, dataset_name)
        distances_dict = get_distance_dataset(monitor_dict, feats_ood)
        distances_ood = [distance for k, v in distances_dict.items() for distance in v]
        scores = np.concatenate([distances_id, distances_ood])*-1
        labels = np.concatenate([np.ones(len(distances_id)), np.zeros(len(distances_ood))])
        fpr95, auroc = compute_ood_metrics(scores, labels)
        fpr95_list.append(fpr95)
        auroc_list.append(auroc)
        ood_counts.append(len(distances_ood))
    mean_fpr95 = round(sum(fpr95_list) / len(fpr95_list), 2)
    fpr95_list.append(mean_fpr95)
    mean_auroc = round(sum(auroc_list) / len(auroc_list), 2)
    auroc_list.append(mean_auroc)
    return fpr95_list, auroc_list, ood_counts

def main(id, density, model_type):
    ood_datasets = ["ID-voc-OOD-coco", "OOD-open"] if id == "voc" else ["ID-bdd-OOD-coco", "OOD-open", "voc-ood"]
    monitor_dict = {}
    feats_dict = npy2feats_dict(model_type, id, f"{id}-train")
    for k, v in tqdm.tqdm(feats_dict.items(), desc="category loop", leave=False):
        if len(v) < density:
            continue
        k_start = round(len(v)/density)
        clustering_results = features_clustering_by_k_start(v, k_start)
        monitor_dict[k] = monitor_construction_from_features(v, clustering_results)
    feats_id = npy2feats_dict(model_type, id, f"{id}-val")
    distances_dict = get_distance_dataset(monitor_dict, feats_id)
    distances_id = [distance for k, v in distances_dict.items() for distance in v]
    
    fpr95_list = []
    auroc_list = []
    ood_counts = []
    for dataset_name in ood_datasets:
        feats_ood = npy2feats_dict(model_type, id, dataset_name)
        distances_dict = get_distance_dataset(monitor_dict, feats_ood)
        distances_ood = [distance for k, v in distances_dict.items() for distance in v]
        scores = np.concatenate([distances_id, distances_ood])*-1
        labels = np.concatenate([np.ones(len(distances_id)), np.zeros(len(distances_ood))])
        fpr95, auroc = compute_ood_metrics(scores, labels)
        fpr95_list.append(round(fpr95*100,2))
        auroc_list.append(round(auroc*100,2))
        ood_counts.append(len(distances_ood))
    mean_fpr95 = round(sum(fpr95_list) / len(fpr95_list), 2)
    fpr95_list.append(mean_fpr95)
    mean_auroc = round(sum(auroc_list) / len(auroc_list), 2)
    auroc_list.append(mean_auroc)
    print("evalution using bam")
    print(fpr95_list)
    print(auroc_list)
    print(ood_counts)
    return auroc_list, ood_counts

postprocess_dict = {"msp": msp_postprocess, "ebo": ebo_postprocess, "mls": maxlogits_postprocess, "mds": mahalanobis_postprocess}
id, scratch = "bdd", ""
ood_datasets = ["ID-voc-OOD-coco", "OOD-open"] if id == "voc" else ["ID-bdd-OOD-coco", "OOD-open", "voc-ood"]
for model_type in ["v10s", "v10m", "v10l"]:
    print(f"evaluation on {id}-{model_type}{scratch} model")
    for k,v in postprocess_dict.items():
        print(f"evalution using {k}")
        print(process_and_evaluate(id, v, dataset_names=ood_datasets))
    main(id, 50, model_type)