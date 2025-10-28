import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import sklearn.neighbors._base
import sys
import os
import torch.cuda.amp as amp
from itertools import chain
import time
import psutil


sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train(model, model1, config, train_loader, valid_loader=None, foldername=""):
    optimizer = AdamW(
        chain(model.parameters(), model1.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )

    scaler = amp.GradScaler()
    p1, p2 = int(0.75 * config["epochs"]), int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)
    criterion = nn.CrossEntropyLoss(reduction='none')

    total_epochs = config["epochs"]

    for epoch_no in range(total_epochs):
        model.train()
        model1.train()

        avg_loss, fl_loss, mse_loss = 0.0, 0.0, 0.0

        initial_noise_w, initial_mse_w, initial_ce_w = 1, 1, 0.1
        final_noise_w, final_mse_w, final_ce_w = 0.1, 0.1, 0.5

        mean = np.array([6.115921e+01, 0.0, 5.012372e+03, 8.577913e+01, 1.231834e+02,
                         6.487961e+01, 1.981481e+01, 3.699407e+01, 9.676768e+01, 5.628323e+00,
                         3.741237e+00, 3.576394e+00, 1.126222e+02, 2.730177e+00, 1.060125e+02,
                         1.392093e+02, 1.506129e+00, 2.725690e+01, 8.284392e+00, 1.047430e+02,
                         1.505578e+00, 1.417590e+02, 1.020023e+01, 3.089938e+01, 2.470126e+01,
                         4.252328e+01, 1.160214e+02, 5.111877e+01, 2.031380e+02, 4.016609e+00,
                         5.781657e+00, 4.263790e+00, 3.486935e+00, 1.390026e+02, 1.182809e+01,
                         5.044114e+01, 4.205614e+02])

        std = np.array([1.596153e+01, 1.0, 7.525188e+03, 1.842967e+01, 2.342812e+01,
                        1.480334e+01, 5.885128e+00, 7.552111e-01, 2.966343e+00, 9.935996e-01,
                        1.682795e+00, 8.242207e-01, 3.537135e+02, 6.699034e-01, 9.850513e+01,
                        5.331702e+02, 3.096187e+00, 2.140450e+01, 7.646539e-01, 6.896862e+00,
                        1.597400e+00, 5.823925e+01, 2.166648e+00, 6.353790e+00, 6.220732e+00,
                        1.169103e+01, 7.017883e+01, 2.324614e+01, 1.085020e+02, 6.091881e-01,
                        9.368000e-01, 1.462472e+00, 7.327223e-01, 5.953719e+00, 6.388267e+00,
                        3.356331e+02, 4.840134e+02])

        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    loss0, generated_data, gt_point, status, original_data, imputed_point, seq_length = model(train_batch)
                    loss1 = ((generated_data - original_data) ** 2 * gt_point).sum() / gt_point.sum()
                    imputed_data = (generated_data * (1 - imputed_point) + original_data * imputed_point)

                    outputs = model1(imputed_data.permute(0, 2, 1), seq_length)
                    labels_expanded = status.unsqueeze(1).expand(-1, outputs.shape[1])

                    lb, ub = torch.tensor([...]), torch.tensor([...])
                    over_ub = torch.clamp(imputed_data * std + mean - ub, min=0)
                    under_lb = torch.clamp(lb - imputed_data * std + mean, min=0)
                    range_loss = ((over_ub ** 2 + under_lb ** 2) * gt_point).sum() / gt_point.sum()

                    loss_per_step = criterion(outputs.reshape(-1, 2), labels_expanded.reshape(-1))
                    loss_per_step = loss_per_step.view(outputs.shape[0], outputs.shape[1])

                    mask = torch.arange(outputs.size(1), device='cuda').unsqueeze(0) < seq_length.unsqueeze(1)
                    loss2 = (loss_per_step * mask.float()).sum() / mask.float().sum()

                    alpha = epoch_no / total_epochs
                    noise_w = initial_noise_w * (1 - alpha) + final_noise_w * alpha
                    mse_w = initial_mse_w * (1 - alpha) + final_mse_w * alpha
                    ce_w = initial_ce_w * (1 - alpha) + final_ce_w * alpha

                    total_loss = ce_w * loss2 + mse_w * range_loss + noise_w * loss0

                    it.set_postfix(avg_df_loss=loss0.item(), avg_fl_loss=loss2.item(), mse_loss=loss1.item(), range_loss=range_loss.item(), epoch=epoch_no)

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                del outputs, generated_data, gt_point, status, original_data, imputed_point, imputed_data
                torch.cuda.empty_cache()

                avg_loss += loss0.item()
                mse_loss += loss1.item()
                fl_loss += loss2.item()

        lr_scheduler.step()

        model_save_path, model1_save_path = os.path.join('logs', f'model_[{epoch_no + 1}].pth'), os.path.join('logs', f'model1_[{epoch_no + 1}].pth')
        torch.save(model.state_dict(), model_save_path)
        torch.save(model1.state_dict(), model1_save_path)

        if valid_loader and (epoch_no + 1) % 10 == 0:
            validate_model(model, model1, valid_loader)


def validate_model(model, model1, valid_loader):
    model.eval()
    model1.eval()
    all_probs, all_labels, all_preds = [], [], []

    with torch.no_grad():
        with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, valid_batch in enumerate(it, start=1):
                lose0, produce_data, gtv_point, label, initial_data, filled_point, seq_length = model(valid_batch, is_train=0)
                valid_data = (produce_data * (1 - filled_point) + initial_data * filled_point)
                outputs = model1(valid_data.permute(0, 2, 1), seq_length)

                probs = torch.softmax(outputs, dim=2)
                mask = (torch.arange(outputs.size(1), device='cuda').unsqueeze(0) < seq_length.unsqueeze(1))

                valid_probs = probs[mask]
                valid_labels = label.unsqueeze(1).expand(-1, outputs.size(1))[mask]

                all_probs.append(valid_probs.cpu().numpy())
                all_labels.append(valid_labels.cpu().numpy())
                all_preds.append(valid_probs.argmax(dim=1).cpu().numpy())

                del outputs, produce_data, gtv_point, label, initial_data, filled_point, valid_data
                torch.cuda.empty_cache()

        all_probs, all_labels, all_preds = np.concatenate(all_probs), np.concatenate(all_labels), np.concatenate(all_preds)
        accuracy = (all_preds == all_labels).mean()
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()

        print(f'Validation Results:\nAccuracy: {accuracy:.2%}, AUC: {auc:.4f}\nConfusion Matrix:\n{cm}\nTN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')


def rounding(imputed_data, data_x):
    """Round imputed data for categorical variables."""
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()
    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])
    return rounded_data


def test(model, model1, ext_loader):
    model.eval()
    model1.eval()
    all_probs, all_labels, all_preds = [], [], []
    initial_all, point_all, produce_all, test_csdi, sample_data_list, test_point, label_all = [], [], [], [], [], [], []
    seq_length_list = []

    mean = np.array([6.115921e+01, 0.0, 5.012372e+03, 8.577913e+01, 1.231834e+02,
                     6.487961e+01, 1.981481e+01, 3.699407e+01, 9.676768e+01, 5.628323e+00,
                     3.741237e+00, 3.576394e+00, 1.126222e+02, 2.730177e+00, 1.060125e+02,
                     1.392093e+02, 1.506129e+00, 2.725690e+01, 8.284392e+00, 1.047430e+02,
                     1.505578e+00, 1.417590e+02, 1.020023e+01, 3.089938e+01, 2.470126e+01,
                     4.252328e+01, 1.160214e+02, 5.111877e+01, 2.031380e+02, 4.016609e+00,
                     5.781657e+00, 4.263790e+00, 3.486935e+00, 1.390026e+02, 1.182809e+01,
                     5.044114e+01, 4.205614e+02])

    std = np.array([1.596153e+01, 1.0, 7.525188e+03, 1.842967e+01, 2.342812e+01,
                    1.480334e+01, 5.885128e+00, 7.552111e-01, 2.966343e+00, 9.935996e-01,
                    1.682795e+00, 8.242207e-01, 3.537135e+02, 6.699034e-01, 9.850513e+01,
                    5.331702e+02, 3.096187e+00, 2.140450e+01, 7.646539e-01, 6.896862e+00,
                    1.597400e+00, 5.823925e+01, 2.166648e+00, 6.353790e+00, 6.220732e+00,
                    1.169103e+01, 7.017883e+01, 2.324614e+01, 1.085020e+02, 6.091881e-01,
                    9.368000e-01, 1.462472e+00, 7.327223e-01, 5.953719e+00, 6.388267e+00,
                    3.356331e+02, 4.840134e+02])

    start_time = time.time()
    process = psutil.Process(os.getpid())
    peak_mem_usage = 0
    n_samples = 0

    with torch.no_grad():
        with tqdm(ext_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, valid_batch in enumerate(it, start=1):
                batch_start = time.time()

                lose0, produce_data, gtv_point, label, initial_data, filled_point, seq_length = model(valid_batch, is_train=0)
                valid_data = (produce_data * (1 - filled_point) + initial_data * filled_point)

                for i in range(valid_data.shape[0]):
                    actual_length = seq_length[i].item()
                    sample = valid_data[i, :, :actual_length].cpu().numpy()

                    sample_data_list.append(sample)
                    seq_length_list.append(actual_length)

                test_csdi.append(valid_data)
                initial_all.append(initial_data)
                point_all.append(gtv_point)
                produce_all.append(produce_data)
                test_point.append(filled_point)
                label_all.append(label)

                outputs = model1(valid_data.permute(0, 2, 1), seq_length)

                probs = torch.softmax(outputs, dim=2)
                mask = torch.arange(outputs.size(1), device='cpu').unsqueeze(0) < seq_length.unsqueeze(1)

                valid_probs = probs[mask]
                valid_labels = label.unsqueeze(1).expand(-1, seq_len)[mask]

                all_probs.append(valid_probs.cpu().numpy())
                all_labels.append(valid_labels.cpu().numpy())
                all_preds.append(valid_probs.argmax(dim=1).cpu().numpy())

                batch_size, seq_len, _ = outputs.shape
                n_samples += batch_size

                mem = process.memory_info().rss / 1024 ** 2
                peak_mem_usage = max(peak_mem_usage, mem)
                batch_time = time.time() - batch_start
                it.set_postfix({"batch_time_sec": batch_time})

                del outputs, produce_data, gtv_point, label, initial_data, filled_point, valid_data
                torch.cuda.empty_cache()

        all_probs, all_labels, all_preds = np.concatenate(all_probs), np.concatenate(all_labels), np.concatenate(all_preds)

        fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
        optimal_idx = np.argmax(tpr - fpr)

        accuracy = (all_preds == all_labels).mean()
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()

        rows = []
        for patient_id, (features_2d, actual_len) in enumerate(zip(sample_data_list, seq_length_list)):
            for time_step in range(actual_len):
                features = features_2d[:, time_step]
                row = [patient_id, time_step] + features.tolist()
                rows.append(row)

        columns = ['ID', 'num'] + [f'feature_{i + 1}' for i in range(features_2d.shape[0])]
        df = pd.DataFrame(rows, columns=columns)

        total_time = time.time() - start_time
        latency_per_sample = total_time / n_samples
        throughput = n_samples / total_time

        print(f'Final Test Results (All Timesteps):\n'
              f'Accuracy: {accuracy:.2%}, AUC: {auc:.4f}\n'
              f'Confusion Matrix:\n{cm}\n'
              f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n'
              f'--- Deployment Metrics ---\n'
              f'Total inference time: {total_time:.2f}s for {n_samples} samples\n'
              f'Latency per sample: {latency_per_sample * 1000:.2f} ms\n'
              f'Throughput: {throughput:.2f} samples/sec\n'
              f'Peak RAM usage: {peak_mem_usage:.2f} MB\n'
              '-----------------------------')


