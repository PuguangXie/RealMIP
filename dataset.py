import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings("ignore")


class ICU_Dataset(Dataset):
    def __init__(self, stage='train', seed=0):
        np.random.seed(seed)

        self.patients = []
        self.labels = []
        self.survival_indices = []
        self.death_indices = []

        filepath = self._get_filepath_for_stage(stage)

        observed_data = pd.read_csv(filepath)
        self.status = observed_data["died_in_icu"]
        self.patient_id = observed_data["patientunitstayid"]

        grouped = observed_data.groupby('patientunitstayid', sort=False)

        for idx, (patientunitstayid, group) in enumerate(grouped):
            group_sorted = group.sort_values('observationoffset')
            features = group_sorted.iloc[:, 2:].values.astype(np.float32)

            label = group_sorted['died_in_icu'].iloc[0]
            observation_offsets = group_sorted['observationoffset'].values
            observed_masks = np.array(1 - np.isnan(features)).astype("float32")
            observed_values = np.nan_to_num(features)
            gt_masks = observed_masks

            self.patients.append({
                "patient_id": patientunitstayid,
                'observed_data': observed_values,
                'offsets': observation_offsets,
                "observed_masks": observed_masks,
                "gt_masks": gt_masks,
                'status': label,
            })
            self.labels.append(label)

            if label == 0:
                self.survival_indices.append(idx)
            else:
                self.death_indices.append(idx)

    def _get_filepath_for_stage(self, stage):
        file_mapping = {
            'train': './data/eicu_train.csv',
            'internal_test': './data/eicu_test.csv',
            'external_test_1': './data/mimiciv.csv',
            'external_test_2': './data/sicdb.csv'
        }
        return file_mapping.get(stage, '')

    def __getitem__(self, idx):
        patient = self.patients[idx]
        patient_id = patient["patient_id"]
        features = patient["observed_data"]
        label = patient['status']
        offsets = patient['offsets'] * 7.525188e+03 + 5.012372e+03
        observed_masks = patient['observed_masks']
        gt_masks = patient['gt_masks']

        seq_length = len(features)

        return (
            torch.tensor(patient_id),
            torch.FloatTensor(features),
            torch.tensor(observed_masks),
            torch.tensor(gt_masks),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(offsets),
            torch.tensor(seq_length)
        )

    def __len__(self):
        return len(self.patients)


class BalancedSampler(Sampler):
    def __init__(self, dataset):
        self.death_indices = dataset.death_indices
        self.survival_indices = dataset.survival_indices
        self.num_deaths = len(self.death_indices)

    def __iter__(self):
        survival_samples = np.random.choice(
            self.survival_indices,
            size=self.num_deaths,
            replace=True
        ).tolist()

        combined = self.death_indices + survival_samples
        np.random.shuffle(combined)
        return iter(combined)

    def __len__(self):
        return 2 * self.num_deaths


def collate_fn(batch):
    patient_id, sequences, observed_mask, gt_mask, labels, offsets, lengths = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    padded_observed_mask = pad_sequence(observed_mask, batch_first=True)
    padded_gt_mask = pad_sequence(gt_mask, batch_first=True)
    padded_offsets = pad_sequence(offsets, batch_first=True)
    labels = torch.stack(labels)

    return {
        "patient_id": torch.tensor(patient_id),
        "observed_data": padded_sequences,
        "observed_mask": padded_observed_mask,
        "gt_mask": padded_gt_mask,
        "status": labels,
        "offsets": padded_offsets,
        'seq_length': torch.tensor(lengths)
    }


def get_dataloader(stage='train', seed=2025, batch_size=8, missing_ratio=0.1):
    if stage == 'train':
        dataset = ICU_Dataset(stage='train', seed=seed)
        sampler = BalancedSampler(dataset)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
        valid_dataset = ICU_Dataset(stage='internal_test', seed=seed)
        valid_loader = DataLoader(valid_dataset, batch_size=100, shuffle=False, collate_fn=collate_fn)
        return loader, valid_loader

    if stage == 'test':
        test_dataset1 = ICU_Dataset(stage='external_test_1', seed=seed)
        valid_loader1 = DataLoader(test_dataset1, batch_size=10, shuffle=False, collate_fn=collate_fn)
        test_dataset2 = ICU_Dataset(stage='external_test_2', seed=seed)
        valid_loader2 = DataLoader(test_dataset2, batch_size=10, shuffle=False, collate_fn=collate_fn)
        return valid_loader1, valid_loader2
