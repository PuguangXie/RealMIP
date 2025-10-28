import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.emb_feature_dim = config["model"]["featureemb"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_feature_dim + 1
        self.embed_layer = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim)

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        self.num_steps = config_diff["num_steps"]
        self.beta = self._init_beta_schedule(config_diff)
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha, dtype=torch.float32, device=self.device).unsqueeze(1).unsqueeze(1)

    def _init_beta_schedule(self, config_diff):
        if config_diff["schedule"] == "quad":
            return np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            return np.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)

    def get_causal_mask(self, observed_mask):
        B, K, L = observed_mask.shape
        causal_mask = torch.tril(torch.ones(L, L, device=self.device), diagonal=0).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, K, -1, -1)
        combined_mask = observed_mask.unsqueeze(-1) * causal_mask
        return combined_mask.max(dim=-1).values

    def get_randmask(self, observed_mask):
        B, K, L = observed_mask.shape
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask_flat = rand_for_mask.view(B, -1)
        observed_mask_flat = observed_mask.view(B, -1)

        for i in range(B):
            start_idx = 3 * L
            end_idx = (K - 2) * L
            valid_observed = observed_mask_flat[i, start_idx:end_idx]
            num_observed = valid_observed.sum().item()

            if num_observed == 0:
                continue

            sample_ratio = np.random.rand()
            num_masked = int(num_observed * sample_ratio)
            valid_rand = rand_for_mask_flat[i, start_idx:end_idx]

            if num_masked > 0:
                selected = valid_rand.topk(num_masked).indices
                global_indices = start_idx + selected
                rand_for_mask_flat[i, global_indices] = -1

        rand_for_mask = rand_for_mask_flat.view(B, K, L)
        rand_for_mask[:, :3, :] = observed_mask[:, :3, :]
        rand_for_mask[:, -2:-1, :] = observed_mask[:, -2:-1, :]
        cond_mask = (rand_for_mask > 0).float()

        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def get_side_info(self, cond_mask):
        B, K, L = cond_mask.shape
        feature_embed = self.embed_layer(torch.arange(self.target_dim, device=self.device))
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = feature_embed.permute(0, 3, 2, 1)
        side_mask = cond_mask.unsqueeze(1)
        side_info = torch.cat([side_info, side_mask], dim=1)
        return side_info

    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train, seq_length):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train, seq_length, set_t=t)
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, seq_length, set_t=-1):
        B, K, L = observed_data.shape
        t = torch.full((B,), set_t if is_train != 1 else torch.randint(0, self.num_steps, [B], device=self.device), dtype=torch.long, device=self.device)

        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + ((1.0 - current_alpha) ** 0.5) * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t, seq_length)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)
        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples, seq_length):
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L, device=self.device)

        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)
            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B, 2, K, L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t], device=self.device), seq_length)
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            patient_id,
            observed_data,
            observed_mask,
            gt_mask,
            for_pattern_mask,
            cut_length,
            status,
            offsets,
            seq_length,
        ) = self.process_data(batch)

        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        generated_data = self.impute(observed_data, cond_mask, side_info, n_samples=3, seq_length=seq_length).to(self.device)

        generated_data_median = torch.median(generated_data.permute(0, 1, 3, 2), dim=1).values

        return (
            loss_func(observed_data, cond_mask, observed_mask, side_info, is_train, seq_length),
            generated_data_median.permute(0, 2, 1),
            (observed_mask - cond_mask),
            status,
            observed_data,
            observed_mask,
            seq_length
        )

    def evaluate(self, batch, n_samples):
        (
            patient_id,
            observed_data,
            observed_mask,
            gt_mask,
            for_pattern_mask,
            cut_length,
            status,
            offsets,
            seq_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples, seq_length)

            for i in range(len(cut_length)):
                target_mask[i, ..., 0:cut_length[i].item()] = 0

        return samples, observed_data, target_mask, observed_mask


class TSB_eICU(CSDI_base):
    def __init__(self, config, device, target_dim=37):
        super(TSB_eICU, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        patient_id = batch["patient_id"].to(self.device).long()
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        status = batch["status"].to(self.device).long()
        offsets = batch["offsets"].to(self.device).float()
        seq_length = batch['seq_length'].to(self.device).long()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data), dtype=torch.long, device=self.device)
        for_pattern_mask = observed_mask

        return (
            patient_id,
            observed_data,
            observed_mask,
            gt_mask,
            for_pattern_mask,
            cut_length,
            status,
            offsets,
            seq_length,
        )


