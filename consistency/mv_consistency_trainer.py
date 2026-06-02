from typing import Dict

import torch
import torch.nn.functional as F
from transformers import Trainer


class MultiViewConsistencyTrainer(Trainer):
    def __init__(
        self,
        *args,
        lambda_dist: float = 0.2,
        lambda_repr: float = 0.05,
        consistency_temperature: float = 1.0,
        use_dist_consistency: bool = True,
        use_repr_consistency: bool = True,
        log_aux_losses: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_dist = lambda_dist
        self.lambda_repr = lambda_repr
        self.consistency_temperature = consistency_temperature
        self.use_dist_consistency = use_dist_consistency
        self.use_repr_consistency = use_repr_consistency
        self.log_aux_losses = log_aux_losses

    @staticmethod
    def _extract_logits_at_positions(logits: torch.Tensor, positions: torch.Tensor):
        """
        logits: [B, T, V]
        positions: [B, K]
        """
        valid_mask = positions >= 0
        if not valid_mask.any():
            return None, valid_mask

        safe_positions = positions.clamp(min=0)
        bsz, k = safe_positions.shape
        batch_idx = torch.arange(bsz, device=logits.device).unsqueeze(1).expand(-1, k)
        gathered = logits[batch_idx, safe_positions]  # [B, K, V]
        gathered = gathered[valid_mask]               # [N_valid, V]
        return gathered, valid_mask

    @staticmethod
    def _extract_hidden_at_positions(hidden: torch.Tensor, positions: torch.Tensor):
        """
        hidden: [B, T, H]
        positions: [B]
        """
        valid_mask = positions >= 0
        if not valid_mask.any():
            return None, valid_mask

        safe_positions = positions.clamp(min=0)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        gathered = hidden[batch_idx, safe_positions]  # [B, H]
        gathered = gathered[valid_mask]               # [N_valid, H]
        return gathered, valid_mask

    def _js_div_from_logits(self, sid_logits: torch.Tensor, title_logits: torch.Tensor) -> torch.Tensor:
        temp = self.consistency_temperature
        sid_log_prob = F.log_softmax(sid_logits / temp, dim=-1)
        title_log_prob = F.log_softmax(title_logits / temp, dim=-1)

        sid_prob = sid_log_prob.exp()
        title_prob = title_log_prob.exp()
        mean_prob = 0.5 * (sid_prob + title_prob)

        js = 0.5 * (
            F.kl_div(sid_log_prob, mean_prob, reduction="batchmean")
            + F.kl_div(title_log_prob, mean_prob, reduction="batchmean")
        )
        return js

    def compute_loss(self, model, inputs: Dict[str, torch.Tensor], return_outputs: bool = False, **kwargs):
        sid_inputs = {
            "input_ids": inputs["sid_input_ids"],
            "attention_mask": inputs["sid_attention_mask"],
            "labels": inputs["sid_labels"],
            "output_hidden_states": True,
            "use_cache": False,
        }
        title_inputs = {
            "input_ids": inputs["title_input_ids"],
            "attention_mask": inputs["title_attention_mask"],
            "labels": inputs["title_labels"],
            "output_hidden_states": True,
            "use_cache": False,
        }

        sid_outputs = model(**sid_inputs)
        title_outputs = model(**title_inputs)

        ce_loss = 0.5 * (sid_outputs.loss + title_outputs.loss)

        # ---------- 分布一致性 ----------
        if self.use_dist_consistency and self.lambda_dist > 0:
            sid_pred_logits, _ = self._extract_logits_at_positions(
                sid_outputs.logits, inputs["sid_pred_positions"]
            )
            title_pred_logits, _ = self._extract_logits_at_positions(
                title_outputs.logits, inputs["title_pred_positions"]
            )

            if sid_pred_logits is not None and title_pred_logits is not None:
                pair_count = min(sid_pred_logits.size(0), title_pred_logits.size(0))
                if pair_count > 0:
                    dist_loss = self._js_div_from_logits(
                        sid_pred_logits[:pair_count],
                        title_pred_logits[:pair_count],
                    )
                else:
                    dist_loss = ce_loss.new_zeros(())
            else:
                dist_loss = ce_loss.new_zeros(())
        else:
            dist_loss = ce_loss.new_zeros(())

        # ---------- 表征一致性 ----------
        if self.use_repr_consistency and self.lambda_repr > 0:
            sid_hidden, _ = self._extract_hidden_at_positions(
                sid_outputs.hidden_states[-1], inputs["sid_anchor_pos"]
            )
            title_hidden, _ = self._extract_hidden_at_positions(
                title_outputs.hidden_states[-1], inputs["title_anchor_pos"]
            )

            if sid_hidden is not None and title_hidden is not None:
                pair_count = min(sid_hidden.size(0), title_hidden.size(0))
                if pair_count > 0:
                    sid_hidden = F.normalize(sid_hidden[:pair_count], dim=-1)
                    title_hidden = F.normalize(title_hidden[:pair_count], dim=-1)
                    repr_loss = (1.0 - (sid_hidden * title_hidden).sum(dim=-1)).mean()
                else:
                    repr_loss = ce_loss.new_zeros(())
            else:
                repr_loss = ce_loss.new_zeros(())
        else:
            repr_loss = ce_loss.new_zeros(())

        total_loss = ce_loss + self.lambda_dist * dist_loss + self.lambda_repr * repr_loss

        if self.log_aux_losses and self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log(
                {
                    "ce_loss": float(ce_loss.detach().cpu()),
                    "dist_loss": float(dist_loss.detach().cpu()),
                    "repr_loss": float(repr_loss.detach().cpu()),
                    "total_mv_loss": float(total_loss.detach().cpu()),
                }
            )

        if return_outputs:
            outputs = {
                "sid_outputs": sid_outputs,
                "title_outputs": title_outputs,
                "ce_loss": ce_loss.detach(),
                "dist_loss": dist_loss.detach(),
                "repr_loss": repr_loss.detach(),
            }
            return total_loss, outputs

        return total_loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        eval 阶段也走自定义 multi-view loss，
        避免 Trainer 默认直接 model(**inputs) 导致输入不匹配。
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

        loss = loss.mean().detach()
        return (loss, None, None)