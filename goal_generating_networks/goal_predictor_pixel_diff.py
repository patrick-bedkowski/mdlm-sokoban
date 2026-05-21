import gc
import os
from collections import defaultdict

import wandb
wandb.login()
# wandb.login("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")

from datetime import datetime
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
    Softmax,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2

from envs import Sokoban
from metric_logging import log_scalar
from supervised import DataCreatorSokobanPixelDiff

import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from metric_logging import log_scalar
from envs import Sokoban
from supervised import DataCreatorSokobanPixelDiff
from goal_generating_networks.sokoban_metrics import compute_sokoban_metrics

import matplotlib.pyplot as plt
import os
import numpy as np


def save_diffusion_grid(boards, steps, dump_folder, batch_idx="0"):
    """
    Plots a 4x4 grid of Sokoban board states for a single sample.
    boards: List of 16 flattened numpy arrays (144 items each).
    steps: List of 16 labels corresponding to each board slot.
    """
    colors = {
        0: 'white',        # Floor
        1: 'black',        # Wall
        2: 'lightgreen',   # Goal
        3: 'saddlebrown',  # Box
        4: 'green',        # BoxOnGoal
        5: 'red',          # Agent
        6: 'darkred',      # AgentOnGoal
        7: 'lightgray'     # MASK
    }

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i < len(boards):
            board = boards[i].reshape(12, 12)

            for r in range(12):
                for c in range(12):
                    tile_color = colors.get(int(board[r, c]), 'white')
                    ax.add_patch(plt.Rectangle((c, 11 - r), 1, 1, color=tile_color))

            ax.set_xlim(0, 12)
            ax.set_ylim(0, 12)
            ax.set_title(str(steps[i]), fontsize=10)
            ax.set_aspect('equal')

        ax.set_xticks([])
        ax.set_yticks([])

    os.makedirs(dump_folder, exist_ok=True)
    save_path = os.path.join(dump_folder, f"diffusion_grid_batch_{batch_idx}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Diffusion grid saved to {save_path}")

    # Optional Weights & Biases tracking:
    if wandb.run is not None:
        wandb.log({f"media/diffusion_grid_{batch_idx}": wandb.Image(save_path)})

    return save_path


class DiTBlock(nn.Module):
    """
    A Transformer block that uses Adaptive Layer Norm (adaLN)
    to modulate signal flow based on the diffusion step.
    """

    def __init__(self, d_model, nhead):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn1 = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn2 = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        # Modulation layer: Produces 6 parameters (3 pairs of scale/shift)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model)
        )

    def forward(self, x, context, t_emb):
        """
        Block structure:
        1. Self-Attention with adaLN modulation
        2. Cross-Attention (conditioning on context) with adaLN modulation
        3. Feed-Forward with adaLN modulation.
        Why need cross attention for initial and goal state board?
        Because the model needs to learn how to use the conditioning information (initial and goal states)
        to guide the denoising process. The cross-attention allows the model to attend to relevant parts of
        the initial and goal state representations when processing the latent board, enabling it to make informed
        updates that move towards the goal configuration.
        """
        # Generate modulation parameters from the time embedding
        # We split the output into 6 vectors: (shift1, scale1, shift2, scale2, shift3, scale3)
        mods = self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        shift1, scale1, shift2, scale2, shift3, scale3 = mods

        # 1. Self-Attention Block with modulation
        x_norm = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + self.attn1(x_norm, x_norm, x_norm)[0]

        # 2. Cross-Attention Block with modulation (Conditioning on Start/Goal)
        x_norm = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + self.attn2(x_norm, context, context)[0]

        # 3. Feed-Forward Block with modulation
        x_norm = self.norm3(x) * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        x = x + self.mlp(x_norm)

        return x


class SokobanTransformer(nn.Module):
    architecture_name = "architecture4"

    def __init__(self, vocab_size=8, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.mask_token_id = 7

        # Token and 2D Positional Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 144, d_model))

        # Inside SokobanTransformer.__init__
        self.pos_emb_row = nn.Parameter(torch.randn(1, 12, 1, d_model // 2))
        self.pos_emb_col = nn.Parameter(torch.randn(1, 1, 12, d_model // 2))

        # Time Embedding (Mask Ratio)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Custom blocks supporting adaLN
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, nhead) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(d_model, 7)
        print("Model MDLM initiated with Separated Condition Embeddings!")

    def get_2d_pos(self):
        # Broadcast row/col embeddings to create a 12x12x128 grid
        pos = torch.cat([
            self.pos_emb_row.expand(-1, -1, 12, -1),
            self.pos_emb_col.expand(-1, 12, -1, -1)
        ], dim=-1)
        return pos.view(1, 144, self.d_model)

    def apply_stochastic_mask(self, target_board):
        """
        Vectorized forward process that masks exactly t% of eligible tokens.

        Steps:
        1. Sample a random mask ratio t ~ U(0, 1) for each sample in the batch.
        2. Create a boolean mask of eligible positions (non-wall tokens).
        3. For each eligible position, assign a random priority score.
        4. Compute the threshold priority that corresponds to the t-th quantile among eligible tokens.
        """
        device = target_board.device
        batch_size, seq_len = target_board.shape

        # 1. Sample t ~ U(0, 1)
        mask_ratio = torch.rand(batch_size, 1, device=device)

        # 2. Eligible mask (non-wall)
        mask_eligible = (target_board != 0)  # (B, 144)

        # 3. For each eligible position, assign a random priority
        #    Non-eligible positions get priority > 1 (will never be selected)
        rand_priorities = torch.rand(batch_size, seq_len, device=device)
        rand_priorities[~mask_eligible] = 2.0  # Push walls to the end

        # 4. Compute the threshold: the t-th quantile among eligible tokens
        #    Sort priorities, find the cutoff index
        num_eligible = mask_eligible.float().sum(dim=1, keepdim=True)  # (B, 1)
        num_to_mask = torch.clamp((mask_ratio * num_eligible).floor(), min=1)  # (B, 1) at least 1

        # 5. Use topk to find the positions to mask
        #    We want the num_to_mask smallest priorities (among eligible)
        #    Equivalent: threshold = sorted_priorities[num_to_mask]
        sorted_priorities, _ = rand_priorities.sort(dim=1)  # (B, 144)
        
        # Gather the threshold value for each sample
        threshold_idx = (num_to_mask - 1).long().clamp(0, seq_len - 1)  # (B, 1)
        thresholds = sorted_priorities.gather(1, threshold_idx)  # (B, 1)

        # 6. Mask positions with priority <= threshold AND eligible
        mask_indices = (rand_priorities <= thresholds) & mask_eligible

        masked_inputs = target_board.clone()
        masked_inputs[mask_indices] = self.mask_token_id

        return masked_inputs, mask_indices, mask_ratio

    def forward(self, current_state, final_goal, masked_subgoal, t):
        # 1. Embeddings + 2D Positional Bias
        e_curr = self.token_emb(current_state.view(-1, 144)) + self.get_2d_pos()
        e_goal = self.token_emb(final_goal.view(-1, 144)) + self.get_2d_pos()
        e_mask = self.token_emb(masked_subgoal.view(-1, 144)) + self.get_2d_pos()

        # 2. Prepare Context and Time info
        context = torch.cat([e_curr, e_goal], dim=1)  # (Batch, 288, d_model)
        t_emb = self.time_mlp(t)  # (Batch, d_model)

        # 3. Process Latent Board
        x = e_mask
        for block in self.blocks:
            x = block(x, context, t_emb)

        return self.output_layer(x)


# --- Modified Class ---
class GoalPredictorPixelDiff:
    def __init__(self, num_layers=4, model_id=None, learning_rate=0.001, temperature=.8, batch_size=32):
        self.core_env = Sokoban()
        self.dim_room = self.core_env.get_dim_room()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_layers = num_layers
        self.model_id = model_id
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dump_folder = "out/mdlm_28_04_26/"

        self._model = None
        self._predictions_counter = 0
        self.mask_token_id = 7
        self.data_creator = DataCreatorSokobanPixelDiff()
        self._temperature = temperature

        self.date_now = datetime.now().strftime('%H-%M-%d-%m-%Y')
        self.predictor_name = "v3_full_remask"


    def construct_networks(self):
        print("Constructing MDLM Transformer with num_layers={}, learning_rate={}, batch_size={}".format(
            self.num_layers, self.learning_rate, self.batch_size))
        if self._model is None:
            # Initialize the Transformer
            self._model = SokobanTransformer(num_layers=self.num_layers).to(self.device)

            # Print Parameter Count
            params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            print(f"MDLM Transformer initialized with {params:,} trainable parameters.")

            self.optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)
            self.token_weights = torch.tensor([0.1, 0.1, 5.0, 2.0, 10.0, 5.0, 10.0], device=self.device)
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=self.token_weights)
        else:
            self.load_parameters()

        # init wandb
        run_name = f"{self.predictor_name}_{self._model.architecture_name}_{self.date_now}"
        wandb.init(
            project="sokoban-diffusion-llm",
            entity="bedkowski-patrick",
            name=run_name,
            config={
                "architecture": self._model.architecture_name,
                "layers": self.num_layers,
                "lr": self.learning_rate,
                "batch_size": self.batch_size,
                "date": self.date_now
            }
        )

    def _denoise(
            self,
            inp,
            cond,
            inf_steps=256,
            stochastic=True,
            temperature=1.0,
            n_of_samples_to_collect=5):
        """
        Performs the iterative denoising process.
        Returning a list of intermediate batch states and the final predicted tensor.
        """
        batch_size, seq_len = inp.shape
        device = inp.device

        wall_mask = (inp == 0)
        eligible_mask = ~wall_mask
        num_eligible = eligible_mask.sum(dim=1)

        cur_dream = torch.full_like(inp, self.mask_token_id)
        cur_dream[wall_mask] = 0

        timesteps = torch.linspace(1.0, 0.0, inf_steps + 1, device=device)

        # Calculate 12 evenly spaced step indices to collect intermediate states
        collect_steps = torch.linspace(0, inf_steps - 1, steps=12, dtype=torch.long).tolist()
        intermediate_boards = []

        for step in range(inf_steps):
            # Collect intermediate frames at the scheduled steps
            if step in collect_steps:
                intermediate_boards.append(cur_dream.clone().cpu().numpy())

            t_current = timesteps[step]
            t_next = timesteps[step + 1]
            t_input = t_current.view(1, 1).expand(batch_size, 1)

            logits = self._model(inp, cond, cur_dream, t_input)
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            confidences, predictions = probs.max(dim=-1)

            num_to_mask = (t_next * num_eligible).long()
            next_dream = predictions.clone()
            next_dream[wall_mask] = 0

            if step != inf_steps - 1:
                for b in range(batch_size):
                    k = min(num_to_mask[b].item(), eligible_mask[b].sum().item())
                    if k == 0:
                        continue

                    scores = confidences[b].clone()
                    scores[wall_mask[b]] = 1.0

                    if stochastic:
                        uncertainties = 1.0 - scores
                        uncertainties[wall_mask[b]] = 0
                        probs_remask = uncertainties / (uncertainties.sum() + 1e-8)
                        remask_indices = torch.multinomial(probs_remask, num_samples=k, replacement=False)
                    else:
                        uncertainties = 1.0 - scores
                        _, remask_indices = torch.topk(uncertainties, k=k)

                    next_dream[b, remask_indices] = self.mask_token_id

            cur_dream = next_dream

        return intermediate_boards, cur_dream

    def load_parameters(self):
        path = self.model_id
        print("loading model parameters from {}".format(path))
        checkpoint = torch.load(path, map_location=self.device)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model parameters loaded from {path}")


    def fit_and_dump(self, training_data, validation_data, epochs,
                     dump_folder, checkpoints=None,
                     n_samples_to_collect=5):
        # 1. Flatten the inputs to (N, 144) to avoid 3D broadcasting errors
        (t_inputs, t_targets) = training_data
        train_x = torch.tensor(t_inputs[0], dtype=torch.long).reshape(t_inputs[0].shape[0], -1)
        train_cond = torch.tensor(t_inputs[1], dtype=torch.long).reshape(t_inputs[1].shape[0], -1)
        train_y = torch.tensor(t_targets, dtype=torch.long).reshape(t_targets.shape[0], -1)

        (v_inputs, v_targets) = validation_data
        val_x = torch.tensor(v_inputs[0], dtype=torch.long).reshape(v_inputs[0].shape[0], -1)
        val_g = torch.tensor(v_inputs[1], dtype=torch.long).reshape(v_inputs[1].shape[0], -1)
        val_y = torch.tensor(v_targets, dtype=torch.long).reshape(v_targets.shape[0], -1)

        print("DEBUG: Training X shape: {} | Training Y shape: {} | Training Cond shape: {}".format(
            train_x.shape, train_y.shape, train_cond.shape))
        print(f"DEBUG: Validation X shape: {val_x.shape} | Validation Y shape: {val_y.shape} | Validation G shape: {val_g.shape}")

        num_samples = train_x.shape[0]

        for epoch in range(epochs):
            # --- TRAINING PHASE ---
            self._model.train()
            epoch_train_loss = 0
            indices = torch.randperm(num_samples)

            # forward
            for i in range(0, num_samples, self.batch_size):  # calculate the batches
                batch_idx = indices[i:i + self.batch_size]
                b_x = train_x[batch_idx].to(self.device)
                b_cond = train_cond[batch_idx].to(self.device)
                b_y = train_y[batch_idx].to(self.device)

                # 1. MASKING LOGIC
                masked_inputs, mask_indices, mask_ratio = self._model.apply_stochastic_mask(b_y)
                logits = self._model(b_x, b_cond, masked_inputs, mask_ratio)

                # 2. LOSS: only on masked positions, normalized by count
                loss_raw = self.criterion(logits.transpose(1, 2), b_y)
                masked_loss = (loss_raw * mask_indices.float()).sum(dim=1)
                num_masked = mask_indices.float().sum(dim=1)
                final_loss = (masked_loss / (num_masked + 1e-6)).mean()

                self.optimizer.zero_grad()
                final_loss.backward()
                self.optimizer.step()
                epoch_train_loss += final_loss.item()

            avg_train_loss = epoch_train_loss / (num_samples / self.batch_size)

            # --- ENHANCED VALIDATION PHASE ---
            self._model.eval()
            val_loss, val_delta_acc, val_wall_acc = 0, 0, 0
            val_full_denoise_acc = 0

            with torch.no_grad():
                all_metrics = defaultdict(list)
                for i in range(0, val_x.size(0), self.batch_size):
                    bv_x = val_x[i:i + self.batch_size].to(self.device)
                    bv_cond = val_g[i:i + self.batch_size].to(self.device)
                    bv_y = val_y[i:i + self.batch_size].to(self.device)

                    # Standard Masking for Delta Check
                    v_masked_inputs, v_mask_indices, v_mask_ratio = self._model.apply_stochastic_mask(bv_y)
                    v_logits = self._model(bv_x, bv_cond, v_masked_inputs, v_mask_ratio)

                    scaled_logits = v_logits / self._temperature
                    probs = torch.softmax(scaled_logits, -1)
                    v_preds = torch.distributions.Categorical(
                        probs
                    ).sample()

                    # WALL STABILITY ACCURACY
                    wall_mask = (bv_y == 0)
                    if wall_mask.sum() > 0:
                        wall_correct = (v_preds == bv_y) & wall_mask
                        val_wall_acc += wall_correct.sum().item() / (wall_mask.sum().item() + 1e-6)

                    # DELTA ACCURACY
                    change_mask = (bv_x != bv_y)
                    critical_mask = v_mask_indices & change_mask
                    if critical_mask.sum() > 0:
                        correct = (v_preds == bv_y) & critical_mask
                        val_delta_acc += correct.sum().item() / (critical_mask.sum().item() + 1e-6)

                    # ============================================================
                    # FULL DENOISING: Approach 1 — Full Re-evaluation at Each Step
                    # ============================================================
                    if i == 0:
                        cur_dream = self._denoise(
                            bv_x,
                            bv_cond,
                            inf_steps=256,
                            stochastic=True,
                            temperature=self._temperature
                        )
                        val_full_denoise_acc = (cur_dream == bv_y).float().mean().item()

                        # Compute all metrics
                        batch_metrics = compute_sokoban_metrics(
                            pred_board=cur_dream,
                            input_board=bv_x,
                            target_board=bv_y,
                            goal_board=bv_cond
                        )
                        for k, v in batch_metrics.items(): all_metrics[k].append(v)

            # Average across all batches
            avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
            wandb.log({f"val_rules/{k}": v for k, v in avg_metrics.items()})

            avg_val_delta = val_delta_acc / (val_x.size(0) / self.batch_size)
            avg_val_wall = val_wall_acc / (val_x.size(0) / self.batch_size)

            wandb.log({
                "train/loss": avg_train_loss,
                "val/delta_accuracy": avg_val_delta,
                "val/wall_stability": avg_val_wall,
                "val/full_denoise_acc": val_full_denoise_acc,
                "epoch": epoch
            })

            if epoch % 5 == 0:
                self.predict_and_get_wandb_image(val_x[n_samples_to_collect:], val_g[n_samples_to_collect:],
                                                 ground_truth_subgoal=val_y[n_samples_to_collect:])

            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Delta Acc: {avg_val_delta:.4f} | Wall Stability: {avg_val_wall:.4f} | Full Denoise Acc: {val_full_denoise_acc:.4f}")
            log_scalar('val_delta_accuracy', epoch, avg_val_delta)
            log_scalar('val_full_denoise_acc', epoch, val_full_denoise_acc)
            log_scalar('train_loss', epoch, avg_train_loss)

            if checkpoints is not None and epoch in checkpoints:
                self.save_model(os.path.join(dump_folder, f'epoch_{epoch}.pt'))

    def predict_and_get_wandb_image(self, input_boards, conditions, ground_truth_subgoal=None):
        """Helper to run prediction and return a wandb Image object."""
        _, plot_path = self.predict_pdf_batch(
            input_boards.cpu().numpy(), conditions.cpu().numpy(), ground_truth_subgoal=ground_truth_subgoal.cpu().numpy() if ground_truth_subgoal is not None else None, steps=256
        )
        return wandb.Image(plot_path, caption="Diffusion Steps")

    def predict_pdf_batch(self, input_boards, conditions, ground_truth_subgoal=None,
                          steps=256, n_of_samples_to_collect=5):
        self._predictions_counter += 1
        self._model.eval()

        # 1. Standard Pre-processing
        if input_boards.ndim == 3: input_boards = np.expand_dims(input_boards, axis=0)
        if conditions.ndim == 3: conditions = np.expand_dims(conditions, axis=0)

        # Handle Input Boards (144 vs 1008)
        if input_boards.shape[-1] == 1008:
            input_tokens = np.argmax(input_boards.reshape(-1, 144, 7), axis=-1)
        elif input_boards.shape[-1] == 7:
            input_tokens = np.argmax(input_boards, axis=-1)
        else:
            input_tokens = input_boards

        # Handle Conditions / Final Goal
        if conditions.shape[-1] == 1008:
            cond_tokens = np.argmax(conditions.reshape(-1, 144, 7), axis=-1)
        elif conditions.shape[-1] == 7:
            cond_tokens = np.argmax(conditions, axis=-1)
        else:
            cond_tokens = conditions

        # Handle Ground Truth Subgoal (if provided)
        if ground_truth_subgoal is not None:
            if ground_truth_subgoal.ndim == 3:
                ground_truth_subgoal = np.expand_dims(ground_truth_subgoal, axis=0)
            if ground_truth_subgoal.shape[-1] == 1008:
                gt_subgoal_tokens = np.argmax(ground_truth_subgoal.reshape(-1, 144, 7), axis=-1)
            elif ground_truth_subgoal.shape[-1] == 7:
                gt_subgoal_tokens = np.argmax(ground_truth_subgoal, axis=-1)
            else:
                gt_subgoal_tokens = ground_truth_subgoal
        else:
            gt_subgoal_tokens = None

        batch_size = input_tokens.shape[0]

        # 2. Setup Tensors and Run Denoising Pipeline
        inp = torch.tensor(input_tokens.reshape(batch_size, -1), dtype=torch.long).to(self.device)
        cond = torch.tensor(cond_tokens.reshape(batch_size, -1), dtype=torch.long).to(self.device)

        intermediate_history, cur_dream = self._denoise(
            inp,
            cond,
            inf_steps=steps,
            stochastic=True,
            temperature=self._temperature,
            n_of_samples_to_collect=n_of_samples_to_collect
        )

        dream_tokens_flat = cur_dream.cpu().numpy().reshape(batch_size, 144)
        input_tokens_flat = input_tokens.reshape(batch_size, 144)
        cond_tokens_flat = cond_tokens.reshape(batch_size, 144)

        # 3. Generate separate diagnostic grids for the first N samples
        num_visualize = min(batch_size, n_of_samples_to_collect)
        save_paths = []
        collect_steps = torch.linspace(0, steps - 1, steps=12, dtype=torch.long).tolist()

        for b in range(num_visualize):
            sample_boards = []
            sample_labels = []

            # Slot 0: Input state
            sample_boards.append(input_tokens_flat[b])
            sample_labels.append("Initial Input")

            # Slot 1: Ground Truth Subgoal (or fallback placeholder)
            if gt_subgoal_tokens is not None:
                gt_flat = gt_subgoal_tokens.reshape(batch_size, 144)
                sample_boards.append(gt_flat[b])
                sample_labels.append("GT Subgoal")
            else:
                sample_boards.append(input_tokens_flat[b])
                sample_labels.append("(No GT)")

            # Slots 2-13: The 12 Intermediate Diffusion Steps
            for idx, step_idx in enumerate(collect_steps):
                sample_boards.append(intermediate_history[idx][b])
                sample_labels.append(f"Step {step_idx}")

            # Slot 14: Final Dream (Prediction)
            sample_boards.append(dream_tokens_flat[b])
            sample_labels.append("Final Dream")

            # Slot 15: Final Board (Solved State Goal)
            sample_boards.append(cond_tokens_flat[b])
            sample_labels.append("Final Board")

            # Save the plot grid specifically for this sample item
            unique_batch_idx = f"{self._predictions_counter}_sample_{b}"
            path = save_diffusion_grid(
                boards=sample_boards,
                steps=sample_labels,
                dump_folder=self.dump_folder,
                batch_idx=unique_batch_idx
            )
            save_paths.append(path)

        # 4. Corrected Adaptive Translation Logic
        pdf_1009 = np.zeros((batch_size, 1009))

        for b in range(batch_size):
            # Find all tiles where the current node board differs from our diffusion dream
            diff_indices = np.where(input_tokens_flat[b] != dream_tokens_flat[b])[0]
            num_diffs = len(diff_indices)

            if num_diffs == 0:
                # The GoalBuilder successfully finished building our dream! Trigger STOP.
                pdf_1009[b, 1008] = 1.0
            else:
                # Provide an "Exit Strategy": give the stop bit a baseline probability (e.g., 20%)
                # This allows smart_sample to gracefully consider a node finished even if
                # minor non-essential tiles haven't perfectly shifted yet.
                stop_prob = 0.20
                pdf_1009[b, 1008] = stop_prob

                # Distribute the remaining 80% probability evenly across ALL missing edits.
                # This restores the tree's ability to branch out and try different path variants.
                remaining_prob = 1.0 - stop_prob
                prob_per_edit = remaining_prob / num_diffs

                for idx in diff_indices:
                    target_tile = dream_tokens_flat[b, idx]
                    flat_action_index = (idx * 7) + target_tile
                    pdf_1009[b, flat_action_index] = prob_per_edit

        return pdf_1009, save_paths


    def save_model(self, path):
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_layers': self.num_layers,
        }, path)

    def load_model(self, path):
        self.construct_networks()
        self._model.load_state_dict(torch.load(path, map_location=self.device))

    # Keep utility methods for search compatibility
    def reset_predictions_counter(self):
        self._predictions_counter = 0

    def read_predictions(self):
        return self._predictions_counter

    def load_data(self, dataset_file):
        self.data_creator.load(dataset_file)

    def flat_to_2d(self, n):
        if n == 1008: return -1, -1, -1  # Handle the stop bit
        element = n % 7
        base_n = n // 7
        x = base_n // self.dim_room[0]
        y = base_n % self.dim_room[1]
        return x, y, element

    def smart_sample(self, pdf, internal_confidence_level):
        pdf = np.array(pdf).squeeze()  # ensure shape (1009,)
        assert pdf.ndim == 1, f"Expected 1D pdf, got shape {pdf.shape}"
        
        out, out_p = [], []
        for idx in reversed(np.argsort(pdf)):
            out.append(self.flat_to_2d(int(idx)))  # explicit cast too
            out_p.append(float(pdf[idx]))
            if sum(out_p) > internal_confidence_level:
                break
        return out, out_p


def log_diffusion_details(file_path, step=None, board=None, input_board=None, diffs=None, mode="trace"):
    """
    Handles all debug logging for the MDLM reasoning process.
    """
    mapping = {0: ' ', 1: '#', 2: '.', 3: '$', 4: '*', 5: '@', 6: '+', 7: '?'}

    with open(file_path, 'a') as f:
        if mode == "input":
            f.write("============================================================\n")
            f.write("NEW INFERENCE JOB: INPUT BOARD (Initial State)\n")
            f.write("============================================================\n")
            grid = input_board.reshape(12, 12)
            for row in grid:
                f.write("".join([mapping.get(int(t), 'E') for t in row]) + "\n")
            f.write("\n")

        elif mode == "trace":
            f.write(f"--- Diffusion Step: {step} ---\n")
            grid = board.reshape(12, 12)
            for row in grid:
                f.write("".join([mapping.get(int(t), 'E') for t in row]) + "\n")
            f.write("\n")

        elif mode == "final":
            f.write("============================================================\n")
            f.write("FINAL ANALYSIS (Dream vs. Input)\n")
            f.write(f"Total Tile Differences: {len(diffs)}\n")
            f.write(f"Difference Indices: {diffs.tolist()}\n")
            f.write("============================================================\n\n")