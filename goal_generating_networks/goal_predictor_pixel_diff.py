import gc
import os
from collections import defaultdict

import wandb
# wandb.login(key="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")
wandb.login("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")

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


def save_diffusion_grid(boards, steps, dump_folder, batch_idx=0):
    """
    Plots a 4x4 grid of Sokoban board states.
    boards: List of 16 numpy arrays (12x12).
    steps: List of strings/ints identifying the step for each board.
    """
    # Mapping: 0:Floor, 1:Wall, 2:Goal, 3:Box, 4:BoxOnGoal, 5:Agent, 6:AgentOnGoal, 7:MASK
    colors = {
        0: 'white',  # Floor
        1: 'black',  # Wall
        2: 'lightgreen',  # Goal
        3: 'saddlebrown',  # Box
        4: 'green',  # BoxOnGoal
        5: 'red',  # Agent
        6: 'darkred',  # AgentOnGoal
        7: 'lightgray'  # MASK
    }

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i < len(boards):
            board = boards[i].reshape(12, 12)
            # Create a color matrix
            color_matrix = np.vectorize(colors.get)(board)

            # Since ax.imshow needs RGB or specific types, we plot tiles
            for r in range(12):
                for c in range(12):
                    tile_color = colors.get(int(board[r, c]), 'white')
                    ax.add_patch(plt.Rectangle((c, 11 - r), 1, 1, color=tile_color))

            ax.set_xlim(0, 12)
            ax.set_ylim(0, 12)
            ax.set_title(f"Step: {steps[i]}")
            ax.set_aspect('equal')

        ax.set_xticks([])
        ax.set_yticks([])

    os.makedirs(dump_folder, exist_ok=True)
    save_path = os.path.join(dump_folder, f"diffusion_grid_batch_{batch_idx}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Diffusion grid saved to {save_path}")

    # log the image to wandb
    return save_path


# --- PyTorch Transformer Module ---
# class SokobanTransformer(nn.Module):
#     architecture_name = "architecture1"
#     def __init__(self, vocab_size=8, seq_len=144, d_model=128, nhead=8, num_layers=4):
#         super().__init__()
#         # 1. Create separate embedding dictionaries for Time/Context separation
#         self.emb_curr = nn.Embedding(vocab_size, d_model)
#         self.emb_goal = nn.Embedding(vocab_size, d_model)
#         self.emb_mask = nn.Embedding(vocab_size, d_model)
#         self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
#         self.input_projection = nn.Linear(d_model * 3, d_model)
#         self.output_layer = nn.Linear(d_model, 7)  # Output 7 classes (IDs 0-6)
#         self.mask_token_id = 7
#
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=d_model * 4,
#             batch_first=True,
#             activation='gelu'
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         print("Model MDLM initiated with Separated Condition Embeddings!")
#
#     def apply_stochastic_mask(self, target_board):
#         """
#         Refined Forward Process:
#         1. Samples a random mask ratio 't'.
#         2. Protects walls (Token 0) from being masked.
#         3. Replaces eligible tokens with MASK_ID.
#         """
#         device = target_board.device
#         # 1. Sample ratio t ~ U(0, 1)
#         mask_ratio = torch.rand(target_board.size(0), 1, device=device)
#
#         # 2. Identify eligible non-wall tokens
#         mask_eligible = (target_board != 0)
#
#         # 3. Generate mask indices based on ratio[cite: 1]
#         mask_indices = (torch.rand(target_board.shape, device=device) < mask_ratio) & mask_eligible
#
#         masked_inputs = target_board.clone()
#         masked_inputs[mask_indices] = self.mask_token_id
#
#         return masked_inputs, mask_indices, mask_ratio
#
#     def forward(self, current_state, final_goal, masked_subgoal):
#         curr = current_state.view(current_state.size(0), -1).long()
#         goal = final_goal.view(final_goal.size(0), -1).long()
#         mask = masked_subgoal.view(masked_subgoal.size(0), -1).long()
#
#         # 3. Embed each state using its specific dictionary
#         e_curr = self.emb_curr(curr)
#         e_goal = self.emb_goal(goal)
#         e_mask = self.emb_mask(mask)
#
#         # 4. Concatenate along the feature dimension (dim=-1)
#         # Shape goes from (Batch, 144, 128) -> (Batch, 144, 384)
#         x_concat = torch.cat([e_curr, e_goal, e_mask], dim=-1)
#
#         # 5. Project back to d_model -> (Batch, 144, 128)
#         x = self.input_projection(x_concat)
#
#         # Add positional encoding
#         x = x + self.pos_embedding
#
#         feat = self.transformer(x)
#         return self.output_layer(feat)


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
    architecture_name = "architecture2"

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
        e_curr = self.token_emb(current_state.view(-1, 144)) + self.pos_emb
        e_goal = self.token_emb(final_goal.view(-1, 144)) + self.pos_emb
        e_mask = self.token_emb(masked_subgoal.view(-1, 144)) + self.pos_emb

        # 2. Prepare Context and Time info
        context = torch.cat([e_curr, e_goal], dim=1) # (Batch, 288, d_model)
        t_emb = self.time_mlp(t) # (Batch, d_model)

        # 3. Process Latent Board
        x = e_mask
        for block in self.blocks:
            x = block(x, context, t_emb)

        return self.output_layer(x)


# --- Modified Class ---
class GoalPredictorPixelDiff:
    def __init__(self, num_layers=4, model_id=None, learning_rate=0.001, batch_size=32):
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

    
    def _denoise_full_reeval(self, inp, cond, inf_steps=256):
        """
        Approach 1: Full Re-evaluation with Remasking.
        
        At each step:
        1. Run the model on the CURRENT dream state (partially masked).
        2. Get predictions and confidence for ALL positions.
        3. Determine how many tokens should be revealed at this timestep.
        4. From all eligible (non-wall) positions, keep only the top-k
            most confident predictions. Everything else gets RE-MASKED.
        
        This allows the model to CORRECT early mistakes — a token revealed
        at step 50 can be re-masked at step 100 if the model becomes less
        confident about it given new context from other revealed tokens.
        
        Args:
            inp:  (B, 144) input board tokens
            cond: (B, 144) solved board tokens
            inf_steps: number of diffusion steps
        
        Returns:
            cur_dream: (B, 144) final denoised board
        """
        batch_size, seq_len = inp.shape
        device = inp.device

        # Initialize: all eligible tokens masked, walls pre-filled
        cur_dream = torch.full((batch_size, seq_len), self.mask_token_id, 
                            dtype=torch.long, device=device)
        
        # Walls are NEVER masked — they are structural and known from input
        wall_mask = (inp == 0)  # (B, 144) True where walls are
        cur_dream[wall_mask] = 0

        # Count eligible (non-wall) tokens per sample
        eligible_mask = ~wall_mask  # (B, 144) True where non-wall
        num_eligible = eligible_mask.float().sum(dim=1)  # (B,)

        # Timestep schedule: t goes from 1.0 (fully masked) → 0.0 (fully revealed)
        timesteps = torch.linspace(1.0, 0.0, inf_steps + 1, device=device)

        for step in range(inf_steps):
            t_current = timesteps[step]      # current masking level
            t_next = timesteps[step + 1]     # target masking level after this step

            # 1. Tell the model the current masking ratio
            t_input = t_current.view(1, 1).expand(batch_size, 1)

            # 2. Full forward pass — model predicts ALL positions
            logits = self._model(inp, cond, cur_dream, t_input)
            probs = torch.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)  # (B, 144)

            # 3. Determine how many eligible tokens to REVEAL at this point
            #    At t_next, we want (1 - t_next) fraction of eligible tokens revealed
            num_to_reveal = ((1.0 - t_next) * num_eligible).long()  # (B,)

            # 4. For each sample, rebuild the dream state from scratch
            #    Only the top-k most confident eligible predictions survive
            new_dream = torch.full_like(cur_dream, self.mask_token_id)
            new_dream[wall_mask] = 0  # walls always visible

            for b in range(batch_size):
                # Get confidence scores only for eligible positions
                scores = confidences[b].clone()
                scores[~eligible_mask[b]] = -1.0  # exclude walls from competition

                # Select the top num_to_reveal positions by confidence
                k = min(num_to_reveal[b].item(), eligible_mask[b].sum().item())
                if k > 0:
                    _, top_indices = torch.topk(scores, k=k)
                    new_dream[b, top_indices] = predictions[b, top_indices]

            cur_dream = new_dream

        return cur_dream

    def load_parameters(self):
        path = self.model_id
        print("loading model parameters from {}".format(path))
        checkpoint = torch.load(path, map_location=self.device)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model parameters loaded from {path}")


    def fit_and_dump(self, training_data, validation_data, epochs, dump_folder, checkpoints=None):
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
                    v_preds = torch.argmax(v_logits, dim=-1)

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
                        cur_dream = self._denoise_full_reeval(
                            bv_x, bv_cond, inf_steps=256
                        )
                        val_full_denoise_acc = (cur_dream == bv_y).float().mean().item()

                        # Compute all metrics
                        batch_metrics = compute_sokoban_metrics(
                            pred_board   = cur_dream,
                            input_board  = bv_x,
                            target_board = bv_y,
                            goal_board   = bv_cond
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
                grid_img = self.predict_and_get_wandb_image(val_x[0:1], val_g[0:1],
                                                            ground_truth_subgoal=val_y[0:1])
                wandb.log({"media/diffusion_process_sample1": grid_img})

                grid_img = self.predict_and_get_wandb_image(val_x[1:2], val_g[1:2],
                                                            ground_truth_subgoal=val_y[1:2])
                wandb.log({"media/diffusion_process_sample2": grid_img})

                grid_img = self.predict_and_get_wandb_image(val_x[2:3], val_g[2:3],
                                                            ground_truth_subgoal=val_y[2:3])
                wandb.log({"media/diffusion_process_sample3": grid_img})

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

    def predict_pdf_batch(self, input_boards, conditions, ground_truth_subgoal=None, steps=256):
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

        # Data collection for 4x4 grid (16 slots):
        # Slot 0: Input Board
        # Slot 1: Ground Truth Subgoal (if available)
        # Slots 2-13: 12 Intermediate Diffusion Steps
        # Slot 14: Final Dream (Prediction)
        # Slot 15: Final Board (Solved State)
        collected_boards = []
        collected_labels = []
        # Slot 0: Input state
        collected_boards.append(input_tokens[0].copy())
        collected_labels.append("Initial Input")

        # Slot 1: Ground Truth Subgoal (or placeholder)
        if gt_subgoal_tokens is not None:
            collected_boards.append(gt_subgoal_tokens[0].copy())
            collected_labels.append("GT Subgoal")
        else:
            collected_boards.append(input_tokens[0].copy())  # duplicate input as placeholder
            collected_labels.append("(No GT)")


        # 2. Setup Tensors
        inp = torch.tensor(input_tokens.reshape(batch_size, -1), dtype=torch.long).to(self.device)
        cond = torch.tensor(cond_tokens.reshape(batch_size, -1), dtype=torch.long).to(self.device)

        # Initialize fully masked dream (walls pre-filled)
        seq_len = 144
        cur_dream = torch.full((batch_size, seq_len), self.mask_token_id,
                            dtype=torch.long, device=self.device)
        wall_mask = (inp == 0)
        cur_dream[wall_mask] = 0

        # Eligible tokens (non-wall)
        eligible_mask = ~wall_mask  # (B, 144)
        num_eligible = eligible_mask.float().sum(dim=1)  # (B,)

        # 3. Diffusion Loop — Full Re-evaluation with Remasking
        save_indices = np.linspace(0, steps - 1, 12, dtype=int)
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=self.device)

        for step in range(steps):
            t_current = timesteps[step]
            t_next = timesteps[step + 1]

            with torch.no_grad():
                # Pass current masking ratio to model
                t_input = t_current.view(1, 1).expand(batch_size, 1)
                logits = self._model(inp, cond, cur_dream, t_input)
                probs = torch.softmax(logits, dim=-1)
                confidences, predictions = torch.max(probs, dim=-1)

                # How many eligible tokens should be revealed after this step
                num_to_reveal = ((1.0 - t_next) * num_eligible).long()

                # Rebuild dream from scratch — full re-evaluation
                new_dream = torch.full_like(cur_dream, self.mask_token_id)
                new_dream[wall_mask] = 0

                for b in range(batch_size):
                    scores = confidences[b].clone()
                    scores[~eligible_mask[b]] = -1.0

                    k = min(num_to_reveal[b].item(), eligible_mask[b].sum().item())
                    if k > 0:
                        _, top_indices = torch.topk(scores, k=k)
                        new_dream[b, top_indices] = predictions[b, top_indices]

                cur_dream = new_dream

            # Capture intermediate states for visualization
            if step in save_indices:
                collected_boards.append(cur_dream[0].cpu().numpy().copy())
                collected_labels.append(f"Step {step} (t={t_current:.2f})")

        # Add final state
        dream_tokens_flat = cur_dream.cpu().numpy().reshape(batch_size, 144)
        collected_boards.append(dream_tokens_flat[0])
        collected_labels.append("Final Dream")

        # Slot 15: Final Board (solved state)
        collected_boards.append(cond_tokens[0])
        collected_labels.append("Final Board")

        # 4. Visualization
        save_path = save_diffusion_grid(
            collected_boards, collected_labels, self.dump_folder, self._predictions_counter
        )

        # 5. Legacy Adapter Logic (Returning PDF to Solver)
        input_tokens_flat = input_tokens.reshape(batch_size, 144)
        pdf_1009 = np.zeros((batch_size, 1009))
        for b in range(batch_size):
            diffs = np.where(input_tokens_flat[b] != dream_tokens_flat[b])[0]
            if len(diffs) == 0:
                pdf_1009[b, 1008] = 1.0
            else:
                idx = diffs[0]
                target_tile = dream_tokens_flat[b, idx]
                pdf_1009[b, (idx * 7) + target_tile] = 1.0

        return pdf_1009, save_path

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
#
# class GoalPredictorPixelDiff:
#     def __init__(
#         self,
#         num_layers=5,
#         batch_norm=True,
#         model_id=None,
#         learning_rate=0.01,
#         kernel_size=(5, 5),
#         weight_decay=0.,
#         batch_size=32
#     ):
#
#         self.core_env = Sokoban()
#         self.dim_room = self.core_env.get_dim_room()
#
#         self.num_layers = num_layers
#         self.batch_norm = batch_norm
#         self.model_id = model_id
#
#         self._model = None
#         self._predictions_counter = 0
#         self.learning_rate = learning_rate
#         self.kernel_size = kernel_size
#         self.weight_decay = weight_decay
#         self.batch_size = batch_size
#
#     def construct_networks(self):
#         if self._model is None:
#             if self.model_id is None:
#                 input_state = Input(batch_shape=(None, None, None, 7))
#                 input_condition = Input(batch_shape=(None, None, None, 7))
#
#                 layer = Concatenate()([input_state, input_condition])
#
#                 for _ in range(self.num_layers):
#                     layer = Conv2D(
#                         filters=64,
#                         kernel_size=self.kernel_size,
#                         padding='same',
#                         activation='relu',
#                         kernel_regularizer=l2(self.weight_decay),
#                     )(layer)
#
#                     if self.batch_norm:
#                         layer = BatchNormalization()(layer)
#
#                 branch1 = Dense(7, activation='relu', kernel_regularizer=l2(self.weight_decay))(layer)
#                 branch1 = Flatten()(branch1)
#
#                 branch2 = Dense(1, activation='relu', kernel_regularizer=l2(self.weight_decay))(layer)
#                 # branch2 = Flatten()(branch2)
#                 branch2 = GlobalAveragePooling2D()(branch2)
#
#                 output = Concatenate()([branch1, branch2])
#                 output = Softmax()(output)
#
#                 self._model = Model(inputs=[input_state, input_condition], outputs=output)
#                 self._model.compile(
#                     loss='categorical_crossentropy',
#                     metrics='accuracy',
#                     optimizer=Adam(learning_rate=self.learning_rate)
#                 )
#
#                 self.data_creator = DataCreatorSokobanPixelDiff()
#             else:
#                 self.load_model(self.model_id)
#
#     def reset_predictions_counter(self):
#         self._predictions_counter = 0
#
#     def read_predictions(self):
#         return self._predictions_counter
#
#     def load_data(self, dataset_file):
#         self.data_creator.load(dataset_file)
#
#     def fit_and_dump(self, x, y, validation_data, epochs, dump_folder, checkpoints=None):
#
#         # --- ENHANCED DEBUG PRINTS ---
#         import numpy as np
#
#         # --- UPDATED DEBUG FOR TOKENS ---
#         print(f"DEBUG: Input X shape: {x.shape} | Target Y shape: {y.shape}")
#         unique_tokens_x = np.unique(x)
#         unique_tokens_y = np.unique(y)
#         print(f"DEBUG: Unique tokens in X: {unique_tokens_x}")
#         print(f"DEBUG: Unique tokens in Y: {unique_tokens_y}")
#
#         # Check if any tokens are outside the 0-6 range (ID 7 is reserved for MASK)
#         if np.any(unique_tokens_y > 6):
#             print("WARNING: Found tokens > 6 in target. Ensure ID 7 is reserved for MASK only.")
#         # ---
#
#         for epoch in range(epochs):
#             history = self._model.fit(x, y, batch_size=self.batch_size, epochs=1, validation_data=validation_data)
#             train_history = history.history
#             for metric, value in train_history.items():
#                 log_scalar(metric, epoch, value[0])
#             if checkpoints is not None and epoch in checkpoints:
#                 print(f'saving model after {epoch} epochs.')
#                 self.save_model(os.path.join(dump_folder, f'epoch_{epoch}'))
#             gc.collect()
#         self.save_model(os.path.join(dump_folder, f'epoch_{epoch}'))
#
#     def predict_pdf(self, input, condition):
#         self._predictions_counter += 1
#
#         test_input = np.array([input])
#         print(f"DEBUG: Inference Input Shape: {test_input.shape}")
#
#         raw =  self._model.predict([np.array([input]), np.array([condition])])[0]
#         return raw
#
#     def predict_pdf_batch(self, input_boards, conditions):
#         self._predictions_counter += 1
#         raw =  self._model.predict([input_boards, conditions])
#         return raw
#
#     def save_model(self, model_id):
#         self._model.save(model_id)
#
#     def load_model(self, model_id):
#         self._model = keras.models.load_model(model_id)
#
#     def flat_to_2d(self, n):
#         element = n % 7
#         base_n = n // 7
#         x = base_n // self.dim_room[0]
#         y = base_n % self.dim_room[1]
#
#         return x, y, element
#
#     def smart_sample(self, pdf, internal_confidence_level):
#         assert internal_confidence_level > 0 and internal_confidence_level < 1, 'confidence_level must be between 0 and 1'
#         out = []
#         out_p = []
#
#         for idx in reversed(np.argsort(pdf)):
#             out.append(self.flat_to_2d(idx))
#             out_p.append(pdf[idx])
#
#             if sum(out_p) > internal_confidence_level:
#                 break
#
#         return out, out_p


import os
import numpy as np


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