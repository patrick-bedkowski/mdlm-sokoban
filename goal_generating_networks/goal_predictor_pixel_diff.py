import gc
import os

import wandb
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
        Refined Forward Process:
        1. Samples a random mask ratio 't'.
        2. Protects walls (Token 0) from being masked.
        3. Replaces eligible tokens with MASK_ID.
        """
        device = target_board.device
        # 1. Sample ratio t ~ U(0, 1)
        mask_ratio = torch.rand(target_board.size(0), 1, device=device)

        # 2. Identify eligible non-wall tokens
        mask_eligible = (target_board != 0)

        # 3. Generate mask indices based on ratio[cite: 1]
        mask_indices = (torch.rand(target_board.shape, device=device) < mask_ratio) & mask_eligible

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
        run_name = f"{self._model.architecture_name}_{self.date_now}"
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

    def load_parameters(self):
        path = self.model_id
        print("loading model parameters from {}".format(path))
        checkpoint = torch.load(path, map_location=self.device)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model parameters loaded from {path}")

    def fit_and_dump(self, training_data, validation_data, epochs, dump_folder, checkpoints=None):
        # 1. Flatten the inputs to (N, 144) to avoid 3D broadcasting errors
        # x[0] is current, x[1] is final goal
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

        # print('DATA SAVED')
        # # save example inputs to the txt file
        # with open(os.path.join(dump_folder, 'example_inputs.txt'), 'w') as f:
        #     f.write(f"Example Training Input (current state):\n{train_x[0].view(12, 12)}\n")
        #     f.write(f"Example Training Condition (final goal):\n{train_cond[0].view(12, 12)}\n")
        #     f.write(f"Example Training Target (midpoint):\n{train_y[0].view(12, 12)}\n")
        #     f.write(f"Example Validation Input (current state):\n{val_x[0].view(12, 12)}\n")
        #     f.write(f"Example Validation Condition (final goal):\n{val_g[0].view(12, 12)}\n")
        #     f.write(f"Example Validation Target (midpoint):\n{val_y[0].view(12, 12)}\n")

        num_samples = train_x.shape[0]

        for epoch in range(epochs):
            # --- TRAINING PHASE ---
            self._model.train()
            epoch_train_loss = 0
            indices = torch.randperm(num_samples)  # used for randomly shuffling the data at each epoch

            for i in range(0, num_samples, self.batch_size):  # process all the batches
                batch_idx = indices[i:i + self.batch_size]
                b_x = train_x[batch_idx].to(self.device)
                b_cond = train_cond[batch_idx].to(self.device)
                b_y = train_y[batch_idx].to(self.device)

                # 1. MASKING LOGIC: Only mask non-wall tokens
                masked_inputs, mask_indices, mask_ratio = self._model.apply_stochastic_mask(b_y)
                logits = self._model(b_x, b_cond, masked_inputs, mask_ratio)

                # 2. WEIGHTED LOSS: Apply token weights and LLaDA 1/t weighting
                # CrossEntropy expects (B, C, L)
                loss_raw = self.criterion(logits.transpose(1, 2), b_y)

                # LLaDA Objective: Weight the loss on masked tokens by 1/t
                # This prevents the model from ignoring samples with high masking ratios.
                t_weighted_loss = (loss_raw * mask_indices).sum(dim=1) / (mask_ratio.squeeze() * b_y.size(1) + 1e-6)
                final_loss = t_weighted_loss.mean()

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
                for i in range(0, val_x.size(0), self.batch_size):
                    bv_x = val_x[i:i + self.batch_size].to(self.device)
                    bv_cond = val_g[i:i + self.batch_size].to(self.device)
                    bv_y = val_y[i:i + self.batch_size].to(self.device)

                    # Standard Masking for Delta Check
                    # If you are calculating validation loss or accuracy
                    v_masked_inputs, v_mask_indices, v_mask_ratio = self._model.apply_stochastic_mask(bv_y)
                    v_logits = self._model(bv_x, bv_cond, v_masked_inputs, v_mask_ratio)
                    v_preds = torch.argmax(v_logits, dim=-1)

                    # 3. WALL STABILITY ACCURACY
                    # Check accuracy only on tokens that are walls in the ground truth
                    wall_mask = (bv_y == 0)
                    if wall_mask.sum() > 0:
                        wall_correct = (v_preds == bv_y) & wall_mask
                        val_wall_acc += wall_correct.sum().item() / (wall_mask.sum().item() + 1e-6)

                    # DELTA ACCURACY: Only check accuracy on tiles that changed (Agent/Boxes)
                    change_mask = (bv_x != bv_y)
                    critical_mask = v_mask_indices & change_mask
                    if critical_mask.sum() > 0:
                        correct = (v_preds == bv_y) & critical_mask
                        val_delta_acc += correct.sum().item() / (critical_mask.sum().item() + 1e-6)

                    # FULL DENOISING CHECK (LLaDA Low-Confidence Remasking)
                    if i == 0:
                        inf_steps = 256
                        cur_dream = torch.full_like(bv_y, self.mask_token_id)
                        # PROTECT WALLS in starting dream: Start with walls already visible
                        cur_dream = torch.where(bv_x == 0, 0, cur_dream)

                        timesteps = torch.linspace(1, 0, inf_steps + 1)
                        for step in range(inf_steps):
                            t_next = timesteps[step + 1]
                            i_logits = self._model(bv_x, bv_cond, cur_dream, t_next.unsqueeze(0).to(self.device))
                            i_probs = torch.softmax(i_logits, dim=-1)
                            confidences, predictions = torch.max(i_probs, dim=-1)

                            is_masked = (cur_dream == self.mask_token_id)
                            num_to_keep = int((1 - t_next) * bv_y.size(1))

                            for b in range(bv_x.size(0)):
                                score = confidences[b].clone()
                                score[~is_masked[b]] = 1.1
                                _, top_indices = torch.topk(score, k=num_to_keep)

                                new_state = torch.full_like(cur_dream[b], self.mask_token_id)
                                # Keep walls visible even if they weren't in top-k
                                new_state[bv_x[b] == 0] = 0
                                new_state[top_indices] = predictions[b][top_indices]
                                cur_dream[b] = new_state

                        val_full_denoise_acc = (cur_dream == bv_y).float().mean().item()

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
                grid_img = self.predict_and_get_wandb_image(val_x[0:1], val_g[0:1])
                wandb.log({"media/diffusion_process": grid_img})

            print(f"Epoch {epoch} | Delta Acc: {avg_val_delta:.4f} | Wall Stability: {avg_val_wall:.4f} | Full Denoise Acc: {val_full_denoise_acc:.4f}")
            log_scalar('val_delta_accuracy', epoch, avg_val_delta)
            log_scalar('val_full_denoise_acc', epoch, val_full_denoise_acc)

            # Logging
            print(
                f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f}")
            log_scalar('train_loss', epoch, avg_train_loss)

            if checkpoints is not None and epoch in checkpoints:
                self.save_model(os.path.join(dump_folder, f'epoch_{epoch}.pt'))

    def predict_and_get_wandb_image(self, input_boards, conditions):
        """Helper to run prediction and return a wandb Image object."""
        # This calls your predict_pdf_batch logic but returns the plot
        _, plot_path = self.predict_pdf_batch(input_boards.cpu().numpy(), conditions.cpu().numpy(), steps=256)
        # save_diffusion_grid(..., dump_folder=self.dump_folder)
        return wandb.Image(plot_path, caption="Diffusion Steps")

    def predict_pdf_batch(self, input_boards, conditions, steps=256):
        self._predictions_counter += 1
        self._model.eval()

        # 1. Standard Pre-processing
        if input_boards.ndim == 3: input_boards = np.expand_dims(input_boards, axis=0)
        if conditions.ndim == 3: conditions = np.expand_dims(conditions, axis=0)

        # Handle Input Boards (144 vs 1008)
        if input_boards.shape[-1] == 1008:
            # Reshape (B, 1008) -> (B, 144, 7) then argmax to (B, 144)
            input_tokens = np.argmax(input_boards.reshape(-1, 144, 7), axis=-1)
        elif input_boards.shape[-1] == 7:
            input_tokens = np.argmax(input_boards, axis=-1)
        else:
            input_tokens = input_boards

        # Handle Conditions / Final Goal (The likely source of the 1008)
        if conditions.shape[-1] == 1008:
            # Reshape (B, 1008) -> (B, 144, 7) then argmax to (B, 144)
            cond_tokens = np.argmax(conditions.reshape(-1, 144, 7), axis=-1)
        elif conditions.shape[-1] == 7:
            conditions = np.argmax(conditions, axis=-1)

        batch_size = input_tokens.shape[0]

        # Data collection for 4x4 grid (16 slots)
        # Slot 0: Input, Slot 15: Final, Slots 1-14: Intermediate
        collected_boards = []
        collected_labels = []

        # Save the absolute input state as the first slot
        collected_boards.append(input_tokens[0].copy())
        collected_labels.append("Initial Input")

        # 2. Setup Tensors
        inp = torch.tensor(input_tokens.reshape(batch_size, -1), dtype=torch.long).to(self.device)
        cond = torch.tensor(conditions.reshape(batch_size, -1), dtype=torch.long).to(self.device)
        cur_subgoals = torch.full((batch_size, 144), self.mask_token_id, dtype=torch.long).to(self.device)

        # 3. Diffusion Loop
        # Calculate which indices to save to fill the 14 intermediate slots
        # 256 / 14 ~ every 18 steps
        save_indices = np.linspace(0, steps - 1, 14, dtype=int)

        for i in range(steps):
            with torch.no_grad():
                logits = self._model(inp, cond, cur_subgoals)
                probs = torch.softmax(logits, dim=-1)
                max_probs, pred_ids = torch.max(probs, dim=-1)

                is_masked = (cur_subgoals == self.mask_token_id)
                num_masked = is_masked[0].sum().item()
                num_to_reveal = int(np.ceil(num_masked / (steps - i)))

                for b in range(batch_size):
                    if num_to_reveal > 0:
                        conf = max_probs[b].clone()
                        conf[~is_masked[b]] = -1.0
                        _, top_idx = torch.topk(conf, k=min(num_to_reveal, num_masked))
                        cur_subgoals[b, top_idx] = pred_ids[b, top_idx]

            # Log to list if it's one of our 14 capture points
            if i in save_indices:
                collected_boards.append(cur_subgoals[0].cpu().numpy().copy())
                collected_labels.append(f"Diff Step {i}")

        # Add the Final Dream state as the 16th slot
        dream_tokens_flat = cur_subgoals.cpu().numpy().reshape(batch_size, 144)
        collected_boards.append(dream_tokens_flat[0])
        collected_labels.append("Final Dream")

        # 4. Trigger Visualization
        save_path = save_diffusion_grid(collected_boards, collected_labels, self.dump_folder, self._predictions_counter)

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
        out, out_p = [], []
        for idx in reversed(np.argsort(pdf)):
            out.append(self.flat_to_2d(idx))
            out_p.append(pdf[idx])
            if sum(out_p) > internal_confidence_level: break
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