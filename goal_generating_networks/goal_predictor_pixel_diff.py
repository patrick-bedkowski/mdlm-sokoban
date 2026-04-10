import gc
import os

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


# --- PyTorch Transformer Module ---
class SokobanTransformer(nn.Module):
    def __init__(self, vocab_size=8, seq_len=144, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output 7 classes (IDs 0-6) for each of the 144 positions
        self.output_layer = nn.Linear(d_model, 7)

    def forward(self, current_state, final_goal, masked_subgoal):
        # Ensure everything is (Batch, 144)
        curr = current_state.view(current_state.size(0), -1).long()
        goal = final_goal.view(final_goal.size(0), -1).long()
        mask = masked_subgoal.view(masked_subgoal.size(0), -1).long()

        # Sum the embeddings (Conditioning)
        # The model now knows: "I am HERE, I want to be THERE, and this is my CURRENT GUESS for the midpoint"
        x = self.embedding(curr) + self.embedding(goal) + self.embedding(mask)
        x = x + self.pos_embedding

        feat = self.transformer(x)
        return self.output_layer(feat)  # (Batch, 144, 7)


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

        self._model = None
        self._predictions_counter = 0
        self.mask_token_id = 7
        self.data_creator = DataCreatorSokobanPixelDiff()

    def construct_networks(self):
        if self._model is None:
            # Initialize the Transformer
            self._model = SokobanTransformer(num_layers=self.num_layers).to(self.device)

            # Print Parameter Count
            params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            print(f"MDLM Transformer initialized with {params:,} trainable parameters.")

            self.optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)
            self.criterion = nn.CrossEntropyLoss(reduction='none')

    def fit_and_dump(self, x, y, validation_data, epochs, dump_folder, checkpoints=None):
        # 1. Flatten the inputs to (N, 144) to avoid 3D broadcasting errors
        # x[0] is current, x[1] is final goal
        train_x = torch.tensor(x[0], dtype=torch.long).view(x[0].shape[0], -1)
        train_cond = torch.tensor(x[1], dtype=torch.long).view(x[1].shape[0], -1)
        train_y = torch.tensor(y, dtype=torch.long).view(y.shape[0], -1)

        val_x = torch.tensor(validation_data[0][0], dtype=torch.long).view(validation_data[0][0].shape[0], -1)
        val_cond = torch.tensor(validation_data[0][1], dtype=torch.long).view(validation_data[0][1].shape[0], -1)
        val_y = torch.tensor(validation_data[1], dtype=torch.long).view(validation_data[1].shape[0], -1)

        num_samples = train_x.shape[0]

        for epoch in range(epochs):
            self._model.train()
            epoch_loss = 0
            indices = torch.randperm(num_samples)

            for i in range(0, num_samples, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                b_x = train_x[batch_idx].to(self.device)
                b_cond = train_cond[batch_idx].to(self.device)
                b_y = train_y[batch_idx].to(self.device)  # Now (Batch, 144)

                # --- CORRECTED MASKING LOGIC ---
                # mask_ratio: (Batch, 1)
                mask_ratio = torch.rand(b_y.size(0), 1, device=self.device) * 0.8 + 0.2

                # This now works because b_y is (Batch, 144)
                # mask_indices: (Batch, 144)
                mask_indices = torch.rand(b_y.shape, device=self.device) < mask_ratio

                masked_inputs = b_y.clone()
                masked_inputs[mask_indices] = self.mask_token_id

                # Forward (Ensure your model forward accepts b_x, b_cond, masked_inputs)
                logits = self._model(b_x, b_cond, masked_inputs)

                # Compute loss only on masked tokens
                # logits: (B, 144, 7), b_y: (B, 144)
                loss = self.criterion(logits.transpose(1, 2), b_y)
                masked_loss = (loss * mask_indices).sum() / (mask_indices.sum() + 1e-6)

                self.optimizer.zero_grad()
                masked_loss.backward()
                self.optimizer.step()
                epoch_loss += masked_loss.item()

            avg_loss = epoch_loss / (num_samples / self.batch_size)
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
            log_scalar('train_loss', epoch, avg_loss)

            if checkpoints is not None and epoch in checkpoints:
                print(f"Checkpointing model at epoch {epoch} with loss {avg_loss:.4f}, directory: {dump_folder}")
                self.save_model(os.path.join(dump_folder, f'epoch_{epoch}.pt'))

    def predict_pdf(self, input, condition):
        """
        Modified to output the 1009-dim vector expected by the Search code.
        """
        self._predictions_counter += 1
        self._model.eval()

        # Flatten to 144 tokens if they arrive as (12, 12)
        inp = torch.tensor(input.flatten(), dtype=torch.long).unsqueeze(0).to(self.device)

        # Start with all tokens masked for the subgoal (Standard MDLM inference start)
        masked_goal = torch.full((1, 144), self.mask_token_id, dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self._model(inp, masked_goal)  # (1, 144, 7)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # (144, 7)

        # Flatten probabilities: 144 * 7 = 1008
        pdf_1008 = probs.flatten()

        # Add a dummy 1009th bit (end-of-subgoal) to match original Search signature
        # We set it to a low probability so the search explores tile changes first
        return np.concatenate([pdf_1008, [0.01]])

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
