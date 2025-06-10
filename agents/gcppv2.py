import torch
import numpy as np
from agents.base_gc import BaseLearnerGC
from torch.optim import Adam
import torch.nn.functional as F
from utils.setup_elements import input_size_match
from utils.data import Dataloader_from_numpy, extract_samples_according_to_labels
from models.timeVAE import VariationalAutoencoderConv
from utils.utils import EarlyStopping


class GenerativeClassiferPPv2(BaseLearnerGC):
    """
    data-free prototype-based generative classifier with latent distillation
    """

    def __init__(self, args):
        super(GenerativeClassiferPPv2, self).__init__(args)
        self.input_size = input_size_match[args.data]
        self.batch_size = args.batch_size
        self.generators = {}

    def learn_task(self, task):
        (x_train, y_train), (x_val, y_val), _ = task
        self.before_task(y_train)

        if self.verbose:
            print(f"Creating the shared encoder")

        vae = VariationalAutoencoderConv(
            seq_len=self.input_size[0],
            feat_dim=self.input_size[1],
            latent_dim=self.args.feature_dim,  # 2 for visualization
            hidden_layer_sizes=[64, 128, 256, 512],  # [128, 256]
            device=self.device,
            recon_wt=self.args.recon_wt,
        )

        epochs_g = self.args.epochs_g
        ckpt_path_g = self.ckpt_path.replace("/ckpt", f"/vae_ckpt_{id}")

        for id in self.classes_in_task:
            if self.verbose:
                print(f"Creating the disentangled decoder {id}")

            vae.create_decoder(id)
            optimizer_g = Adam(vae.parameters(), lr=self.args.lr_g, betas=(0.9, 0.999))

            if self.verbose:
                print(f"Training the generator {id}")

            early_stopping = EarlyStopping(
                path=ckpt_path_g,
                patience=self.args.patience,
                mode="min",
                verbose=False,
            )

            (x_train_id, y_train_id) = extract_samples_according_to_labels(
                x_train, y_train, [id]
            )
            (x_val_id, y_val_id) = extract_samples_according_to_labels(
                x_val, y_val, [id]
            )

            train_dataloader = Dataloader_from_numpy(
                x_train_id, y_train_id, self.batch_size, shuffle=True
            )
            val_dataloader = Dataloader_from_numpy(
                x_val_id, y_val_id, self.batch_size, shuffle=False
            )

            for epoch in range(epochs_g):
                for batch_id, (x, y) in enumerate(train_dataloader):
                    x = x.to(self.device)
                    x_ = None

    def evaluate(self, task_stream, path=None):
        pass
