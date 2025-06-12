import torch
import numpy as np
from agents.base_gc import BaseLearnerGC
from torch.optim import Adam
import torch.nn.functional as F
from utils.setup_elements import input_size_match
from utils.data import Dataloader_from_numpy, extract_samples_according_to_labels
from models.gcppVAE import GCVariationalAutoencoderConv
from utils.utils import EarlyStopping


class GenerativeClassiferPPv2(BaseLearnerGC):
    """
    data-free prototype-based generative classifier with latent distillation
    """

    def __init__(self, args):
        super(GenerativeClassiferPPv2, self).__init__(args)
        self.input_size = input_size_match[args.data]
        self.batch_size = args.batch_size

    def learn_task(self, task):
        (x_train, y_train), (x_val, y_val), _ = task
        self.before_task(y_train)

        if self.verbose:
            print(f"Creating the base generator with the shared encoder")

        generator = GCVariationalAutoencoderConv(
            seq_len=self.input_size[0],
            feat_dim=self.input_size[1],
            latent_dim=self.args.feature_dim,  # 2 for visualization
            hidden_layer_sizes=[64, 128, 256, 512],  # [128, 256]
            device=self.device,
            recon_wt=self.args.recon_wt,
        )

        ckpt_path_g = self.ckpt_path.replace("/ckpt", f"/generator_ckpt")
        epochs_g = self.args.epochs_g

        for id in self.classes_in_task:
            if self.verbose:
                print(f"Creating the disentangled decoder {id}")

            generator.create_decoder(id)
            optimizer_g = Adam(
                generator.parameters(), lr=self.args.lr_g, betas=(0.9, 0.999)
            )

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

                    # generator's input should be (N, L, C)
                    rnt = 1 / (self.task_now + 1) if self.args.adaptive_weight else 0.5
                    generator_loss_dict = generator.train_a_batch(
                        x=x.transpose(1, 2),
                        optimizer=optimizer_g,
                        decoder_id=id,
                        x_=x_,
                        rnt=rnt,
                    )

                train_mse_loss, train_kl_loss = generator.evaluate(train_dataloader, id)
                # Validate on val set for early stop
                val_mse_loss, val_kl_loss = generator.evaluate(val_dataloader, id)

                if self.verbose:
                    print(
                        "Epoch {}/{}: Recons Loss = {}, KL Divergence = {}".format(
                            epoch + 1,
                            epochs_g,
                            train_mse_loss,
                            train_kl_loss,
                        )
                    )

                early_stopping(val_mse_loss, generator)
                if early_stopping.early_stop:
                    if self.verbose:
                        print("Early stopping")
                    break

        self.after_task(None, None)

    def evaluate(self, task_stream, path=None):
        print("OK")
        pass
