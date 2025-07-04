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

        if self.verbose:
            print(f"Creating the base generator with the shared encoder")

        self.generator = GCVariationalAutoencoderConv(
            seq_len=self.input_size[0],
            feat_dim=self.input_size[1],
            latent_dim=self.args.feature_dim,  # 2 for visualization
            hidden_layer_sizes=[64, 128, 256, 512],  # [128, 256]
            device=self.device,
            recon_wt=self.args.recon_wt,
        )

    def learn_task(self, task):
        (x_train, y_train), (x_val, y_val), _ = task
        self.before_task(y_train)

        for id in self.classes_in_task:
            if self.verbose:
                print(f"Creating the disentangled decoder {id}")

            self.generator.create_decoder(id)

            ckpt_path_g = self.ckpt_path.replace("/ckpt", f"/generator_ckpt_{id}")
            epochs_g = self.args.epochs_g

            if self.verbose:
                print(f"Training the generator {id}")

            encoder_params = list(self.generator.encoder.parameters())
            decoder_params = list(self.generator.decoders[str(id)].parameters())

            params = encoder_params + decoder_params

            optimizer_g = Adam(params, lr=self.args.lr_g, betas=(0.9, 0.999))
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
                    generator_loss_dict = self.generator.train_a_batch(
                        x=x.transpose(1, 2),
                        optimizer=optimizer_g,
                        decoder_id=id,
                        x_=x_,
                        rnt=rnt,
                    )

                train_mse_loss, train_kl_loss = self.generator.evaluate(
                    train_dataloader, id
                )
                # Validate on val set for early stop
                val_mse_loss, val_kl_loss = self.generator.evaluate(val_dataloader, id)

                if self.verbose:
                    print(
                        "Epoch {}/{}: Recons Loss = {}, KL Divergence = {}".format(
                            epoch + 1,
                            epochs_g,
                            train_mse_loss,
                            train_kl_loss,
                        )
                    )

                early_stopping(val_mse_loss, self.generator)
                if early_stopping.early_stop:
                    if self.verbose:
                        print("Early stopping")
                    break

        self.after_task(None, None)

    # override method
    def after_task(self, x_train, y_train):
        self.learned_classes += self.classes_in_task
        self.generator.copy_encoder()

    def evaluate(self, task_stream, path=None):
        if self.task_now == 0:
            self.num_tasks = task_stream.n_tasks
            self.Acc_tasks = {
                "valid": np.zeros((self.num_tasks, self.num_tasks)),
                "test": np.zeros((self.num_tasks, self.num_tasks)),
            }

        prototypes = []  # list of prototypes (L, C)
        for id in self.learned_classes:
            prototype = self.generator.estimate_prototype(size=100, decoder_id=id)
            prototypes.append(prototype)
        # prototypes = torch.stack(prototypes, dim=0)  # (#prototypes, L, C)

        eval_modes = ["valid", "test"]  # 'valid' is for checking generalization.
        for mode in eval_modes:
            if self.verbose:
                print("\n ======== Evaluate on {} set ========".format(mode))

            for i in range(self.task_now + 1):
                (x_eval, y_eval) = (
                    task_stream.tasks[i][1]
                    if mode == "valid"
                    else task_stream.tasks[i][2]
                )
                eval_dataloader_i = Dataloader_from_numpy(
                    x_eval, y_eval, self.batch_size, shuffle=False
                )

                total = 0
                correct = 0

                for batch_id, (x, y) in enumerate(eval_dataloader_i):
                    x, y = x.to(self.device), y.to(self.device)
                    x = x.transpose(1, 2)

                    total += y.size(0)

                    if y.size == 1:
                        y.unsqueeze()

                    dists = []  # list of distances (#prototypes, )
                    for id, p in enumerate(prototypes):
                        dist = self.generator.estimate_distance(x, p)
                        dists.append(dist)
                    dists = torch.cat(dists, dim=1)  # (batch_size, #prototypes)

                    preds = torch.argmin(dists, dim=1)
                    correct += preds.eq(y).sum().item()

                    # cf matrix
                    if (
                        self.cf_matrix
                        and self.task_now + 1 == self.num_tasks
                        and mode == "test"
                    ):
                        self.y_pred_cf.extend(preds.data.cpu().numpy())
                        self.y_true_cf.extend(y.data.cpu().numpy())
                eval_acc_i = 100.0 * (correct / total)

                if self.verbose:
                    print("Task {}: Accuracy == {} ;".format(i, eval_acc_i))

                self.Acc_tasks[mode][self.task_now][i] = np.around(
                    eval_acc_i, decimals=2
                )

            if self.task_now + 1 == self.num_tasks and self.verbose:
                with np.printoptions(suppress=True):  # Avoid Scientific Notation
                    print("Accuracy matrix of all tasks:")
                    print(self.Acc_tasks[mode])
