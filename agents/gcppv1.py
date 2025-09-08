import torch
import numpy as np
from agents.base_gcpp import BaseLearnerGCPP
from torch.optim import Adam
import torch.nn.functional as F
from utils.setup_elements import input_size_match
from utils.data import Dataloader_from_numpy, extract_samples_according_to_labels
from models.timeVAE import VariationalAutoencoderConv
from utils.utils import EarlyStopping


class GenerativeClassiferPlusPlusV1(BaseLearnerGCPP):
    """
    data-free prototype-based generative classifier incremental learning
    """

    def __init__(self, args):
        super(GenerativeClassiferPlusPlusV1, self).__init__(args)
        self.input_size = input_size_match[args.data]
        self.batch_size = args.batch_size

    def learn_task(self, task):
        """
        Basic workflow for learning a task. For particular methods, this function will be overwritten.
        """
        (x_train, y_train), (x_val, y_val), _ = task
        self.before_task(y_train)

        for id in self.classes_in_task:
            if self.verbose:
                print(f"Creating the generator {id}")

            generator = VariationalAutoencoderConv(
                seq_len=self.input_size[0],
                feat_dim=self.input_size[1],
                latent_dim=self.args.feature_dim,  # 2 for visualization
                hidden_layer_sizes=[64, 128, 256, 512],  # [128, 256]
                device=self.device,
                recon_wt=self.args.recon_wt,
                classifier=self.args.classifier,
            )
            setattr(self, "generator{}".format(id), generator)

            ckpt_path_g = self.ckpt_path.replace("/ckpt", f"/generator_ckpt_{id}")
            epochs_g = self.args.epochs_g

            if self.verbose:
                print(f"Training the generator {id}")

            optimizer_g = Adam(
                generator.parameters(), lr=self.args.lr_g, betas=(0.9, 0.999)
            )
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

                    rnt = 1 / (self.task_now + 1) if self.args.adaptive_weight else 0.5
                    generator_loss_dict = getattr(
                        self, "generator{}".format(id)
                    ).train_a_batch(
                        x=x.transpose(1, 2),
                        optimizer=optimizer_g,
                        x_=x_,
                        rnt=rnt,
                    )

                train_mse_loss, train_kl_loss = getattr(
                    self, "generator{}".format(id)
                ).evaluate(train_dataloader)
                # Validate on val set for early stop
                val_mse_loss, val_kl_loss = getattr(
                    self, "generator{}".format(id)
                ).evaluate(val_dataloader)
                # check comment on evaluate generator

                if self.verbose:
                    print(
                        "Epoch {}/{}: Recons Loss = {}, KL Divergence = {}".format(
                            epoch + 1,
                            epochs_g,
                            train_mse_loss,
                            train_kl_loss,
                        )
                    )

                # func save_checkpoint in early_stopping is why it took so long
                early_stopping(val_mse_loss, getattr(self, "generator{}".format(id)))
                if early_stopping.early_stop:
                    if self.verbose:
                        print("Early stopping")
                    break

            # self.after_task for generator
            if hasattr(self.generator, "decoders") and self.generator.decoders:
                self.generator.copy_encoder()
            self.learned_classes += [id]

        # self.after_task(x_train, y_train) # for learner

    def evaluate(self, task_stream, path=None):
        if self.task_now == 0:
            self.num_tasks = task_stream.n_tasks
            self.Acc_tasks = {
                "valid": np.zeros((self.num_tasks, self.num_tasks)),
                "test": np.zeros((self.num_tasks, self.num_tasks)),
            }

        prototypes = (
            []
        )  # list of prototypes with each shape (C, L) if not fmap else (latent_dim, )
        for id in self.learned_classes:
            prototype = getattr(self, "generator{}".format(id)).estimate_prototype(
                size=100
            )
            prototypes.append(prototype)
        # prototypes = torch.stack(prototypes, dim=0)  # (#prototypes, C, L) if not fmap else (#prototypes, latent_dim, )

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
                # here is most suitable, if applicable, for evaluating generator loss (acts as a generator)

                # evaluating generator acc (acts as a pseudo-learner)
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

                    distances = []  # list of distances with each shape (#prototypes, )
                    for id, p in enumerate(prototypes):
                        distance = getattr(
                            self, "generator{}".format(id)
                        ).estimate_distance(x, p)
                        distances.append(distance)
                    distances = torch.cat(distances, dim=1)  # (batch_size, #prototypes)

                    prediction = torch.argmin(distances, dim=1)
                    correct += prediction.eq(y).sum().item()

                    if (
                        self.cf_matrix
                        and self.task_now + 1 == self.num_tasks
                        and mode == "test"
                    ):
                        self.test_for_cf_matrix(
                            eval_dataloader_i,
                            prediction.data.cpu().numpy(),
                            y.data.cpu().numpy(),
                        )
                eval_acc_i = 100.0 * (correct / total)

                if self.verbose:
                    print("Task {}: Accuracy == {} ;".format(i, eval_acc_i))

                self.Acc_tasks[mode][self.task_now][i] = np.around(
                    eval_acc_i, decimals=2
                )

            # Print accuracy matrix of the tasks on this run
            if self.task_now + 1 == self.num_tasks and self.verbose:
                with np.printoptions(suppress=True):  # Avoid Scientific Notation
                    print("Accuracy matrix of all tasks:")
                    print(self.Acc_tasks[mode])

    def after_task(self, x_train, y_train):
        # TODO
        # buffer matter and or proto matter
        pass
