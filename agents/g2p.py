import torch
import numpy as np
from agents.base_mv import BaseLearnerMV
from torch.optim import Adam
import copy
from utils.setup_elements import input_size_match
from models.gcppVAE import GCPPVariationalAutoencoderConv
from utils.data import Dataloader_from_numpy, extract_samples_according_to_labels
from utils.utils import EarlyStopping
from torch.optim import lr_scheduler
from utils.optimizer import adjust_learning_rate
from agents.utils import g2p, gcpp


class Generative2Prototype(BaseLearnerMV):
    """ """

    def __init__(self, model, args):
        super(Generative2Prototype, self).__init__(model, args)
        input_size = input_size_match[args.data]
        self.batch_size = args.batch_size

        if self.verbose:
            print(f"Creating the base generator with the shared encoder")

        self.generator = GCPPVariationalAutoencoderConv(
            seq_len=input_size[0],
            feat_dim=input_size[1],
            latent_dim=args.feature_dim,  # 2 for visualization
            hidden_layer_sizes=[64, 128, 256, 512],  # [128, 256]
            device=self.device,
            recon_wt=args.recon_wt,
        )

        self.previous_generator = None
        self.previous_model = None
        self.means_of_exemplars = None

        self.warmup_epochs = 50
        self.max_mem_per_class = 100

    def learn_task(self, task):
        """
        Basic workflow for learning a task. For particular methods, this function will be overwritten.
        """

        (x_train, y_train), (x_val, y_val), _ = task

        self.before_task(y_train)
        train_dataloader = Dataloader_from_numpy(
            x_train, y_train, self.batch_size, shuffle=True
        )
        val_dataloader = Dataloader_from_numpy(
            x_val, y_val, self.batch_size, shuffle=False
        )
        early_stopping = EarlyStopping(
            path=self.ckpt_path, patience=self.args.patience, mode="min", verbose=False
        )
        self.scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            steps_per_epoch=len(train_dataloader),
            epochs=self.epochs,
            max_lr=self.args.lr,
        )
        # Train the learner
        if self.verbose:
            print("Training the learner...")

        for epoch in range(self.epochs):
            # Train for one epoch
            epoch_loss_train, epoch_acc_train = self.train_epoch(
                train_dataloader, epoch=epoch
            )

            # Test on val set for early stop
            epoch_loss_val, epoch_acc_val = self.cross_entropy_epoch_run(
                val_dataloader, mode="val"
            )

            if self.args.lradj != "TST":
                adjust_learning_rate(
                    self.optimizer, self.scheduler, epoch + 1, self.args
                )

            if self.verbose:
                self.epoch_loss_printer(epoch, epoch_acc_train, epoch_loss_train)

            early_stopping(epoch_loss_val, self.model, save=True)
            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break

        # Train the generator
        for id in self.classes_in_task:
            # if self.verbose:
            #     print(f"Creating the disentangled decoder {id}")

            # self.generator.create_decoder(id)

            # ckpt_path_g = self.ckpt_path.replace("/ckpt", f"/generator_ckpt_{id}")
            # epochs_g = self.args.epochs_g

            # if self.verbose:
            #     print(f"Training the generator {id}")

            # encoder_params = list(self.generator.encoder.parameters())
            # decoder_params = list(self.generator.decoders[str(id)].parameters())

            # params = encoder_params + decoder_params

            # optimizer_g = Adam(params, lr=self.args.lr_g, betas=(0.9, 0.999))
            # early_stopping = EarlyStopping(
            #     path=ckpt_path_g,
            #     patience=self.args.patience,
            #     mode="min",
            #     verbose=False,
            # )

            # (x_train_id, y_train_id) = extract_samples_according_to_labels(
            #     x_train, y_train, [id]
            # )
            # (x_val_id, y_val_id) = extract_samples_according_to_labels(
            #     x_val, y_val, [id]
            # )

            # train_dataloader = Dataloader_from_numpy(
            #     x_train_id, y_train_id, self.batch_size, shuffle=True
            # )
            # val_dataloader = Dataloader_from_numpy(
            #     x_val_id, y_val_id, self.batch_size, shuffle=False
            # )

            # for epoch in range(epochs_g):
            #     for batch_id, (x, y) in enumerate(train_dataloader):
            #         x = x.to(self.device)
            #         x_ = None

            #         # generator's input should be (N, C, L)
            #         rnt = 1 / (self.task_now + 1) if self.args.adaptive_weight else 0.5
            #         generator_loss_dict = self.generator.train_a_batch(
            #             x=x.transpose(1, 2),
            #             optimizer=optimizer_g,
            #             decoder_id=id,
            #             x_=x_,
            #             rnt=rnt,
            #         )

            #     train_mse_loss, train_kl_loss = self.generator.evaluate(
            #         train_dataloader, id
            #     )
            #     # Validate on val set for early stop
            #     val_mse_loss, val_kl_loss = self.generator.evaluate(val_dataloader, id)

            #     if self.verbose:
            #         print(
            #             "Epoch {}/{}: Recons Loss = {}, KL Divergence = {}".format(
            #                 epoch + 1,
            #                 epochs_g,
            #                 train_mse_loss,
            #                 train_kl_loss,
            #             )
            #         )

            #     # func save_checkpoint in early_stopping is why it took so long
            #     early_stopping(val_mse_loss, self.generator, save=False)
            #     if early_stopping.early_stop:
            #         if self.verbose:
            #             print("Early stopping")
            #         break

            # # self.after_task for generator
            # if hasattr(self.generator, "decoders") and self.generator.decoders:
            #     self.generator.copy_encoder()
            self.learned_classes += [id]

        self.after_task(x_train, y_train)  # for learner

    def train_epoch(self, dataloader, epoch):
        total = 0
        epoch_loss = 0

        self.model.train()

        x_buff, y_buff = None, None
        if self.args.replay_l == "raw" and self.buffer:
            x_buff, y_buff = gcpp.retrieve_buffer(
                self.buffer, self.batch_size, self.learned_classes
            )

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)  # x is in shape of (N, L, C)
            total += y.size(0)

            if y.size == 1:
                y.unsqueeze()

            self.optimizer.zero_grad()
            loss_ce = 0

            combined_batch, combined_labels = None, None
            if self.task_now > 0:
                if self.args.replay_l == "raw" and x_buff is not None:
                    rp = torch.randperm(x_buff.size(0))
                    x_, y_ = x_buff[rp].to(self.device), y_buff[rp].to(self.device)

                    combined_batch = torch.cat((x, x_))
                    combined_labels = torch.cat((y, y_))

            if combined_batch is not None or combined_labels is not None:
                outputs = g2p.calc_logits(self.model, combined_batch)
                loss_ce += self.criterion(outputs, combined_labels)
            else:
                outputs = g2p.calc_logits(self.model, x)
                loss_ce += self.criterion(outputs, y)

            loss_ce.backward()
            self.optimizer_step(epoch=epoch)

            epoch_loss += loss_ce

        epoch_loss /= batch_id + 1

        return epoch_loss, None

    def after_task(self, x_train, y_train):
        self.model.load_state_dict(torch.load(self.ckpt_path))
        self.previous_model = copy.deepcopy(self.model).eval()

        if self.args.replay_g == "raw" and self.max_mem_per_class != 0:
            self.buffer = gcpp.update_buffer(
                self.buffer,
                self.learned_classes,
                x_train,
                y_train,
                self.max_mem_per_class,
            )

        all_cls = self.learned_classes
        (x_train, y_train), _, _ = g2p.all_data(self.task_stream.tasks)

        X = torch.Tensor(x_train).to(self.device)
        Y = torch.Tensor(y_train).long().to(self.device)

        # all_cls = np.array(torch.unique(Y).to("cpu"))
        all_means = []

        with torch.no_grad():
            all_means = g2p.calc_cls_feature_mean_buffer(self.model, X, Y, all_cls)
            self.means_of_exemplars = all_means

        # if not hasattr(self, "means_of_exemplars") or self.means_of_exemplars is None:
        #     self.means_of_exemplars = all_means
        # else:
        #     self.means_of_exemplars = torch.cat(
        #         [self.means_of_exemplars, all_means], dim=0
        #     )
