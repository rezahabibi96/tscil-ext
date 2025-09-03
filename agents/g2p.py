import torch
from agents.base import BaseLearner
from torch.optim import Adam
import copy
from utils.setup_elements import input_size_match
from models.gcppVAE import GCPPVariationalAutoencoderConv
from utils.data import Dataloader_from_numpy, extract_samples_according_to_labels
from utils.utils import EarlyStopping
from torch.optim import lr_scheduler
from utils.optimizer import adjust_learning_rate
from agents.utils import g2p


class Generative2Prototype(BaseLearner):
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

        self.transform = g2p.TSAugmentPipeline()

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

            early_stopping(epoch_loss_val, self.model)
            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break

        # Train the generator
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

                    # generator's input should be (N, C, L)
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

                # func save_checkpoint in early_stopping is why it took so long
                early_stopping(val_mse_loss, self.generator)
                if early_stopping.early_stop:
                    if self.verbose:
                        print("Early stopping")
                    break

            # self.after_task for generator
            self.learned_classes += [id]
            self.generator.copy_encoder()

        self.after_task(x_train, y_train)  # for learner

    def train_epoch(self, dataloader, epoch):
        total = 0
        correct = 0
        epoch_loss = 0
        self.model.train()

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)  # x is in shape of (N, L, C)
            total += y.size(0)

            if y.size == 1:
                y.unsqueeze()

            self.optimizer.zero_grad()
            loss_ce = 0

            if self.task_now > 0:  # Generative Replay after 1st task
                # TODO : sample (learned class) data from generator
                pass

                with torch.no_grad():
                    # TODO : old model loss on sample data
                    pass

                # Train the classifier model on this batch
                # TODO : new model loss on sample data

            x_aug = self.transform(x.transpose(1, 2))

            x_feat = self.model.feature(x.transpose(1, 2)).unsqueeze(1)
            x_aug_feat = self.model.feature(x_aug.transpose(1, 2)).unsqueeze(1)

            features = torch.cat([x_feat, x_aug_feat], dim=1)

            outputs = features
            loss_ce += self.criterion(outputs, y)
            loss_ce.backward()
            self.optimizer_step(epoch=epoch)

            epoch_loss += loss_ce
            prediction = torch.argmax(outputs, dim=1)  # TODO :
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100.0 * (correct / total)
        epoch_loss /= batch_id + 1

        return epoch_loss, epoch_acc

    def after_task(self, x_train, y_train):
        self.model.load_state_dict(torch.load(self.ckpt_path))
        self.previous_model = copy.deepcopy(self.model).eval()
