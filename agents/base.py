# -*- coding: UTF-8 -*-
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import abc
import os
from abc import abstractmethod
from utils.data import Dataloader_from_numpy
from utils.metrics import plot_confusion_matrix
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.optimizer import set_optimizer, adjust_learning_rate
from utils.utils import EarlyStopping, BinaryCrossEntropy
from torch.optim import lr_scheduler
import copy
from agents.utils.functions import compute_cls_feature_mean_buffer
from utils.data import Dataloader_from_numpy, extract_samples_according_to_labels


class BaseLearner(nn.Module, metaclass=abc.ABCMeta):
    """
    This is the base class for the agent. The workflow is as follows:
    agent.learn_task(data=[train, val]):
        agent.before_task():
            each epoch:
                agent.train_epoch(data=train, mode='train')
                agent.cross_entropy_epoch_run(data=val, mode='val')
        agent.after_task()

    agent.evaluate(data=[val, test])
        if task is final:
            agent.cross_entropy_epoch_run(data=val, mode='test')
            agent.test_for_cf_matrix(data=test, mode='test')
        else:
            agent.cross_entropy_epoch_run(data=val, mode='test')
            agent.cross_entropy_epoch_run(data=test, mode='test')
    Notes:
    - train_epoch(data=train, mode='train') never calls cross_entropy_epoch_run(data=train, mode='train').
        It uses self.criterion, which is defined in before_task or in the agent-specific implementation.

    - cross_entropy_epoch_run(data=val, mode='val') is used only for early stopping, since only its loss is considered (not accuracy).
        In this mode, ncm_classifier is not checked. Even though this is intentional (accuracy is not needed)
        This behavior is inconsistent because it always assumes the loss is applied to the final head.
        As a result, this design does not support methods that do not rely on CE/BCE loss on the final head.
        In other words, this design does not support methods that do not rely on model logits on the final head.

    - cross_entropy_epoch_run(data=val, mode='test') is deliberately run with mode='test' (even though the data is validation data).
        This ensures that ncm_classifier is checked, since accuracy is required for checking generalization.

    - test_for_cf_matrix(data=test, mode='test') does not check ncm_classifier, even though cross_entropy_epoch_run does in test mode.
        This inconsistency may be invalid for methods that solely use ncm_classifier

    - Another inconsistency appears in evaluate: for tasks 0 to final-1, evaluation uses cross_entropy_epoch_run(test mode),
        which does check ncm_classifier. However, for the final task, evaluation switches to test_for_cf_matrix,
        which does not check ncm_classifier.

    - Another inconsistency appears in test_for_cf_matrix: when computing predictions, `prediction` matches cross_entropy_epoch_run
        and is used for accuracy calculation, but `predictions` is computed differently, used only for the confusion matrix,
        and does not occur in cross_entropy_epoch_run.
        Although self.criterion (CE/BCE) is used in cross_entropy_epoch_run,
        torch.nn.CrossEntropyLoss() is always used in test_for_cf_matrix.

    In summary, this codebase:
    - Assumes methods use CE/BCE loss,
    - Assumes training relies on the final head,
    - Does not natively support methods that train without final head,
    - Does not natively support methods that train without CE/BCE loss.

    For agents that differ from this approach, do not forget to modify:
    - before_task: to set self.criterion,
    - cross_entropy_epoch_run: to define how self.criterion is applied,
    - evaluate: to specify how evaluation is performed,
    - test_for_cf_matrix: to define behavior on the final task (which may be unnecessary).

    Comments added:
    - agents/base.py on behavioral inconsistencies
    - agents/gcppv1.py on shape-awareness of learner, generator, x, x_
    - agents/gcppv2.py on shape-awareness of learner, generator, x, x_
    - utils/utils.py on loss func choice consideration
    """

    def __init__(self, model: nn.Module, args: argparse.Namespace):
        super(BaseLearner, self).__init__()
        self.model = model
        self.optimizer = set_optimizer(self.model, args)
        self.scheduler = None

        self.args = args
        self.run_id = args.run_id  # index of 'run', for saving ckpt
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.device = args.device
        self.scenario = args.scenario
        self.verbose = args.verbose
        self.tsne = args.tsne
        self.cf_matrix = args.cf_matrix

        self.buffer = None
        self.er_mode = args.er_mode
        self.teacher = None
        self.use_kd = False

        # Only applicable for replay-based methods
        self.ncm_classifier = False

        if not self.args.tune:
            self.ckpt_path = args.exp_path + "/ckpt_r{}.pt".format(self.run_id)
        else:
            # To avoid conflicts between multiple running trials
            self.ckpt_path = args.exp_path + "/ckpt_{}_r{}.pt".format(
                os.getpid(), self.run_id
            )

        self.task_now = -1  # ID of the current task

        # ToDO: Consider the case that class order can change!
        self.learned_classes = (
            []
        )  # Joint ohv (one-hot vector) labels for all the seen classes
        self.classes_in_task = (
            []
        )  # Joint ohv (one-hot vector) labels for classes in the current task

        if not self.args.early_stop:
            self.args.patience = self.epochs  # Set Early stop patience as # epochs

        if self.cf_matrix:
            self.y_pred_cf, self.y_true_cf = (
                [],
                [],
            )  # Collected results for Confusion matrix

    def before_task(self, y_train):
        """
        Preparations before training a task, called in 'learn_task'.
        # Note that we assume there is no overlapping among classes across tasks.

        - update Task ID: self.task_now
        - Check the data in the new task, update the label set of learned classes.
        - Expand the model's head & update the optimizer to include the new parameters

        Args
            y_train: np array of training labels of the current task

        """
        self.task_now += 1
        self.classes_in_task = list(
            set(y_train.tolist())
        )  # labels in order, not original randomized-order labels
        # it is in oder due to stream.py (# map the class labels to the ordered ones)

        n_new_classes = len(self.classes_in_task)
        assert n_new_classes > 1, "A task must contain more than 1 class"

        # ############## Single-head Model #############
        # Adapt the model and optimizer for the new task
        if self.task_now != 0:
            # self.model.increase_neurons(n_new=n_new_classes)
            self.model.update_head(n_new=n_new_classes, task_now=self.task_now)
            self.model.to(self.device)
            self.optimizer = set_optimizer(
                self.model, self.args, task_now=self.task_now
            )

        # Initialize the main criterion for classification
        if self.args.criterion == "BCE":
            self.criterion = BinaryCrossEntropy(
                dim=self.model.head.out_features, device=self.device
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        if self.verbose:
            print(
                "\n--> Task {}: {} classes in total".format(
                    self.task_now, len(self.learned_classes + self.classes_in_task)
                )
            )
            print("\n--> Class: {}".format(self.classes_in_task))

    # this with train_epoch are the tightly coupled with agent-specific
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

        self.after_task(x_train, y_train)

    # this with train_epoch are the tightly coupled with agent-specific
    @abstractmethod
    def train_epoch(self, dataloader, epoch):
        """
        Train the agent for 1 epoch.
        Return:
            - Average Accuracy of the epoch
            - Average Loss(es) of the epoch
        """
        raise NotImplementedError

    def cross_entropy_epoch_run(self, dataloader, epoch=None, mode="train"):
        """
        Train / eval with cross entropy.

        Args:
            dataloader: dataloader for train/val/test
            epoch: used for lr_adj
            train: set True for training, False for eval

        Returns:
            epoch_loss: average cross entropy loss on this epoch
            epoch_acc: average accuracy on this epoch
        """
        total = 0
        correct = 0
        epoch_loss = 0

        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)

            if y.size == 1:
                y.unsqueeze()

            if mode == "train":
                self.optimizer.zero_grad()
                outputs = self.model(x)
                step_loss = self.criterion(outputs, y)
                step_loss.backward()
                self.optimizer_step(epoch)

            elif mode in ["val", "test"]:
                with torch.no_grad():
                    outputs = self.model(x)
                    step_loss = self.criterion(outputs, y)

                    if mode == "test" and self.ncm_classifier:
                        features = self.model.feature(x)
                        distance = torch.cdist(
                            F.normalize(features, p=2, dim=1),
                            F.normalize(self.means_of_exemplars, p=2, dim=1),
                        )
                        outputs = -distance  # select class with min distance

            epoch_loss += step_loss

            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100.0 * (correct / total)
        epoch_loss /= batch_id + 1  # avg loss of a mini batch

        return epoch_loss, epoch_acc

    def optimizer_step(self, epoch):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        self.optimizer.step()

        # if self.args.norm == "BIN":
        #     bin_gates = [
        #         p for p in self.model.parameters() if getattr(p, "bin_gate", False)
        #     ]
        #     for p in bin_gates:
        #         p.data.clamp_(min=0, max=1)

        if self.args.lradj == "TST":
            adjust_learning_rate(
                self.optimizer, self.scheduler, epoch + 1, self.args, printout=False
            )
            self.scheduler.step()

    def epoch_loss_printer(self, epoch, acc, loss):
        print(
            "Epoch {}/{}: Accuracy = {}, Loss = {}".format(
                epoch + 1, self.epochs, acc, loss
            )
        )

    def after_task(self, x_train, y_train):
        self.learned_classes += self.classes_in_task
        self.model.load_state_dict(torch.load(self.ckpt_path))  # eval()

        if (
            self.buffer and self.er_mode == "task"
        ):  # Additional pass to collect memory samples
            dataloader = Dataloader_from_numpy(
                x_train, y_train, self.batch_size, shuffle=True
            )
            for batch_id, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.buffer.update(x, y)

        # Compute means of classes if using ncm_classifier
        if self.ncm_classifier:
            # agents use ncm_classifier are icarl and only icarl
            self.means_of_exemplars = compute_cls_feature_mean_buffer(
                self.buffer, self.model
            )

        # Save the teacher model if using kd
        if self.use_kd:
            # agents use kd are lwf and dt2w
            self.teacher = copy.deepcopy(self.model)  # eval() mode
            if not self.args.teacher_eval:
                self.teacher.train()  # train() mode

    @torch.no_grad()
    def evaluate(self, task_stream, path=None):
        """
        Evaluate on the test sets of all the learned tasks (<= task_now).
        Save the test accuracies of the learned tasks in the Acc matrix.
        Visualize the feature space with TSNE, if self.tsne == True.

        Args:
            task_stream: Object of Task Stream, list of ((x_train, y_train), (x_val, y_val), (x_test, y_test)).
            path: path prefix to save the TSNE png files.

        """
        # Get num_tasks and create Accuracy Matrix for 'val set and 'test set'
        if self.task_now == 0:
            self.num_tasks = task_stream.n_tasks
            self.Acc_tasks = {
                "valid": np.zeros((self.num_tasks, self.num_tasks)),
                "test": np.zeros((self.num_tasks, self.num_tasks)),
            }

        # Reload the original optimal model to prevent the changes of statistics in BN layers.
        self.model.load_state_dict(torch.load(self.ckpt_path))

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

                # Use eval data to evaluate learner
                eval_dataloader_i = Dataloader_from_numpy(
                    x_eval, y_eval, self.batch_size, shuffle=False
                )

                if (
                    self.cf_matrix
                    and self.task_now + 1 == self.num_tasks
                    and mode == "test"
                ):  # Collect results for CM
                    eval_loss_i, eval_acc_i = self.test_for_cf_matrix(eval_dataloader_i)
                else:
                    eval_loss_i, eval_acc_i = self.cross_entropy_epoch_run(
                        eval_dataloader_i, mode="test"
                    )

                if self.verbose:
                    print(
                        "Task {}: Accuracy == {}, Test CE Loss == {} ;".format(
                            i, eval_acc_i, eval_loss_i
                        )
                    )

                self.Acc_tasks[mode][self.task_now][i] = np.around(
                    eval_acc_i, decimals=2
                )

                # Use eval data to evaluate generator
                if self.args.agent == "GR" and self.verbose:
                    eval_mse_loss, eval_kl_loss = self.generator.evaluate(
                        eval_dataloader_i
                    )
                    print(
                        "        Recons Loss (MAE) == {}, KL Div == {} ;".format(
                            eval_mse_loss, eval_kl_loss
                        )
                    )

            """
            In base.py extended agents: 
                during training, the learner is checked for its loss and accuracy (train and val), 
                    while the generator (acts as a generator) is checked for its loss only (train and val). 
                during evaluation, the learner is checked for its loss and accuracy (val and test), 
                    while the generator (acts as a generator) is checked for its loss only (val and test).

            In base_gcpp.py extended agents: 
                during training, there is no actual learner, 
                    while the generator (acts as a generator) is checked for its loss only (train and val). 
                during evaluation, there is no actual learner, 
                    while the generator (acts as a pseudo-learner) is checked for its accuracy only (val and test)
                    and cheking for its loss, although possible, is complex due to the disentangled decoder.

            In base_mv.py extended agents: 
                during training, the learner is checked for its loss only (train and val), 
                    while the generator (acts as a generator) is checked for its loss only (train and val). 
                during evaluation, the learner is checked for its accuracy and loss (val and test), 
                    while the generator (acts as a generator) is not checked at all due to the disentangled decoder
                    and cheking for its loss, although possible, is complex due to the disentangled decoder.
            """

            # Print accuracy matrix of the tasks on this run
            if self.task_now + 1 == self.num_tasks and self.verbose:
                with np.printoptions(suppress=True):  # Avoid Scientific Notation
                    print("Accuracy matrix of all tasks:")
                    print(self.Acc_tasks[mode])

        # TODO: ZZ: TSNE visualization for all learned classes.
        if self.tsne and not self.args.tune:
            tsne_path = path + "t{}".format(self.task_now)
            self.feature_space_tsne_visualization(task_stream, path=tsne_path)

        # TODO: ZZ: TSNE visualization ??
        if self.args.tsne_g and self.args.agent == "GR" and not self.args.tune:
            tsne_path = path + "t{}_g".format(self.task_now)
            self.feature_space_tsne_visualization(
                task_stream, path=tsne_path, view_generator=True
            )

    @torch.no_grad()
    def test_for_cf_matrix(self, dataloader):
        """
        Test for one epoch before getting the confusion matrix.
        Run this after learning the final task.

        Args:
            dataloader: dataloader for train/test

        Returns:
            epoch_loss: average cross entropy loss on this epoch
            epoch_acc: average accuracy on this epoch
        """
        total = 0
        correct = 0
        epoch_loss = 0
        ce_loss = torch.nn.CrossEntropyLoss()

        self.model.eval()
        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)

            if y.size == 1:
                y.unsqueeze()

            with torch.no_grad():
                outputs = self.model(x)
                step_loss = ce_loss(outputs, y)

                # using torch.exp assumes outputs are log-probabilities, however, the model outputs are raw logits
                # torch.argmax(outputs, dim=1) is sufficient for class prediction
                predictions = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
                labels = y.data.cpu().numpy()

                self.y_pred_cf.extend(predictions)  # Save Prediction
                self.y_true_cf.extend(labels)  # Save Truth

            epoch_loss += step_loss

            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100.0 * (correct / total)
        epoch_loss /= batch_id + 1  # avg loss of a mini batch

        return epoch_loss, epoch_acc

    def plot_cf_matrix(self, path, classes):
        plot_confusion_matrix(self.y_true_cf, self.y_pred_cf, classes, path)

    @torch.no_grad()
    def feature_space_tsne_visualization(
        self, task_stream, path, view_generator=False, shared_encoder=False
    ):
        """featured in evaluate func"""
        z = None
        x_all, y_all = None, None
        for i in range(self.task_now + 1):
            _, _, (x_i, y_i) = task_stream.tasks[i]

            if x_all is None:
                x_all, y_all = x_i, y_i
            else:
                x_all = np.concatenate((x_all, x_i))
                y_all = np.concatenate((y_all, y_i))

            if view_generator and not shared_encoder:
                for id in list(set(y_i.tolist())):
                    (x_id, y_id) = extract_samples_according_to_labels(x_i, y_i, [id])
                    generator = getattr(self, "generator{}".format(id))
                    x_id = torch.Tensor(x_id).to(self.device)
                    _, _, z_id = generator.encoder(x_id.transpose(1, 2))
                    z = torch.cat((z, z_id), dim=0) if z is not None else z_id

        # Save the nparrays of features
        x_all = torch.Tensor(x_all).to(self.device)

        if view_generator:
            if shared_encoder:
                z_mean, z_log_var, z = self.generator.encoder(x_all.transpose(1, 2))
            features = z.cpu().detach().numpy()
        else:
            features = self.model.feature(x_all).cpu().detach().numpy()

        np.save(path + "f", features)
        np.save(path + "y", y_all)

        tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=50)
        tsne_result = tsne.fit_transform(features, y_all)

        df = pd.DataFrame(tsne_result, columns=["d1", "d2"])
        df["class"] = y_all

        plt.figure(figsize=(6, 6), dpi=128)
        color_palette = {
            0: "tab:red",
            1: "tab:blue",
            2: "tab:green",
            3: "tab:cyan",
            4: "tab:pink",
            5: "tab:gray",
            6: "tab:orange",
            7: "tab:brown",
            8: "tab:olive",
            9: "tab:purple",
            10: "darkseagreen",
            11: "black",
        }

        g1 = sns.scatterplot(
            x="d1",
            y="d2",
            hue="class",
            # palette=sns.color_palette("hls", self.n_cur_classes),
            palette=color_palette,
            s=10,
            data=df,
            legend="full",
            alpha=1,
        )
        g1.set(xticklabels=[])  # remove the tick labels
        g1.set(xlabel=None)  # remove the axis label
        g1.set(yticklabels=[])  # remove the tick labels
        g1.set(ylabel=None)  # remove the axis label
        g1.tick_params(bottom=False, left=False)  # remove the ticks

        plt.savefig(path, bbox_inches="tight")
        plt.show()


class SequentialFineTune(BaseLearner):
    def __init__(self, model, args):
        super(SequentialFineTune, self).__init__(model, args)

    def train_epoch(self, dataloader, epoch):
        (
            epoch_acc_train,
            epoch_loss_train,
        ) = self.cross_entropy_epoch_run(
            dataloader=dataloader, epoch=epoch, mode="train"
        )  # only here uses cross_entropy_epoch_run in train mode
        return epoch_loss_train, epoch_acc_train
