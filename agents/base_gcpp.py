import os
import abc
import torch
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abc import abstractmethod
from sklearn.manifold import TSNE
from utils.metrics import plot_confusion_matrix
from utils.data import Dataloader_from_numpy, extract_samples_according_to_labels


class BaseLearnerGCPP(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, args: argparse.Namespace):
        super(BaseLearnerGCPP, self).__init__()
        self.scheduler = None
        self.task_stream = args.task_stream

        self.args = args
        self.run_id = args.run_id  # index of 'run', for saving ckpt
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.device = args.device
        self.scenario = args.scenario
        self.verbose = args.verbose
        self.tsne = args.tsne
        self.cf_matrix = args.cf_matrix

        self.buffer = dict()
        self.er_mode = args.er_mode

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

        if self.cf_matrix:
            self.y_pred_cf, self.y_true_cf = (
                [],
                [],
            )  # Collected results for Confusion matrix

    def before_task(self, y_train):
        self.task_now += 1
        self.classes_in_task = list(
            set(y_train.tolist())
        )  # labels in order, not original randomized-order labels

        n_new_classes = len(self.classes_in_task)
        assert n_new_classes > 1, "A task must contain more than 1 class"

        if self.verbose:
            print(
                "\n--> Task {}: {} classes in total".format(
                    self.task_now, len(self.learned_classes + self.classes_in_task)
                )
            )
            print("\n--> Class: {}".format(self.classes_in_task))

    def learn_task(self, task):
        (x_train, y_train), (x_val, y_val), _ = task
        self.before_task(y_train)

    # def train_epoch(self, dataloader, epoch):
    #     pass

    # def cross_entropy_epoch_run(self, dataloader, epoch=None, mode="train"):
    #     pass

    # def optimizer_step(self, epoch):
    #     pass

    # def epoch_loss_printer(self, epoch, acc, loss):
    #     pass

    def after_task(self, x_train, y_train):
        print(f"Classes learned just now: {self.classes_in_task}")
        print(f"Classes learned so far: {self.learned_classes}")

    @abstractmethod
    def evaluate(self, task_stream, path=None):
        raise NotImplementedError

    @torch.no_grad()
    def test_for_cf_matrix(self, dataloader, y_pred, y_true):
        self.y_pred_cf.extend(y_pred)
        self.y_true_cf.extend(y_true)

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
