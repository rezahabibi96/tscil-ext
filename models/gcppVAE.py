import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from models.utils import Flatten, Reshape
import matplotlib.pyplot as plt
from utils.utils import load_pickle, AverageMeter
from utils.data import Dataloader_from_numpy
import math
from models.utils import *
import time


class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def forward(self, z_mean, z_log_var):
        batch, dim = z_mean.size()
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class UndoPadding1d(nn.Module):
    def __init__(self, padding=(0, 1)):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        out = x[:, :, self.padding[0] : -self.padding[-1]]
        return out


class VaeEncoder(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, hidden_layer_sizes):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.use_padding = []
        self.in_lengths = [seq_len]

        self._get_encoder()

    def _get_encoder(self):
        modules = []
        in_channels = self.feat_dim
        in_len = self.seq_len

        for i, out_channels in enumerate(self.hidden_layer_sizes):
            if in_len % 2 == 1:
                modules.append(nn.ConstantPad1d(padding=(0, 1), value=0))
            modules.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )
            modules.append(nn.ReLU())
            in_len = in_len // 2 if in_len % 2 == 0 else in_len // 2 + 1
            self.in_lengths.append(in_len)
            in_channels = out_channels

        self.encoder_conv = nn.Sequential(*modules)
        self.encoder_fc1 = nn.Linear(
            in_features=in_channels * in_len, out_features=self.latent_dim
        )
        self.encoder_fc2 = nn.Linear(
            in_features=in_channels * in_len, out_features=self.latent_dim
        )

    def forward(self, x):
        """
        x: (N, C, L)
        """
        hx = self.encoder_conv(x)
        hx = Flatten()(hx)
        z_mean = self.encoder_fc1(hx)
        z_log_var = self.encoder_fc2(hx)
        # ==================== DIAGNOSIS ====================
        # comment if not using kd and rd
        z_mean = torch.clamp(z_mean, min=-5.0, max=5.0)
        z_log_var = torch.clamp(z_log_var, min=-5.0, max=5.0)
        # comment if not using kd and rd
        # ==================== DIAGNOSIS ====================
        z = Sampling()(z_mean, z_log_var)
        return z_mean, z_log_var, z

    def _get_fmap(self, x):
        hx = self.encoder_conv(x)
        hx = Flatten()(hx)
        z_mean = self.encoder_fc1(hx)
        z_log_var = self.encoder_fc2(hx)
        # ==================== DIAGNOSIS ====================
        # comment if not using kd and rd
        z_mean = torch.clamp(z_mean, min=-10, max=10)
        z_log_var = torch.clamp(z_log_var, min=-10, max=10)
        # comment if not using kd and rd
        # ==================== DIAGNOSIS ====================
        return z_mean
        # z = Sampling()(z_mean, z_log_var)
        # return z_mean, z_log_var, z


class VaeDecoder(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, hidden_layer_sizes, in_lengths):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.in_lengths = in_lengths
        self._get_decoder()

    def _get_decoder(self):
        self.decoder_input = nn.Linear(
            self.latent_dim, self.hidden_layer_sizes[-1] * self.in_lengths[-1]
        )
        modules = []
        reversed_layers = list(reversed(self.hidden_layer_sizes[:-1]))
        in_channels = self.hidden_layer_sizes[-1]

        for i, out_channels in enumerate(reversed_layers):
            modules.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            modules.append(nn.ReLU())
            if self.in_lengths[-i - 2] % 2 == 1:
                modules.append(UndoPadding1d(padding=(0, 1)))
            in_channels = out_channels

        self.decoder_conv = nn.Sequential(*modules)
        self.decoder_conv_final = nn.ConvTranspose1d(
            in_channels,
            self.feat_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.decoder_fc_final = nn.Linear(
            self.seq_len * self.feat_dim, self.seq_len * self.feat_dim
        )

    def forward(self, z):
        hz = self.decoder_input(z)
        hz = nn.ReLU()(hz)
        hz = Reshape(ts_channels=self.hidden_layer_sizes[-1])(hz)
        hz = self.decoder_conv(hz)
        hz = self.decoder_conv_final(hz)
        if self.seq_len % 2 == 1:
            hz = UndoPadding1d(padding=(0, 1))(hz)
        hz_flat = Flatten()(hz)
        hz_flat = self.decoder_fc_final(hz_flat)
        x_decoded = Reshape(ts_channels=self.feat_dim)(hz_flat)
        return x_decoded


class GCPPVariationalAutoencoderConv(nn.Module):
    def __init__(
        self,
        seq_len,
        feat_dim,
        latent_dim,
        hidden_layer_sizes,
        device,
        fmap,
        kd,
        lambda_kd=0.1,
        recon_wt=3.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.fmap = fmap
        self.kd = kd
        self.lambda_kd = lambda_kd
        self.recon_wt = recon_wt

        self.total_loss_tracker = AverageMeter()
        self.recon_loss_tracker = AverageMeter()
        self.kl_loss_tracker = AverageMeter()
        self.kd_loss_tracker = AverageMeter()
        self.replay_recon_loss_tracker = AverageMeter()
        self.replay_kl_loss_tracker = AverageMeter()
        self.replay_kd_loss_tracker = AverageMeter()

        self.device = device
        self.encoder = VaeEncoder(seq_len, feat_dim, latent_dim, hidden_layer_sizes).to(
            device
        )
        # self.decoder = VaeDecoder(
        #     seq_len, feat_dim, latent_dim, hidden_layer_sizes, self.encoder.in_lengths
        # ).to(device)
        self.decoders = nn.ModuleDict()

    def forward(self, x, decoder_id):
        """
        x: shape of (N, C, L)
        """
        z_mean, z_log_var, z = self.encoder(x)
        # x_decoded = self.decoder(z)
        # decoder = getattr(self, "decoder{}".format(decoder_id))
        x_decoded = self.decoders[str(decoder_id)](z)
        return x_decoded

    def copy_encoder(self):
        self.encoder_teacher = copy.deepcopy(self.encoder)

    def _get_kd_loss(self, x):
        z_teacher = self.encoder_teacher._get_fmap(x)
        z_student = self.encoder._get_fmap(x)

        z_teacher_norm = F.normalize(z_teacher, p=2, dim=1)  # (batch_size, latent_dim)
        z_student_norm = F.normalize(z_student, p=2, dim=1)  # (batch_size, latent_dim)

        kd_loss = 1 - F.cosine_similarity(z_teacher_norm, z_student_norm, dim=1).mean()
        return kd_loss

    def _get_recon_loss(self, x, x_recons):
        def get_reconst_loss_by_axis(x, x_c, dim):
            x_r = torch.mean(x, dim=dim)
            x_c_r = torch.mean(x_c, dim=dim)
            err = torch.square(x_r - x_c_r)
            loss = torch.sum(err)
            return loss

        # overall
        err = torch.square(x - x_recons)
        reconst_loss = torch.sum(err)  # Not mean value, but sum.
        # ToDo: Is adding this loss_by_axis a common practice for training TS VAE?
        reconst_loss += get_reconst_loss_by_axis(x, x_recons, dim=2)  # by time axis
        # reconst_loss += get_reconst_loss_by_axis(x, x_recons, axis=[1])    # by feature axis
        return reconst_loss

    def _get_loss(self, x, decoder_id, x_=None):
        z_mean, z_log_var, z = self.encoder(x)

        # recon = self.decoder(z)
        # decoder = getattr(self, "decoder{}".format(decoder_id))
        recon = self.decoders[str(decoder_id)](z)
        recon_loss = self._get_recon_loss(x, recon)
        kl_loss = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
        kd_loss = torch.tensor(0.0, device=x.device)

        if hasattr(self, "encoder_teacher") and self.encoder_teacher is not None:
            if self.kd:
                if x_ is not None:
                    kd_loss = self._get_kd_loss(x_)
                else:
                    kd_loss = self._get_kd_loss(x)

        # kl_loss = torch.mean(torch.sum(kl_loss, dim=1)) / (self.seq_len * self.feat_dim)
        # total_loss = self.recon_wt * recon_loss + (1-self.recon_wt) * kl_loss

        kl_loss = torch.sum(torch.sum(kl_loss, dim=1))
        total_loss = self.recon_wt * recon_loss + kl_loss + self.lambda_kd * kd_loss

        return total_loss, recon_loss, kl_loss, kd_loss

    def train_a_batch(self, x, optimizer, decoder_id, x_=None, rnt=0.5):
        self.train()
        optimizer.zero_grad()

        # Current data
        total_loss, recon_loss, kl_loss, kd_loss = self._get_loss(x, decoder_id, x_)

        # Original code
        # Replay data
        # if x_ is not None:
        #     total_loss_r, recon_loss_r, kl_loss_r, kd_loss_r = self._get_loss(
        #         x_, decoder_id
        #     )
        #     # total_loss = total_loss + total_loss_r
        #     total_loss = rnt * total_loss + (1 - rnt) * total_loss_r
        # it no longer applies
        # x and x_ may differ in class and may affect the decoder_id

        total_loss.backward()
        optimizer.step()

        self.total_loss_tracker.update(
            total_loss, x.size(0) if x_ is None else x.size(0) + x_.size(0)
        )

        self.recon_loss_tracker.update(recon_loss, x.size(0))
        self.kl_loss_tracker.update(kl_loss, x.size(0))
        self.kd_loss_tracker.update(kd_loss, x.size(0))

        # Replay loss
        # if x_ is not None:
        #     self.replay_recon_loss_tracker.update(recon_loss_r, x_.size(0))
        #     self.replay_kl_loss_tracker.update(kl_loss_r, x_.size(0))
        #     self.replay_kd_loss_tracker.update(kd_loss_r, x_.size(0))

        return {
            "loss": self.total_loss_tracker.avg(),
            "recon_loss": self.recon_loss_tracker.avg(),
            "kl_loss": self.kl_loss_tracker.avg(),
            "kd_loss": self.kd_loss_tracker.avg(),
            "replay_recon_loss": self.replay_recon_loss_tracker.avg(),
            "replay_kl_loss": self.replay_kl_loss_tracker.avg(),
            "replay_kd_loss": self.replay_kd_loss_tracker.avg(),
        }

    def _get_eval_loss(self, x, decoder_id):
        from agents.utils.functions import euclidean_dist

        self.eval()

        z_mean, z_log_var, z = self.encoder(x)
        # recon = self.decoder(z)
        # decoder = getattr(self, "decoder{}".format(decoder_id))
        recon = self.decoders[str(decoder_id)](z)

        mse_loss = torch.nn.MSELoss()(x, recon)
        kl_loss = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
        kl_loss = torch.mean(torch.sum(kl_loss, dim=1)) / (
            self.seq_len * self.feat_dim
        )  # mean over batch, then divide by # of input-pixels

        return mse_loss, kl_loss

    @torch.no_grad()
    def evaluate(self, dataloader, decoder_id):
        """
        Compute the recons and KL div on testing data
        """
        self.eval()

        total = 0
        epoch_mse_loss = 0
        epoch_kl_loss = 0

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            x = x.transpose(1, 2)

            total += y.size(0)
            if y.size == 1:
                y.unsqueeze()

            mse_loss, kl_loss = self._get_eval_loss(x, decoder_id)

            epoch_mse_loss += mse_loss
            epoch_kl_loss += kl_loss

        epoch_mse_loss /= batch_id + 1  # avg loss of a mini batch
        epoch_kl_loss /= batch_id + 1

        return epoch_mse_loss, epoch_kl_loss

    def sample(self, size, decoder_id):
        self.eval()
        z = torch.randn(size, self.latent_dim).to(self.device)
        with torch.no_grad():
            # x = self.decoder(z)
            # decoder = getattr(self, "decoder{}".format(decoder_id))
            x = self.decoders[str(decoder_id)](z)
        return x

    def reset_trackers(self):
        self.total_loss_tracker.reset()
        self.recon_loss_tracker.reset()
        self.kl_loss_tracker.reset()
        self.kd_loss_tracker.reset()
        self.replay_recon_loss_tracker.reset()
        self.replay_kl_loss_tracker.reset()
        self.replay_kd_loss_tracker.reset()

    # COMMENT REMOVED
    # CHECK THE ORIGIN REPO

    # IMPORTANT COMMENT
    # PLEASE NOTE

    # x is in shape of (N, L, C) (from original raw data)
    # x_ is in shape of (N, C, L) (from generative sample data)
    # generator's input should be (N, C, L)
    # model's input should be (N, L, C)

    def create_decoder(self, decoder_id):
        decoder = VaeDecoder(
            self.seq_len,
            self.feat_dim,
            self.latent_dim,
            self.hidden_layer_sizes,
            self.encoder.in_lengths,
        ).to(self.device)
        self.decoders[str(decoder_id)] = decoder

    @torch.no_grad()
    def estimate_prototype(self, size, decoder_id):
        self.eval()
        x = self.sample(size, decoder_id)

        if self.fmap:
            x = self.encoder._get_fmap(x)

        # proto is in shape of (C, L) if not fmap else (latent_dim, )
        prototype = torch.mean(x, dim=0)
        return prototype

    @torch.no_grad()
    def estimate_distance(self, x, p):
        self.eval()
        # x: (batch_size, C, L)
        # p: from (C, L) to (1, C, L) if not fmap else from (latent_dim, ) to (1, latent_dim) <=> 1 is #prototypes
        p = p.unsqueeze(0)  # equivalent to p = torch.stack([p], dim=0)

        if self.fmap:
            x = self.encoder._get_fmap(x)  # (batch_size, latent_dim)
        else:
            x = x.reshape(x.size(0), -1)  # (batch_size, C*L)
            p = p.reshape(p.size(0), -1)  # (#prototypes, C*L)

        x = F.normalize(x, p=2, dim=1)
        p = F.normalize(p, p=2, dim=1)

        similarity = torch.matmul(x, p.T)
        dist = 1 - similarity
        return dist
