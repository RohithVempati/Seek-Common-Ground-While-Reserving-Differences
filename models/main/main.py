import pickle

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import os
from train_utils import AverageMeter

from .main_utils import Get_Scalar
from train_utils import ce_loss, wd_loss, EMA, Bn_Controller, MultiClassFocalLossWithAlpha
import json

from sklearn.metrics import *
from copy import deepcopy
from train_utils import ce_loss
import contextlib
from models.nets import fusion_model, dmd
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

def fgmr_loss(visual_private_feat, text_private_feat, tau=0.1):
    # tau is retained in the requested API; cosine alignment has no temperature term.
    batch_size = visual_private_feat.shape[0]
    visual_private_feat = visual_private_feat.reshape(batch_size, -1)
    text_private_feat = text_private_feat.reshape(batch_size, -1).detach()
    visual_private_feat = F.normalize(visual_private_feat, p=2, dim=1)
    text_private_feat = F.normalize(text_private_feat, p=2, dim=1)
    similarity = (visual_private_feat * text_private_feat).sum(dim=1)
    return (1.0 - similarity).mean()


def rac_calibrated_pseudo_label(logits_visual, logits_text, logits_fusion, temperature):
    if temperature <= 0:
        raise ValueError('RAC temperature must be greater than zero.')
    probabilities = torch.stack((
        torch.softmax(logits_visual / temperature, dim=1),
        torch.softmax(logits_text / temperature, dim=1),
        torch.softmax(logits_fusion / temperature, dim=1),
    ), dim=1)
    view_confidences = probabilities.max(dim=2).values
    weights = torch.softmax(view_confidences / temperature, dim=1)
    pseudo_probabilities = (probabilities * weights.unsqueeze(-1)).sum(dim=1)
    conf, pseudo_label = pseudo_probabilities.max(dim=1)
    return pseudo_probabilities, pseudo_label, conf, weights


def masked_mean(losses, mask):
    weights = mask.to(dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


class DVADScheduler:
    def __init__(self, total_steps, warmup_steps=512, final_margin=0.7,
                 mode='margin', percentile_start=20.0, percentile_end=80.0):
        self.total_steps = total_steps
        self.start_step = warmup_steps
        self.end_step = int(0.9 * total_steps)
        if self.start_step < 0 or self.start_step >= self.end_step:
            raise ValueError('warmup_steps must be between 0 and the DVAD end step.')
        self.initial_margin = 0.5
        self.final_margin = final_margin
        self.mode = mode
        self.percentile_start = percentile_start
        self.percentile_end = percentile_end

    def margin(self, step):
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        progress = min(max(progress, 0.0), 1.0)
        return self.initial_margin + progress * (self.final_margin - self.initial_margin)

    def percentile(self, step):
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        progress = min(max(progress, 0.0), 1.0)
        percentile = self.percentile_start + progress * (
            self.percentile_end - self.percentile_start)
        return min(max(percentile, 0.0), 100.0)


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def compute_cosine(self, x, y):
        # x = self.compute_compact_s(x)
        # y = self.compute_compact_s(y)
        x_norm = torch.sqrt(torch.sum(torch.pow(x, 2), 1)+1e-8)
        x_norm = torch.max(x_norm, 1e-8*torch.ones_like(x_norm))
        y_norm = torch.sqrt(torch.sum(torch.pow(y, 2), 1)+1e-8)
        y_norm = torch.max(y_norm, 1e-8*torch.ones_like(y_norm))
        cosine = torch.sum(x * y, 1) / (x_norm * y_norm)
        return cosine

    def forward(self, ids, feats, margin=0.1):
        B, F = feats.shape

        s = feats.repeat(1, B).view(-1, F) # B**2 X F
        s_ids = ids.view(B, 1).repeat(1, B) # B X B
        
        t = feats.repeat(B, 1) # B**2 X F
        t_ids = ids.view(1, B).repeat(B, 1) # B X B 

        cosine = self.compute_cosine(s, t) # B**2
        equal_mask = torch.eye(B, dtype=torch.bool) # B X B
        s_ids = s_ids[~equal_mask].view(B, B-1) # B X (B-1)
        t_ids = t_ids[~equal_mask].view(B, B-1) # B X (B-1)
        cosine = cosine.view(B, B)[~equal_mask].view(B, B-1) # B X (B-1)

        sim_mask = (s_ids == t_ids) # B X (B-1)
        margin = 0.15 * abs(s_ids - t_ids)#[~sim_mask].view(B, B - 3)

        loss = 0
        loss_num = 0
        
        for i in range(B):
            sim_num = sum(sim_mask[i])
            dif_num = B - 1 - sim_num
            if not sim_num or not dif_num:
                continue
            sim_cos = cosine[i, sim_mask[i]].reshape(-1, 1).repeat(1, dif_num)
            dif_cos = cosine[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)
            t_margin = margin[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)

            loss_i = torch.max(torch.zeros_like(sim_cos), t_margin - sim_cos + dif_cos).mean()
            loss += loss_i
            loss_num += 1

        if loss_num == 0:
            loss_num = 1

        loss = loss / loss_num
        return loss

class S2_VER:
    # def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, \
    #              hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, \
                 hard_label=True, t_fn=None, p_fn=None, it=0, tb_log=None, args=None, logger=None):

        super(S2_VER, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        # self.model = net_builder(num_classes=num_classes)
        # self.fusion_model = fusion_model.FusionModel(num_classes=num_classes)
        self.model = dmd.DMD(args)
        self.ema_model = None

        # self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.lst = [[] for i in range(10)]
        self.abs_lst = [[] for i in range(10)]
        self.clsacc = [[] for i in range(10)]
        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        self._pla_batch = None

        self.bn_controller = Bn_Controller()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.MSE = MSE()
        self.sim_loss = HingeLoss()
        self.cosine = nn.CosineEmbeddingLoss()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    @torch.no_grad()
    def evaluate_pla(self, args, step, max_samples=256):
        if self._pla_batch is None:
            cached_images = []
            cached_texts = []
            cached_labels = []
            sample_count = 0
            for _, images, texts, labels in self.loader_dict['eval']:
                take = min(max_samples - sample_count, labels.shape[0])
                cached_images.append(images[:take])
                cached_texts.extend(texts[:take])
                cached_labels.append(labels[:take])
                sample_count += take
                if sample_count >= max_samples:
                    break
            if sample_count == 0:
                raise ValueError('The evaluation loader has no samples for PLA.')
            self._pla_batch = (
                torch.cat(cached_images, dim=0),
                cached_texts,
                torch.cat(cached_labels, dim=0),
            )
        images, texts, labels = self._pla_batch
        images = images.cuda(args.gpu)
        labels = labels.cuda(args.gpu)
        text_inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        text_inputs = {key: value.cuda(args.gpu) for key, value in text_inputs.items()}

        self.model.eval()
        self.ema.apply_shadow()
        try:
            output = self.model(images, text_inputs)
            if args.use_rac:
                _, predictions, _, _ = rac_calibrated_pseudo_label(
                    output['logits_visual_private'],
                    output['logits_text_private'],
                    output['logits_fusion'],
                    args.rac_temperature,
                )
            else:
                predictions = output['logits_fusion'].argmax(dim=1)
            accuracy = (predictions == labels).float().mean().item()
            visual_logits = output['logits_visual_private']
            text_logits = output['logits_text_private']
            fusion_logits = output['logits_fusion']
            view_accuracies = [
                (logits.argmax(dim=1) == labels).float().mean().item()
                for logits in (visual_logits, text_logits, fusion_logits)
            ]
            view_confidences = [
                torch.softmax(logits, dim=1).max(dim=1).values.mean().item()
                for logits in (visual_logits, text_logits, fusion_logits)
            ]
            self.print_fn(
                "Step {}: VIEWS acc(v/t/f)={:.3f}/{:.3f}/{:.3f} "
                "conf(v/t/f)={:.3f}/{:.3f}/{:.3f}".format(
                    step, *view_accuracies, *view_confidences))
        finally:
            self.ema.restore()
            self.model.train()
        return accuracy

    @torch.no_grad()
    def _diagnostic_forward(self, images, text_inputs):
        was_training = self.model.training
        self.model.eval()
        try:
            return self.model(images, text_inputs)
        finally:
            self.model.train(was_training)

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args, epoch, best_eval_acc, logger=None):

        ngpus_per_node = torch.cuda.device_count()
        if args.eval_interval <= 0:
            raise ValueError('eval_interval must be greater than zero.')
        TOTAL_STEPS = args.epoch * args.num_train_iter
        dvad_scheduler = DVADScheduler(
            TOTAL_STEPS, args.warmup_steps, args.dvad_final_margin,
            args.dvad_mode, args.dvad_percentile_start, args.dvad_percentile_end)

        # EMA Init
        self.model.train()
        # TODO
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        
        sup_losses = AverageMeter()
        unsup_losses = AverageMeter()
        decouple_losses = AverageMeter()
        # contrast_losses = AverageMeter()
        total_losses = AverageMeter()
        mask_ratios = AverageMeter()
        # distribution_losses = AverageMeter()
        lr_last = 0
        batch_data_time = AverageMeter()
        batch_model_time = AverageMeter()

        pseudo_true_ratios = AverageMeter()
        rac_conf_values = []

        start_batch.record()

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)
        # focal_loss = MultiClassFocalLossWithAlpha()
        iter_num = 0

        for (_, x_lb, t_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s0, x_ulb_s1, t_ulb, y_ulb) in tqdm(zip(self.loader_dict['train_lb'],
                                                                    self.loader_dict['train_ulb']), total=len(self.loader_dict['train_ulb'])):
            # break
            step = self.it
            display_step = step + 1
            iter_num += 1
            end_batch.record()
            torch.cuda.synchronize()
            batch_data_time.update(start_batch.elapsed_time(end_batch) / 1000)
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s0.shape[0] and num_ulb == x_ulb_s1.shape[0]
            try:
                x_lb.cuda(args.gpu)
            except Exception as e:
                print("An error occurred:", e)           
            x_lb, x_ulb_w, x_ulb_s0, x_ulb_s1 = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s0.cuda(args.gpu), x_ulb_s1.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)

            img_inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s0, x_ulb_s1))
            t_lb = list(t_lb)
            t_ulb = list(t_ulb)
            # TODO text的输入格式
            text_input = t_lb + t_ulb + t_ulb + t_ulb
            text_input = self.tokenizer(text_input, return_tensors='pt', padding=True, truncation=True)
            text_input = {key: value.cuda(args.gpu) for key, value in text_input.items()}

            # hyper-params for update
            # T = self.t_fn(self.it)
            # p_cutoff = self.p_fn(self.it)

            # inference and calculate sup/unsup losses
            with amp_cm():
                # logits, features = self.model(img_inputs, text_input)
                # res = self.fusion_model(img_inputs, text_input)
                output = self.model(img_inputs, text_input)
                # logits_m = output['pre_m']
                logits_m = output['logits_fusion']
                logits_t = output['logits_text_private']
                logits_v = output['logits_visual_private']

                if display_step == 1:
                    diag_text_inputs = {
                        key: value[:num_lb] for key, value in text_input.items()}
                    diagnostic_output = self._diagnostic_forward(x_lb, diag_text_inputs)
                    vis_counts = diagnostic_output['logits_visual_private'].argmax(
                        dim=1).bincount(minlength=3).tolist()
                    text_counts = diagnostic_output['logits_text_private'].argmax(
                        dim=1).bincount(minlength=3).tolist()
                    fusion_counts = diagnostic_output['logits_fusion'].argmax(
                        dim=1).bincount(minlength=3).tolist()
                    label_counts = y_lb.bincount(minlength=3).tolist()
                    self.print_fn(
                        "DIAG step=1: vis_pred_counts={}, text_pred_counts={}, "
                        "fusion_pred_counts={}, label_counts={}".format(
                            vis_counts, text_counts, fusion_counts, label_counts))
                    self.print_fn("FGMR anchor = {}, weight = {}".format(
                        args.fgmr_anchor, args.fgmr_weight))

                features_m = output['c_l'] + output['c_v']
                features_t = output['c_l']
                features_v = output['c_v']

                logits_x_lb = logits_m[:num_lb]
                logits_t_lb = logits_t[:num_lb] 
                logits_v_lb = logits_v[:num_lb] 
                # logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                logits_x_ulb_w, logits_x_ulb_s0, logits_x_ulb_s1 = torch.split(logits_m[num_lb:], num_ulb)
                logits_t_ulb_w, logits_t_ulb_s0, logits_t_ulb_s1 = torch.split(logits_t[num_lb:], num_ulb)
                logits_v_ulb_w, logits_v_ulb_s0, logits_v_ulb_s1 = torch.split(logits_v[num_lb:], num_ulb)

                features_lb = features_m[:num_lb]
                features_ulb_w, features_ulb_s0, features_ulb_s1 = torch.split(features_m[num_lb:], num_ulb)
                # features_t_ulb_w, features_c_ulb_s0, features_c_ulb_s1 = torch.split(features_t[num_lb:], num_ulb)
                # features_v_ulb_w, features_v_ulb_s0, features_v_ulb_s1 = torch.split(features_v[num_lb:], num_ulb)
                
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

                pre_v_in_m = output['pre_v_in_m'][:num_lb]
                pre_t_in_m = output['pre_t_in_m'][:num_lb]
                
                if epoch <= 10:
                    select_v = []
                    select_t = []
                    v_sup_loss = 0
                    t_sup_loss = 0
                    label_filter_threshold = 0.5 * sup_loss
                    for i in range(len(pre_v_in_m)):
                        if(ce_loss(pre_v_in_m[i], y_lb[i]) < label_filter_threshold):
                            select_v.append(i)
                        if(ce_loss(pre_t_in_m[i], y_lb[i]) < label_filter_threshold):
                            select_t.append(i)
                    if select_v != []:
                        v_sup_loss =  ce_loss(logits_v_lb[select_v], y_lb[select_v], reduction='mean')  
                    if select_t != []:    
                        t_sup_loss =  ce_loss(logits_t_lb[select_t], y_lb[select_t], reduction='mean')   
                    sup_loss = sup_loss + v_sup_loss + t_sup_loss             


                modal_index = output['modal_index']
                modal_index_x_ulb_w, _, _ = torch.split(modal_index[num_lb:], num_ulb)
                pre_m_in_t = output['pre_m_in_t']
                pre_m_in_t,_,_ = torch.split(pre_m_in_t[num_lb:], num_ulb)
                pre_m_in_v = output['pre_m_in_v']
                pre_m_in_v,_,_ = torch.split(pre_m_in_v[num_lb:], num_ulb)

                with torch.no_grad():
                    logits_x_ulb_w = logits_x_ulb_w.detach()
                    logits_t_ulb_w = logits_t_ulb_w.detach()
                    logits_v_ulb_w = logits_v_ulb_w.detach()

                    if display_step == 1000:
                        diag_text_inputs = {
                            key: value[num_lb:num_lb + num_ulb]
                            for key, value in text_input.items()}
                        diagnostic_output = self._diagnostic_forward(
                            x_ulb_w, diag_text_inputs)
                        vis_counts = diagnostic_output['logits_visual_private'].argmax(
                            dim=1).bincount(minlength=3).tolist()
                        text_counts = diagnostic_output['logits_text_private'].argmax(
                            dim=1).bincount(minlength=3).tolist()
                        fusion_counts = diagnostic_output['logits_fusion'].argmax(
                            dim=1).bincount(minlength=3).tolist()
                        self.print_fn(
                            "DIAG step=1000: vis={}, text={}, fusion={}".format(
                                vis_counts, text_counts, fusion_counts))

                    modal_index_x_ulb_w = modal_index_x_ulb_w.detach()

                    features_lb = features_lb.detach()
                    features_ulb_w = features_ulb_w.detach()  # [bs*,2816]
                    # features_t_ulb_w = features_t_ulb_w.detach()
                    # features_v_ulb_w = features_v_ulb_w.detach()

                    pre_m_in_t = pre_m_in_t.detach()
                    pre_m_in_v = pre_m_in_v.detach()

                    ulb_probs = torch.softmax(logits_x_ulb_w, dim=1)
                    t_ulb_probs = torch.softmax(logits_t_ulb_w, dim=1)
                    v_ulb_probs = torch.softmax(logits_v_ulb_w, dim=1)
                    
                    scores, lbs_u_guess = torch.max(ulb_probs, dim=1)
                    t_scores, t_lbs_u_guess = torch.max(t_ulb_probs, dim=1)
                    v_scores, v_lbs_u_guess = torch.max(v_ulb_probs, dim=1)

                    fusion_lbs = lbs_u_guess
                    fusion_scores = scores
                    _, rac_lbs, rac_conf, _ = rac_calibrated_pseudo_label(
                        logits_v_ulb_w, logits_t_ulb_w, logits_x_ulb_w,
                        args.rac_temperature)
                    rac_conf_values.append(rac_conf.detach().float())
                    if args.use_rac:
                        lbs_u_guess = rac_lbs
                        scores = rac_conf
                        gate_conf = rac_conf
                    else:
                        gate_conf = fusion_scores

                    apply_dominant_filter = (
                        (args.use_dvad and args.use_dominant_filter) or
                        (not args.use_dvad and epoch <= 10)
                    )
                    dominant_mask = torch.ones_like(scores, dtype=torch.bool)
                    if apply_dominant_filter:
                        persudo_filter_threshold = sup_loss
                        for i,idx in enumerate(modal_index_x_ulb_w):
                            if idx == 2:
                                loss_m_in_t = ce_loss(pre_m_in_t[i], lbs_u_guess[i])
                                loss_m_in_v = ce_loss(pre_m_in_v[i], lbs_u_guess[i])
                                if loss_m_in_t > persudo_filter_threshold or loss_m_in_v > persudo_filter_threshold:
                                    dominant_mask[i] = False

                    threshold = args.threshold
                    warmup_active = (args.use_rac or args.use_dvad) and step < args.warmup_steps
                    dvad_percentile = None
                    batch_threshold = None
                    if args.use_dvad:
                        if args.dvad_mode == 'margin':
                            mask = (lbs_u_guess == fusion_lbs) & gate_conf.gt(dvad_scheduler.margin(step))
                            fallback_active = False
                            fixed_mask = None
                        elif args.dvad_mode == 'percentile':
                            dvad_percentile = dvad_scheduler.percentile(step)
                            batch_threshold = torch.quantile(
                                rac_conf.detach().float(), dvad_percentile / 100.0)
                            mask = (lbs_u_guess == fusion_lbs) & rac_conf.ge(batch_threshold)
                            fallback_active = False
                            fixed_mask = None
                        else:  # hybrid_fallback
                            fixed_mask = gate_conf.gt(args.threshold)
                            if fixed_mask.sum() == 0:
                                fallback_p = args.dvad_percentile_end
                                fallback_threshold = torch.quantile(
                                    gate_conf.float(), fallback_p / 100.0)
                                fallback_mask = gate_conf.ge(fallback_threshold)
                                mask = (lbs_u_guess == fusion_lbs) & fallback_mask
                                fallback_active = True
                            else:
                                mask = (lbs_u_guess == fusion_lbs) & fixed_mask
                                fallback_active = False
                        if args.use_dominant_filter:
                            mask = mask & dominant_mask
                    elif args.use_rac:
                        mask = scores.ge(threshold) & dominant_mask
                    else:
                        scores = scores.masked_fill(~dominant_mask, 0)
                        mask = scores.ge(threshold)

                    if warmup_active:
                        mask = torch.zeros_like(mask)
                    if args.use_rac or args.use_dvad:
                        t_mask = mask
                        v_mask = mask
                    else:
                        t_mask = t_scores.ge(threshold)
                        v_mask = v_scores.ge(threshold)

                    y_ulb = y_ulb.cuda(args.gpu)
                    pseudo_true_ratios.update(((lbs_u_guess == y_ulb) * mask).sum() / (mask.sum()+1e-7))

                if warmup_active:
                    unsup_loss = logits_x_ulb_s0.sum() * 0.0
                elif args.use_rac or args.use_dvad:
                    unsup_loss_m = F.cross_entropy(logits_x_ulb_s0, lbs_u_guess, reduction='none')
                    unsup_loss_t = F.cross_entropy(logits_t_ulb_s0, lbs_u_guess, reduction='none')
                    unsup_loss_v = F.cross_entropy(logits_v_ulb_s0, lbs_u_guess, reduction='none')
                    unsup_loss = (masked_mean(unsup_loss_m, mask) +
                                  masked_mean(unsup_loss_t, mask) +
                                  masked_mean(unsup_loss_v, mask))
                else:
                    unsup_loss_m = F.cross_entropy(logits_x_ulb_s0, lbs_u_guess, reduction='none') * mask
                    unsup_loss_t = F.cross_entropy(logits_t_ulb_s0, t_lbs_u_guess, reduction='none') * t_mask
                    unsup_loss_v = F.cross_entropy(logits_v_ulb_s0, v_lbs_u_guess, reduction='none') * v_mask
                    unsup_loss = unsup_loss_m.mean() + unsup_loss_t.mean() + unsup_loss_v.mean()

                fgmr = logits_x_lb.new_zeros(())
                if args.use_fgmr:
                    text_anchor = (output['text_private_feat'] if args.fgmr_anchor == 's_l'
                                   else output['c_l'])
                    fgmr = fgmr_loss(output['visual_private_feat'], text_anchor)

                # decouple loss
                # reconstruction loss
                loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                loss_recon = loss_recon_l + loss_recon_v

                # cycle consistency loss between s_x and s_x_r
                loss_sl_slr = self.MSE(output['s_l'], output['s_l_r'])
                loss_sv_slv = self.MSE(output['s_v'], output['s_v_r'])
                loss_s_sr = loss_sl_slr + loss_sv_slv

                # ort loss
                cosine_similarity_s_c_l = self.cosine(output['s_l'], output['c_l'],
                                                        torch.tensor([-1]).cuda()).mean(0)
                cosine_similarity_s_c_v = self.cosine(output['s_v'], output['c_v'],
                                                        torch.tensor([-1]).cuda()).mean(0)
                loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v
                # margin loss
                c_l, c_v = output['c_l_sim'], output['c_v_sim']
                c_l_lb, c_v_lb = c_l[:num_lb], c_v[:num_lb]
                c_l_ulb_w, c_l_ulb_s0, c_l_ulb_s1 = torch.split(c_l[num_lb:], num_ulb)
                c_v_ulb_w, c_v_ulb_s0, c_v_ulb_s1 = torch.split(c_v[num_lb:], num_ulb)
                ids, feats = [], []
                for i in range(y_lb.size(0)):
                    feats.append(c_l_lb[i].view(1, -1))
                    feats.append(c_v_lb[i].view(1, -1))
                    ids.append(y_lb[i].view(1, -1))
                    ids.append(y_lb[i].view(1, -1))

                for i in range(y_ulb.size(0)):
                    feats.append(c_l_ulb_w[i].view(1, -1))
                    feats.append(c_v_ulb_w[i].view(1, -1))
                    ids.append(lbs_u_guess[i].view(1, -1))
                    ids.append(lbs_u_guess[i].view(1, -1))               
                feats = torch.cat(feats, dim=0)
                ids = torch.cat(ids, dim=0)
                loss_sim = self.sim_loss(ids, feats)

                decouple_loss = loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1 

                total_loss = (sup_loss + self.lambda_u * unsup_loss +
                              args.fgmr_weight * fgmr + decouple_loss)
                # only attention
                # total_loss = sup_loss + self.lambda_u * unsup_loss_m.mean() + decouple_loss
                # only SMC
                # total_loss = sup_loss + self.lambda_u * unsup_loss + decouple_loss
                # base
                # total_loss = sup_loss + self.lambda_u * unsup_loss_m.mean()
                

            sup_losses.update(sup_loss.cpu().detach())
            unsup_losses.update(unsup_loss.cpu().detach())
            decouple_losses.update(decouple_loss.cpu().detach())
            total_losses.update(total_loss.cpu().detach())
            mask_ratios.update(mask.float().mean().cpu().detach())
            
            lr_last = self.optimizer.param_groups[0]['lr']
            
            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()
            self.it += 1

            if display_step % 100 == 0:
                accepted_fraction = mask.float().mean().item()
                mean_rac_conf = rac_conf.mean().item()
                confidence_view = 'RAC' if args.use_rac else 'fusion'
                if not args.use_dvad or args.dvad_mode == 'margin':
                    _dvad_extra = f"dvad_margin={dvad_scheduler.margin(step):.4f}"
                elif args.dvad_mode == 'percentile':
                    _dvad_extra = (
                        f"p={dvad_percentile:.2f}, "
                        f"batch_threshold={batch_threshold.item():.4f}")
                else:
                    _dvad_extra = (
                        f"fallback_active={int(fallback_active)}, "
                        f"n_fixed={int(fixed_mask.sum())}")
                self.print_fn(
                    "Step {}: sup_loss={}, unsup_loss={}, fgmr_loss={}, "
                    "decouple_loss={}, total_loss={}, accepted_fraction={}, "
                    "mean_{}_conf={}, {}".format(
                        display_step, sup_loss.item(), unsup_loss.item(), fgmr.item(),
                        decouple_loss.item(), total_loss.item(), accepted_fraction,
                        confidence_view, mean_rac_conf, _dvad_extra))
            if display_step % 500 == 0 or step + 1 >= TOTAL_STEPS:
                pla = self.evaluate_pla(args, display_step)
                self.print_fn("Step {}: held-out PLA ({} samples)={}".format(
                    display_step, self._pla_batch[2].shape[0], pla))

            end_run.record()
            torch.cuda.synchronize()
            batch_model_time.update(start_run.elapsed_time(end_run) / 1000)

            # self.it += 1
            start_batch.record()

        # self.print_fn(self.relation_matrix)

        conf_deciles = torch.quantile(
            torch.cat(rac_conf_values),
            torch.arange(10, dtype=torch.float32, device=rac_conf_values[0].device) / 10.0,
        ).cpu().tolist()
        formatted_deciles = '[{}]'.format(
            ', '.join('{:.3f}'.format(value) for value in conf_deciles))
        self.print_fn("Epoch {}: RAC conf deciles = {}, dvad_margin_final = {}".format(
            epoch + 1, formatted_deciles, args.dvad_final_margin))
        self.print_fn("Epoch {}/{} train: data time: {}, model time: {}, last lr: {}, labeled loss: {}, unlabeled loss: {}, decouple_loss: {}, total_loss: {}, mask ratio: {}, pseudo label correct ratio: {}".
                      format(epoch, args.epoch, batch_data_time.avg, batch_model_time.avg, lr_last, sup_losses.avg, unsup_losses.avg, decouple_losses.avg,total_losses.avg, mask_ratios.avg, pseudo_true_ratios.avg))

        if (epoch + 1) % args.eval_interval != 0 and epoch + 1 != args.epoch:
            return best_eval_acc

        eval_dict = self.evaluate(args=args)
        best_eval_acc = max(best_eval_acc, eval_dict['eval/top-1-acc'])
        self.print_fn("Epoch {}/{} test: test loss: {}, top-1 acc: {}, top-5 acc: {}, best top-1 acc: {}".format(
            epoch, args.epoch, eval_dict['eval/loss'], eval_dict['eval/top-1-acc'], eval_dict['eval/top-5-acc'], best_eval_acc
        ))
        
        save_path = os.path.join(args.save_dir, args.save_name)

        if eval_dict['eval/top-1-acc'] == best_eval_acc:
            self.save_model('model_best.pth', save_path)
        return eval_dict['eval/top-1-acc']

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, text_input, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            text_input = self.tokenizer(text_input, return_tensors='pt', padding=True, truncation=True)
            text_input = {key: value.cuda(args.gpu) for key, value in text_input.items()}
            num_batch = x.shape[0]
            total_num += num_batch
            output = self.model(x, text_input)
            # logits = output['pre_m']
            logits = output['pre_m_att']
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().detach().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().detach().tolist())
            total_loss += loss.cpu().detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5}

    def save_model(self, save_name, save_path):
        # if self.it < 1000000:
        #     return
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model},
                   save_filename)
        if self.num_classes == 10:
            tb_path = os.path.join(save_path, 'tensorboard')
            if not os.path.exists(tb_path):
                os.makedirs(tb_path, exist_ok=True)
            with open(os.path.join(save_path, 'tensorboard', 'lst_fix.pkl'), 'wb') as f:
                pickle.dump(self.lst, f)
            with open(os.path.join(save_path, 'tensorboard', 'abs_lst.pkl'), 'wb') as h:
                pickle.dump(self.abs_lst, h)
            with open(os.path.join(save_path, 'tensorboard', 'clsacc.pkl'), 'wb') as g:
                pickle.dump(self.clsacc, g)
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn('model loaded')

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


if __name__ == "__main__":
    pass
