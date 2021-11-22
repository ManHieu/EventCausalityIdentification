import copy
from typing import Dict
from statistics import mean
import torch
import torch.nn.functional as F
import logging 
import json 
import pytorch_lightning as pl
import torch.optim as optim
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from data_modules.input_formats import INPUT_FORMATS
from data_modules.output_formats import OUTPUT_FORMATS
from model.selector_model import Selector
from utils.utils import compute_f1
import numpy as np


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class GenEERModel(pl.LightningModule):
    def __init__(self, tokenizer_name: str, model_name_or_path: str, selector_name_or_path:str,
                input_format: str, oupt_format: str, max_input_len: int, max_oupt_len: int,
                number_step: int, num_train_epochs: int,
                p_learning_rate: float, s_learning_rate: float,
                adam_epsilon: float, fn_activate: str='leakyrelu', weight_decay: float=0, warmup: float=0) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_for_generating = copy.deepcopy(self.tokenizer)
        # when generating, we will use the logits of right-most token to predict the next token
        # so the padding should be on the left
        self.tokenizer_for_generating.padding_side = 'left'
        self.tokenizer_for_generating.pad_token = self.tokenizer_for_generating.eos_token # to avoid an error
        
        self.input_formater = INPUT_FORMATS[input_format]()
        self.oupt_formater = OUTPUT_FORMATS[oupt_format]()
        self.number_templates = len(self.input_formater.templates)

        self.number_step = number_step

        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.selector = Selector(sentence_encoder=selector_name_or_path, 
                                number_layers=self.number_templates, 
                                tokenizer=self.tokenizer,
                                max_input_len=max_input_len, 
                                input_format=input_format, 
                                fn_activate=fn_activate)
        
        self.rewards = []
        self.val_rewards = []
    
    def reward(self, sample, golds):
        # prepare input for generating to compute reward
        inputs_encoding_for_generating = self.tokenizer_for_generating(sample, padding='longest',
                                                                    max_length=self.hparams.max_input_len,
                                                                    truncation=True,return_tensors="pt")
        sample_outputs = self.t5.generate(
                                        input_ids=inputs_encoding_for_generating.input_ids.cuda(), 
                                        do_sample=True, 
                                        top_k=20, top_p=0.95, 
                                        max_length=self.hparams.max_oupt_len, 
                                        num_return_sequences=1, num_beams=1,)
        sample_outputs = self.tokenizer_for_generating.batch_decode(sample_outputs, skip_special_tokens=True)
        # print(sample_outputs)
        f1_reward = compute_f1(sample_outputs, golds)[0]
        return f1_reward
    
    # def on_epoch_start(self) -> None:
    #     for i, optimizer in enumerate(self.optimizers()):
    #         for j, param_group in enumerate(optimizer.param_groups):
    #             print(f"Optimizer {i} - group {j}: {param_group['lr']}")

    def on_train_epoch_start(self) -> None:
        self.rewards = []
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        # input_sentences, output_sentences, context_sentences, ED_templates, labels = batch
        action, log_probs, probs = self.selector(batch)

        # prepare input for predictor
        template = [int(index) for index in action]
        task_prefix = 'causality identification'
        inputs_for_classifier = [self.input_formater.format_input(example=example, template_type=temp_id, task_descriptor=task_prefix)[-1]
                                for temp_id, example in zip(template, batch)]
        inputs_encoding_for_classifier = self.tokenizer(inputs_for_classifier, padding='longest',
                                                        max_length=self.hparams.max_input_len,
                                                        truncation=True,return_tensors="pt")
        
        output_sentences = [self.oupt_formater.format_output(example=example, template_type=temp_id)
                for temp_id, example in zip(template, batch)]
        output_sentence_encoding = self.tokenizer(output_sentences,
                                                                padding='longest',
                                                                max_length=self.hparams.max_input_len,
                                                                truncation=True,
                                                                return_tensors="pt")
        labels = output_sentence_encoding.input_ids
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100 # replace padding token id's of the labels by -100

        predicted_loss = self.t5(
                    input_ids=inputs_encoding_for_classifier.input_ids.cuda(), 
                    attention_mask=inputs_encoding_for_classifier.attention_mask.cuda(), 
                    labels=labels.cuda()
        ).loss
        predicted_loss = torch.mean(predicted_loss)
        self.log_dict({"predicted_loss": predicted_loss}, prog_bar=True)
        if optimizer_idx == 0:
            return predicted_loss

        # compute policy loss
        reward = self.reward(inputs_for_classifier, output_sentences)
        self.rewards.append(reward)
        normalized_reward = reward - mean(self.rewards)
        # print(f"Reward: {normalized_reward} - log_prob: {log_probs}")
        
        policy_loss = []

        for log_prob in  log_probs:
            policy_loss.append(-log_prob * normalized_reward)
        policy_loss = sum(policy_loss)
        self.log_dict({"policy_loss": policy_loss}, prog_bar=True)
        if optimizer_idx == 1:
            return policy_loss

    def on_validation_epoch_start(self) -> None:
        self.val_rewards = []
    
    def validation_step(self,batch, batch_idx):
        # input_sentences, output_sentences, context_sentences, ED_templates, labels = batch
        action, log_probs, probs = self.selector(batch)

        # prepare input for predictor
        template = [int(index) for index in action]
        task_prefix = 'causality identification'
        inputs_for_classifier = [self.input_formater.format_input(example=example, template_type=temp_id, task_descriptor=task_prefix)[-1] 
                                for temp_id, example in zip(template, batch)]
        inputs_encoding_for_classifier = self.tokenizer(inputs_for_classifier, padding='longest',
                                                        max_length=self.hparams.max_input_len,
                                                        truncation=True,return_tensors="pt")
        
        # prepare output for predictor
        output_sentences = [self.oupt_formater.format_output(example=example, template_type=temp_id)
                for temp_id, example in zip(template, batch)]
        output_sentence_encoding = self.tokenizer(output_sentences,
                                                                padding='longest',
                                                                max_length=self.hparams.max_input_len,
                                                                truncation=True,
                                                                return_tensors="pt")
        labels = output_sentence_encoding.input_ids
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100 # replace padding token id's of the labels by -100
        
        predicted_loss = self.t5(
                    input_ids=inputs_encoding_for_classifier.input_ids.cuda(), 
                    attention_mask=inputs_encoding_for_classifier.attention_mask.cuda(), 
                    labels=labels.cuda()
        ).loss
        predicted_loss = torch.mean(predicted_loss)
        self.log_dict({"policy_loss": predicted_loss}, prog_bar=True)

        # compute policy loss
        reward = self.reward(inputs_for_classifier, output_sentences)
        self.val_rewards.append(reward)
        normalized_reward = reward - mean(self.val_rewards)

        policy_loss = []
        for log_prob in log_probs:
            policy_loss.append(-log_prob * normalized_reward)
        policy_loss = sum(policy_loss)
        self.log_dict({"policy_loss": policy_loss}, prog_bar=True)

        return policy_loss, predicted_loss

    def validation_epoch_end(self, outputs):
        avg_policy_loss = torch.mean(torch.stack([output[0] for output in outputs]))
        avg_predicted_loss = torch.mean(torch.stack([output[1] for output in outputs]))
        self.log_dict({"policy_loss": avg_policy_loss, "predicted_loss": avg_predicted_loss}, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        # input_sentences, output_sentences, context_sentences, ED_templates, labels = batch
        action, log_probs, probs = self.selector(batch)

        # prepare input for predictor
        template = [int(index) for index in action]
        task_prefix = 'causality identification'
        inputs_for_classifier = [self.input_formater.format_input(example=example, template_type=temp_id, task_descriptor=task_prefix)[-1]
                                for temp_id, example in zip(template, batch)]
        inputs_encoding_for_generating = self.tokenizer_for_generating(inputs_for_classifier, padding='longest',
                                                                    max_length=self.hparams.max_input_len,
                                                                    truncation=True,return_tensors="pt")
        
        # generate output 
        sample_outputs = self.t5.generate(input_ids=inputs_encoding_for_generating.input_ids.cuda(), do_sample=True, 
                                         top_k=20, top_p=0.95, max_length=self.hparams.max_oupt_len, 
                                         num_return_sequences=1, num_beams=8,)
        sample_outputs = self.tokenizer_for_generating.batch_decode(sample_outputs, skip_special_tokens=True)

        # gold output
        output_sentences = [self.oupt_formater.format_output(example=example, template_type=temp_id)
                for temp_id, example in zip(template, batch)]

        return sample_outputs, output_sentences, inputs_for_classifier

    def test_epoch_end(self, outputs):
        # evaluate F1
        preds = []
        for output in outputs:
            for sample in zip(*output):
                preds.append({
                    'sentence': sample[2],
                    'predicted': sample[0],
                    'gold': sample[1]
                })

        with open('./predictions.json','w') as writer:
            writer.write(json.dumps(preds, indent=6)+'\n')

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        t_total = self.number_step * self.hparams.num_train_epochs
        
        # config optimizer for predictor
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_predictor_parameters = [
            {
                "params": [p for n, p in self.t5.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.t5.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        p_optimizer = AdamW(optimizer_grouped_predictor_parameters, lr=self.hparams.p_learning_rate, eps=self.hparams.adam_epsilon)
        
        num_warmup_steps = self.hparams.warmup * t_total
        p_scheduler = get_linear_schedule_with_warmup(
            p_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
        )

        # config optimizer for selector
        optimizer_grouped_selector_parameters = [
            {
                "params": [p for n, p in self.selector.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.selector.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        s_optimizer = AdamW(optimizer_grouped_selector_parameters, lr=self.hparams.s_learning_rate)
        
        def m_lr_lambda(current_step: int):
            return 0.5 ** int(current_step / (2*self.number_step))
        s_scheduler = optim.lr_scheduler.LambdaLR(s_optimizer, lr_lambda=m_lr_lambda)

        return ({
            "optimizer": p_optimizer,
            "lr_scheduler": {
                "scheduler": p_scheduler,
                'interval': 'step'
            }
        },
            {
            "optimizer": s_optimizer,
            "lr_scheduler": s_scheduler,
        })
