import copy
from typing import Dict, List
from statistics import mean
import torch
import torch.nn.functional as F
import logging 
import json 
import pytorch_lightning as pl
import torch.optim as optim
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from data_modules.input_example import InputExample, Relation, RelationType
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
                number_step: int, num_train_epochs: int,p_learning_rate: float, s_learning_rate: float, 
                predict_weight: float, reconstruct_weight: float, kl_weight: float, f1_weight: float,
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
        self.number_templates = len(self.input_formater.templates) - 1

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
    
    def compute_reward(self, examples, templates, sample_outputs, golds):
        #-------------------F1_REWARD-----------------------
        # prepare input for generating to compute reward
        # task_prefix = 'causality identification'
        # inputs_for_classifier = [self.input_formater.format_input(example=example, template_type=temp_id, task_descriptor=task_prefix)[-1]
        #                         for temp_id, example in zip(templates, examples)]
        f1_reward = compute_f1(sample_outputs, golds)[0]

        #---------------------KL REWARD-----------------------
        self.t5.eval()
        output_sentences = [self.oupt_formater.format_output(example=example, template_type=temp_id)
                            for temp_id, example in zip(templates, examples)]
        output_sentence_encoding = self.tokenizer(output_sentences,
                                                padding='longest',
                                                max_length=self.hparams.max_oupt_len,
                                                truncation=True,
                                                return_tensors="pt")
        labels = output_sentence_encoding.input_ids
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100 # replace padding token id's of the labels by -100
        
        task_prefix = 'causality identification'
        augmented_inputs = [self.input_formater.format_input(example=example, template_type=temp_id, task_descriptor=task_prefix)[-1]
                                for temp_id, example in zip(templates, examples)]
        augmented_inputs_encoding = self.tokenizer(augmented_inputs, padding='longest',
                                                            max_length=self.hparams.max_input_len,
                                                            truncation=True,return_tensors="pt")
        final_logits = self.t5(
                            input_ids=augmented_inputs_encoding.input_ids.cuda(), 
                            attention_mask=augmented_inputs_encoding.attention_mask.cuda(),
                            labels=labels.cuda()
                            ).logits
        
        baseline_templates = [-1]*len(templates)
        baseline_inputs = [self.input_formater.format_input(example=example, template_type=temp_id, task_descriptor=task_prefix)[-1]
                                for temp_id, example in zip(baseline_templates, examples)]
        baseline_inputs_encoding = self.tokenizer(baseline_inputs, padding='longest',
                                                            max_length=self.hparams.max_input_len,
                                                            truncation=True,return_tensors="pt")
        baseline_logits = self.t5(
                            input_ids=baseline_inputs_encoding.input_ids.cuda(), 
                            attention_mask=baseline_inputs_encoding.attention_mask.cuda(),
                            labels=labels.cuda()
                            ).logits
        
        KL_reward = F.kl_div(baseline_logits.log_softmax(0), final_logits.softmax(0), reduction='mean')
        # print(sample_outputs)
        # former_scores, _ = self.nll_inference(examples, [-1]*len(templates))
        # final_scores, final_predicts = self.nll_inference(examples, templates)

        # f1_reward = compute_f1(final_predicts, golds)[0]
        # KL_reward = F.kl_div(former_scores.log_softmax(0), final_scores.softmax(0), reduction='mean')
        self.log_dict({'f1_reward': f1_reward, 'KL_reward': KL_reward}, prog_bar=True)
        self.t5.train()
        # print(f"Reward: {f1_reward + KL_reward}")
        return float(self.hparams.f1_weight * f1_reward + self.hparams.kl_weight * KL_reward)
    
    def nll_inference(self, examples: List[InputExample], template: List[int], relations=['Yes', 'No']):
        scores = []
        predicts = []
        for temp_id, example in zip(template, examples):
            task_prefix = 'causality identification'
            inputs_for_classifier = self.input_formater.format_input(example=example, template_type=temp_id, task_descriptor=task_prefix)[-1]
            inputs_encoding_for_classifier = self.tokenizer(inputs_for_classifier, padding='longest',
                                                            max_length=self.hparams.max_input_len,
                                                            truncation=True,return_tensors="pt")
            score = []
            for relation in relations:
                candidate = copy.deepcopy(example)
                if relation == 'No':
                    candidate.relations = []
                if relation == 'Yes':
                    candidate.relations = [Relation(type=RelationType(short='cause', natural='cause'),
                                                    head=candidate.triggers[0],
                                                    tail=candidate.triggers[1])]
                
                output_sentences = self.oupt_formater.format_output(example=candidate, template_type=temp_id)
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
                score.append(float(predicted_loss))
            score = torch.FloatTensor(score)
            predict = relation[score.max(0)[1]]
            
            predicts.append(predict)
            scores.append(score)
        
        scores = torch.stack(scores, dim=0) # (batch_size, num_class)
        # print(scores.size())
        return scores, predicts

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
                                                max_length=self.hparams.max_oupt_len,
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

        # generate output sentences
        inputs_encoding_for_generating = self.tokenizer_for_generating(inputs_for_classifier, padding='longest',
                                                                    max_length=self.hparams.max_input_len,
                                                                    truncation=True,return_tensors="pt")
        sample_outputs = self.t5.generate(
                                        input_ids=inputs_encoding_for_generating.input_ids.cuda(), 
                                        do_sample=True, 
                                        top_k=20, top_p=0.95, 
                                        max_length=self.hparams.max_oupt_len, 
                                        num_return_sequences=1, num_beams=1,)
        sample_outputs = self.tokenizer_for_generating.batch_decode(sample_outputs, skip_special_tokens=True)

        # reconstruct loss
        task_prefix = "Generate question and context"
        inputs_encoding_for_reconstruct = self.tokenizer([f"{task_prefix}\n\n{sent}" for sent in sample_outputs], padding='longest',
                                                        max_length=self.hparams.max_oupt_len,
                                                        truncation=True,return_tensors="pt")

        outputs_for_reconstruct = [self.input_formater.format_input(example=example, template_type=temp_id, task_descriptor='')[-1]
                                for temp_id, example in zip(template, batch)]                                                  
        outputs_encoding_for_reconstruct = self.tokenizer(outputs_for_reconstruct, padding='longest',
                                                        max_length=self.hparams.max_input_len,
                                                        truncation=True,return_tensors="pt")
        labels = outputs_encoding_for_reconstruct.input_ids
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100 # replace padding token id's of the labels by -100
        
        reconstruct_loss = self.t5(
                                input_ids=inputs_encoding_for_reconstruct.input_ids.cuda(), 
                                attention_mask=inputs_encoding_for_reconstruct.attention_mask.cuda(), 
                                labels=labels.cuda()
                            ).loss
        reconstruct_loss = torch.mean(reconstruct_loss)
        self.log_dict({'predicted_loss': predicted_loss, 'reconstruct_loss': reconstruct_loss}, prog_bar=True)
        if optimizer_idx == 0:
            return self.hparams.predict_weight * predicted_loss + self.hparams.reconstruct_weight * reconstruct_loss

        # compute policy loss
        reward = self.compute_reward(examples=batch, templates=template, sample_outputs=sample_outputs, golds=output_sentences)

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
                                                max_length=self.hparams.max_oupt_len,
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

        # generate output sentences
        inputs_encoding_for_generating = self.tokenizer_for_generating(inputs_for_classifier, padding='longest',
                                                                    max_length=self.hparams.max_input_len,
                                                                    truncation=True,return_tensors="pt")
        sample_outputs = self.t5.generate(
                                        input_ids=inputs_encoding_for_generating.input_ids.cuda(), 
                                        do_sample=True, 
                                        top_k=20, top_p=0.95, 
                                        max_length=self.hparams.max_oupt_len, 
                                        num_return_sequences=1, num_beams=1,)
        sample_outputs = self.tokenizer_for_generating.batch_decode(sample_outputs, skip_special_tokens=True)

        # reconstruct loss
        task_prefix = "Generate question and context"
        inputs_encoding_for_reconstruct = self.tokenizer([f"{task_prefix}\n\n{sent}" for sent in sample_outputs], padding='longest',
                                                        max_length=self.hparams.max_oupt_len,
                                                        truncation=True,return_tensors="pt")

        outputs_for_reconstruct = [self.input_formater.format_input(example=example, template_type=temp_id, task_descriptor='')[-1]
                                for temp_id, example in zip(template, batch)]                                                  
        outputs_encoding_for_reconstruct = self.tokenizer(outputs_for_reconstruct, padding='longest',
                                                        max_length=self.hparams.max_input_len,
                                                        truncation=True,return_tensors="pt")
        labels = outputs_encoding_for_reconstruct.input_ids
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100 # replace padding token id's of the labels by -100
        
        reconstruct_loss = self.t5(
                                input_ids=inputs_encoding_for_reconstruct.input_ids.cuda(), 
                                attention_mask=inputs_encoding_for_reconstruct.attention_mask.cuda(), 
                                labels=labels.cuda()
                            ).loss
        reconstruct_loss = torch.mean(reconstruct_loss)
        self.log_dict({'predicted_loss': predicted_loss, 'reconstruct_loss': reconstruct_loss}, prog_bar=True)

        # compute policy loss
        reward = self.compute_reward(examples=batch, templates=template, sample_outputs=sample_outputs, golds=output_sentences)
        self.val_rewards.append(reward)
        normalized_reward = reward - mean(self.val_rewards)

        policy_loss = []
        for log_prob in log_probs:
            policy_loss.append(-log_prob * normalized_reward)
        policy_loss = sum(policy_loss)
        self.log_dict({"policy_loss": policy_loss}, prog_bar=True)

        return policy_loss, self.hparams.predict_weight * predicted_loss + self.hparams.reconstruct_weight * reconstruct_loss

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
