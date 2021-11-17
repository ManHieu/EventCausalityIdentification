import copy
from typing import Dict
import torch
import torch.nn.functional as F
import logging 
import json 
import pytorch_lightning as pl
import torch.optim as optim
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from data_modules.input_formats import INPUT_FORMATS
from model.selector_model import Selector
from utils.utils import compute_f1


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class GenEERModel(pl.LightningModule):
    def __init__(self, tokenizer_name: str, model_name_or_path: str, selector_name_or_path:str,
                templates: Dict[int, str], input_format: str, max_input_len: int, max_oupt_len: int,
                num_train_epochs: int, s_weight: float, p_weight: float, 
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
        self.templates = templates

        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.selector = Selector(sentence_encoder=selector_name_or_path, 
                                number_layers=len(templates.keys()), 
                                tokenizer=self.tokenizer,
                                max_input_len=max_input_len, 
                                input_format=input_format, 
                                fn_activate=fn_activate)
    
    
    def training_step(self, batch, batch_idx):
        input_sentences, output_sentences, context_sentences, ED_templates = batch

        target_encoding = self.tokenizer(output_sentences,
                                        padding='longest',
                                        max_length=self.hparams.max_oupt_len,
                                        truncation=True,
                                        return_tensors="pt")
        labels = target_encoding.input_ids
        # replace padding token id's of the labels by -100
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100 

        # action, log_probs, probs = self.selector(input_sentences)

        # template = [self.templates[int(index)] for index in action]
        template = [self.templates[0] for index in range(len(input_sentences))]
        task_prefix = 'causality identification'
        inputs_for_classifier = [self.input_formater.format_input_for_predictor(ctx=ctx, task_prefix=task_prefix, additional_info=temp) 
                                for temp, ctx in zip(template, input_sentences)]
        inputs_encoding_for_classifier = self.tokenizer(inputs_for_classifier, padding='longest',
                                                        max_length=self.hparams.max_input_len,
                                                        truncation=True,return_tensors="pt")
        
        predicted_loss = self.t5(
                    input_ids=inputs_encoding_for_classifier.input_ids.cuda(), 
                    attention_mask=inputs_encoding_for_classifier.attention_mask.cuda(), 
                    labels=labels.cuda()
        ).loss
        predicted_loss = torch.mean(predicted_loss)

        # inputs_encoding_for_generating = self.tokenizer_for_generating(inputs_for_classifier, padding='longest',
        #                                                             max_length=self.hparams.max_input_len,
        #                                                             truncation=True,return_tensors="pt")
        # sample_outputs = self.t5.generate(input_ids=inputs_encoding_for_generating.input_ids.cuda(), do_sample=True, 
        #                                  top_k=20, top_p=0.95, max_length=self.hparams.max_oupt_len, 
        #                                  num_return_sequences=1, num_beams=1,)
        # sample_outputs = self.tokenizer_for_generating.batch_decode(sample_outputs, skip_special_tokens=True)
        
        # reward = compute_f1(sample_outputs, output_sentences)[0]
        # policy_loss = []
        # for log_prob in log_probs:
        #     policy_loss.append(-log_prob * reward)
        # policy_loss = sum(policy_loss)
        policy_loss = 0
        loss = self.hparams.s_weight * policy_loss + self.hparams.p_weight * predicted_loss
        self.log_dict({"policy_loss": policy_loss, "predicted_loss": predicted_loss}, prog_bar=True)

        return loss
    
    def validation_step(self,batch, batch_idx):
        input_sentences, output_sentences, context_sentences, ED_templates = batch

        target_encoding = self.tokenizer(output_sentences,
                                        padding='longest',
                                        max_length=self.hparams.max_oupt_len,
                                        truncation=True,
                                        return_tensors="pt")
        labels = target_encoding.input_ids
        # replace padding token id's of the labels by -100
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100 

        # action, log_probs, probs = self.selector(input_sentences)

        # template = [self.templates[int(index)] for index in action]
        template = [self.templates[0] for index in range(len(input_sentences))]
        task_prefix = 'causality identification'
        inputs_for_classifier = [self.input_formater.format_input_for_predictor(ctx=ctx, task_prefix=task_prefix, additional_info=temp) 
                                for temp, ctx in zip(template, input_sentences)]
        inputs_encoding_for_classifier = self.tokenizer(inputs_for_classifier, padding='longest',
                                                        max_length=self.hparams.max_input_len,
                                                        truncation=True,return_tensors="pt")
        
        predicted_loss = self.t5(
                    input_ids=inputs_encoding_for_classifier.input_ids.cuda(), 
                    attention_mask=inputs_encoding_for_classifier.attention_mask.cuda(), 
                    labels=labels.cuda()
        ).loss
        predicted_loss = torch.mean(predicted_loss)

        # inputs_encoding_for_generating = self.tokenizer_for_generating(inputs_for_classifier, padding='longest',
        #                                                             max_length=self.hparams.max_input_len,
        #                                                             truncation=True,return_tensors="pt")
        # sample_outputs = self.t5.generate(input_ids=inputs_encoding_for_generating.input_ids.cuda(), do_sample=True, 
        #                                  top_k=20, top_p=0.95, max_length=self.hparams.max_oupt_len, 
        #                                  num_return_sequences=1, num_beams=1,)
        # sample_outputs = self.tokenizer_for_generating.batch_decode(sample_outputs, skip_special_tokens=True)
        
        # reward = compute_f1(sample_outputs, output_sentences)[0]
        # policy_loss = []
        # for log_prob in log_probs:
        #     policy_loss.append(-log_prob * reward)
        # policy_loss = sum(policy_loss)

        policy_loss = 0
        loss = self.hparams.s_weight * policy_loss + self.hparams.p_weight * predicted_loss

        self.log_dict({"policy_loss": policy_loss, "predicted_loss": predicted_loss}, prog_bar=True)

        return policy_loss, predicted_loss, loss

    def validation_epoch_end(self, outputs):
        # avg_policy_loss = torch.mean(torch.stack([output[0] for output in outputs]))
        avg_predicted_loss = torch.mean(torch.stack([output[1] for output in outputs]))
        self.log_dict({"policy_loss": 0, "predicted_loss": avg_predicted_loss}, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        input_sentences, output_sentences, context_sentences, ED_templates = batch

        target_encoding = self.tokenizer(output_sentences,
                                        padding='longest',
                                        max_length=self.hparams.max_oupt_len,
                                        truncation=True,
                                        return_tensors="pt")
        labels = target_encoding.input_ids
        # replace padding token id's of the labels by -100
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100 

        # action, log_probs, probs = self.selector(input_sentences)

        # template = [self.templates[int(index)] for index in action]
        template = [self.templates[0] for index in range(len(input_sentences))]
        task_prefix = 'causality identification'
        inputs_for_classifier = [self.input_formater.format_input_for_predictor(ctx=ctx, task_prefix=task_prefix, additional_info=temp) 
                                for temp, ctx in zip(template, input_sentences)]

        inputs_encoding_for_generating = self.tokenizer_for_generating(inputs_for_classifier, padding='longest',
                                                                    max_length=self.hparams.max_input_len,
                                                                    truncation=True,return_tensors="pt")
        
        sample_outputs = self.t5.generate(input_ids=inputs_encoding_for_generating.input_ids.cuda(), do_sample=True, 
                                         top_k=20, top_p=0.95, max_length=self.hparams.max_oupt_len, 
                                         num_return_sequences=1, num_beams=8,)

        sample_outputs = self.tokenizer_for_generating.batch_decode(sample_outputs, skip_special_tokens=True)

        return sample_outputs, output_sentences, input_sentences

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
        t_total = len(self.train_dataloader()) * self.hparams.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_predictor_parameters = [
            {
                "params": [p for n, p in self.t5.named_parameters() if not any(nd in n for nd in no_decay)]
                        # + [p for n, p in self.selector.named_parameters() if not any(nd in n for nd in no_decay)]
                        ,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.t5.named_parameters() if any(nd in n for nd in no_decay)]
                        # + [p for n, p in self.selector.named_parameters() if any(nd in n for nd in no_decay)]
                        ,
                "weight_decay": 0.0,
            },
        ]
        p_optimizer = AdamW(optimizer_grouped_predictor_parameters, lr=self.hparams.p_learning_rate, eps=self.hparams.adam_epsilon)
        
        num_warmup_steps = self.hparams.warmup * t_total
        p_scheduler = get_linear_schedule_with_warmup(
            p_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
        )

        optimizer_grouped_selector_parameters = [
            {
                "params": [p for n, p in self.selector.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.selector.named_parameters() if any(nd in n for nd in no_decay)]
                        ,
                "weight_decay": 0.0,
            },
        ]
        s_optimizer = AdamW(optimizer_grouped_selector_parameters, lr=self.hparams.s_learning_rate)
        
        def m_lr_lambda(current_step: int):
            return 0.5 ** int(current_step / (2*len(self.train_dataloader())))
        s_scheduler = optim.lr_scheduler.LambdaLR(s_optimizer, lr_lambda=m_lr_lambda)

        return ({
            "optimizer": p_optimizer,
            "lr_scheduler": {
                "scheduler": p_scheduler,
                'interval': 'step'
            }
        },)
        #     {
        #     "optimizer": s_optimizer,
        #     "lr_scheduler": s_scheduler,
        # })
