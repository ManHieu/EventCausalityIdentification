from typing import Callable, Optional
import torch 
import logging 
import json 
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer 
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from arguments import ModelArguments, TrainingArguments


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class GenEERModel(pl.LightningModule):
    def __init__(self, model_args: ModelArguments, training_args: TrainingArguments,
                learning_rate: float, adam_epsilon: float, 
                weight_decay: float=0,warmup: float=0) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_args.tokenizer_name)
    
    def forward(self, input_ids, attention_mask=None, 
                decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels =lm_labels,
        )
    
    def training_step(self, batch, batch_idx):
        # print(f'Epoch {self.trainer.current_epoch} / Step {self.trainer.global_step}: lr {self.trainer.optimizers[0].param_groups[0]["lr"]}')
        lm_labels = batch['tgt_token_ids']
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
                    input_ids=batch['input_token_ids'],
                    attention_mask=batch['input_attn_mask'],
                    lm_labels =lm_labels,
                    decoder_attention_mask=batch['tgt_attn_mask']
                    )
        loss = outputs[0]
        loss = torch.mean(loss)
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self,batch, batch_idx):
        lm_labels = batch["tgt_token_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
                        input_ids=batch["input_token_ids"],
                        attention_mask=batch["input_attn_mask"],
                        lm_labels=lm_labels,
                        decoder_attention_mask=batch["tgt_attn_mask"]
                    )
        loss = outputs[0]
        loss = torch.mean(loss)
        self.log("val_loss", loss)
        return loss  

    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.stack(outputs))
        self.log('avg_val_loss', avg_loss)
    
    def test_step(self, batch, batch_idx):
        sample_output = self.model.generate(input_ids=batch['input_token_ids'], do_sample=True, 
                                top_k=20, top_p=0.95, max_length=16, num_return_sequences=1,num_beams=1,)
        sample_output = sample_output.reshape(batch['input_token_ids'].size(0), 1, -1)
        # doc_key = batch['doc_key'] # list 
        tgt_token_ids = batch['tgt_token_ids']

        return (sample_output, tgt_token_ids) 

    def test_epoch_end(self, outputs):
        # evaluate F1 
        with open('./predictions.jsonl','w') as writer:
            for tup in outputs:
                for idx in range(tup[0].size(0)):
                    
                    pred = {
                        # 'doc_key': tup[0][idx],
                        'predicted': self.tokenizer.decode(tup[0][idx].squeeze(0), skip_special_tokens=True),
                        'gold': self.tokenizer.decode(tup[1][idx].squeeze(0), skip_special_tokens=True) 
                    }
                    print(pred)
                    writer.write(json.dumps(pred)+'\n')
        return {}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        t_total = len(self.train_dataloader())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        
        num_warmup_steps = self.hparams.warmup * t_total * self.hparams.training_args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step'
            }
        }
