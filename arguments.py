from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated list of dataset names, for training."}
    )

    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to data directory"}
    )

    tokenizer: str = field(
        default = None,
        metadata= {"help": "The tokenizer used to prepare data"}
    )

    tokenizer_for_encode_input_sentence: str = field(
        default = None,
        metadata= {"help": "The tokenizer used to prepare data"}
    )

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, shorter sequences will be padded."
        },
    )

    max_output_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum output sequence length (default is the same as input)"
        },
    )

    input_format: str = field(
        default=None, metadata={"help": "Input format"}
    )
    
    output_format: str = field(
        default=None, metadata={"help": "Output format"}
    )

    multitask: bool = field(
        default=False, metadata={"help": "If true, each input sentence is prepended with the dataset name"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Arguments for the Trainer.
    """
    output_dir: str = field(
        default='experiments',
        metadata={"help": "The output directory where the results and model weights will be written."}
    )
    
    zero_shot: bool = field(
        default=False,
        metadata={"help": "Zero-shot setting"}
    )

    batch_size: int = field(
        default=8,
        metadata={"help": "Batch_size for training and evaluating"}
    )
    p_learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Learning rate of predictor"}
    )
    s_learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Learning rate of selector"}
    )
    gradient_clip_val: float = field(
        default=0.0,
        metadata={"help":"Gradient clipping value"}
    )
    seed: int = field(
        default=1741,
        metadata={"help": "seeding for reproductivity"}
    )
    selector_weight: int = field(
        default=1.0,
        metadata={'help': "The weight of selector loss"}
    )
    predictor_weight: int = field(
        default=1.0,
        metadata={'help': "The weight of predictor loss"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    selector_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    fn_activate: Optional[str] = field(
        default='leakyrelu',
        metadata={'help': "Type of activate function using in selector"}
    )

