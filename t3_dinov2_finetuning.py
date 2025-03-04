import transformers
import accelerate
import peft
from datasets import load_dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import AutoImageProcessor, AutoModelForImageClassification, Dinov2ForImageClassification, Dinov2Config, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate


def main():
    model_checkpoint = "facebook/dinov2-large"
    dataset = load_dataset("imagefolder", data_dir="data/plates/plates")

    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(image_processor.size["height"]),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize
        ]
    )

    val_transforms = Compose(
        [
            RandomResizedCrop(image_processor.size["height"]),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize
        ]
    )

    train_ds = dataset["train"]
    val_ds = dataset["val"]

    # transformation
    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)

    model = Dinov2ForImageClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    print_trainable_parameters(model)

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"]
    )
    lora_model = get_peft_model(model, config)
    print_trainable_parameters(lora_model)

    model_name = model_checkpoint.split("/")[-1]
    batch_size = 128

    args = TrainingArguments(
        f"{model_name}-finetuned-plates",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        num_train_epochs=5,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
        label_names=["labels"],
    )


def preprocess_train(example_batch, train_transforms):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def preprocess_val(example_batch, val_transforms):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def print_trainable_parameters(model):
    """
    print the number of trainable parameters in the model
    :param model:
    :return:
    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        if param.requires_grad:
            trainable_params += num_params
        all_param += num_params
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")