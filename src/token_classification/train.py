from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import classification_report

from src.config import ConfigManager


def train(model, dataset, data_collator, tokenizer, id2label):
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = []
        true_predictions = []

        for pred, label in zip(predictions, labels):
            temp_pred = []
            temp_lab = []
            for p, l in zip(pred, label):
                if l != -100:
                    temp_pred.append(id2label[p])
                    temp_lab.append(id2label[l])
            true_labels.extend(temp_lab)
            true_predictions.extend(temp_pred)
        
        report = classification_report(true_labels, true_predictions, zero_division=0)
        print(report)
        return {}
    
    training_args = TrainingArguments(
        output_dir=ConfigManager().get("training")["output_dir"],
        num_train_epochs=ConfigManager().get("training")["num_train_epochs"],
        per_device_train_batch_size=ConfigManager().get("training")["per_device_train_batch_size"],
        per_device_eval_batch_size=ConfigManager().get("training")["per_device_eval_batch_size"],
        warmup_steps=ConfigManager().get("training")["warmup_steps"],
        weight_decay=ConfigManager().get("training")["weight_decay"],
        logging_dir=ConfigManager().get("training")["logging_dir"],
        logging_steps=ConfigManager().get("training")["logging_steps"],
        evaluation_strategy=ConfigManager().get("training")["evaluation_strategy"],
        save_strategy=ConfigManager().get("training")["save_strategy"],
        save_total_limit=ConfigManager().get("training")["save_total_limit"]
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model(ConfigManager().get("training")["save_dir"])
    tokenizer.save_pretrained(ConfigManager().get("training")["save_dir"])
