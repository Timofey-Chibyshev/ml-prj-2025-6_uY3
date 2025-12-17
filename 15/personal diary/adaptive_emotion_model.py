import os
import json
import joblib
from datetime import datetime
from typing import Dict, Optional

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig, TaskType

# отключаем wandb
os.environ["WANDB_DISABLED"] = "true"


class AdaptiveEmotionModel:
    def __init__(self, model_dir: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_dir = model_dir

        # Директория дообученных весов
        self.ft_dir = f"{self.model_dir}_fine_tuned"
        load_dir = self.ft_dir if os.path.exists(self.ft_dir) else self.model_dir

        # Загружаем модель и токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            load_dir
        ).to(device)

        # Конфигурация (из оригинальной папки модели)
        with open(
            os.path.join(model_dir, "config_custom.json"), "r", encoding="utf-8"
        ) as f:
            self.config = json.load(f)

        # label_encoder (если есть)
        label_encoder_path = os.path.join(model_dir, "label_encoder.joblib")
        if os.path.exists(label_encoder_path):
            try:
                self.label_encoder = joblib.load(label_encoder_path)
            except Exception as e:
                print(f" Не удалось загрузить label_encoder.joblib: {e}")
                self.label_encoder = None
        else:
            print(" label_encoder.joblib не найден, продолжаю без него")
            self.label_encoder = None

        self.id2label = {int(k): v for k, v in self.config["id2label"].items()}
        self.label2id = self.config["label2id"]
        self.num_labels = self.config["num_labels"]
        self.class_weights = torch.tensor(
            self.config["class_weights"], dtype=torch.float, device=device
        )

        # История feedback'ов
        self.feedback_file = os.path.join(model_dir, "feedback_history.json")
        self.feedback_history = []
        self._load_feedback_history()

        # Авто‑дообучение
        self.auto_fine_tune_threshold = 10  # каждые 10 новых исправлений
        self.last_fine_tune_count = len(self.feedback_history)

        print(f"✓ Модель загружена из {load_dir} на {device}")
        print(f"✓ Классы эмоций: {self.config['labels']}")
        print(f"✓ Загружено feedback-ов: {len(self.feedback_history)}\n")

    # ===== PREDICT =====

    def predict(self, text: str, return_probs: bool = True) -> Dict:
        """Предсказание эмоции для одного текста."""
        self.base_model.eval()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.base_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        pred_id = int(probs.argmax())
        pred_label = self.id2label[pred_id]
        confidence = float(probs[pred_id])

        result = {
            "emotion": pred_label,      # сюда смотрит фронт
            "confidence": confidence,   # 0..1
            "id": pred_id,
        }

        if return_probs:
            result["probs"] = {
                self.id2label[i]: float(probs[i]) for i in range(len(probs))
            }

        return result

    # ===== FEEDBACK =====

    def add_feedback(
        self, text: str, predicted_emotion: str, corrected_emotion: str
    ) -> bool:
        """
        Добавляет feedback и при необходимости запускает авто‑дообучение.

        text: исходный текст
        predicted_emotion: ярлык модели (русский, как в id2label / label2id)
        corrected_emotion: исправленный ярлык (должен быть ключом label2id)
        """
        if corrected_emotion not in self.label2id:
            print(f"Эмоция '{corrected_emotion}' не найдена в списке классов")
            print(f"   Доступные: {list(self.label2id.keys())}")
            return False

        entry = {
            "text": text,
            "predicted_emotion": predicted_emotion,
            "corrected_emotion": corrected_emotion,
            "timestamp": datetime.now().isoformat(),
        }
        self.feedback_history.append(entry)
        self._save_feedback_history()

        print(f"✓ Feedback добавлен: {predicted_emotion} → {corrected_emotion}")

        # === АВТО‑ДООБУЧЕНИЕ ===
        new_count = len(self.feedback_history)
        if new_count >= self.auto_fine_tune_threshold:
            print(
                f"🚀 Достигнут порог auto fine-tune: {new_count} feedback'ов. "
                f"Запускаю дообучение (усиленный режим)..."
            )
            success = self.fine_tune()
            if success:
                # Подгружаем дообученную модель
                if os.path.exists(self.ft_dir):
                    self.base_model = AutoModelForSequenceClassification.from_pretrained(
                        self.ft_dir
                    ).to(self.device)
                    self.tokenizer = AutoTokenizer.from_pretrained(self.ft_dir)
                    print("✓ self.base_model обновлена до дообученной версии")
                # после fine_tune feedback_history уже очищен
                self.last_fine_tune_count = len(self.feedback_history)
            else:
                print("auto fine-tune не выполнен (недостаточно данных или ошибка)")

        return True

    def _save_feedback_history(self):
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        with open(self.feedback_file, "w", encoding="utf-8") as f:
            json.dump(self.feedback_history, f, ensure_ascii=False, indent=2)

    def _load_feedback_history(self):
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                self.feedback_history = json.load(f)
        else:
            self.feedback_history = []

    # ===== DATASET PREPARATION =====

    def _prepare_fine_tune_dataset(self) -> Optional[Dataset]:
        if not self.feedback_history:
            print("Нет feedback'ов для дообучения")
            return None

        df = pd.DataFrame(self.feedback_history)
        df["label_id"] = df["corrected_emotion"].map(self.label2id)
        df = df.dropna(subset=["label_id"]).reset_index(drop=True)

        if len(df) == 0:
            print("После маппинга датасет пуст (проверь названия эмоций)")
            return None

        print(f"Примеров для дообучения: {len(df)}")
        print("Распределение исправленных эмоций:")
        print(df["corrected_emotion"].value_counts())

        dataset = Dataset.from_pandas(df[["text", "label_id"]])
        return dataset

    # ===== FINE-TUNING (усиленный) =====

    def fine_tune(
        self, num_epochs: int = 3, learning_rate: float = 5e-5, use_lora: bool = True
    ) -> bool:
        """
        Дообучает модель на накопленных feedback'ах.
        Усиленный режим: делает сильный акцент на пользовательских исправлениях.
        """
        feedback_dataset = self._prepare_fine_tune_dataset()
        if feedback_dataset is None or len(feedback_dataset) == 0:
            print("Нет данных для дообучения, fine_tune прерван")
            return False

        dataset_size = len(feedback_dataset)

        # Для маленького датасета усиливаем обучение
        if dataset_size <= 50:
            num_epochs = max(num_epochs, 10)
            learning_rate = max(learning_rate, 2e-4)
            weight_decay = 0.0
        else:
            weight_decay = 0.01

        print(
            f"Использую усиленный fine-tune: "
            f"dataset_size={dataset_size}, epochs={num_epochs}, lr={learning_rate}, weight_decay={weight_decay}"
        )

        # Токенизация
        def tokenize_function(examples):
            texts = [str(t) for t in examples["text"]]
            enc = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=128,
            )
            enc["labels"] = examples["label_id"]
            return enc

        dataset_encoded = feedback_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=8,
            remove_columns=feedback_dataset.column_names,
        )
        dataset_encoded.set_format(type="torch")

        print(f"Токенизировано: {len(dataset_encoded)} примеров\n")

        model = self.base_model
        if use_lora:
            print("Использую LoRA для эффективного fine-tuning (усиленный режим)")
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=16,          # сильнее адаптеры
                lora_alpha=64,
                lora_dropout=0.1,
                bias="none",
                target_modules=["query", "value"],
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        outer_self = self

        class WeightedTrainer(Trainer):
            def compute_loss(
                inner_self, model, inputs, return_outputs: bool = False, **kwargs
            ):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits

                # Базовые веса классов
                class_weights = outer_self.class_weights.clone().to(model.device)

                # Дополнительно усиливаем веса классов,
                # которые реально присутствуют в этом батче
                with torch.no_grad():
                    unique_labels = torch.unique(labels)
                    for lbl in unique_labels:
                        class_weights[lbl] *= 2.0  # в 2 раза сильнее

                loss_fct = CrossEntropyLoss(weight=class_weights)
                loss = loss_fct(
                    logits.view(-1, outer_self.num_labels), labels.view(-1)
                )
                return (loss, outputs) if return_outputs else loss

        training_args = TrainingArguments(
            output_dir="./fine_tune_checkpoints",
            learning_rate=learning_rate,
            per_device_train_batch_size=8,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            logging_steps=10,
            save_strategy="no",
            seed=42,
            fp16=False,              # на CPU
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset_encoded,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
        )

        trainer.train()

        # Сохраняем дообученную модель
        os.makedirs(self.ft_dir, exist_ok=True)
        model.save_pretrained(self.ft_dir)
        self.tokenizer.save_pretrained(self.ft_dir)

        info = {
            "fine_tuned_date": datetime.now().isoformat(),
            "num_feedback_examples": len(feedback_dataset),
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "use_lora": use_lora,
            "mode": "strong_feedback",
        }
        with open(
            os.path.join(self.ft_dir, "fine_tune_info.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        print(f"\n Дообученная модель сохранена в: {self.ft_dir}")

        # Очистка использованных feedback'ов
        used_count = len(feedback_dataset)
        if used_count > 0 and used_count <= len(self.feedback_history):
            self.feedback_history = self.feedback_history[used_count:]
            self._save_feedback_history()
            print(f"✓ Удалено {used_count} использованных feedback'ов")
        self.last_fine_tune_count = len(self.feedback_history)

        return True

    def get_feedback_stats(self) -> Dict:
        """Возвращает статистику по feedback'ам."""
        if not self.feedback_history:
            return {"total": 0}

        df = pd.DataFrame(self.feedback_history)
        return {
            "total": len(df),
            "unique_texts": len(df["text"].unique()),
            "corrections_by_emotion": df["corrected_emotion"]
            .value_counts()
            .to_dict(),
            "misclassified_as": df["predicted_emotion"]
            .value_counts()
            .to_dict(),
        }
