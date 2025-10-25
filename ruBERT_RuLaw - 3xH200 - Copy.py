import subprocess
import sys

def install_package(package: str):
    """Устанавливает пакет через pip в том же интерпретаторе."""
    print(f"→ Установка пакета: {package}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", *package.split()],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    print(f"✓ Установлен: {package}")

# Список пакетов с явным указанием, что импортировать
required_packages = [
    ("install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130", "torch"),
    ("install transformers", "transformers"),
    ("install datasets", "datasets"),
    ("install tensorboard", "tensorboard"),
    ("install accelerate", "accelerate")
]


for install_str, import_name in required_packages:
    try:
        __import__(import_name)
        print(f"{import_name} уже установлен ✅")
    except ImportError:
        install_package(install_str)


import os
from pathlib import Path
import tempfile
import shutil
import torch
import random
import numpy as np

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")

# Теперь импорты HF библиотек после настройки env
from datasets import load_dataset, load_from_disk, config
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
# Подтверждение кэша datasets
print("Datasets cache directory:", config.HF_DATASETS_CACHE)


# Флаги и пути
test_mode = False  # True для теста на 3k примерах
base_dir = Path.home() / "ruBERT_data"  # ~/ruBERT_data

tokenized_path = base_dir / ('tokenized_rulaw_test' if test_mode else 'tokenized_rulaw')
model_output_dir = base_dir / "ruBERT-ruLaw"
logs_dir = base_dir / "rulaw_logs"

# Создаём все нужные директории
tokenized_path.parent.mkdir(parents=True, exist_ok=True)
model_output_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Для воспроизводимости на GPU
    torch.backends.cudnn.benchmark = False    # Отключает оптимизации, влияющие на воспроизводимость

set_seed(42)

def main():
    # Загрузка токенизатора
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Функция токенизации
    def tokenize_function(examples):
        texts = [text for text in examples['textIPS'] if text is not None and str(text).strip()]
        if not texts:
            return {'input_ids': [], 'attention_mask': []}

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            padding=False
        )
        # Чтобы Trainer точно работал, убедимся, что есть только 'input_ids' и 'attention_mask'
        result = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
        return result

    # Проверка и токенизация/загрузка датасета
    if not tokenized_path.exists():
        print("Загрузка датасета...")
        ds = load_dataset("irlspbru/RusLawOD")
        print(f"Датасет загружен: {len(ds['train'])} примеров")

        # Фильтрация: удаляем None или пустые 'textIPS'
        print("Фильтрация пустых/None текстов...")
        ds = ds.filter(lambda x: x['textIPS'] is not None and str(x['textIPS']).strip() != '', desc="Фильтрация")
        print(f"После фильтрации: {len(ds['train'])} примеров")

        # Тестовый режим: subsample 3k документов ПОСЛЕ фильтрации, ПЕРЕД токенизацией
        if test_mode:
            ds['train'] = ds['train'].shuffle(seed=42).select(range(3000))
            print(f"Тестовый режим: subsample до {len(ds['train'])} примеров перед токенизацией")

        print("Токенизация...")
        # Используем num_proc=8 для ускорения на multi-core CPU
        tokenized_ds = ds.map(
            tokenize_function,
            batched=True,
            remove_columns=ds['train'].column_names,
            num_proc=24,
            desc="Токенизация"
        )
        tokenized_ds.save_to_disk(tokenized_path)
        print("Токенизированный датасет сохранён.")

        # Очистка временных файлов после токенизации, если нужно
        # ds.cleanup_cache_files()  # Раскомментируйте, если нужно очистить кэш датасета
    else:
        print("Загрузка сохранённого токенизированного датасета...")
        tokenized_ds = load_from_disk(tokenized_path)
        print(f"Токенизированный датасет загружен: {len(tokenized_ds['train'])} примеров")

    # Получение train split
    full_train = tokenized_ds['train']

    # Разделение на train и eval (90/10) с использованием HF Dataset method
    train_test = full_train.train_test_split(
        test_size=0.1, seed=42, shuffle=True
    )
    train_dataset = train_test['train']
    eval_dataset = train_test['test']
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    print(f"Train dataset size after tokenization: {len(train_dataset)}")
    # Data collator для MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Модель для MLM
    print("Загрузка модели...")
    model = AutoModelForMaskedLM.from_pretrained("DeepPavlov/rubert-base-cased")

    # Для теста: меньше эпох (3 вместо 5)
    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        overwrite_output_dir=False,
        num_train_epochs=3 if test_mode else 8,  # Меньше для теста
        max_steps = 40000,
        per_device_train_batch_size=160,
        per_device_eval_batch_size=160,
        gradient_accumulation_steps=1,  # Минимально для стабильности с большим batch
        warmup_steps=500 if test_mode else 2000,  # Меньше для теста
        logging_steps=100,
        save_steps=500 if test_mode else 2000,  # Частые сохранения для теста
        eval_steps=500 if test_mode else 2000,
        save_total_limit=6,
        eval_strategy="steps",
        prediction_loss_only=True,  # Ускоряет eval на multi-GPU
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=8,  # Параллельная загрузка для multi-GPU
        bf16=True,  # BF16 для A100 (лучше FP16 по точности и скорости)
        report_to="tensorboard",  # Для детальных логов
        logging_dir=str(logs_dir),
        push_to_hub=False,  # Если хочешь пушить, измени на True и добавь login
        ddp_find_unused_parameters=False,
        seed=42,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # Для новых версий Transformers
    )

    # Обучение
    print("Запуск обучения...")
    trainer.train(resume_from_checkpoint=True)

    # Сохранение модели
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))
    print(f"Модель сохранена в {model_output_dir}")
    print(f"Логи TensorBoard в {logs_dir} (запустите: tensorboard --logdir {logs_dir})")


    # Опциональный пуш на HF Hub (если у тебя есть токен)
    # Получи токен на huggingface.co/settings/tokens (write access)
    # Затем раскомментируй и укажи:
    # from huggingface_hub import login
    # login(token="твой_hf_token")
    # trainer.push_to_hub("твой-username/rubertrulaw", commit_message="Fine-tuned on RusLawOD")

if __name__ == '__main__':
    main()

