import os
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset, Audio, Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 设置随机种子确保结果可复现
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)
processor = Wav2Vec2Processor.from_pretrained(r"C:\wav2vec2-base-960h")
num_labels = 6
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    r"C:\wav2vec2-base-960h",
    num_labels=num_labels,
    problem_type="single_label_classification"
)
model.freeze_feature_extractor()

def load_audio_dataset(data_dir):
    audio_files = []
    labels = []
    label_map = {}

    # 构建标签映射
    for i, dialect in enumerate(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, dialect)):
            label_map[dialect] = i

    # 收集音频文件和标签
    for dialect, label_id in label_map.items():
        dialect_dir = os.path.join(data_dir, dialect)
        for audio_file in os.listdir(dialect_dir):
            if audio_file.endswith('.wav'):
                audio_files.append(os.path.join(dialect_dir, audio_file))
                labels.append(label_id)

    # 划分训练集和测试集
    train_files, test_files, train_labels, test_labels = train_test_split(
        audio_files, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 创建datasets格式
    train_dataset = Dataset.from_dict({
        'audio': train_files,
        'label': train_labels
    })

    test_dataset = Dataset.from_dict({
        'audio': test_files,
        'label': test_labels
    })

    # 设置音频特性
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

    return train_dataset, test_dataset, label_map

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = processor(
        audio_arrays,
        sampling_rate=processor.feature_extractor.sampling_rate,
        max_length=16000 * 10,  # 10秒音频
        truncation=True,
        padding="max_length"
    )
    inputs["labels"] = examples["label"]
    return inputs

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}

def main():
    # 加载数据
    data_dir = r"C:\code\PythonProject try\mp3_dialect2"
    train_dataset, test_dataset, label_map = load_audio_dataset(data_dir)

    # 预处理数据
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 训练模型
    trainer.train()

    # 保存模型和处理器
    model.save_pretrained("./dialect_classifier")
    processor.save_pretrained("./dialect_classifier")

    # 评估最终模型
    metrics = trainer.evaluate()
    print(f"最终评估结果: {metrics}")

    # 保存标签映射
    import json
    with open("./dialect_classifier/label_map.json", "w") as f:
        json.dump(label_map, f)

if __name__ == "__main__":
    main()