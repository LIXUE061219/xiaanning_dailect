import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import librosa
import json

model_path = r"C:\code\PythonProject try\dialect_classifier"
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)

with open(f"{model_path}/label_map.json", "r") as f:
    label_map = json.load(f)
id_to_label = {v: k for k, v in label_map.items()}

def predict_dialect(audio_file):
    # 加载音频文件
    audio, sr = librosa.load(audio_file, sr=16000)

    # 预处理音频
    inputs = processor(
        audio,
        sampling_rate=16000,
        max_length=16000 * 10,  # 10秒音频
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).item()

    # 获取预测的方言标签
    predicted_dialect = id_to_label[predictions]

    # 获取置信度
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    confidence = probabilities[0][predictions].item()

    return {
        "predicted_dialect": predicted_dialect,
        "confidence": confidence,
        "all_probabilities": {id_to_label[i]: probabilities[0][i].item() for i in range(len(id_to_label))}
    }

if __name__ == "__main__":
    audio_file = r"C:\code\PythonProject try\wav_dailect_test\大爹细爹.wav"
    target_length = 16000 * 10
    if len(audio_file) > target_length:
        audio = audio_file[:target_length]
    else:
        audio = np.pad(audio_file, (0, max(0, target_length - len(audio_file))))
    result = predict_dialect(audio_file)
    print(f"预测方言: {result['predicted_dialect']}")
    print(f"置信度: {result['confidence']:.4f}")
    print("各类别概率分布:")
    for dialect, prob in result["all_probabilities"].items():
        print(f"  {dialect}: {prob:.4f}")