import os
import numpy as np
import librosa
import keras
from sklearn.model_selection import train_test_split
from keras import layers, models, utils

# 超参数配置
SAMPLE_RATE = 22050
DURATION = None  # 秒
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
TARGET_TIME_STEPS = 130  # 根据音频长度和hop_length计算得到
NUM_CLASSES = 10  # 根据实际类别数修改
BATCH_SIZE = 32
EPOCHS = 50


def extract_features(file_path):
    """从音频文件中提取原始梅尔频谱图（功率谱）"""
    features = []
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        non_silent_intervals = librosa.effects.split(y, top_db=8,
                                                     frame_length=2048,
                                                     hop_length=512)

        target_length = (TARGET_TIME_STEPS - 1) * HOP_LENGTH + N_FFT

        for interval in non_silent_intervals:
            start, end = interval
            segment = y[start:end]

            # 标准化音频长度
            processed_audio = librosa.util.fix_length(segment, size=target_length)

            # 生成原始梅尔功率谱
            mel_spec = librosa.feature.melspectrogram(y=processed_audio,
                                                      sr=sr,
                                                      n_fft=N_FFT,
                                                      hop_length=HOP_LENGTH,
                                                      n_mels=N_MELS)
            features.append(mel_spec)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
    return features


def load_dataset(data_dir):
    """加载数据集并提取原始特征"""
    features = []
    labels = []

    for label_idx, class_dir in enumerate(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_dir)
        if not os.path.isdir(class_path):
            continue

        for audio_file in os.listdir(class_path):
            file_path = os.path.join(class_path, audio_file)
            try:
                specs = extract_features(file_path)
                if specs:
                    features.extend(specs)
                    labels.extend([label_idx] * len(specs))
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    # 转换为numpy数组
    X = np.array(features)
    y = utils.to_categorical(np.array(labels), num_classes=NUM_CLASSES)

    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_cnn_model(input_shape):
    """构建带规范化层的CNN模型"""
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # 添加规范化层（自动处理训练/测试差异）
        layers.Normalization(axis=-1),

        # 频率轴卷积
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # 时空联合卷积
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # 高层特征提取
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),

        layers.Dense(256, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    DATA_DIR = r"C:\code\PythonProject try\mp3_dialect"

    # 加载原始数据集
    X_train_raw, X_test_raw, y_train, y_test = load_dataset(DATA_DIR)

    # 全局分贝转换（使用训练集统计量）
    global_max = np.max(X_train_raw)
    X_train = librosa.power_to_db(X_train_raw, ref=global_max)
    X_test = librosa.power_to_db(X_test_raw, ref=global_max)

    # 添加通道维度并验证形状
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    print(f"Final train shape: {X_train.shape}, test shape: {X_test.shape}")

    # 构建模型
    model = build_cnn_model(X_train.shape[1:])

    # 适配规范化层（仅使用训练数据）
    normalization_layer = model.layers[0]
    normalization_layer.adapt(X_train)

    # 训练模型
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test)
    )

    # 评估和保存模型（保持不变）
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")
    model.save("speech_classifier.keras")