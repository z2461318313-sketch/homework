"""
新闻文本分类 - Transformer模型
适配格式：每行 "label_id 数字1 数字2 数字3 ..."（标签已经是数字ID）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# ==================== 类别映射 ====================
# 数字ID到中文类别的映射（根据你的数据，标签已经是0-13的数字）
ID_TO_CATEGORY = {
    0: '科技',
    1: '股票',
    2: '体育',
    3: '娱乐',
    4: '时政',
    5: '社会',
    6: '教育',
    7: '财经',
    8: '家居',
    9: '游戏',
    10: '房产',
    11: '时尚',
    12: '彩票',
    13: '星座'
}
CATEGORY_TO_ID = {v: k for k, v in ID_TO_CATEGORY.items()}
NUM_CLASSES = len(ID_TO_CATEGORY)

print(f"类别映射: {ID_TO_CATEGORY}")

# ==================== 数据加载函数 ====================

def load_data(data_path, max_samples=None):
    """
    加载数据 - 适配格式：每行 "标签ID 数字1 数字2 数字3 ..."
    例如：1 9073 9073 5444 3811 4744 ...（标签ID已经是0-13的数字）
    """
    texts = []
    labels = []

    print(f"正在读取文件: {data_path}")

    # 统计跳过的行数
    skipped = 0
    valid_labels = set(range(NUM_CLASSES))  # 0-13

    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if max_samples and line_num >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            # 按空格分割
            parts = line.split()
            if len(parts) < 2:
                skipped += 1
                continue

            # 第一个是标签ID（应该是数字）
            label_str = parts[0]

            # 尝试转换为整数
            try:
                label_id = int(label_str)
                # 检查是否在有效范围内（0-13）
                if label_id in valid_labels:
                    # 后面的都是文本token
                    text = ' '.join(parts[1:])
                    labels.append(label_id)
                    texts.append(text)
                else:
                    skipped += 1
                    if skipped <= 20:  # 只打印前20个警告
                        print(f"警告：标签ID {label_id} 超出范围(0-{NUM_CLASSES-1})，第{line_num+1}行跳过")
            except ValueError:
                # 如果标签不是数字，可能是中文标签，尝试转换
                if label_str in CATEGORY_TO_ID:
                    label_id = CATEGORY_TO_ID[label_str]
                    text = ' '.join(parts[1:])
                    labels.append(label_id)
                    texts.append(text)
                else:
                    skipped += 1
                    if skipped <= 20:
                        print(f"警告：无法识别的标签 '{label_str}'，第{line_num+1}行跳过")

    print(f"成功加载 {len(texts)} 条数据")
    if skipped > 0:
        print(f"跳过 {skipped} 条无效数据")

    if len(texts) == 0:
        raise ValueError("没有成功加载任何数据，请检查文件格式！")

    # 打印标签分布
    unique, counts = np.unique(labels, return_counts=True)
    print("\n标签分布：")
    for u, c in zip(unique, counts):
        print(f"  {u}({ID_TO_CATEGORY[u]}): {c} ({c/len(labels)*100:.1f}%)")

    return texts, labels

def inspect_file_first_lines(data_path, num_lines=10):
    """检查文件的前几行，帮助调试"""
    print(f"\n=== 文件前{num_lines}行预览 ===")
    with open(data_path, 'r', encoding='utf-8') as f:
        for i in range(num_lines):
            line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) >= 2:
                print(f"行{i+1}: 标签='{parts[0]}', 文本前5个token={parts[1:6]}...")
            else:
                print(f"行{i+1}: {line.strip()[:100]}")

# ==================== 词汇表构建 ====================

def build_vocab(texts, min_freq=2, max_vocab_size=15000):
    """构建词汇表（数字token级别）"""
    from collections import Counter

    word_freq = Counter()
    for text in texts:
        tokens = text.split()
        word_freq.update(tokens)

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.most_common(max_vocab_size - 2):
        if freq >= min_freq:
            vocab[word] = len(vocab)

    print(f"\n词汇表统计:")
    print(f"  词汇表大小: {len(vocab)}")
    print(f"  总token数: {sum(word_freq.values())}")
    print(f"  唯一token数: {len(word_freq)}")
    print(f"  平均文本长度: {sum(len(t.split()) for t in texts) / len(texts):.1f} tokens")

    return vocab

def text_to_sequence(text, vocab, max_len=512):
    """将文本（数字序列）转换为索引序列"""
    tokens = text.split()
    seq = [vocab.get(t, vocab['<UNK>']) for t in tokens]

    # 截断或填充
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [vocab['<PAD>']] * (max_len - len(seq))

    return seq

# ==================== Dataset类 ====================

class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        seq = text_to_sequence(text, self.vocab, self.max_len)
        return torch.LongTensor(seq), torch.LongTensor([label])[0]

# ==================== Transformer模型 ====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.1, num_classes=NUM_CLASSES):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        if mask is None:
            mask = (x == 0)

        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)  # 全局平均池化
        output = self.classifier(x)

        return output

# ==================== 训练函数 ====================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for data, target in tqdm(dataloader, desc="Training"):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = output.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, f1, accuracy, all_preds, all_labels

# ==================== 可视化函数 ====================

def plot_training_curves(train_losses, train_f1s, val_losses, val_f1s):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_f1s, 'b-', label='Train F1')
    axes[1].plot(epochs, val_f1s, 'r-', label='Val F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro F1 Score')
    axes[1].set_title('Training and Validation F1 Score')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[ID_TO_CATEGORY[i] for i in range(NUM_CLASSES)],
                yticklabels=[ID_TO_CATEGORY[i] for i in range(NUM_CLASSES)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_class_distribution(labels, title='Class Distribution'):
    plt.figure(figsize=(12, 6))
    unique, counts = np.unique(labels, return_counts=True)
    class_names = [ID_TO_CATEGORY[i] for i in unique]

    bars = plt.bar(class_names, counts)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150)
    plt.show()

# ==================== 主程序 ====================

def main():
    # 超参数配置
    BATCH_SIZE = 512          # 根据GPU显存调整
    MAX_LEN = 256            # 最大序列长度
    EMBED_DIM = 256          # 嵌入维度
    NUM_HEADS = 8            # 注意力头数
    NUM_LAYERS = 4           # Transformer层数
    FFN_DIM = 512            # 前馈网络维度
    DROPOUT = 0.1            # Dropout比例
    LEARNING_RATE = 1e-4     # 学习率
    NUM_EPOCHS = 10          # 训练轮数
    MIN_FREQ = 2             # 最小词频
    MAX_VOCAB = 15000        # 最大词汇表大小

    print("=" * 50)
    print("新闻文本分类 - Transformer模型")
    print("=" * 50)
    print(f"设备: {device}")
    print(f"类别数: {NUM_CLASSES}")

    # 数据文件路径（请修改为你的实际路径）
    DATA_PATH = r'D:\train_set.csv'

    # 先检查文件格式
    inspect_file_first_lines(DATA_PATH, num_lines=10)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    # 如果数据量很大，可以先测试一小部分：
    # texts, labels = load_data(DATA_PATH, max_samples=10000)
    texts, labels = load_data(DATA_PATH)  # 加载全部数据

    # 可视化类别分布
    plot_class_distribution(labels, 'Training Set Class Distribution')

    # 2. 构建词汇表
    print("\n[2/6] 构建词汇表...")
    vocab = build_vocab(texts, min_freq=MIN_FREQ, max_vocab_size=MAX_VOCAB)

    # 3. 划分训练集和验证集 (8:2)
    print("\n[3/6] 划分数据集 (8:2)...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"训练集: {len(train_texts)} 样本")
    print(f"验证集: {len(val_texts)} 样本")

    # 4. 创建DataLoader
    print("\n[4/6] 创建DataLoader...")
    train_dataset = NewsDataset(train_texts, train_labels, vocab, MAX_LEN)
    val_dataset = NewsDataset(val_texts, val_labels, vocab, MAX_LEN)

    # Windows下num_workers设为0避免多进程问题
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 5. 初始化模型
    print("\n[5/6] 初始化Transformer模型...")
    model = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=EMBED_DIM,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=FFN_DIM,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 6. 训练
    print("\n[6/6] 开始训练...")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    train_losses, train_f1s = [], []
    val_losses, val_f1s = [], []
    best_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)

        # 训练
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_f1s.append(train_f1)

        # 验证
        val_loss, val_f1, val_acc, val_preds, val_true = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        # 更新学习率
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_transformer_model.pth')
            print(f"✓ 保存最佳模型 (F1: {best_f1:.4f})")

    # 7. 可视化结果
    print("\n" + "=" * 50)
    print("训练完成！生成可视化结果...")
    print("=" * 50)

    # 训练曲线
    plot_training_curves(train_losses, train_f1s, val_losses, val_f1s)

    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load('best_transformer_model.pth'))
    final_loss, final_f1, final_acc, final_preds, final_true = evaluate(model, val_loader, criterion, device)

    print(f"\n最终测试结果:")
    print(f"准确率: {final_acc:.4f}")
    print(f"Macro F1: {final_f1:.4f}")

    # 分类报告
    print("\n分类报告:")
    target_names = [ID_TO_CATEGORY[i] for i in range(NUM_CLASSES)]
    print(classification_report(final_true, final_preds, target_names=target_names))

    # 混淆矩阵
    plot_confusion_matrix(final_true, final_preds)

    print("\n完成！")
    print("生成的文件：")
    print("  - training_curves.png: 训练曲线")
    print("  - confusion_matrix.png: 混淆矩阵")
    print("  - class_distribution.png: 类别分布")
    print("  - best_transformer_model.pth: 最佳模型权重")

if __name__ == "__main__":
    main()