import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

warnings.filterwarnings('ignore')


# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


set_seed(42)

# GPU配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ 未检测到GPU，使用CPU训练")


# 定义1D CNN模型
class HeartbeatCNN(nn.Module):
    def __init__(self, input_channels, sequence_length, num_classes):
        super(HeartbeatCNN, self).__init__()

        # 卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )

        # 计算经过卷积后的特征维度
        self.feature_size = self._calculate_feature_size(sequence_length)

        # 全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def _calculate_feature_size(self, seq_len):
        """计算经过卷积池化后的特征维度"""
        # 模拟前向传播计算维度
        x = torch.randn(1, 1, seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # 全局平均池化
        x = torch.mean(x, dim=2)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # 全局平均池化替代flatten
        x = torch.mean(x, dim=2)
        x = self.classifier(x)
        return x


# 简化版模型（更稳定）
class SimpleHeartbeatCNN(nn.Module):
    def __init__(self, input_channels, sequence_length, num_classes):
        super(SimpleHeartbeatCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 计算池化后的序列长度
        conv_output_length = sequence_length // 16  # 4次池化，每次/2
        self.fc_input_size = 256 * conv_output_length

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.fc_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# 解析心跳信号
def parse_heartbeat_signal(signal_str):
    """解析心跳信号字符串为浮点数列表"""
    if isinstance(signal_str, str):
        parts = signal_str.split(',')
        values = []
        for part in parts:
            part = part.strip()
            if part:
                try:
                    values.append(float(part))
                except ValueError:
                    continue
        return values
    return []


# 数据加载和预处理函数
def load_and_preprocess_data(csv_path):
    """加载CSV数据并进行预处理"""
    # 读取数据
    df = pd.read_csv(csv_path)
    print(f"数据集形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # 处理心跳信号
    print("\n正在解析心跳信号...")
    signals_list = []
    for signal in df['heartbeat_signals']:
        parsed_signal = parse_heartbeat_signal(signal)
        signals_list.append(parsed_signal)

    # 检查信号长度
    signal_lengths = [len(s) for s in signals_list]
    max_length = max(signal_lengths)
    print(f"信号长度: 最小={min(signal_lengths)}, 最大={max_length}, 平均={np.mean(signal_lengths):.2f}")

    # 填充到相同长度
    X = np.array([sig + [0] * (max_length - len(sig)) for sig in signals_list])

    # 处理标签
    y = df['label'].values.astype(np.int64)

    print(f"\n特征形状: {X.shape}")
    print(f"类别数量: {len(np.unique(y))}")
    print(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, max_length


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    train_times = []

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)

        # 学习率调整
        if scheduler:
            scheduler.step(val_loss)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_heartbeat_model.pth')

        epoch_time = time.time() - epoch_start
        train_times.append(epoch_time)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}] '
                  f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}% | '
                  f'Time: {epoch_time:.2f}s')

    avg_epoch_time = np.mean(train_times)
    print(f"\n平均每个epoch训练时间: {avg_epoch_time:.2f}秒")

    return train_losses, val_losses, train_accs, val_accs


# 可视化结果
def visualize_results(train_losses, val_losses, train_accs, val_accs, model, test_loader, device, num_classes):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 损失曲线
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    axes[0, 0].plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[0, 1].plot(train_accs, label='Train Accuracy', color='blue', linewidth=2)
    axes[0, 1].plot(val_accs, label='Validation Accuracy', color='red', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 模型评估
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                cbar_kws={'label': 'Count'})
    axes[1, 0].set_xlabel('Predicted Label', fontsize=12)
    axes[1, 0].set_ylabel('True Label', fontsize=12)
    axes[1, 0].set_title('Confusion Matrix', fontsize=14)

    # 类别准确率
    class_acc = []
    for i in range(len(cm)):
        if cm[i].sum() > 0:
            class_acc.append(cm[i][i] / cm[i].sum() * 100)
        else:
            class_acc.append(0)

    axes[1, 1].bar(range(len(class_acc)), class_acc, color='steelblue', alpha=0.7)
    axes[1, 1].set_xlabel('Class Label', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 1].set_title('Per-class Accuracy', fontsize=14)
    axes[1, 1].set_xticks(range(len(class_acc)))
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    for i, acc in enumerate(class_acc):
        axes[1, 1].text(i, acc + 1, f'{acc:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()

    # 打印分类报告
    print("\n" + "=" * 60)
    print("分类报告:")
    print("=" * 60)
    target_names = [f'Class_{i}' for i in range(num_classes)]
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n✅ 测试集准确率: {test_accuracy * 100:.2f}%")
    print("=" * 60)

    return test_accuracy


# 可视化样本预测
def visualize_predictions(model, test_loader, device, num_samples=5):
    model.eval()
    samples_shown = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)

            for i in range(min(len(batch_x), num_samples - samples_shown)):
                plt.figure(figsize=(12, 4))
                signal = batch_x[i].cpu().numpy().squeeze()
                plt.plot(signal, color='blue', linewidth=1.5)

                color = 'green' if batch_y[i].item() == predicted[i].item() else 'red'
                status = '✓ Correct' if batch_y[i].item() == predicted[i].item() else '✗ Wrong'
                plt.title(f'Sample {samples_shown + 1}: True = {batch_y[i].item()}, '
                          f'Predicted = {predicted[i].item()} [{status}]',
                          fontsize=12, color=color)
                plt.xlabel('Time Steps', fontsize=10)
                plt.ylabel('Normalized Amplitude', fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

                samples_shown += 1
                if samples_shown >= num_samples:
                    return


# 主函数
def main():
    # ============================================================
    # 🔧 在这里替换您的CSV文件路径！！！
    # ============================================================
    csv_path = "D:\\train.csv"  # 改成您的实际文件路径
    # ============================================================

    try:
        # 加载数据
        print("📂 加载数据中...")
        X, y, scaler, seq_length = load_and_preprocess_data(csv_path)

        # 重塑数据 [batch, channels, sequence_length]
        X = X.reshape(X.shape[0], 1, X.shape[1])

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\n📊 数据划分:")
        print(f"  训练集: {X_train.shape[0]} 样本")
        print(f"  测试集: {X_test.shape[0]} 样本")
        print(f"  信号长度: {X.shape[2]} 时间步")

        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # 划分验证集
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

        # 创建数据加载器
        batch_size = 64 if torch.cuda.is_available() else 32
        num_workers = 0

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=torch.cuda.is_available())

        # 初始化模型（使用简化版）
        input_channels = 1
        num_classes = len(np.unique(y))

        # 选择模型
        use_simple_model = True  # 使用简化版模型避免维度问题
        if use_simple_model:
            model = SimpleHeartbeatCNN(input_channels, X.shape[2], num_classes).to(device)
        else:
            model = HeartbeatCNN(input_channels, X.shape[2], num_classes).to(device)

        print(f"\n🏗️ 模型信息:")
        print(f"  模型类型: {'SimpleCNN' if use_simple_model else 'ComplexCNN'}")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Batch大小: {batch_size}")
        print(f"  使用设备: {device}")

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        # 训练
        num_epochs = 50  # 减少epoch数加快训练
        print("\n" + "=" * 60)
        print("🚀 开始训练...")
        print("=" * 60)

        start_time = time.time()
        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device
        )
        total_time = time.time() - start_time
        print(f"\n⏱️ 总训练时间: {total_time:.2f}秒 ({total_time / 60:.2f}分钟)")

        # 加载最佳模型
        model.load_state_dict(torch.load('best_heartbeat_model.pth'))
        print("✅ 已加载最佳模型")

        # 可视化
        test_acc = visualize_results(train_losses, val_losses, train_accs, val_accs,
                                     model, test_loader, device, num_classes)

        print("\n📊 样本预测:")
        visualize_predictions(model, test_loader, device, num_samples=5)

        # 保存模型
        save_dict = {
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'seq_length': seq_length,
            'test_accuracy': test_acc,
            'input_shape': X.shape[1:]
        }
        torch.save(save_dict, 'heartbeat_model_complete.pth')

        print("\n💾 模型已保存:")
        print("  - best_heartbeat_model.pth (最佳模型权重)")
        print("  - heartbeat_model_complete.pth (完整模型)")

    except FileNotFoundError:
        print(f"\n❌ 找不到文件: {csv_path}")
        print("\n请修改代码中的 csv_path 变量为您的实际文件路径")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()