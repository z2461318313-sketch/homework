import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler  # 修复GradScaler弃用问题
from tqdm import tqdm
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 修复Windows多进程报错

    # ===================== 1. 超参数设置（适配4070） =====================
    BATCH_SIZE = 4096  # 显存不足可以调小到32
    EPOCHS = 50
    LR = 0.001
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    CSV_PATH = "D:\\train.csv"  # 改成你的CSV文件路径
    USE_AMP = True  # 开启混合精度训练，4070支持

    # ===================== 2. 关键修复：读取并拆分字符串格式的心跳信号 =====================
    # 读取数据
    df = pd.read_csv(CSV_PATH)
    print("原始数据前5行：")
    print(df.head())


    # 核心修复：把heartbeat_signals列的逗号分隔字符串拆分成数值数组
    def parse_signal(signal_str):
        # 去掉字符串两端的引号（如果有的话），再按逗号拆分
        if isinstance(signal_str, str):
            signal_str = signal_str.strip('"')
            return np.array(signal_str.split(','), dtype=np.float32)
        else:
            return np.array([], dtype=np.float32)


    # 对每一行的信号进行拆分
    signals = df['heartbeat_signals'].apply(parse_signal)
    # 把列表变成二维数组
    X = np.vstack(signals.values)
    y = df['label'].values

    print(f"\n处理后信号形状：{X.shape}，标签形状：{y.shape}")
    print(f"信号数据类型：{X.dtype}")

    # 分层划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )


    # 数据集类
    class HeartbeatDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            # 适配Conv1d输入格式：(N, 1, L)
            return self.X[idx].unsqueeze(0), self.y[idx]


    # 创建DataLoader（关键修复：num_workers设为0，Windows下多进程问题直接规避）
    train_dataset = HeartbeatDataset(X_train, y_train)
    test_dataset = HeartbeatDataset(X_test, y_test)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )


    # ===================== 3. 构建1D-CNN模型 =====================
    class HeartbeatCNN(nn.Module):
        def __init__(self, input_length, num_classes):
            super(HeartbeatCNN, self).__init__()
            self.conv_layers = nn.Sequential(
                # 第一层卷积
                nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                # 第二层卷积
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                # 第三层卷积
                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            )
            # 计算卷积后的特征长度
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, input_length)
                out = self.conv_layers(dummy_input)
                flattened_size = out.numel()

            self.fc_layers = nn.Sequential(
                nn.Linear(flattened_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
            return x


    # 自动获取类别数和输入长度
    num_classes = len(np.unique(y))
    input_length = X.shape[1]
    model = HeartbeatCNN(input_length=input_length, num_classes=num_classes)

    # ===================== 4. 训练配置（GPU版） =====================
    # 自动检测设备，优先用CUDA（你的4070）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler('cuda') if USE_AMP else None  # 修复GradScaler弃用问题

    # 记录训练过程
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    # ===================== 5. 训练循环（GPU+混合精度） =====================
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for inputs, labels in pbar:
            # 数据移动到GPU
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

            if USE_AMP:
                # 混合精度训练
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 普通训练
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader.dataset)
        train_acc = correct_train / total_train
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # 测试阶段
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if USE_AMP:
                    with autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        test_acc = correct_test / total_test
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # ===================== 6. 结果可视化 =====================
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(test_loss_list, label="Test Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(test_acc_list, label="Test Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    classes = np.arange(num_classes)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # 在混淆矩阵上添加数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    # 输出分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))