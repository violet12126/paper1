import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image
from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import tqdm



def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强和预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=6),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.labels = [int(f.split('-')[1].split('.')[0]) for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        return self.transform(image) if self.transform else image, label


class CNN(nn.Module):
    def __init__(self, num_classes=6, base_channels=32, dropout_rate=0.3,
                 extra_conv_layers=1, use_attention=True, kernel_size=3):
        super().__init__()
        self.features = self._make_layers(base_channels, extra_conv_layers, kernel_size)

        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(base_channels * (2 ** extra_conv_layers),
                          base_channels * (2 ** extra_conv_layers) // 8, 1),
                nn.ReLU(),
                nn.Conv2d(base_channels * (2 ** extra_conv_layers) // 8,
                          base_channels * (2 ** extra_conv_layers), 1),
                nn.Sigmoid()
            )
        else:
            self.attention = None

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * (2 ** extra_conv_layers), 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def _make_layers(self, base_channels, num_layers, kernel_size):
        layers = []
        current_channels = base_channels
        layers += [
            nn.Conv2d(3, current_channels, kernel_size, padding=1),
            nn.BatchNorm2d(current_channels),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        ]

        for _ in range(num_layers):
            layers += [
                nn.Conv2d(current_channels, current_channels * 2, kernel_size, padding=1),
                nn.BatchNorm2d(current_channels * 2),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2, 2)
            ]
            current_channels *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        if self.attention is not None:
            x = x * self.attention(x)
        return self.classifier(x)


def run_experiment(seed, num_epochs=60, visualize=False):
    seed_everything(seed)

    # 数据加载
    train_dataset = SpectrogramDataset(r'D:\aa_my_learning\my_learining\项目1\短时傅里叶时频\train_img',
                                       train_transform)
    val_dataset = SpectrogramDataset(r'D:\aa_my_learning\my_learining\项目1\短时傅里叶时频\valid_img', test_transform)
    test_dataset = SpectrogramDataset(r'D:\aa_my_learning\my_learining\项目1\短时傅里叶时频\test_img', test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 模型初始化
    model = CNN(num_classes=6, base_channels=50, extra_conv_layers=0,
                use_attention=False, dropout_rate=0.19906,kernel_size=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.000205, weight_decay=4.4696e-05)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    best_val_acc = 0.0
    best_model_state = None
    patience = 20
    no_improve = 0

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{num_epochs}',
                         bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += batch_size


            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{(predicted == labels).sum().item() / batch_size:.3f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        # 计算训练指标
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # 验证步骤
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_bar = tqdm(val_loader, desc='Validating',
                       bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', leave=False)

        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                batch_size = inputs.size(0)
                val_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += batch_size

                val_bar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'val_acc': f'{(predicted == labels).sum().item() / batch_size:.3f}'
                })

        # 计算验证指标
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # 打印epoch总结
        print(f"\nEpoch {epoch + 1:02d}/{num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            no_improve = 0
            print(f"🔥 New best validation accuracy: {val_acc:.4f}")
        else:
            no_improve += 1
            print(f"⏳ No improvement for {no_improve}/{patience} epochs")
            if no_improve >= patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break

        scheduler.step()

    # 加载最佳模型进行测试
    model.load_state_dict(best_model_state)
    test_acc, test_f1, test_recall, test_precision = evaluate_model(model, test_loader, seed, visualize)

    return test_acc, test_f1, test_recall, test_precision


def evaluate_model(model, test_loader, seed, visualize):
    model.eval()
    all_labels = []
    all_preds = []
    all_features = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            features = model.features(inputs)
            features = torch.flatten(nn.AdaptiveAvgPool2d(1)(features), start_dim=1)
            outputs = model(inputs)

            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    # 计算所有指标
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')

    if visualize:
        # t-SNE可视化（添加图例版本）
        tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
        features_2d = tsne.fit_transform(np.concatenate(all_features))

        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(all_labels)
        cmap = plt.get_cmap('tab10', len(unique_labels))  # 使用tab10颜色映射

        # 为每个类别单独绘制点并添加标签
        for label in unique_labels:
            mask = np.array(all_labels) == label
            plt.scatter(features_2d[mask, 0],
                        features_2d[mask, 1],
                        color=cmap(label),  # 根据标签值选择颜色
                        label=f'Class {label}',
                        alpha=0.6,
                        edgecolors='w')

        # 添加图例并调整布局
        plt.legend(title='Classes',
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left',
                   borderaxespad=0.)
        plt.title(f't-SNE Visualization (Seed {seed})')
        plt.tight_layout()  # 防止图例被截断
        plt.savefig(f'tsne_seed_{seed}.png', bbox_inches='tight')  # 保存完整图像
        plt.close()

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix (Seed {seed})')
        plt.savefig(f'cm_seed_{seed}.png')
        plt.close()

    return accuracy, f1, recall, precision


if __name__ == "__main__":
    num_runs = 1
    seeds = [42 + i for i in range(num_runs)]

    # 初始化所有指标的存储列表
    accuracies = []
    f1_scores = []
    recalls = []
    precisions = []

    # 创建并打开结果文件
    with open('experiment_results.txt', 'w') as f:
        f.write("========== 实验报告 ==========\n\n")
        f.write(f"运行次数: {num_runs}\n")
        f.write(f"使用的随机种子: {seeds}\n\n")

        for i, seed in enumerate(seeds):
            print(f"\n=== Running with seed {seed} ({i + 1}/{num_runs}) ===")
            f.write(f"\n=== 第 {i + 1}/{num_runs} 次运行（种子 {seed}） ===\n")

            # 获取所有指标
            acc, f1, recall, precision = run_experiment(seed,num_epochs=2, visualize=(i == num_runs - 1))

            # 存储结果
            accuracies.append(acc)
            f1_scores.append(f1)
            recalls.append(recall)
            precisions.append(precision)

            # 打印并保存详细结果
            print(f"Seed {seed} Results:")
            print(f"Test Accuracy: {acc:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"Precision: {precision:.4f}")

            f.write(f"测试准确率: {acc:.4f}\n")
            f.write(f"F1分数: {f1:.4f}\n")
            f.write(f"召回率: {recall:.4f}\n")
            f.write(f"精确率: {precision:.4f}\n")

        # 计算统计信息
        final_report = [
            "\n========== 最终统计结果 ==========",
            f"平均测试准确率: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
            f"平均F1分数: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
            f"平均召回率: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
            f"平均精确率: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
            "\n详细数据:",
            "准确率列表: " + ", ".join([f"{x:.4f}" for x in accuracies]),
            "F1分数列表: " + ", ".join([f"{x:.4f}" for x in f1_scores]),
            "召回率列表: " + ", ".join([f"{x:.4f}" for x in recalls]),
            "精确率列表: " + ", ".join([f"{x:.4f}" for x in precisions])
        ]

        # 输出到控制台和文件
        print("\n".join(final_report))
        f.write("\n".join(final_report))

    print("\n实验结果已保存到 experiment_results.txt")