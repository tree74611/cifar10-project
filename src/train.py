# 训练脚本（简化版）
import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.data_loader import get_dataloaders
from src.models import get_model
from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss/total, correct/total

def test(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return running_loss/total, correct/total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='simple_cnn')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--save-dir', default='checkpoints')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    model = get_model(name=args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter()

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        print(f'Epoch {epoch}: train_acc={train_acc:.4f} test_acc={test_acc:.4f}')
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/test', test_acc, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model_state': model.state_dict(), 'acc': best_acc}, os.path.join(args.save_dir, 'best.pth'))
    writer.close()

if __name__ == '__main__':
    main()
