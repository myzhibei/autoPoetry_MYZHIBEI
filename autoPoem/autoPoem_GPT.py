import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 定义模型


class PoetryModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(PoetryModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size,
                            num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        attention_weights = torch.softmax(self.attention(output), dim=1)
        attended_output = torch.sum(attention_weights * output, dim=1)
        output = self.fc(attended_output)
        return output

# 定义数据集类


class PoetryDataset(Dataset):
    def __init__(self, poems):
        self.poems = poems

    def __len__(self):
        return len(self.poems)

    def __getitem__(self, index):
        poem = self.poems[index]
        return poem

# 加载数据集


def load_dataset(data_file):
    data = np.load(data_file, allow_pickle=True)
    poems = data['data']
    ix2word = data['ix2word'].item()
    word2ix = data['word2ix'].item()
    return poems, ix2word, word2ix

# 训练模型


def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = batch.to(device)
        target = batch[:, 1:].contiguous().view(-1).to(device)
        output = model(batch[:, :-1])
        loss = criterion(output.view(-1, len(word2ix)), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# 测试模型


def test_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            target = batch[:, 1:].contiguous().view(-1).to(device)
            output = model(batch[:, :-1])
            loss = criterion(output.view(-1, len(word2ix)), target)
            total_loss += loss.item()

    return total_loss / len(dataloader)

# 生成诗句


def generate_poem(model, start_sentence, word2ix, ix2word, device, max_length=125, temperature=1.0):
    model.eval()
    start_words = [word2ix.get(word, word2ix['<unk>'])
                   for word in start_sentence]
    test_inputs = torch.tensor(
        [word2ix['<s>']], dtype=torch.long).unsqueeze(0).to(device)
    generated_poem = start_words.copy()

    for _ in range(max_length):
        outputs = model(test_inputs)
        predicted = torch.softmax(outputs.squeeze() / temperature, dim=1)
        word = torch.multinomial(predicted, num_samples=1).item()

        if word == word2ix['</s>']:
            break

        generated_poem.append(word)
        test_inputs = torch.tensor(
            [word], dtype=torch.long).unsqueeze(0).to(device)

    generated_poem = [ix2word[word] for word in generated_poem]
    generated_poem = ''.join(generated_poem)

    return generated_poem

# 生成藏头诗


def generate_acrostic_poem(model, start_sentence, word2ix, ix2word, device, max_length=125, temperature=1.0):
    model.eval()
    start_words = [word2ix.get(word, word2ix['<unk>'])
                   for word in start_sentence]
    test_inputs = torch.tensor(
        [word2ix['<s>']], dtype=torch.long).unsqueeze(0).to(device)
    generated_poem = start_words.copy()

    for i in range(max_length):
        outputs = model(test_inputs)
        predicted = torch.softmax(outputs.squeeze() / temperature, dim=1)

        if i < len(start_words):
            word = start_words[i]
        else:
            _, word = torch.max(predicted, 1)
            word = word.item()

        if word == word2ix['</s>']:
            break

        generated_poem.append(word)
        test_inputs = torch.tensor(
            [word], dtype=torch.long).unsqueeze(0).to(device)

    generated_poem = [ix2word[word] for word in generated_poem]
    generated_poem = ''.join(generated_poem)

    return generated_poem

# 保存模型


def save_model(model, save_dir):
    torch.save(model.state_dict(), save_dir)

# 加载模型


def load_model(model, save_dir):
    model.load_state_dict(torch.load(save_dir))

# 主函数


def main():
    # 设置超参数
    input_size = len(word2ix)
    hidden_size = 256
    num_layers = 2
    dropout = 0.5
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = './model/poetry_model.pth'

    # 创建模型实例
    model = PoetryModel(input_size, hidden_size,
                        num_layers, dropout).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)

    # 创建数据集和数据加载器
    dataset = PoetryDataset(poems)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)

    # 开始训练
    for epoch in range(num_epochs):
        train_loss = train_model(
            model, dataloader, optimizer, criterion, device)
        test_loss = test_model(model, dataloader, criterion, device)

        lr_scheduler.step()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # 保存最优模型
    save_model(model, save_dir)

    # 加载最优模型进行测试
    best_model = PoetryModel(input_size, hidden_size,
                             num_layers, dropout).to(device)
    load_model(best_model, save_dir)
    best_model.eval()

    test_loss = test_model(best_model, dataloader, criterion, device)
    generated_poem = generate_poem(
        best_model, "湖光秋月两相和", word2ix, ix2word, device)
    generated_poem_head = generate_acrostic_poem(
        best_model, "湖光秋月两相和", word2ix, ix2word, device)

    print(f"Best Model Test Loss: {test_loss:.4f}")
    print(f"Generated Poem: {generated_poem}")
    print(f"Generated Acrostic Poem: {generated_poem_head}")


if __name__ == '__main__':
    # 加载数据集
    poems, ix2word, word2ix = load_dataset('./data/tang.npz')

    # 运行主函数
    main()
