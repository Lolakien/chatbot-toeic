import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Mở file intents.json với mã hóa UTF-8 để đọc dữ liệu
with open('formatted_data1.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []  # Danh sách chứa tất cả các từ
tags = []       # Danh sách chứa các nhãn (tags)
xy = []        # Danh sách chứa cặp (câu hỏi, nhãn)

# Lặp qua từng câu trong mẫu intents
for intent in intents['intents']:
    tag = intent['tag']  # Lấy nhãn của intent
    tags.append(tag)     # Thêm nhãn vào danh sách tags
    for pattern in intent['patterns']:
        # Phân tách từng từ trong câu
        w = tokenize(pattern)
        all_words.extend(w)  # Thêm các từ vào danh sách all_words
        xy.append((w, tag))  # Thêm cặp (câu hỏi, nhãn) vào danh sách xy

# Gốc hóa và chuyển đổi từng từ về chữ thường
ignore_words = ['?', '.', '!']  # Các ký tự không cần thiết
all_words = [stem(w) for w in all_words if w not in ignore_words]  # Gốc hóa từ
# Loại bỏ các từ trùng lặp và sắp xếp
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# In ra thông tin về số lượng mẫu và từ
print(len(xy), "câu hỏi")
print(len(tags), "nhãn:", tags)
print(len(all_words), "từ đã gốc hóa duy nhất:", all_words)

# Tạo dữ liệu huấn luyện
X_train = []  # Danh sách chứa dữ liệu đầu vào
y_train = []  # Danh sách chứa nhãn đầu ra
for (pattern_sentence, tag) in xy:
    # X: bag of words cho mỗi câu hỏi
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)  # Thêm bag of words vào dữ liệu đầu vào
    # y: PyTorch CrossEntropyLoss cần nhãn lớp, không phải one-hot
    label = tags.index(tag)  # Lấy chỉ số của nhãn
    y_train.append(label)    # Thêm nhãn vào dữ liệu đầu ra

X_train = np.array(X_train)  # Chuyển đổi danh sách thành mảng NumPy
y_train = np.array(y_train)  # Chuyển đổi danh sách thành mảng NumPy

# Tham số siêu
num_epochs = 1000          # Số lần lặp huấn luyện
batch_size = 8             # Kích thước lô
learning_rate = 0.001      # Tốc độ học
input_size = len(X_train[0])  # Kích thước đầu vào
hidden_size = 8            # Kích thước lớp ẩn
output_size = len(tags)    # Số lượng nhãn đầu ra
print(input_size, output_size)

# Tạo lớp dữ liệu cho Chatbot
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)  # Số mẫu trong tập dữ liệu
        self.x_data = X_train          # Dữ liệu đầu vào
        self.y_data = y_train          # Nhãn đầu ra

    # Hỗ trợ lập chỉ mục để lấy mẫu thứ i
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # Gọi len(dataset) để trả về kích thước
    def __len__(self):
        return self.n_samples

# Khởi tạo dataset và DataLoader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# Kiểm tra xem có GPU hay không
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo mô hình NeuralNet
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Hàm mất mát và bộ tối ưu hóa
criterion = nn.CrossEntropyLoss()  # Hàm mất mát
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Bộ tối ưu hóa Adam

# Huấn luyện mô hình
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)  # Chuyển dữ liệu vào GPU nếu có
        labels = labels.to(dtype=torch.long).to(device)  # Chuyển nhãn vào GPU
        
        # Bước tiến
        outputs = model(words)  # Dự đoán đầu ra từ mô hình
        loss = criterion(outputs, labels)  # Tính toán hàm mất mát
        
        # Bước lùi và tối ưu hóa
        optimizer.zero_grad()  # Đặt gradient về 0
        loss.backward()        # Tính toán gradient
        optimizer.step()       # Cập nhật trọng số
        
    # In ra thông tin về quá trình huấn luyện
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# In ra kết quả mất mát cuối cùng
print(f'mất mát cuối: {loss.item():.4f}')

# Lưu trạng thái của mô hình và các thông tin khác
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"  # Tên file lưu
torch.save(data, FILE)  # Lưu dữ liệu vào file

print(f'huấn luyện hoàn thành. Tập tin đã được lưu tại {FILE}')