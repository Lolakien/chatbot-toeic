import torch
import torch.nn as nn  # Nhập thư viện PyTorch và các lớp mạng nơ-ron

# Định nghĩa lớp NeuralNet kế thừa từ nn.Module
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # Khởi tạo lớp cha (nn.Module)
        super(NeuralNet, self).__init__()

        # Định nghĩa các lớp trong mạng nơ-ron
        self.l1 = nn.Linear(input_size, hidden_size)  # Lớp đầu vào đến lớp ẩn đầu tiên
        self.l2 = nn.Linear(hidden_size, hidden_size) # Lớp ẩn đầu tiên đến lớp ẩn thứ hai
        self.l3 = nn.Linear(hidden_size, num_classes)  # Lớp ẩn thứ hai đến lớp đầu ra

        self.relu = nn.ReLU()  # Hàm kích hoạt ReLU

    # Phương thức tiến (forward) để xác định cách dữ liệu đi qua mạng
    def forward(self, x):
        out = self.l1(x)           # Đưa dữ liệu qua lớp 1
        out = self.relu(out)       # Áp dụng hàm kích hoạt ReLU
        out = self.l2(out)         # Đưa dữ liệu qua lớp 2
        out = self.relu(out)       # Áp dụng hàm kích hoạt ReLU
        out = self.l3(out)         # Đưa dữ liệu qua lớp 3 (lớp đầu ra)
        
        # Không áp dụng hàm kích hoạt và softmax ở cuối
        return out  # Trả về đầu ra của lớp cuối cùng
    

    
    #File này định nghĩa mô hình mạng nơ-ron cho chatbot 
    # hoặc ứng dụng xử lý ngôn ngữ tự nhiên (NLP). 
    # Nó sẽ được sử dụng trong quá trình huấn luyện và suy diễn (inference) 
    # để phân loại câu hỏi hoặc tin nhắn từ người dùng thành các nhãn tương ứng, 
    # qua đó trả về phản hồi phù hợp.