import random
import json
import torch

from model import NeuralNet  # Nhập mô hình NeuralNet từ file model.py
from nltk_utils import bag_of_words, tokenize  # Nhập các hàm trợ giúp từ nltk_utils

# Kiểm tra xem có GPU hay không, nếu có thì sử dụng GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mở file dữ liệu intents với mã hóa UTF-8
with open('formatted_data1.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)  # Đọc nội dung file JSON vào biến intents

FILE = "data.pth"  # Tên file lưu trữ mô hình
data = torch.load(FILE)  # Tải trạng thái mô hình từ file

# Lấy các tham số cần thiết từ dữ liệu đã tải
input_size = data["input_size"]         # Kích thước đầu vào
hidden_size = data["hidden_size"]       # Kích thước lớp ẩn
output_size = data["output_size"]       # Kích thước đầu ra
all_words = data['all_words']           # Danh sách các từ đã gốc hóa
tags = data['tags']                     # Danh sách các nhãn (tags)
model_state = data["model_state"]       # Trạng thái mô hình

# Khởi tạo mô hình NeuralNet với các tham số đã lấy
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)  # Tải trọng số cho mô hình
model.eval()  # Chuyển mô hình sang chế độ đánh giá (không cập nhật trọng số)

bot_name = "Sam"  # Tên của bot

# Hàm để lấy phản hồi từ chatbot
def get_response(msg):
    sentence = tokenize(msg)  # Phân tách câu thành các từ
    X = bag_of_words(sentence, all_words)  # Tạo bag of words từ câu
    X = X.reshape(1, X.shape[0])  # Định dạng lại mảng để phù hợp với đầu vào của mô hình
    X = torch.from_numpy(X).to(device)  # Chuyển đổi thành tensor và đưa vào GPU nếu có

    output = model(X)  # Dự đoán đầu ra từ mô hình
    _, predicted = torch.max(output, dim=1)  # Lấy chỉ số của nhãn được dự đoán

    tag = tags[predicted.item()]  # Lấy nhãn từ danh sách tags

    probs = torch.softmax(output, dim=1)  # Tính xác suất cho các nhãn
    prob = probs[0][predicted.item()]  # Lấy xác suất của nhãn dự đoán
    if prob.item() > 0.75:  # Nếu xác suất lớn hơn 0.75
        for intent in intents['intents']:
            if tag == intent["tag"]:  # Tìm tag trong intents
                return random.choice(intent['responses'])  # Trả về một phản hồi ngẫu nhiên từ danh sách phản hồi
    
    return "I do not understand..."  # Nếu không tìm thấy phản hồi phù hợp

# Phần chính của chương trình
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")  # Thông báo cho người dùng
    while True:
        # Nhập câu hỏi từ người dùng
        sentence = input("You: ")
        if sentence == "quit":  # Nếu người dùng nhập 'quit', thoát khỏi vòng lặp
            break

        resp = get_response(sentence)  # Lấy phản hồi từ hàm get_response
        print(resp)  # In phản hồi ra màn hình