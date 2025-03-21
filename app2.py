import random
import json
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Kiểm tra xem có GPU hay không, nếu có thì sử dụng GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mở file dữ liệu intents
with open('formatted_data1.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# Tải mô hình đã huấn luyện
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Khởi tạo mô hình NeuralNet
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()



# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Chatbot API")
# Thêm CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các frontend gọi API (có thể đổi thành ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả phương thức HTTP (GET, POST, PUT, DELETE,...)
    allow_headers=["*"],  # Cho phép tất cả headers
)
# Định nghĩa schema đầu vào
class ChatRequest(BaseModel):
    message: str

# Hàm xử lý chatbot
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."

# API endpoint để chatbot phản hồi
@app.post("/chat")
async def chat(request: ChatRequest):
    response = get_response(request.message)
    return {"response": response}

# Chạy ứng dụng FastAPI với Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
