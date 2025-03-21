from flask import Flask, render_template, request, jsonify
from chat import get_response

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Route cho trang chủ
@app.get("/")
def index_get():
    return render_template("base.html")

# Route xử lý dự đoán
@app.post("/predict")
def predict():
    # Lấy dữ liệu từ yêu cầu POST
    data = request.get_json()
    text = data.get("message")

    # Kiểm tra đầu vào hợp lệ
    if not text or not isinstance(text, str):
        return jsonify({"error": "Invalid input"}), 400

    # Gọi hàm get_response từ module chat
    response = get_response(text)

    # Trả về kết quả dưới dạng JSON
    message = {"answer": response}
    return jsonify(message)

# Chạy ứng dụng
if __name__ == "__main__":
    app.run(debug=True)