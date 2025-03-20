import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Tải mô hình và vectorizer
model = load_model('chatbot_model.h5')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def chatbot():
    print("Chào bạn! Tôi là chatbot AI hỗ trợ ôn luyện TOEIC.")
    while True:
        user_input = input("Bạn: ").strip()
        if user_input.lower() == 'thoát':
            print("Chatbot: Tạm biệt!")
            break
        
        # Dự đoán câu trả lời từ mô hình
        X_test = vectorizer.transform([user_input]).toarray()
        prediction = model.predict(X_test)
        response = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        
        print(f"Chatbot: {response}")

chatbot()