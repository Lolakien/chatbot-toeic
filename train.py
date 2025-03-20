import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Tải dữ liệu
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Tiền xử lý dữ liệu
questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]

# Mã hóa câu hỏi
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions).toarray()

# Mã hóa câu trả lời
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(answers)

# Xây dựng mô hình
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(len(set(y)), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10)

# Lưu mô hình
model.save('chatbot_model.h5')
# Lưu vectorizer và label encoder
import joblib
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')