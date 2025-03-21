import numpy as np  # Nhập thư viện NumPy để xử lý mảng
import nltk  # Nhập thư viện NLTK để xử lý ngôn ngữ tự nhiên
# nltk.download('punkt')  # Tải bộ dữ liệu phân tách câu (chỉ cần chạy một lần)
from nltk.stem.porter import PorterStemmer  # Nhập lớp PorterStemmer để gốc hóa từ
stemmer = PorterStemmer()  # Khởi tạo đối tượng PorterStemmer

def tokenize(sentence):
    """
    Phân tách câu thành mảng các từ/tokens.
    Một token có thể là một từ, ký tự dấu câu, hoặc số.
    """
    return nltk.word_tokenize(sentence)  # Sử dụng NLTK để phân tách câu

def stem(word):
    """
    Gốc hóa từ = tìm dạng gốc của từ.
    Ví dụ:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())  # Chuyển từ về chữ thường và gốc hóa

def bag_of_words(tokenized_sentence, words):
    """
    Trả về mảng bag of words:
    1 cho mỗi từ đã biết có trong câu, 0 nếu không có.
    Ví dụ:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # Gốc hóa từng từ trong câu
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    # Khởi tạo mảng bag với 0 cho mỗi từ
    bag = np.zeros(len(words), dtype=np.float32)
    
    # Kiểm tra từng từ trong danh sách words
    for idx, w in enumerate(words):
        if w in sentence_words:  # Nếu từ có trong câu
            bag[idx] = 1  # Đặt giá trị 1 tại vị trí tương ứng

    return bag  # Trả về mảng bag of words