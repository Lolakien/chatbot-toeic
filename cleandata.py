import json
import re

def clean_json_file(input_file, output_file):
    """
    Làm sạch các ký tự đặc biệt khỏi tệp JSON.

    Args:
        input_file (str): Đường dẫn đến tệp JSON đầu vào.
        output_file (str): Đường dẫn đến tệp JSON đầu ra đã được làm sạch.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        cleaned_data = clean_data(data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

        print(f"Đã làm sạch dữ liệu và lưu vào {output_file}")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp {input_file}")
    except json.JSONDecodeError:
        print(f"Lỗi: Tệp {input_file} không phải là JSON hợp lệ")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

def clean_data(data):
    """
    Làm sạch các ký tự đặc biệt khỏi dữ liệu JSON.

    Args:
        data (dict or list): Dữ liệu JSON cần làm sạch.

    Returns:
        dict or list: Dữ liệu JSON đã được làm sạch.
    """
    if isinstance(data, dict):
        return {clean_string(key): clean_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_data(item) for item in data]
    elif isinstance(data, str):
        return clean_string(data)
    else:
        return data

def clean_string(text):
    """
    Làm sạch các ký tự đặc biệt khỏi chuỗi.

    Args:
        text (str): Chuỗi cần làm sạch.

    Returns:
        str: Chuỗi đã được làm sạch.
    """
    # Loại bỏ các ký tự không phải chữ và số, dấu cách, dấu chấm, dấu hỏi hoặc dấu gạch dưới
    text = re.sub(r'[^\w\s\.\?]', '', text)
    # Loại bỏ các khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    input_file = "formatted_data1.json"  # Tên tệp đầu vào của bạn
    output_file = "formatted_data1_cleaned.json" # Tên file sau khi làm sạch

    clean_json_file(input_file, output_file)