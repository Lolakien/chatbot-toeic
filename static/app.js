class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
        };
        this.state = false; // Trạng thái chatbox (mở/đóng)
        this.messages = []; // Lưu trữ tin nhắn
    }

    // Hiển thị hoặc ẩn chatbox
    display() {
        const { openButton, chatBox, sendButton } = this.args;

        // Bật/tắt chatbox khi nhấn nút mở
        openButton.addEventListener('click', () => this.toggleState(chatBox));

        // Gửi tin nhắn khi nhấn nút gửi
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        // Gửi tin nhắn khi nhấn phím Enter
        const inputField = chatBox.querySelector('input');
        inputField.addEventListener('keyup', (event) => {
            if (event.key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    // Bật/tắt trạng thái chatbox
    toggleState(chatBox) {
        this.state = !this.state;

        // Hiển thị hoặc ẩn chatbox
        if (this.state) {
            chatBox.classList.add('chatbox--active');
        } else {
            chatBox.classList.remove('chatbox--active');
        }
    }

    // Xử lý khi nhấn nút gửi
    onSendButton(chatBox) {
        const inputField = chatBox.querySelector('input');
        const text = inputField.value.trim();

        if (text !== "") {
            // Thêm tin nhắn của người dùng vào mảng messages
            this.messages.push({ name: "User", message: text });

            // Gửi tin nhắn đến http://127.0.0.1:5000/predict
            fetch($SCRIPT_ROOT + '/predict', {
                method: 'POST',
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: text }),
            })
                .then((response) => response.json())
                .then((data) => {
                    // Thêm phản hồi từ chatbot vào mảng messages
                    this.messages.push({ name: "Bot", message: data.answer });

                    // Hiển thị tin nhắn trong chatbox
                    this.updateChatText(chatBox);

                    // Xóa nội dung trong ô nhập
                    inputField.value = '';
                })
                .catch((error) => {
                    console.error('Error:', error);
                    this.updateChatText(chatBox);
                    inputField.value = '';
                });
        }
        else
            return;
    }



    // Cập nhật nội dung chatbox
    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function (item,) {
            if (item.name === "User") {
                html += '<div class = "messages__item messages__item--operator">' + item.message + '</div>';
            }
            else
                html += '<div class = "messages__item messages__item--visitor">' + item.message + '</div>';
        });
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}

// Khởi tạo và hiển thị chatbox
const chatbox = new Chatbox();
chatbox.display();