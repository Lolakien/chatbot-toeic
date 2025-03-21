class Chatbox {
    constructor() {
        // Khởi tạo các tham số cho Chatbox
        this.args = {
            openButton: document.querySelector('.chatbox__button'), // Nút mở chatbox
            chatBox: document.querySelector('.chatbox__support'),   // Khu vực chat
            sendButton: document.querySelector('.send__button')     // Nút gửi tin nhắn
        }

        this.state = false; // Trạng thái hiển thị của chatbox (đóng/mở)
        this.messages = []; // Mảng lưu trữ các tin nhắn trong cuộc trò chuyện
    }

    // Phương thức hiển thị và quản lý các sự kiện
    display() {
        const { openButton, chatBox, sendButton } = this.args;

        // Thêm sự kiện click cho openButton để mở/đóng chatbox
        openButton.addEventListener('click', () => this.toggleState(chatBox))

        // Thêm sự kiện click cho sendButton để gửi tin nhắn
        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        // Thêm sự kiện keyup cho ô nhập liệu để gửi tin nhắn khi nhấn phím Enter
        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox) // Gửi tin nhắn nếu phím Enter được nhấn
            }
        })
    }

    // Phương thức để chuyển đổi trạng thái của chatbox
    toggleState(chatbox) {
        this.state = !this.state; // Đảo ngược trạng thái

        // Hiện hoặc ẩn chatbox dựa trên trạng thái
        if (this.state) {
            chatbox.classList.add('chatbox--active') // Thêm lớp CSS để hiển thị chatbox
        } else {
            chatbox.classList.remove('chatbox--active') // Xóa lớp CSS để ẩn chatbox
        }
    }

    // Phương thức xử lý khi người dùng nhấn nút gửi
    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input'); // Lấy ô nhập liệu
        let text1 = textField.value; // Lấy giá trị của ô nhập liệu
        if (text1 === "") {
            return; // Không làm gì nếu ô nhập liệu trống
        }

        // Tạo đối tượng tin nhắn từ người dùng
        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1); // Thêm tin nhắn vào mảng messages

        // Gửi yêu cầu POST tới server để nhận phản hồi
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }), // Chuyển đổi tin nhắn thành định dạng JSON
            mode: 'cors', // Cho phép yêu cầu cross-origin
            headers: {
                'Content-Type': 'application/json' // Đặt tiêu đề Content-Type cho yêu cầu
            },
        })
            .then(r => r.json()) // Chuyển đổi phản hồi từ JSON
            .then(r => {
                // Tạo đối tượng tin nhắn từ bot
                let msg2 = { name: "Sam", message: r.answer };
                this.messages.push(msg2); // Thêm tin nhắn của bot vào mảng messages
                this.updateChatText(chatbox) // Cập nhật hiển thị chatbox
                textField.value = '' // Xóa ô nhập liệu
            }).catch((error) => {
                console.error('Error:', error); // In lỗi nếu có
                this.updateChatText(chatbox) // Cập nhật hiển thị chatbox
                textField.value = '' // Xóa ô nhập liệu
            });
    }

    // Phương thức cập nhật nội dung tin nhắn trong chatbox
    updateChatText(chatbox) {
        var html = ''; // Biến chứa mã HTML cho tin nhắn
        this.messages.slice().reverse().forEach(function (item, index) {
            // Đảo ngược mảng tin nhắn để hiển thị mới nhất đầu tiên
            if (item.name === "Sam") { // Nếu tin nhắn từ bot
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            } else { // Nếu tin nhắn từ người dùng
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
        });

        // Cập nhật nội dung tin nhắn trong chatbox
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html; // Đặt nội dung HTML mới cho chatbox
    }
}

// Khởi tạo một đối tượng Chatbox và hiển thị chatbox
const chatbox = new Chatbox();
chatbox.display();