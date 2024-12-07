import whisper
from transformers import pipeline
import re
from underthesea import word_tokenize

# Bước 1: Trích xuất âm thanh từ video và chuyển thành văn bản
def extract_text_from_video(video_path):
    # Tải mô hình Whisper
    model = whisper.load_model("base") 
    # có mô hình base, large và small --- > mô hình base là mô hình có kích thước trung bình , nhanh và hiệu quả, nhưng độ chính xác thấp hơn 2 cái kia 
    # Chuyển đổi video thành văn bản
    result = model.transcribe(video_path)
    return result['text']
# ==> result gồm text: văn bản chuyển từ giọng nói ,segment : thông tin về đoạn âm thanh nhận diện, language: ngôn ngữ nhận diện dc

# Bước 2: Tiền xử lý văn bản
def preprocess_text(text):
    # Chuyển tất cả thành chữ thường
    text = text.lower()

    # Loại bỏ các ký tự không cần thiết (chữ cái, chữ số, dấu câu...)
    text = re.sub(r'[^a-zA-Z0-9\u00C0-\u1EF9\s]', '', text)

    # Tách từ (Tokenize) bằng thư viện Underthesea
    words = word_tokenize(text)
    
    # Loại bỏ từ dừng (stopwords) - có thể thêm từ dừng tùy chỉnh
    stopwords = ['và', 'của', 'là', 'với', 'để', 'cái', 'các', 'này', 'có','quá','rất']
    words = [word for word in words if word not in stopwords]

    # Trả về văn bản đã được tiền xử lý
    return ' '.join(words)

# Bước 3: Phân tích cảm xúc của văn bản
def analyze_sentiment(text):
    # Tải mô hình phân tích cảm xúc Phobert cho tiếng Việt
    classifier = pipeline("sentiment-analysis", model="vinai/phobert-base")

    # Phân tích cảm xúc
    result = classifier(text)
    
    # Kiểm tra kết quả phân tích cảm xúc và lọc độ tin cậy
    sentiment_label = result[0]['label']
    confidence = result[0]['score']
    
    print(f"confidence: {confidence}")
    # Kiểm tra nếu độ tin cậy đủ cao để quyết định cảm xúc
    if confidence > 0.5:  
        return sentiment_label
    else:
        return "Uncertain"  # Trả về "Không chắc chắn" nếu độ tin cậy quá thấp

# Bước 4: Quy trình hoàn chỉnh
def process_video(video_path):
    # Trích xuất văn bản từ video
    extracted_text = extract_text_from_video(video_path)
    print(f"Văn bản được trích xuất từ video: {extracted_text}\n")

    # Tiền xử lý văn bản
    preprocessed_text = preprocess_text(extracted_text)
    print(f"Văn bản đã được tiền xử lý: {preprocessed_text}\n")

    # Phân tích cảm xúc của văn bản đã tiền xử lý
    sentiment = analyze_sentiment(preprocessed_text)
    print(f"Cảm xúc nhận được: {sentiment}")

    # Kết quả
    if sentiment == "LABEL_1":
        print("Nội dung của video là tích cực.")
    elif sentiment == "LABEL_0":
        print("Nội dung của video là tiêu cực.")
    else:
        print("Cảm xúc không chắc chắn. Cần kiểm tra lại văn bản.")

# Ví dụ sử dụng
video_path = "D:/opencvP/emotion_extract/hdang2.mp4"  
process_video(video_path)
