# Nhận diện và phân loại các phương tiện giao thông trong video bằng YOLO*
**Truong Gia Vy**

## Mô Tả
Mục tiêu của đề tài nhằm xây dựng hệ thống nhận diện phương tiện giao thông (xe 2 bánh, 4 chỗ, 7 chỗ, xe buýt, xe tải) qua camera:

## Các tiêu chí đánh giá
- **Precision, Recall, True-False Positive, True-False Negative, mAP, IoU, Confidence Score**

## Môi trường thực nghiệm
- **Anaconda**: Nền tảng cài đặt đề tài
- **Python**: Thư viện chính của đề tài
- **Roboflow**: Tổng hợp và phân tích tập dữ liệu
- **JupyterHub**: Xây dựng và huấn luyện mô hình
- **Streamlit**: Hiển thị kết quả mô hình

## Cài Đặt
Bước 1: Tạo môi trường mới trên nền tảng Anaconda với phiên bản Python 3.10
```bash
conda create -n ENV python=3.10
```
Bước 2: Kích hoạt môi trường
```bash
conda activate ENV
```
Bước 3: Di chuyển đến thư mục chứa dữ liệu (Ví dụ: Nếu dữ liệu trong thư mục ở ổ đĩa D)
```bash
cd 20133115_TruongGiaVy_Code (Version 1-May-7)
```
Bước 4: Cài đặt các thư viện cần thiết trong file requirements.txt
```bash
pip install -r requirements.txt
```
Bước 5: Cài đặt phiên bản Streamlit từ 1.39 trở lên
```bash
pip install streamlit==1.39
```
Bước 6: Cài đặt Roboflow cho user
```bash
pip3 install roboflow --user
```
Bước 7: Chạy file trên Streamlit
```bash
streamlit run 1_Demo_Model.py
```

## Cấu Trúc Dự Án
```
YOLO-Vehicle-Objects-Detection/
│
├── requirements.txt        
├── 1_Demo_Model.py     
├── yolov5     
├── deep_sort        
├── pages           
└── README.md            
```
