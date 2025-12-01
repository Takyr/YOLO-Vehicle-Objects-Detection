# Nhận diện và phân loại các phương tiện giao thông trong video bằng YOLO*
**Truong Gia Vy**

## Mô Tả
Mục tiêu của đề tài nhằm xây dựng hệ thống nhận diện phương tiện giao thông (xe 2 bánh, 4 chỗ, 7 chỗ, xe buýt, xe tải) qua camera:
<img width="1468" height="545" alt="Image" src="https://github.com/user-attachments/assets/43467ca0-b427-4ff8-a991-e913cb70ece5" />
<img width="1579" height="668" alt="Image" src="https://github.com/user-attachments/assets/326efeaf-5074-47d5-80f1-f8bc30ecee23" />
<img width="1493" height="649" alt="Image" src="https://github.com/user-attachments/assets/7c8cd8c5-b8a3-4a82-81b6-35cc6215f561" />
<img width="1475" height="659" alt="Image" src="https://github.com/user-attachments/assets/d0879d76-3de2-42fe-808a-ec6b875b7c6d" />

## Các tiêu chí đánh giá
- **Precision, Recall, True-False Positive, True-False Negative, mAP, IoU, Confidence Score**
<img width="1236" height="523" alt="Image" src="https://github.com/user-attachments/assets/1605d816-d920-426b-9c8b-5cd2248e45fa" />
<img width="994" height="517" alt="Image" src="https://github.com/user-attachments/assets/b646f39b-decb-47b7-89e0-aa8284da5253" />
<img width="1402" height="643" alt="Image" src="https://github.com/user-attachments/assets/6f7d8514-6bb0-493d-85c1-312c33747558" />
<img width="1044" height="742" alt="Image" src="https://github.com/user-attachments/assets/96a3533d-b2d6-4385-bc6c-dbdbd4408e1b" />

## Môi trường thực nghiệm
- **Anaconda**: Nền tảng cài đặt đề tài
- **Python**: Thư viện chính của đề tài
- **Roboflow**: Tổng hợp và phân tích tập dữ liệu
- **JupyterHub**: Xây dựng và huấn luyện mô hình
- **Streamlit**: Hiển thị kết quả mô hình
<img width="477" height="479" alt="Image" src="https://github.com/user-attachments/assets/91ad22fc-28dc-41f7-8646-d00cbd32bb28" />

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
<img width="815" height="414" alt="Image" src="https://github.com/user-attachments/assets/1c12fe59-c780-4703-bc84-4459cf737e19" />

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
