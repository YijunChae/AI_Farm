import cv2
import glob
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk, ExifTags

# 이미지 데이터와 라벨을 불러오는 함수
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for image_file in glob.glob(os.path.join(label_folder, '*.jpg')):
                img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (128, 128))  # 이미지 크기를 128x128으로 조정
                images.append(img.flatten())  # 1차원 배열로 변환
                # 바질을 0으로, 인삼을 1로 라벨링
                labels.append(0 if label == "basil" else 1)
    return np.array(images), np.array(labels)

# 데이터셋 경로
dataset_path = './dataset'

# 이미지 데이터와 라벨 불러오기
x_data, y_data = load_images_from_folder(dataset_path)

# 피드백 3: 바질과 인삼의 라벨을 0과 1로 라벨링

# 이미지 스케일링
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

# 데이터셋 분할
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)  # 8:2로 분할

# 피드백 4: 훈련 집합과 테스트 집합 비율을 8:2로 조정

# 라벨 인코딩
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# SVM 모델 생성 및 하이퍼파라미터 튜닝
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
grid.fit(x_train, y_train_encoded)

# 최적의 하이퍼파라미터 출력
print(f"Best Parameters: {grid.best_params_}")

# 최적의 모델로 예측
best_model = grid.best_estimator_
predictions = best_model.predict(x_test)

# 정확도 계산
accuracy = np.mean(predictions == y_test_encoded)
print(f"테스트 집합에 대한 정확률은 {accuracy * 100:.2f}%입니다.")

# 이미지 파일을 선택하고 예측하는 함수
def upload_image():
    global img_display, img_path
    file_path = filedialog.askopenfilename()
    if file_path:
        img_path = file_path
        img = Image.open(file_path)

        # 피드백 2: EXIF 데이터를 확인하여 이미지가 올바른 방향으로 표시되도록 수정
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        try:
            exif = dict(img._getexif().items())
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            # Exif 정보가 없는 경우 회전하지 않음
            pass

        img = img.resize((128, 128))
        img_display = ImageTk.PhotoImage(img)
        canvas.create_image(20, 20, anchor=tk.NW, image=img_display)

def recognize_image():
    if img_path:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128)).flatten()  # 원래 크기 128x128로 조정
        # 피드백 1: 이미지 크기를 통일하여 모델의 성능 저하 방지
        img = scaler.transform([img])  # 스케일링 적용
        prediction = best_model.predict(img)
        predicted_label = le.inverse_transform(prediction)[0]
        result_label.config(text=f"이 식물은: {predicted_label}")

# GUI 생성
root = tk.Tk()
root.title("AI Farm 식물 인식기")
root.geometry("800x600")
root.configure(bg='#000000')

# 스타일 설정
theme_color = '#0ABAB5'
button_font = ('Helvetica', 18)
label_font = ('Helvetica', 18, 'bold')

style = ttk.Style()
style.configure('TButton', font=button_font, padding=10, background='#ffffff', foreground=theme_color)
style.configure('TLabel', font=label_font, background='#000000', foreground=theme_color)

# 프레임 생성
frame = tk.Frame(root, bg='#000000')
frame.pack(pady=20)

# 상단 라벨 생성
title_label = tk.Label(frame, text="식물 인식기", font=('Helvetica', 28, 'bold'), bg=theme_color, fg='#ffffff')
title_label.pack(pady=20)

# 캔버스 생성 (이미지 표시용)
canvas = tk.Canvas(frame, width=250, height=250, bg='#000000')
canvas.pack(pady=20)

# 결과 레이블
result_label = ttk.Label(frame, text="이미지를 업로드하세요", font=label_font)
result_label.pack(pady=10)

# 버튼 생성
button_frame = tk.Frame(frame, bg='#000000')
button_frame.pack(pady=20)

upload_button = tk.Button(button_frame, text="이미지 업로드", command=upload_image, width=25, height=2, bg='#ffffff', fg=theme_color, font=button_font)
upload_button.grid(row=0, column=0, padx=10, pady=10)

recognize_button = tk.Button(button_frame, text="인식 시작", command=recognize_image, width=25, height=2, bg='#ffffff', fg=theme_color, font=button_font)
recognize_button.grid(row=0, column=1, padx=10, pady=10)

exit_button = tk.Button(button_frame, text="종료", command=root.quit, width=25, height=2, bg='#ffffff', fg=theme_color, font=button_font)
exit_button.grid(row=0, column=2, padx=10, pady=10)

# 전역 변수 초기화
img_display = None
img_path = None

# GUI 시작
root.mainloop()
