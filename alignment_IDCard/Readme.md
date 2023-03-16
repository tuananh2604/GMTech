<div align="center">Model Machine learning to detect 4 corner of CCCD and CMND</div>
B1: Tải anaconda qua website: https://www.anaconda.com/</br>
B2: Mở anaconda navigator.</br>
B3: Launch "CMD.exe Prompt"</br>
B4: Chạy câu lệnh: "conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch" để cài đặt torch chạy model Machine Learning detect corner của CMND và CCCD trên GPU. Nếu không cài đặt thì model sẽ chạy trên CPU và chạy rất lâu.</br>
B5: Sau khi cài đặt torch, launch "VS Code"</br>
B6: Cài đặt "pip install detecto" để cài đặt mô hình Detecto để chạy model Machine Learning.</br>
B7: Chạy câu lệnh "from detecto import core, utils, visualize"</br>
B8: Chạy câu lênh:</br>
* dataset = core.Dataset('C:/Users/caotu/Downloads/train')   					#nơi chứa data để training</br>
* model = core.Model(['top_left', 'top_right', 'bottom_left', 'bottom_right']) 		#các label tương đương 4 góc của CMND và CCCD</br>

B9: Chạy câu lệnh:</br>
* losses = model.fit(dataset, epochs=20, verbose=True, learning_rate=0.001)		#epochs càng nhiều thì mô hình detect càng tốt</br>

B10: Chạy câu lệnh:</br>
* model.save('/content/gdrive/MyDrive/training.pth')						#lưu model sau khi training tại đường dẫn với file training tên là "training.pth"</br>

B11: Nếu muốn training tiếp thì chạy câu lệnh sau:</br>
* dataset = core.Dataset('/content/gdrive/MyDrive/cccd_cmnd')					#load lại dataset</br>
* model = core.Model.load('/content/gdrive/MyDrive/training.pth',</p>
		['top_left', 'top_right', 'bottom_left', 'bottom_right'])				#load lại file training đã train ở trước và các label tương ứng 4 góc</br>

B12: Chạy câu lệnh:</br>
* losses = model.fit(dataset, epochs=30, verbose=True, learning_rate=0.001)		#Tiếp tục training</br>

B13: Chạy câu lệnh:</br>
* model.save('/content/gdrive/MyDrive/training.pth')						#Lưu model đè lên file training cũ</br>

<div align="center">Align CCCD and CMND</div>
B1: Bật "VS Code"</br>
B2: Mở folder "alignment ID_Card" và mở align.py</br>
B3: Install các library cần thiết nếu chưa install:</br>
pip install cv2</br>
pip install numpy</br>
pip install matplotlib</br>
pip install detecto</br>
B4: Bật CMD trong VS Code bằng câu lệnh "Ctrl + `"</br>
B5: Chạy câu lệnh: "python align.py" sau đó sẽ yêu cầu "Enter the path to the image: ", sẽ đưa cái đường dẫn chứa hình ảnh cần crop và xem kết quả nó trả về.</br>
