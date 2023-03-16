						Model Machine learning to detect 4 corner of CCCD and CMND
B1: Tải anaconda qua website: https://www.anaconda.com/
B2: Mở anaconda navigator.
B3: Launch "CMD.exe Prompt"
B4: Chạy câu lệnh: "conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch" để cài đặt torch chạy model Machine Learning detect corner của CMND và CCCD trên GPU. Nếu không cài đặt thì model sẽ chạy trên CPU và chạy rất lâu.
B5: Sau khi cài đặt torch, launch "VS Code"
B6: Cài đặt "pip install detecto" để cài đặt mô hình Detecto để chạy model Machine Learning.
B7: Chạy câu lệnh "from detecto import core, utils, visualize"
B8: Chạy câu lênh: 	
	dataset = core.Dataset('C:/Users/caotu/Downloads/train')   					#nơi chứa data để training
	model = core.Model(['top_left', 'top_right', 'bottom_left', 'bottom_right']) 		#các label tương đương 4 góc của CMND và CCCD

B9: Chạy câu lệnh: 
	losses = model.fit(dataset, epochs=20, verbose=True, learning_rate=0.001)		#epochs càng nhiều thì mô hình detect càng tốt

B10: Chạy câu lệnh: 
	model.save('/content/gdrive/MyDrive/training.pth')						#lưu model sau khi training tại đường dẫn với file training tên là "training.pth"

B11: Nếu muốn training tiếp thì chạy câu lệnh sau:
	dataset = core.Dataset('/content/gdrive/MyDrive/cccd_cmnd')					#load lại dataset
	model = core.Model.load('/content/gdrive/MyDrive/training.pth', 	
		['top_left', 'top_right', 'bottom_left', 'bottom_right'])				#load lại file training đã train ở trước và các label tương ứng 4 góc

B12: Chạy câu lệnh:
	losses = model.fit(dataset, epochs=30, verbose=True, learning_rate=0.001)		#Tiếp tục training

B13: Chạy câu lệnh:
	model.save('/content/gdrive/MyDrive/training.pth')						#Lưu model đè lên file training cũ

						Align CCCD and CMND
B1: Bật "VS Code"
B2: Mở folder "alignment ID_Card" và mở align.py
B3: Install các library cần thiết nếu chưa install:
			pip install cv2
			pip install numpy
			pip install matplotlib
			pip install detecto
B4: Bật CMD trong VS Code bằng câu lệnh "Ctrl + `"
B5: Chạy câu lệnh: "python align.py" sau đó sẽ yêu cầu "Enter the path to the image: ", sẽ đưa cái đường dẫn chứa hình ảnh cần crop và xem kết quả nó trả về.