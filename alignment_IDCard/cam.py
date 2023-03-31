import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Start the camera
cap = cv2.VideoCapture(0)

# Create a window using Tkinter
root = tk.Tk()
root.title("Taking ID Card Picture")

# Create a Canvas widget for displaying the camera feed
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Define the rectangle dimensions and position
rect_width = int(320 * 1.25)
rect_height = int(200 * 1.25)
rect_x = int((640 - rect_width) / 2)
rect_y = int((480 - rect_height) / 2)

# Create a button for taking a picture
photo_button = tk.Button(root, text="Chụp", font=("Helvetica", 13), bg="green", fg="white")

# Define a function for taking a picture
def take_picture():
    # Read a frame from the camera
    ret, frame = cap.read()

    # Create a mask for the rectangle
    mask = np.zeros_like(frame)
    cv2.rectangle(mask, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 255, 255), -1)

    # Blur the frame outside the rectangle
    blur = cv2.GaussianBlur(frame, (51, 51), 0)
    output_frame = np.where(mask == np.array([255, 255, 255]), frame, blur)

    # Save the picture as a file
    cv2.imwrite("picture.png", output_frame)

# Bind the "Take Picture" button to the take_picture function
photo_button.config(command=take_picture)
photo_button.pack()

def show_picture():
    img = Image.open("picture.png")
    img.show()

# Create a button for showing the picture
show_button = tk.Button(root, text="Xem ảnh", font=("Helvetica", 13), bg="blue", fg="white", command=show_picture)
show_button.pack()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Create a mask for the rectangle
    mask = np.zeros_like(frame)
    cv2.rectangle(mask, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 255, 255), -1)

    # Draw the rectangle with a green border and a white fill
    thickness = 4
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), thickness)

    # Blur the frame outside the rectangle
    blur = cv2.GaussianBlur(frame, (51, 51), 0)
    output_frame = np.where(mask == np.array([255, 255, 255]), frame, blur)

    # Convert the OpenCV frame to a PIL image
    image = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Convert the PIL image to a Tkinter PhotoImage and display it on the canvas
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    # Update the window
    root.update()

    # Wait for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()