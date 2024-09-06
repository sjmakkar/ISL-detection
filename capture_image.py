import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

cap = cv2.VideoCapture(0)  # Change the index to 0 for default camera

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))
    input("Ready? Press Enter to start...")  # Wait for user input to start

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame. Exiting...")
            break

        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == ord('q'):
            break
        elif key == ord('s'):  # Press 's' to save the frame
            cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
            print("Saved frame {} for class {}".format(counter, j))
            counter += 1

cap.release()
cv2.destroyAllWindows()
