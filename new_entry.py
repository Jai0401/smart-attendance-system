import os
import cv2
import time

def create_student_folder(student_name):
    folder_path = os.path.join("dataset", student_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def capture_snapshots(folder_path, num_snapshots=60, interval=1):
    camera = cv2.VideoCapture(0)

    print(f"Capturing {num_snapshots} snapshots at an interval of {interval} seconds...")
    
    for i in range(num_snapshots):
        _, frame = camera.read()
        snapshot_path = os.path.join(folder_path, f"{i + 1}.png")
        cv2.imwrite(snapshot_path, frame)
        print(f"Snapshot {i + 1} captured.")
        time.sleep(interval)

    camera.release()
    print("Snapshot capture complete.")

if __name__ == "__main__":
    # Get the name of the new student
    student_name = input("Enter the name of the new student: ")

    # Create a folder for the new student
    folder_path = create_student_folder(student_name)

    # Capture snapshots and store them in the student's folder
    capture_snapshots(folder_path)
