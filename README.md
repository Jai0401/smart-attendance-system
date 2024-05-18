# Smart Attendance System using OpenCV
## Getting Started
How to use
```    
cd smart-attendance-system
```
 - Create dataset of face images.
 ```python3 new_entry.py```
 - Extract facial embeddings.
```python3 extract_embeddings.py```
 - Train the SVM model
```python3 train_model.py --recognizer output/recognizer.joblib --le output/le.joblib```
 - Test the model
```python3 app.py```

## Prerequisites

- Python 3.5
- OpenCV
```
sudo apt-get install python-opencv
```
## Screenshots
<img width="728" alt="Screenshot 2024-03-18 at 7 14 37 PM" src="https://github.com/Jai0401/smart-attendance-system/assets/112328542/3b1e5e9c-11a6-4bd9-bbb4-7128ab477192">
<img width="728" alt="Screenshot 2024-03-18 at 7 14 52 PM" src="https://github.com/Jai0401/smart-attendance-system/assets/112328542/0b15c576-97df-43a2-a85c-69d95a7f2f3b">
<img width="766" alt="Screenshot 2024-03-18 at 7 15 14 PM" src="https://github.com/Jai0401/smart-attendance-system/assets/112328542/6eb929d7-bbae-46de-b3db-aeb1aaca0787">



