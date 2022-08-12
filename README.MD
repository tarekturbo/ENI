# ENI
ENI is an api for reading Egyption National Id Cards 
it take's a base64 encoded image of front and back the id card
and return id card info in json format 
# Installation
First Install [Tesseract Executable]((https://github.com/UB-Mannheim/tesseract/wiki)) version 
while installing add Arabic tessdata 
<img src="https://user-images.githubusercontent.com/74266531/169664042-8fb1a519-3564-4852-a0e0-f840f58d219b.png" width="500" height="200" />

add [tessdata_Arabic_Numbers](https://github.com/ahmed-tea/tessdata_Arabic_Numbers/) to your tessdata folder which is normally located at ``C:\Program Files\Tesseract-OCR\tessdata``
Please install the requirements:
```
pip install -r requirements.txt
```
## Deployment
```
python app.py
```
## Request Example 
```python
import requests
url = "http://127.0.0.1:5000/reader" 
data = {"front":"base64ImageOfFrontofCard", "back":"base64ImageOfBackofCard"}
resp = requests.post(url, json=data)
print(resp.json())
```
## Response Example
```json
{"birth": {
"full": "11/11/2005", "year": "2005", "month": "11", "day": "11",
"id": "3051111xxxxxxx", "name": "Tarek", "age": 16, "is_adult": false, "gender": "Male", "religion": "Muslim", "job": "Student", "province" : "cairo", "picture": "http://127.0.0.1:5000/picrute/xxxxxxxxxxxxxxxxxxxxxxx",
"gray_picture": "https://localhost:5000/picture/xxxxxxxxxxxxxxxxxxxxxxx/gray"}}
```
    



