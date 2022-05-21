## ENI
ENI is an api for reading Egyption National Id Cards 
it take's a base64 encoded image of front and back the id card
and return id card info in json format 
#Installation
First Install [Tesseract Executable]((https://github.com/UB-Mannheim/tesseract/wiki)) version 
while installing add Arabic tessdata ![alt](https://user-images.githubusercontent.com/74266531/169664042-8fb1a519-3564-4852-a0e0-f840f58d219b.png)
add [tessdata_Arabic_Numbers](https://github.com/ahmed-tea/tessdata_Arabic_Numbers/) to your tessdata folder which is normally located at ``C:\Program Files\Tesseract-OCR\tessdata``
Please install the requirements:
```
pip install -r requirements.txt
```
#Deployment
```
python app.py
```
# Example of usage
    Request Example 
    ![carbon](https://user-images.githubusercontent.com/74266531/169664590-3ad0b1d8-a047-4672-9ba3-faddc974b0e4.png)
    Response Example
    ![carbon (2)](https://user-images.githubusercontent.com/74266531/169664659-8740cf60-5dc3-4281-9f4c-697b55498311.png)
    ``Sensitive information are hidden``

    



