import codecs
import json, cv2, datetime
from cv2 import cvtColor
from cv2 import imshow
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image
import pytesseract
from pytesseract import image_to_string
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)
from pydantic import BaseModel


def formatArabicSentences(sentences):
    # formatedSentences = arabic_reshaper.reshape(sentences)
    # return get_display(formatedSentences)
    return sentences


# import the necessary packages
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


def scan(path):
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    image = path
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # edged = cv2.Canny(gray, 75, 200)
    edged = cv2.Canny(gray, 150, 200)
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    # show the original and scanned images
    newimg = cv2.resize(warped, (1000, 630))
    return newimg


class Birthday(BaseModel):
    full = "11/11/2005"
    year = 2005
    month = 11
    day = 11


class Person(BaseModel):
    id = "1"
    name = "طارق عبدالرحيم محمد"
    age = 16
    is_adult = False
    gender = "ذكر"
    religion = "مسلم"
    job = "طالب"
    province = "الجيزة"
    picture = "tarekturbo.png"
    gray_picture = "tarekturbogray.png"
    address = ""
    birth: Birthday


class Front:
    def __init__(self, image) -> None:
        self.image = scan(image)

    def __treshhold(self):
        self.thresh = self.gray
        cv2.threshold(self.thresh, 95, 255, cv2.THRESH_BINARY)

    def __split_images(self):
        self.picture = self.image[50:350, 50:275]
        self.name = self.thresh[150:310, 400:1000]
        self.address = self.thresh[300:450, 400:1000]
        self.id = self.thresh[500:560, 400:1000]

    def __images_to_string(self):
        self.person.name = formatArabicSentences(image_to_string(self.name, lang="ara"))
        self.person.address = formatArabicSentences(
            image_to_string(self.address, lang="ara")
        )
        self.person.id = "".join(image_to_string(self.id, lang="ara_number"))
    def __store_picture(self):
        img = self.picture
        path = f"images/{self.person.id}.jpg"
        print(path)
        cv2.imwrite(path,img)
        print("image Stored !")
    def __calculate_birthday(self):
        if self.person.id[0] == "2":
            self.person.birth.year = "19" + self.person.id[1:3]
        else:
            self.person.birth.year = "20" + self.person.id[1:3]
        self.person.birth.month = self.person.id[3:5]
        self.person.birth.day = self.person.id[5:7]
        self.person.birth.full = f"{self.person.birth.day}/{self.person.birth.month}/{self.person.birth.year}"

    def __calculate_age(self):
        today = datetime.date.today()
        self.person.age = (
            today.year
            - int(self.person.birth.year)
            - (
                (today.month, today.day)
                < (int(self.person.birth.month), int(self.person.birth.day))
            )
        )
        if self.person.age >= 18:
            self.person.age = True
    def __calculate_gender(self):
        gender = self.person.id[12:13]
        if int(gender) % 2 == 0:
            self.person.gender = "انثي"
        else:
            self.person.gender = "ذكر"

    def __load_provinces(self):
        return json.load(codecs.open("provinces.json","r","utf-8"))

    def __calculate_province(self):
        id = self.person.id[7:9]
        if id[0] == "0":
            id = id[1]
        print(id)
        if len(id) < 35:
            self.person.province = self.__load_provinces()[int(id)]
        else:
            self.person.province = "أجنبي"

    def process(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.__treshhold()
        self.person = Person(birth=Birthday())
        self.__split_images()
        self.__images_to_string()
        self.person.id = self.person.id.replace("\n", "")
        print(self.person.id +" IS the id")
        self.__calculate_gender()
        self.__calculate_birthday()
        self.__calculate_age()
        self.__calculate_province()
        self.__store_picture()
        formatArabicSentences(self.person.address)
        formatArabicSentences(self.person.province)
        formatArabicSentences(self.person.gender)
        return self.person


class Back:
    def __init__(self, image, person: Person) -> None:
        self.image = scan(image)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.threshhold = self.gray
        self.person = person
        cv2.threshold(self.threshhold, 145, 255, cv2.THRESH_BINARY)

    def shot(self, dim1, dim2, dim3, dim4):
        area = (dim1, dim2, dim3, dim4)
        cropped_img = self.pil_image.crop(area)
        text = image_to_string(cropped_img, lang="ara")
        return formatArabicSentences(text)

    def process(self):
        self.pil_image = Image.fromarray(self.threshhold)
        self.person.job = self.shot(230, 70, 820, 140)
        self.person.religion = self.shot(480, 180, 760, 260)
        return self.person
def data_uri_to_cv2_img(uri):
    import base64
    im_bytes = base64.b64decode(uri)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    # encoded_data = uri.split(',')[1]
    # nparr = np.fromstring(encoded_data.decode('base64'), np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
from flask import Flask,request,jsonify, send_file
app = Flask(__name__)
@app.route("/reader",methods=["GET","POST"])
def reader():
    if request.method == "POST":
        payload = request.get_json()
        if payload.get("front") !=None:
            person = Front(data_uri_to_cv2_img(payload.get("front"))).process()
            person.picture = request.root_url + f"picrute/{person.id}"
            person.gray_picture = person.picture + "/gray"
            if payload.get("back") != None:
                person = Back(payload.get("back"),person).process()
            return person.json(),{"Content-Type":"application/json; charset=utf-8"}
        elif payload.get("back") != None:
            person = Back(payload.get("back"),Person())
            return person.json(),{"Content-Type":"application/json; charset=utf-8"}
        else:
            return jsonify({"success":False,"error":"front image and back image are required !"})
@app.route("/picrute/<id>/")
def picture(id):
    return send_file(f"images/{id}.jpg",mimetype="image/jpg")
# @app.route("/picrute/<id>/gray")
# def gray(id):
#     gray= cv2.imread(f"picrute/{id}.jpg")
#     gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
#     return gray,{"Content-Type":"image/jpg"}
@app.errorhandler(500)
def error_500(err):
    return jsonify({"success":False,"error":"An error occured while processing your id!"})
@app.errorhandler(400)
def error_400(err):
    return jsonify({"success":False,"error":"it seems like you sent an invaild request !"})
if __name__ == "__main__":
    app.run(debug=True)