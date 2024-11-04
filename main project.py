import cv2
import pytesseract
import numpy as np
import pyttsx3

# Set the path for Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the Haar cascade for license plate detection
cascade = cv2.CascadeClassifier(r"C:\Users\LENOVO\OneDrive\Desktop\python cv\project lisence\haarrus.xml")

states = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal"
}

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def extract_num(img_name):
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in nplate:
        a, b = (int(0.02 * img.shape[0]), int(0.025 * img.shape[1]))
        plate = img[y + a:y + h - a, x + b:x + w - b, :]
        
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, plate = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)

        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        stat = read[0:2]

        try:
            state_name = states[stat]
            output_text = f"CAR BELONGS TO THE STATE: {state_name}, NUMBER PLATE: {read}"
        except KeyError:
            output_text = f"State not recognized. Number Plate: {read}"

        print(output_text)
        speak(output_text)  # Speak the output text

        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), (51, 51, 255), -1)
        cv2.putText(img, read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        cv2.imshow("plate", plate)

    cv2.imshow("result", img)
    cv2.imwrite('result.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with the path to the image
extract_num(r"C:\Users\LENOVO\OneDrive\Desktop\a7sp8ji_high-security-number-plates_625x300_12_October_18.webp")