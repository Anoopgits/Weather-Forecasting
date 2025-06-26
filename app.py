from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import pytz
import requests

app = Flask(__name__)

API_KEY = '8f54a19b66e586221d1ec45da54fbdae'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# Load models and encoder
rain_model = joblib.load("rain_predict_model.pkl")
temp_model = joblib.load("temp_predict_model.pkl")
hum_model = joblib.load("hum_predict_model.pkl")
le = joblib.load("wind_dir_encoder.pkl")

def deg_to_compass(wind_deg):
    wind_deg %= 360
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75), ("N", 348.75, 360)
    ]
    for point, start, end in compass_points:
        if start <= wind_deg < end:
            return point
    return "N"

def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': data['main']['humidity'],
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind'].get('deg', 0),
        'pressure': data['main']['pressure'],
        'wind_Gust_Speed': data['wind'].get('speed', 0)
    }

def predict_future(model, current_value):
    predictions = [current_value]
    for _ in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        city = request.form["city"]
        current = get_current_weather(city)
        wind_dir = deg_to_compass(current['wind_gust_dir'])

        compass_encoded = le.transform([wind_dir])[0] if wind_dir in le.classes_ else -1
        input_data = pd.DataFrame([{
            'MinTemp': current['temp_min'],
            'MaxTemp': current['temp_max'],
            'WindGustDir': compass_encoded,
            'Humidity': current['humidity'],
            'Pressure': current['pressure'],
            'Temp': current['current_temp']
        }])

        rain_prediction = rain_model.predict(input_data)[0]
        future_temp = predict_future(temp_model, current['temp_min'])
        future_hum = predict_future(hum_model, current['humidity'])

        timezone = pytz.timezone('Asia/Kolkata')
        now = datetime.now(timezone).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        hours = [(now + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

        return render_template("result.html", city=current['city'], country=current['country'],
                               weather=current, rain="Yes" if rain_prediction else "No",
                               future_temp=zip(hours, future_temp), future_hum=zip(hours, future_hum))

    return render_template("index.html")
