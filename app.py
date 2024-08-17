import json
import random

def generate_weather_data():
    return {
        "temp": {
            "cur": round(random.uniform(60.0, 100.0), 2)  # Temperature between 60 and 100 degrees
        },
        "feelsLike": {
            "cur": round(random.uniform(60.0, 110.0), 2)  # Feels-like temperature can be slightly higher
        },
        "pressure": random.randint(950, 1050),  # Atmospheric pressure in hPa
        "humidity": random.randint(50, 100),  # Humidity percentage
        "clouds": random.randint(0, 100),  # Cloud coverage percentage
        "visibility": random.randint(100, 10000),  # Visibility in meters
        "wind": {
            "deg": random.randint(0, 360),  # Wind direction in degrees
            "gust": round(random.uniform(0.0, 50.0), 1),  # Wind gust in m/s
            "speed": round(random.uniform(0.0, 30.0), 1)  # Wind speed in m/s
        },
        "rain": round(random.uniform(0.0, 100.0), 1) if random.random() > 0.5 else 0,  # Random chance of rain
        "snow": 0,  # Snowfall set to 0
        "conditionId": random.choice([200, 300, 500, 600, 800]),  # Random condition ID for different weather conditions
        "main": random.choice(["Rain", "Clouds", "Clear", "Storm", "Drizzle"]),
        "description": random.choice(["light rain", "heavy intensity rain", "scattered clouds", "clear sky", "thunderstorm"]),
        "icon": {
            "url": "http://openweathermap.org/img/wn/10d@2x.png",
            "raw": "10d"
        }
    }

def generate_disaster_data(num_entries):
    disaster_types = ["Flood", "Storm", "Drought", "Heatwave", "Tornado", "Hurricane"]
    disaster_data = []
    for _ in range(num_entries):
        disaster_type = random.choice(disaster_types)
        weather_data = generate_weather_data()
        disaster_entry = {
            "disasterType": disaster_type,
            "weather": weather_data
        }
        disaster_data.append(disaster_entry)
    return disaster_data

# Generate 10,000 lines of disaster data
disaster_data = generate_disaster_data(10000)

# Write the data to a JSON file
with open("disaster_data.json", "w") as f:
    json.dump(disaster_data, f, indent=4)

print("Generated 10,000 lines of disaster data in 'disaster_data.json'.")
