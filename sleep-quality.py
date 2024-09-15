from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


data = pd.read_csv("hsp.csv")


def time_to_minutes(time_str):
    if pd.isna(time_str):
        return None
    hours, minutes = map(int, time_str.split(":"))
    return hours * 60 + minutes


data["Bedtime Minutes"] = data["Bedtime"].apply(time_to_minutes)
data["Wake-up Minutes"] = data["Wake-up Time"].apply(time_to_minutes)

data.drop(["Bedtime", "Wake-up Time"], axis=1, inplace=True)

data = pd.get_dummies(
    data, columns=["Physical Activity Level", "Dietary Habits"], drop_first=True
)

label_encoders = {}
for col in ["Gender", "Sleep Disorders", "Medication Usage"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

scaler = StandardScaler()
numerical_features = [
    "Age",
    "Sleep Quality",
    "Daily Steps",
    "Calories Burned",
    "Bedtime Minutes",
    "Wake-up Minutes",
]
data[numerical_features] = scaler.fit_transform(data[numerical_features])


X = data.drop(["Sleep Quality", "User ID"], axis=1)
y = data["Sleep Quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = Sequential(
    [
        Dense(64, input_dim=X_train.shape[1], activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="linear"),
    ]
)

model.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
)
history = model.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32
)

loss, mae = model.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {mae}")

import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

