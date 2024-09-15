from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

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

plt.ion()

train_losses = []
val_losses = []

fig, ax = plt.subplots(figsize=(10, 5))

for epoch in range(100):
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=1,
        batch_size=32,
        verbose=0,
    )

    train_losses.append(history.history["loss"][0])
    val_losses.append(history.history["val_loss"][0])

    ax.clear()
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Training and Validation Loss Over Epochs")
    plt.pause(0.1)

plt.ioff()
plt.show()

loss, mae = model.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {mae}")
