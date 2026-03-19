import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


app = Flask(__name__)
sio = socketio.Server(cors_allowed_origins='*')
model = None
args = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.0
        self.error = 0.0
        self.integral = 0.0

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        self.error = self.set_point - measurement
        self.integral += self.error
        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 10
controller.set_desired(set_speed)


def process_image(image):
    image = image[10:130:4, ::2, :]
    image = np.mean(image, axis=2)
    image = (image - image.min()) / 255.0 - 0.5
    image = image.reshape(image.shape[0], image.shape[1], 1)
    return image.astype(np.float32)


def simpleModel(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    return model


@sio.on('telemetry')
def telemetry(sid, data):
    print("TELEMETRY CALLED")

    if not data:
        print("NO DATA")
        send_control(0.0, 0.0)
        return

    try:
        speed = float(data["speed"])
        imgString = data["image"]

        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        image_array = process_image(image_array)

        prediction = model.predict(image_array[None, :, :, :], batch_size=1, verbose=0)
        steering_angle = float(prediction[0][0])

        # testowo stały gaz, żeby auto ruszyło
        throttle = 0.30

        steering_angle = float(np.clip(steering_angle, -1.0, 1.0))

        print(
            f"speed={speed:.3f}, steering={steering_angle:.4f}, "
            f"throttle={throttle:.4f}, shape={image_array.shape}"
        )

        send_control(steering_angle, throttle)

        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save(f'{image_filename}.jpg')

    except Exception as e:
        print("Telemetry error:", e)
        send_control(0.0, 0.0)


@sio.on('connect')
def connect(sid, environ):
    print("CONNECT:", sid)
    send_control(0.0, 0.0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        },
        skip_sid=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model weights file, e.g. model.weights.h5'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Optional image folder to save run images'
    )
    args = parser.parse_args()

    model = simpleModel((30, 160, 1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    model.load_weights(args.model)
    print("Weights loaded successfully")

    if args.image_folder != '':
        print(f"Creating image folder at {args.image_folder}")
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)