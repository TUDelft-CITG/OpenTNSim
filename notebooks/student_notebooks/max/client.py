#!/usr/bin/env python3
import random
import time

import paho.mqtt.client


def create_client():
    """create a new client"""
    broker = 'broker.emqx.io'
    port = 1883
    topic = "python/mqtt/test-notebook"
    def on_connect(*args, **kwargs):
        print('connect', args, kwargs)

    def on_message(client, userdata, msg):
        print(f"Received `{msg.payload.decode()}` from topic: `{msg.topic}` at t:`{time.time()}`")

    client_id = f'python-mqtt-{random.randint(0, 1000)}'
    print('client_id', client_id)
    client = paho.mqtt.client.Client(client_id=client_id)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port)
    client.subscribe(topic)
    client.loop_start()
    return client
