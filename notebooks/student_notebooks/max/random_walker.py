#!/usr/bin/env python3
#
import random
import json
import time

import client


class RandomWalker():
    def __init__(self):
        topics = {
            "target": "python/mqtt/subglobosa/target",
            "position": "python/mqtt/subglobosa/position"
        }
        self.topics = topics
        self.client = client.create_client()
        self.client.subscribe(topics["target"])
        self.client.on_message = self.on_message
        self.position = [0, 0]
        self.t = 0

    def publish(self):
        """publish current state to topic"""
        msg = json.dumps(self.position)
        self.client.publish(topic=self.topics["position"], payload=msg)
    def on_message(self, client, userdata, msg):
        # store the last msg
        self.msg = msg
        self.payload = json.loads(msg.payload.decode())
        print('msg:', self.msg, 'payload:', self.payload, 'of type', type(self.payload))

    def update(self):
        self.t += 1
        self.position[0] = self.position[0] + (random.random() - 0.5)
        self.position[1] = self.position[1] + (random.random() - 0.5)
    def main(self):
        while True:
            self.update()
            self.publish()
            time.sleep(1)


if __name__ == '__main__':
    random_walker = RandomWalker()
    random_walker.main()
