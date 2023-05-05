#!/usr/bin/env python3
import json

import simpy.rt

import client

def virtual_ship(env, client):
    topics = {
        "target": "python/mqtt/subglobosa/target",
        "position": "python/mqtt/subglobosa/position"
    }
    print(env.now)
    client.subscribe(topics["position"])
    def on_message(client, userdata, msg):
        # store the last msg
        msg = msg
        payload = json.loads(msg.payload.decode())
        print('msg:', msg, 'payload:', payload, 'of type', type(payload))

    client.on_message = on_message
    while True:
        msg = json.dumps({"target": 0})
        client.publish(topics["target"], payload=msg)
        yield env.timeout(1)
        print(env.now)

if __name__ == '__main__':
    client = client.create_client()
    env = simpy.rt.RealtimeEnvironment()
    env.process(virtual_ship(env, client))
    env.run()
