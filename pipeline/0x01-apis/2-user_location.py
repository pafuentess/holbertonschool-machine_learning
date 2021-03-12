#!/usr/bin/env python3
""" doc """

import sys
import requests as rq
import time

if __name__ == '__main__':
    url = sys.argv[1]
    payload = {'Accept': "application/vnd.github.v3+json"}
    req = rq.get(url, params=payload)

    if req.status_code == 403:
        limit = req.headers["X-Ratelimit-Reset"]
        x = (int(limit) - int(time.time())) / 60
        print("Reset in {} min".format(int(x)))

    if req.status_code == 200:
        location = req.json()["location"]
        print(location)

    if req.status_code == 404:
        print("Not found")
