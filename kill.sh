#!/usr/bin/env bash
ps -ef|grep -E '_api.py|_8000.py|_8001.py|_8002.py|_8003.py|_8004.py|_8005.py'|awk '{print $2}'| xargs kill -9