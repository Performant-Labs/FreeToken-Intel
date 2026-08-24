"""ZMQ process queues used by the frontend / engine split.

Upstream NVIDIA path: python/freetoken/utils/mp.py
Fill in: GitHub issue `engine-loop` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


class ZmqPushQueue:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def put(self, *args, **kwargs):
        unimplemented("ZmqPushQueue.put", "engine-loop")

    def get(self, *args, **kwargs):
        unimplemented("ZmqPushQueue.get", "engine-loop")

class ZmqPullQueue:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def put(self, *args, **kwargs):
        unimplemented("ZmqPullQueue.put", "engine-loop")

    def get(self, *args, **kwargs):
        unimplemented("ZmqPullQueue.get", "engine-loop")

class ZmqPubQueue:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def put(self, *args, **kwargs):
        unimplemented("ZmqPubQueue.put", "engine-loop")

    def get(self, *args, **kwargs):
        unimplemented("ZmqPubQueue.get", "engine-loop")

class ZmqSubQueue:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def put(self, *args, **kwargs):
        unimplemented("ZmqSubQueue.put", "engine-loop")

    def get(self, *args, **kwargs):
        unimplemented("ZmqSubQueue.get", "engine-loop")

class ZmqAsyncPushQueue:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def put(self, *args, **kwargs):
        unimplemented("ZmqAsyncPushQueue.put", "engine-loop")

    def get(self, *args, **kwargs):
        unimplemented("ZmqAsyncPushQueue.get", "engine-loop")

class ZmqAsyncPullQueue:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def put(self, *args, **kwargs):
        unimplemented("ZmqAsyncPullQueue.put", "engine-loop")

    def get(self, *args, **kwargs):
        unimplemented("ZmqAsyncPullQueue.get", "engine-loop")

