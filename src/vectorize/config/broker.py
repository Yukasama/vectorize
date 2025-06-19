"""Broker configuration for Dramatiq with Redis."""

import os

import dramatiq
from dotenv import load_dotenv
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware.asyncio import AsyncIO

__all__ = ["broker"]


load_dotenv()


def _make_broker() -> dramatiq.Broker:
    url = os.environ["REDIS_URL"]
    broker = RedisBroker(url=url)
    broker.add_middleware(AsyncIO())
    return broker


broker = _make_broker()
dramatiq.set_broker(broker)
