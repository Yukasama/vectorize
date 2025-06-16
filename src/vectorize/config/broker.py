"""Broker configuration for Dramatiq with Redis."""

from dramatiq import set_broker
from dramatiq.brokers.redis import RedisBroker

__all__ = ["redis_broker"]


redis_broker = RedisBroker(url="redis://localhost:6379")
# redis_broker.add_middleware(AsyncIO())
set_broker(redis_broker)
