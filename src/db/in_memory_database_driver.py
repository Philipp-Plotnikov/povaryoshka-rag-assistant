import redis
from typing import List


class PovaryoshkaInMemoryDatabaseDriver:
    def __init__(
        self,
        host="localhost",
        port=6379,
        db=0,
    ):
        self.__client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )

    def _key(self, user_id: str) -> str:
        return f"user:{user_id}:history"

    def add(self, user_id: str, text: str):
        key = self._key(user_id)
        self.__client.rpush(key, text)

    def get(self, user_id: str) -> List[str]:
        key = self._key(user_id)
        return self.__client.lrange(key, 0, -1) # type: ignore

    def clear(self, user_id: str):
        key = self._key(user_id)
        self.__client.delete(key)
