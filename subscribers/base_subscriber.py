from redis import StrictRedis

import settings


class BaseSubscriber:
    def __init__(self, __stream_name__):
        if not __stream_name__:
            raise ValueError("An stream name must be provided")
        self.__stream_name__ = __stream_name__
        self.redis = StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)

    def subscribe(self):
        print('\n\nSubscribing to channel {}\n\n'.format(self.__stream_name__))
        ps = self.redis.pubsub()
        ps.subscribe(self.__stream_name__)
        for message in ps.listen():
            if message['type'] == 'message':
                self.process_message(message['data'])

    def process_message(self, message):
        raise NotImplementedError('Method must be implemented by subclasses')