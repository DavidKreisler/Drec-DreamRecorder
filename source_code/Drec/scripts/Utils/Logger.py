import logging
import os.path


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Logger(metaclass=SingletonMeta):
    def __init__(self):
        # Clear existing handlers from the root logger
        #for handler in logging.root.handlers[:]:
        #    logging.root.removeHandler(handler)

        self.logger = logging.getLogger('Drec')
        self.logger.setLevel('DEBUG')

        self.logger.propagate = False

        # file handler
        self.fh = logging.FileHandler(filename='log.log', encoding='utf-8')
        self.fh.setLevel('DEBUG')

        # console handler
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.ERROR)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)

        self.log_idx = 0

    def log(self, message: str, level: str = 'DEBUG'):
        mes = f'{self.log_idx}: {message}'
        if level.upper() == 'DEBUG':
            self.logger.debug(mes)
        elif level.upper() == 'INFO':
            self.logger.info(mes)
        elif level.upper() == 'WARNING':
            self.logger.warning(mes)
        elif level.upper() == 'ERROR':
            self.logger.error(mes)
        elif level.upper() == 'CRITICAL':
            self.logger.critical(mes)
        self.log_idx += 1

    def close(self):
        """Close all handlers to release the log file."""
        handlers = self.logger.handlers[:]  # Copy the list of handlers
        for handler in handlers:
            self.logger.removeHandler(handler)
            handler.close()  # Close each handler properly

if __name__ == '__main__':
    message = 'example message'
    Logger().log(message, 'DEBUG')
    Logger().log(message, 'INFO')
    Logger().log(message, 'WARNING')
    Logger().log(message, 'ERROR')
    Logger().log(message, 'CRITICAL')

