import logging


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
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        #logging.basicConfig(filename='log.log', encoding='utf-8')
        self.logger = logging.getLogger('')
        self.logger.setLevel('DEBUG')

        # file handler
        fh = logging.FileHandler(filename='log.log', encoding='utf-8')
        fh.setLevel('DEBUG')

        # console handler
        ch = logging.StreamHandler()
        ch.setLevel('ERROR')

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

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


if __name__ == '__main__':
    import time
    message = 'example message'
    Logger().log(message, 'DEBUG')
    Logger().log(message, 'INFO')
    Logger().log(message, 'WARNING')
    Logger().log(message, 'ERROR')
    Logger().log(message, 'CRITICAL')
    while True:
        time.sleep(1)

