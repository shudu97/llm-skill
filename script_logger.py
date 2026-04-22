import logging


class TtyProgressHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        try:
            with open("/dev/tty", "w") as tty:
                tty.write(f"\r\033[2K{msg}\n")
        except OSError:
            print(msg)


def setup_logger(level: int = logging.INFO) -> None:
    handler = TtyProgressHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logging.root.setLevel(level)
    logging.root.handlers.clear()
    logging.root.addHandler(handler)


setup_logger()
