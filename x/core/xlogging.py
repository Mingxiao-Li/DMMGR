import logging
from x.core.registry import registry


@registry.register_logger(name="logger")
class XLogger(logging.Logger):
    def __init__(
        self,
        name: str,
        level: int,
        filename: str = None,
        filemode: str = "a",
        stream=None,
        format_str: str = None,
        dataformat: str = None,
        style: str = "%",
    ):
        super().__init__(name, level)
        if filename is not None:
            handler = logging.FileHandler(filename, filemode)
        else:
            handler = logging.StreamHandler(stream)

        self._formatter = logging.Formatter(format_str, dataformat, style)
        handler.setFormatter((self._formatter))
        super().addHandler(handler)

    def add_filehandler(self, log_filename):
        filehandler = logging.FileHandler(log_filename)
        filehandler.setFormatter(self._formatter)
        self.addHandler(filehandler)


logger = XLogger(
    name="logger", level=logging.INFO, format_str="%(asctime)s- %(message)s"
)
