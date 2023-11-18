import logging

LOGGER_NAME = "auton_survival"


def configure_logging():
    logging.basicConfig()
    logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())
