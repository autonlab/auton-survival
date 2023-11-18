import logging

LOGGER_NAME = "auton_survival"


def configure_logging():
    logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())
