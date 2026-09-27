"""python -m forecast_bust.replay  (build precomputed historical replay cases)"""
import logging

from forecast_bust.replay.build import build

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
build()
