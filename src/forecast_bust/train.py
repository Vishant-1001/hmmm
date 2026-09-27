"""python -m forecast_bust.train [--no-reassemble]"""
import argparse
import logging

from forecast_bust.pipeline import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-reassemble", action="store_true", help="reuse data/interim/states.nc")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    train(reassemble=not a.no_reassemble)
