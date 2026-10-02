"""python -m forecast_bust.train_qgb [--no-reassemble] [--reuse-table]"""
import argparse
import logging

from forecast_bust.pipeline_qgb import train_qgb

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-reassemble", action="store_true", help="reuse data/interim/states.nc")
    ap.add_argument("--reuse-table", action="store_true", help="reuse the existing TABLE parquet "
                    "(skips label/feature/memory/support rebuild; their contents are unchanged by it)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    train_qgb(reassemble=not a.no_reassemble, reuse_table=a.reuse_table)
