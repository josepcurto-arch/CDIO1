import os
import pickle
import logging
import sys


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(levelname)s: %(message)s",
)


def do_complex_stuff_1():
    # Simulate a complex computation
    logging.info("Starting complex_stuff_1 computation...")
    return {"result": "complex_stuff_1_result"}

def do_complex_stuff_2(result_1):
    # Simulate a complex computation
    logging.info(f"Starting complex_stuff_2 computation from {result_1['result']}...")
    return {"result": "complex_stuff_2_result"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Checkpointing and Caching Demo")
    parser.add_argument("--failure-point", choices=["before_1", "after_1", "before_2", "after_2"], help="Simulate failure at a specific point", required=True)
    args = parser.parse_args()

    if not os.path.exists("filename_1.pickle") or os.path.getsize("filename_1.pickle") == 0:
        if args.failure_point == "before_1":
            raise Exception("Simulated failure before complex_stuff_1")
        result_1 = do_complex_stuff_1()
        with open("filename_1.pickle", "wb") as f:
            pickle.dump(result_1, f)
        if args.failure_point == "after_1":
            raise Exception("Simulated failure after complex_stuff_1")
    else:
        with open("filename_1.pickle", "rb") as f:
            result_1 = pickle.load(f)
        logging.info(f"Loaded result_1 from cache: {result_1['result']}")

    if not os.path.exists("filename_2.pickle") or os.path.getsize("filename_2.pickle") == 0:
        if args.failure_point == "before_2":
            raise Exception("Simulated failure before complex_stuff_2")
        result_2 = do_complex_stuff_2(result_1)
        with open("filename_2.pickle", "wb") as f:
            pickle.dump(result_2, f)
        if args.failure_point == "after_2":
            raise Exception("Simulated failure after complex_stuff_2")
    else:
        with open("filename_2.pickle", "rb") as f:
            result_2 = pickle.load(f)
        logging.info(f"Loaded result_2 from cache: {result_2['result']}")
