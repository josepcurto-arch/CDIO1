import sys


def main(name):
    print(f"Hello, {name}!\n")


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        raise Exception(f"Runtime error executing `python {' '.join(args)}`. "
                        f"Usage: `python hello.py <name>`")
    main(sys.argv[1])
