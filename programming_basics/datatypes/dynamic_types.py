def main(name: str) -> str:
    return "Hello " + name + "!"


if __name__ == "__main__":
    print(main("Alice"))
    print(main(b"Alice"))
    print(main(823418))
