def expression(p, q):
    """Evaluate the expression: (p or not q) and not p."""
    return (p or (not q)) and (not p)


def main():
    print("p    q    (p or not q) and not p")
    for p in (False, True):
        for q in (False, True):
            print(f"{p!s:<5} {q!s:<5} {expression(p, q)}")


if __name__ == "__main__":
    main()
