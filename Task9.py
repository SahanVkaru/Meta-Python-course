def implication(p, q, r):
    """Evaluate (p and q) -> r."""
    return (not (p and q)) or r


def main():
    print("p    q    r    (p and q) -> r")
    for p in (False, True):
        for q in (False, True):
            for r in (False, True):
                print(f"{p!s:<5} {q!s:<5} {r!s:<5} {implication(p, q, r)}")


if __name__ == "__main__":
    main()
