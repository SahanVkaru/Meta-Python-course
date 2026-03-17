def exclusive_disjunction(p, q):
    """Return the logical XOR of two boolean values."""
    return (p and not q) or (not p and q)


def main():
    # Print a truth table for XOR
    print("p    q    p XOR q")
    for p in (False, True):
        for q in (False, True):
            print(f"{p!s:<5} {q!s:<5} {exclusive_disjunction(p, q)}")


if __name__ == "__main__":
    main()
