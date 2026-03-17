def implication(p, q):
    """Return the logical implication (p -> q)."""
    # In propositional logic, p -> q is equivalent to (not p) or q
    return (not p) or q


def main():
    print("p    q    p -> q")
    for p in (False, True):
        for q in (False, True):
            print(f"{p!s:<5} {q!s:<5} {implication(p, q)}")


if __name__ == "__main__":
    main()
