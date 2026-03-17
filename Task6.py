def biconditional(p, q):
    """Return the logical biconditional (p <-> q)."""
    # In propositional logic, p <-> q is true when p and q have the same truth value.
    return p == q


def main():
    print("p    q    p <-> q")
    for p in (False, True):
        for q in (False, True):
            print(f"{p!s:<5} {q!s:<5} {biconditional(p, q)}")


if __name__ == "__main__":
    main()
