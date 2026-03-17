def compound_proposition(p, q, r):
    """Evaluate a sample compound proposition.

    Example proposition:
        (p and (q or not r)) or ((not p) and q and r)

    This mixes conjunction, disjunction, and negation.
    """
    return (p and (q or not r)) or ((not p) and q and r)


def main():
    print("p    q    r    result")
    for p in (False, True):
        for q in (False, True):
            for r in (False, True):
                print(f"{p!s:<5} {q!s:<5} {r!s:<5} {compound_proposition(p, q, r)}")


if __name__ == "__main__":
    main()
