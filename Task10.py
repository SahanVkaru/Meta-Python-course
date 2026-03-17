import itertools
# Logical functions
def negation(p):
 return not p
def conjunction(p, q):
 return p and q
def disjunction(p, q):
 return p or q
def implication(p, q):
 return (not p) or q
def biconditional(p, q):
 return p == q
# Evaluate logical expression (P AND Q) -> R
def evaluate_expression(p, q, r):
 return implication(conjunction(p, q), r)
print("P Q R Result")
for p, q, r in itertools.product([True, False], repeat=3):
 result = evaluate_expression(p, q, r)
 print(p, q, r, result)