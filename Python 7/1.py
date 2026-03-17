def conjunction(p, q):
    return p and q
print("P   Q   P AND Q")

for p in[True,False]:
    for q in [True,False]:
    
        result = conjunction(p, q)  
        print(p, q, result)