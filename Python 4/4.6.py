def get_net_price(price,tax_rate=0.05,discount=0.1):
    return price*(1+tax_rate-discount)

print(get_net_price(100,0.1,0.2))