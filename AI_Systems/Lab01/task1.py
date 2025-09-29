def or_(a: int, b: int) -> int:
    return 1 if (a or b) else 0

def and_(a: int, b: int) -> int:
    return 1 if (a and b) else 0

def xor_(a: int, b: int) -> int:
    return and_(or_(a, b), 0 if and_(a, b) else 1)

table = [(0,0),(0,1),(1,0),(1,1)]
for x1, x2 in table:
    print((x1, x2), "->", xor_(x1, x2))
