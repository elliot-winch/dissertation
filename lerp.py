
def lerp(a, b, t):
    return (b - a) * t + a

def inverse_lerp(a, b, n):
    return (n - a) / (b - a)

def lerp_vector(a, b, t):
    if len(a) != len(b):
        print("Error: cannot lerp between vectors with different dimensions")
        exit()

    n = [0] * len(a)
    for i in range(0, len(a)):
        n[i] = lerp(a[i], b[i], t)

    return n
