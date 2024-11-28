from math import floor
import numpy as np


def rgb_to_cmy(r, g, b):
    print(f"r={r:.4f}, g={g:.4f}, b={b:.4f}")
    print(f"[c, m, y] = [1, 1, 1] - [{r:.4f}, {g:.4f}, {b:.4f}]")
    c = 1 - r
    m = 1 - g
    y = 1 - b
    print(f"[c, m, y] = [{c:.4f}, {m:.4f}, {y:.4f}]")
    print("###############\n")


def cmy_to_rgb(c, m, y):
    print(f"c={c:.4f}, m={m:.4f}, y={y:.4f}")
    print(f"[r, g, b] = [1, 1, 1] - [{c:.4f}, {m:.4f}, {y:.4f}]")
    r = 1 - c
    g = 1 - m
    b = 1 - y
    print(f"[r, g, b] = [{r:.4f}, {g:.4f}, {b:.4f}]")
    print("###############\n")


def rgb_to_hsi(r, g, b):
    print(f"r={r:.4f}, g={g:.4f}, b={b:.4f}")
    i = max(r, g, b)
    print(f"I = max({r:.4f}, {g:.4f}, {b:.4f}) = {i:.4f}")
    delta = i - min(r, g, b)
    print(f"delta = {i:.4f} - min({r:.4f}, {g:.4f}, {b:.4f}) = {delta:.4f}")
    s = delta / i
    print(f"S = {delta:.4f}/{i:.4f} = {s:.4f}")
    if r == i:
        print(f"Since R = I")
        h = (g - b) / delta / 6
        print(f"H = 1/6 * ({g:.4f}-{b:.4f})/{delta:.4f} = {h:.4f}")
    elif g == i:
        print(f"Since G = I")
        h = ((b - r) / delta + 2) / 6
        print(f"H = 1/6 * (2 + ({b:.4f}-{r:.4f})/{delta:.4f}) = {h:.4f}")
    else:
        print(f"Since B = I")
        h = ((r - g) / delta + 4) / 6
        print(f"H = 1/6 * (4 + ({r:.4f}-{g:.4f})/{delta:.4f}) = {h:.4f}")
    print(f"[H, S, I] = [{h:.4f}, {s:.4f}, {i:.4f}]")
    print("##############\n")


def hsi_to_rgb(h, s, i):
    print(f"h={h:.4f}, s={s:.4f}, i={i:.4f}")
    h_prime = floor(6 * h)
    print(f"h_prime = floor(6({h:.4f})) = {h_prime:.4f}")
    f = 6 * h - h_prime
    print(f"F = 6({h:.4f}) - {h_prime:.4f} = {f:.4f}")
    p = i * (1 - s)
    print(f"P = {i:.4f}(1 - {s:.4f}) = {p:.4f}")
    q = i * (1 - s * f)
    print(f"Q = {i:.4f}(1 - {s:.4f}({f:.4f})) = {q:.4f}")
    t = i * (1 - s * (1 - f))
    print(f"T = {i:.4f}(1 - {s:.4f}(1 - {f:.4f})) = {t:.4f}")
    if h_prime == 0:
        print(f"Since h_prime = 0")
        r, g, b = i, t, p
        print(f"(R, G, B) = (I, T, P) = ({r:.4f}, {g:.4f}, {b:.4f})")
    elif h_prime == 1:
        print(f"Since h_prime = 1")
        r, g, b = q, i, p
        print(f"(R, G, B) = (Q, I, P) = ({r:.4f}, {g:.4f}, {b:.4f})")
    elif h_prime == 2:
        print(f"Since h_prime = 2")
        r, g, b = p, i, t
        print(f"(R, G, B) = (P, I, T) = ({r:.4f}, {g:.4f}, {b:.4f})")
    elif h_prime == 3:
        print(f"Since h_prime = 3")
        r, g, b = p, q, i
        print(f"(R, G, B) = (P, Q, I) = ({r:.4f}, {g:.4f}, {b:.4f})")
    elif h_prime == 4:
        print(f"Since h_prime = 4")
        r, g, b = t, p, i
        print(f"(R, G, B) = (T, P, I) = ({r:.4f}, {g:.4f}, {b:.4f})")
    elif h_prime == 5:
        print(f"Since h_prime = 5")
        r, g, b = i, p, q
        print(f"(R, G, B) = (I, P, Q) = ({r:.4f}, {g:.4f}, {b:.4f})")
    print("#############\n")

def compliment_of_rgb(r,g,b):
    

def main():
    input = [0.5833, 0.6667, 0.6]
    hsi_to_rgb(*input)


if __name__ == "__main__":
    main()
