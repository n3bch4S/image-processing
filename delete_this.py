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


def compliment_of_rgb(r, g, b):
    print(f"(R, G, B) = ({r:.4f}, {g:.4f}, {b:.4f})")
    R, G, B = 1 - r, 1 - g, 1 - b
    print(
        f"Compliment of (R, G, B) = (1-{r:.4f}, 1-{g:.4f}, 1-{b:.4f}) = ({R:.4f}, {G:.4f}, {B:.4f})"
    )


def show_unique(ndarray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_values, counts = np.unique(ndarray, return_counts=True)

    # Display results
    print("Unique Values:", unique_values)
    print("Counts:", counts)
    # print("#############\n")
    return unique_values, counts


def split_threshold(
    matrix: np.ndarray, threshold: int
) -> tuple[np.ndarray, np.ndarray]:
    background = matrix[matrix < threshold]
    print(f"background: {background}")
    foreground = matrix[matrix >= threshold]
    print(f"foreground: {foreground}")
    # print("################\n")
    return background, foreground


def otsu_method(matrix: np.ndarray):
    unique_values, counts = show_unique(matrix)
    mu_t = np.mean(matrix)
    print(f"mu_t = average of an image = {mu_t:.4f}\n")
    best_t = 0
    max_sigma_square = 0

    for t in range(1, np.max(matrix) + 1):
        print(f"at T = {t}")
        background, foreground = split_threshold(matrix, t)
        mu_0 = np.mean(background)
        print(f"mu_0 = average of background = {mu_0:.4f}")
        omega_0 = len(background) / counts.sum()
        print(f"omega_0 = {len(background)}/{counts.sum()} = {omega_0:.4f}")
        mu_1 = np.mean(foreground)
        print(f"mu_1 = average of foreground = {mu_1:.4f}")
        omega_1 = len(foreground) / counts.sum()
        print(f"omega_1 = {len(foreground)}/{counts.sum()} = {omega_1:.4f}")
        sigma_square = omega_0 * (mu_0 - mu_t) ** 2 + omega_1 * (mu_1 - mu_t) ** 2
        print(
            f"sigma_square = {omega_0:.4f}({mu_0:.4f}-{mu_t:.4f})^2 + {omega_1:.4f}({mu_1:.4f}-{mu_t:.4f})^2 = {sigma_square:.4f}"
        )
        if sigma_square > max_sigma_square:
            max_sigma_square = sigma_square
            best_t = t

        print("#############\n")

    print(f"Best T = {best_t}")
    print(f"at sigma_square = {max_sigma_square:.4f}")
    print("###############\n")


def main():
    input = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1],
            [1, 1, 1, 2, 2, 3],
            [3, 3, 3, 3, 3, 4],
            [4, 4, 4, 4, 4, 4],
            [4, 4, 5, 5, 5, 5],
        ]
    )
    # input = [0.2,0.4,0.6]
    otsu_method(input)


if __name__ == "__main__":
    main()
