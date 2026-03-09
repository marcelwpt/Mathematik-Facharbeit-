import mpmath as mp
import numpy as np
import time

# ==============================
# EINSTELLUNGEN
# ==============================

mp.mp.dps = 30           # Rechengenauigkeit
N = 10000                # Anzahl der Nullstellen
round_digits = 8         # Dezimalstellen

# ==============================
# NULLSTELLEN BERECHNEN
# ==============================

gammas = []
start_time = time.time()

for n in range(1, N + 1):
    zero = mp.zetazero(n)
    gamma = mp.im(zero)
    gammas.append(round(float(gamma), round_digits))

    # Ausgabe jeder Nullstelle
    elapsed = time.time() - start_time
    print(f"Berechne Nullstelle {n}/{N}  |  Zeit bisher: {elapsed:.1f}s")

# ==============================
# FORMATIERTE AUSGABE
# ==============================

total_time = time.time() - start_time
print(f"\nBerechnung abgeschlossen in {total_time:.1f}s.\n")

print("gammas_all = np.array([")
for i, g in enumerate(gammas):
    end = ",\n" if (i + 1) % 4 == 0 else ", "
    print(f"    {g}", end=end)
print("\n])")
