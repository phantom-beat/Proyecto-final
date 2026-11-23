"""Amplía ×4 una o varias imágenes con el generador entrenado.

Uso:
    python scripts/super_resolver.py --pesos checkpoints/generador_srgan.weights.h5 foto.jpg
    python scripts/super_resolver.py --pesos ... --modo simulado foto.jpg   # mide PSNR
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from srgan.inferencia import (ampliar_bicubica, cargar_imagen, guardar_imagen, psnr,  # noqa: E402
                              reducir, super_resolver)
from srgan.modelos import construir_generador  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imagenes", nargs="+")
    ap.add_argument("--pesos", required=True, help="generador_srgan.weights.h5 o generador_srresnet_mejor.weights.h5")
    ap.add_argument("--modo", choices=["real", "simulado"], default="real")
    ap.add_argument("--salida", default="salidas")
    args = ap.parse_args()

    generador = construir_generador()
    generador.load_weights(args.pesos)
    os.makedirs(args.salida, exist_ok=True)

    for ruta in args.imagenes:
        img = cargar_imagen(ruta)
        if args.modo == "simulado":
            lr, hr = reducir(img)
        else:
            lr, hr = img, None
        sr = super_resolver(generador, lr)
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        destino = os.path.join(args.salida, f"SR_x4_{nombre}.png")
        guardar_imagen(sr, destino)
        msg = f"{ruta}: {lr.shape[1]}×{lr.shape[0]} -> {sr.shape[1]}×{sr.shape[0]}  ({destino})"
        if hr is not None:
            msg += f" | PSNR bicúbica {psnr(hr, ampliar_bicubica(lr)):.2f} dB, SRGAN {psnr(hr, sr):.2f} dB"
        print(msg)


if __name__ == "__main__":
    main()
