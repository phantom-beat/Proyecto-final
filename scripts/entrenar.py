"""Entrena SRGAN ×4 desde cero (fase 1 + fase 2).

Uso:
    python scripts/entrenar.py --zip ruta/srgan_dataset_personalizado.zip --salida checkpoints
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

from srgan.config import SEED  # noqa: E402
from srgan.datos import crear_datasets, dividir, preparar_dataset  # noqa: E402
from srgan.entrenamiento import Checkpoints, fase1, fase2  # noqa: E402
from srgan.modelos import construir_discriminador, construir_generador  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="zip del dataset personalizado")
    ap.add_argument("--salida", default="checkpoints", help="carpeta de pesos e historial")
    ap.add_argument("--solo-fase1", action="store_true", help="entrena solo SRResNet")
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    rutas = preparar_dataset(args.zip)
    train, val, test = dividir(rutas)
    print(f"Imágenes -> entrenamiento {len(train)} | validación {len(val)} | prueba {len(test)}")
    ds_train, ds_val, _, pasos = crear_datasets(train, val, test)

    ckpt = Checkpoints(args.salida)
    generador = construir_generador()
    fase1(generador, ds_train, ds_val, pasos, ckpt)
    if not args.solo_fase1:
        fase2(generador, construir_discriminador(), ds_train, ds_val, pasos, ckpt)
    print("Listo. Pesos e historial en", args.salida)


if __name__ == "__main__":
    main()
