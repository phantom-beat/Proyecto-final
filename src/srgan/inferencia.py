"""Super-resolución de imágenes nuevas.

La imagen se procesa por bloques solapados (tiles) para no agotar la memoria de la GPU
con imágenes grandes. Modos:
    "real":      la imagen se toma tal cual y se amplía ×4.
    "simulado":  la imagen se reduce ×4 y se reconstruye, para poder medir PSNR.
"""

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from .config import ESCALA


def cargar_imagen(ruta):
    """Imagen RGB en [0, 1] respetando la orientación EXIF."""
    return np.asarray(ImageOps.exif_transpose(Image.open(ruta)).convert("RGB"), np.float32) / 255.0


def guardar_imagen(img, ruta):
    Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(ruta)


def super_resolver(modelo, lr, tile=128, solape=16, escala=ESCALA):
    """Aplica el generador por bloques solapados. lr: array (H, W, 3) en [0, 1]."""
    H, W, _ = lr.shape
    salida = np.zeros((H * escala, W * escala, 3), np.float32)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            y0, x0 = max(y - solape, 0), max(x - solape, 0)
            y1, x1 = min(y + tile + solape, H), min(x + tile + solape, W)
            sr = modelo(lr[None, y0:y1, x0:x1], training=False)[0].numpy()
            th, tw = (min(y + tile, H) - y) * escala, (min(x + tile, W) - x) * escala
            oy, ox = (y - y0) * escala, (x - x0) * escala
            salida[y * escala:y * escala + th, x * escala:x * escala + tw] = sr[oy:oy + th, ox:ox + tw]
    return np.clip(salida, 0, 1)


def ampliar_bicubica(lr, escala=ESCALA):
    h, w = lr.shape[0] * escala, lr.shape[1] * escala
    return np.clip(tf.image.resize(lr[None], (h, w), method="bicubic")[0].numpy(), 0, 1)


def reducir(img, escala=ESCALA):
    """Modo simulado: recorta a múltiplo de la escala y reduce con bicúbica + antialiasing."""
    h, w = img.shape[0] // escala * escala, img.shape[1] // escala * escala
    hr = img[:h, :w]
    lr = tf.image.resize(hr, (h // escala, w // escala), method="bicubic", antialias=True)
    return np.clip(lr.numpy(), 0, 1), hr


def psnr(a, b):
    return float(tf.image.psnr(a, b, 1.0))
