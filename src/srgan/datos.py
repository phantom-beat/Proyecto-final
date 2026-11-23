"""Dataset personalizado, modelo de degradación y pipelines tf.data.

El dataset (2 800 imágenes de DIV2K, Oxford 102 Flowers y Oxford-IIIT Pet) se distribuye
como un único zip con un manifiesto.csv. La división 80/10/10 es estratificada por fuente.

Modelo de degradación:  I_LR = JPEG_q( bicúbica↓4(I_HR) + n )
(ruido y JPEG solo si DEGRADACION_REALISTA = True).
"""

import os
import random
import zipfile

import pandas as pd
import tensorflow as tf

from .config import (BATCH, DEGRADACION_REALISTA, ESCALA, FUENTES, HR_SIZE, LR_SIZE,
                     RECORTES_POR_IMAGEN, SEED, TEST_MAX, VAL_SIZE)

AUTOTUNE = tf.data.AUTOTUNE


# ---------------- Dataset ----------------
def preparar_dataset(ruta_zip, destino="datos/dataset_personalizado"):
    """Descomprime el zip (si hace falta) y devuelve la lista de rutas de imágenes."""
    manifiesto = os.path.join(destino, "manifiesto.csv")
    if not os.path.exists(manifiesto):
        with zipfile.ZipFile(ruta_zip) as z:
            z.extractall(destino)
    return [os.path.join(destino, a) for a in pd.read_csv(manifiesto)["archivo"]]


def fuente_de(ruta):
    return os.path.basename(ruta).split("_")[0]


def dividir(rutas, seed=SEED):
    """División 80/10/10 estratificada por fuente -> (train, val, test)."""
    train, val, test = [], [], []
    for f in FUENTES:
        grupo = sorted(r for r in rutas if fuente_de(r) == f)
        random.Random(seed).shuffle(grupo)
        n_eval = max(5, int(0.1 * len(grupo)))
        test += grupo[:n_eval]
        val += grupo[n_eval:2 * n_eval]
        train += grupo[2 * n_eval:]
    random.Random(seed).shuffle(val)
    return train, val, test


# ---------------- Degradación y recortes ----------------
def cargar(ruta):
    img = tf.io.decode_jpeg(tf.io.read_file(ruta), channels=3)
    return tf.image.convert_image_dtype(img, tf.float32)   # [0, 1]


def degradar(hr, h_lr, w_lr):
    """Crea la imagen LR a partir de la HR."""
    lr = tf.clip_by_value(tf.image.resize(hr, (h_lr, w_lr), method="bicubic", antialias=True), 0.0, 1.0)
    if DEGRADACION_REALISTA:
        sigma = tf.random.uniform([], 0.0, 0.02)
        lr = tf.clip_by_value(lr + tf.random.normal(tf.shape(lr), stddev=sigma), 0.0, 1.0)
        lr = tf.image.adjust_jpeg_quality(lr, tf.random.uniform([], 60, 95, dtype=tf.int32))
    return lr


def recorte_entrenamiento(img):
    """Recorte aleatorio 96×96 + volteo + rotación de 90° (aumento de datos)."""
    hr = tf.image.random_crop(img, (HR_SIZE, HR_SIZE, 3))
    hr = tf.image.random_flip_left_right(hr)
    hr = tf.image.rot90(hr, tf.random.uniform([], 0, 4, dtype=tf.int32))
    lr = degradar(hr, LR_SIZE, LR_SIZE)
    return tf.ensure_shape(lr, (LR_SIZE, LR_SIZE, 3)), hr


def par_validacion(ruta):
    hr = tf.image.resize_with_crop_or_pad(cargar(ruta), VAL_SIZE, VAL_SIZE)   # recorte central
    lr = degradar(hr, VAL_SIZE // ESCALA, VAL_SIZE // ESCALA)
    return tf.ensure_shape(lr, (VAL_SIZE // ESCALA, VAL_SIZE // ESCALA, 3)), hr


def par_prueba(ruta):
    hr = cargar(ruta)
    s = tf.shape(hr)
    h = tf.minimum(s[0], TEST_MAX) // ESCALA * ESCALA
    w = tf.minimum(s[1], TEST_MAX) // ESCALA * ESCALA
    hr = tf.image.resize_with_crop_or_pad(hr, h, w)
    return degradar(hr, h // ESCALA, w // ESCALA), hr


# ---------------- Pipelines ----------------
def crear_datasets(rutas_train, rutas_val, rutas_test, seed=SEED):
    """Devuelve (ds_train, ds_val, ds_test, pasos_por_epoca)."""
    ds_train = (tf.data.Dataset.from_tensor_slices(rutas_train)
                .shuffle(len(rutas_train), seed=seed).repeat()
                .map(cargar, num_parallel_calls=AUTOTUNE)
                .flat_map(lambda img: tf.data.Dataset.from_tensors(img).repeat(RECORTES_POR_IMAGEN))
                .map(recorte_entrenamiento, num_parallel_calls=AUTOTUNE)
                .shuffle(256)
                .batch(BATCH, drop_remainder=True)
                .prefetch(AUTOTUNE))
    ds_val = (tf.data.Dataset.from_tensor_slices(rutas_val)
              .map(par_validacion, num_parallel_calls=AUTOTUNE).batch(8).cache())
    ds_test = tf.data.Dataset.from_tensor_slices(rutas_test).map(par_prueba).batch(1)
    pasos = len(rutas_train) * RECORTES_POR_IMAGEN // BATCH   # 2240 × 4 / 16 = 560
    return ds_train, ds_val, ds_test, pasos
