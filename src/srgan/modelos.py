"""Generador (SRResNet modificado) y discriminador de SRGAN.

Modificaciones respecto al artículo (Ledig et al., 2017):
    1. Sin Batch Normalization en el generador (EDSR, ESRGAN).
    2. Aprendizaje residual global: SR = bicúbica(LR) + detalle (VDSR); la última
       convolución se inicializa en ceros, así el modelo arranca con la calidad bicúbica.
    3. Salida acotada a [0, 1] con ReLU(max_value=1): las pérdidas y el discriminador
       ven durante el entrenamiento las mismas imágenes que se evalúan.
"""

import math

import keras
import tensorflow as tf
from keras import layers

from .config import ESCALA, HR_SIZE


@keras.saving.register_keras_serializable(package="srgan")
class PixelShuffle(layers.Layer):
    """Sub-pixel convolution: (h, w, c·r²) -> (h·r, w·r, c)."""

    def __init__(self, factor=2, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return tf.nn.depth_to_space(x, self.factor)

    def get_config(self):
        return {**super().get_config(), "factor": self.factor}


@keras.saving.register_keras_serializable(package="srgan")
class AmpliacionBicubica(layers.Layer):
    """Amplía la imagen LR con interpolación bicúbica (sin parámetros entrenables)."""

    def __init__(self, factor=4, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return tf.image.resize(x, tf.shape(x)[1:3] * self.factor, method="bicubic")

    def get_config(self):
        return {**super().get_config(), "factor": self.factor}


def bloque_residual(x, filtros=64):
    """[Conv3×3 → PReLU → Conv3×3] + x  (sin Batch Normalization)."""
    y = layers.Conv2D(filtros, 3, padding="same")(x)
    y = layers.PReLU(shared_axes=[1, 2])(y)
    y = layers.Conv2D(filtros, 3, padding="same")(y)
    return layers.Add()([x, y])


def construir_generador(n_bloques=16, filtros=64, escala=ESCALA):
    """SRResNet modificado. Totalmente convolucional: acepta imágenes de cualquier tamaño."""
    entrada = layers.Input(shape=(None, None, 3), name="imagen_LR")
    x = layers.Conv2D(filtros, 9, padding="same")(entrada)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    salto = x
    for _ in range(n_bloques):
        x = bloque_residual(x, filtros)
    x = layers.Conv2D(filtros, 3, padding="same")(x)
    x = layers.Add()([x, salto])
    for _ in range(int(math.log2(escala))):        # 2 bloques de ×2 = ×4
        x = layers.Conv2D(filtros * 4, 3, padding="same")(x)
        x = PixelShuffle(2)(x)
        x = layers.PReLU(shared_axes=[1, 2])(x)
    # La red predice solo el DETALLE que le falta a la ampliación bicúbica.
    detalle = layers.Conv2D(3, 9, padding="same", kernel_initializer="zeros", name="detalle")(x)
    suma = layers.Add(name="suma_residual")([AmpliacionBicubica(escala)(entrada), detalle])
    salida = layers.ReLU(max_value=1.0, name="imagen_SR")(suma)   # recorta la salida a [0, 1]
    return keras.Model(entrada, salida, name="generador_SRResNet")


def bloque_d(x, filtros, stride, bn=True):
    x = layers.Conv2D(filtros, 3, strides=stride, padding="same")(x)
    if bn:
        x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(0.2)(x)


def construir_discriminador(tam=HR_SIZE):
    """Discriminador de SRGAN: 8 convoluciones + Dense 1024; salida logit (real vs. generada)."""
    entrada = layers.Input(shape=(tam, tam, 3), name="imagen_HR_o_SR")
    x = layers.Rescaling(2.0, offset=-1.0)(entrada)          # [0,1] -> [-1,1]
    x = bloque_d(x, 64, 1, bn=False)
    x = bloque_d(x, 64, 2)
    x = bloque_d(x, 128, 1)
    x = bloque_d(x, 128, 2)
    x = bloque_d(x, 256, 1)
    x = bloque_d(x, 256, 2)
    x = bloque_d(x, 512, 1)
    x = bloque_d(x, 512, 2)
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(0.2)(x)
    salida = layers.Dense(1, name="logit_real")(x)
    return keras.Model(entrada, salida, name="discriminador")
