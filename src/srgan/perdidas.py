"""Funciones de pérdida.

Fase 1 (SRResNet):  L = MSE
Fase 2 (SRGAN):     L_G = L_VGG + 1e-3 · L_adv + λ_pix · L_MSE   (λ_pix = 1)
                    L_D = BCE(D(HR), 1) + BCE(D(G(LR)), 0)

L_VGG compara los mapas de la capa block5_conv4 de una VGG19 (ImageNet), divididos
entre 12.75 como en el artículo. L_adv es la forma no saturante −log D(G(LR)).
"""

import keras
import tensorflow as tf

from .config import PESO_ADV, PESO_PIXEL

mse = keras.losses.MeanSquaredError()
bce = keras.losses.BinaryCrossentropy(from_logits=True)

_extractor_vgg = None


def extractor_vgg():
    """VGG19 hasta block5_conv4 ("VGG54"), congelada. Se construye una sola vez."""
    global _extractor_vgg
    if _extractor_vgg is None:
        vgg = keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(None, None, 3))
        _extractor_vgg = keras.Model(vgg.input, vgg.get_layer("block5_conv4").output, name="vgg54")
        _extractor_vgg.trainable = False
    return _extractor_vgg


def caracteristicas_vgg(img):
    """img en [0, 1]. La VGG espera [0, 255] con el preprocesamiento de ImageNet."""
    x = keras.applications.vgg19.preprocess_input(img * 255.0)
    return extractor_vgg()(x, training=False) / 12.75


def perdida_contenido(hr, sr):
    return mse(caracteristicas_vgg(hr), caracteristicas_vgg(sr))


def perdida_adversarial(logit_falso):
    """Forma no saturante: el generador quiere que D clasifique sus imágenes como reales."""
    return bce(tf.ones_like(logit_falso), logit_falso)


def perdida_generador(hr, sr, logit_falso, peso_adv=PESO_ADV, peso_pixel=PESO_PIXEL):
    """Devuelve (L_G total, L_VGG, L_adv, L_MSE)."""
    p_contenido = perdida_contenido(hr, sr)
    p_adv = perdida_adversarial(logit_falso)
    p_pixel = mse(hr, sr)
    return p_contenido + peso_adv * p_adv + peso_pixel * p_pixel, p_contenido, p_adv, p_pixel


def perdida_discriminador(logit_real, logit_falso):
    return (bce(tf.ones_like(logit_real), logit_real) +
            bce(tf.zeros_like(logit_falso), logit_falso))
