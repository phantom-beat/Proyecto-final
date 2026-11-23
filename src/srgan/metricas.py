"""Métricas de evaluación.

PSNR y SSIM (fidelidad de píxeles, mayor es mejor), pérdida VGG (perceptual, menor es
mejor) y LPIPS con AlexNet (perceptual, menor es mejor; requiere `pip install lpips torch`).
"""

import numpy as np
import tensorflow as tf

from .perdidas import perdida_contenido


def bicubica(lr, tam_hr):
    """Línea base: ampliación bicúbica de un lote LR al tamaño HR."""
    tam_hr = [int(d) for d in tam_hr]
    return tf.clip_by_value(tf.image.resize(lr, tam_hr, method="bicubic"), 0.0, 1.0)


def predecir(modelo, lr):
    return tf.clip_by_value(modelo(lr, training=False), 0.0, 1.0)


def evaluar(modelo, ds_val):
    """MSE, PSNR, SSIM y pérdida VGG promedio sobre un dataset (modelo=None -> bicúbica)."""
    m = {"mse": [], "psnr": [], "ssim": [], "vgg": []}
    for lr, hr in ds_val:
        sr = bicubica(lr, hr.shape[1:3]) if modelo is None else predecir(modelo, lr)
        m["mse"].append(tf.reduce_mean(tf.square(hr - sr), axis=[1, 2, 3]))
        m["psnr"].append(tf.image.psnr(hr, sr, 1.0))
        m["ssim"].append(tf.image.ssim(hr, sr, 1.0))
        m["vgg"].append(tf.fill([tf.shape(hr)[0]], perdida_contenido(hr, sr)))
    return {k: float(tf.reduce_mean(tf.concat(v, 0))) for k, v in m.items()}


class LPIPS:
    """LPIPS (Zhang et al., 2018) con la implementación oficial en PyTorch, ejecutada en CPU."""

    def __init__(self, red="alex"):
        import warnings

        import lpips
        import torch
        warnings.filterwarnings("ignore", category=UserWarning)
        self.torch = torch
        self.modelo = lpips.LPIPS(net=red, verbose=False).eval()

    def __call__(self, a, b):
        """a, b: arrays (H, W, 3) en [0, 1]. Devuelve la distancia LPIPS."""
        def a_tensor(x):
            x = np.array(x, dtype=np.float32)               # copia escribible
            return self.torch.from_numpy(x).permute(2, 0, 1)[None] * 2 - 1   # [-1, 1]
        with self.torch.no_grad():
            return float(self.modelo(a_tensor(a), a_tensor(b)))


def evaluar_prueba(modelos, ds_test, lpips_fn=None):
    """Métricas por imagen en el conjunto de prueba.

    modelos: dict {nombre: modelo o None (bicúbica)}. Devuelve {nombre: {"psnr": [...], ...}}.
    """
    res = {n: {"psnr": [], "ssim": [], "lpips": []} for n in modelos}
    for lr, hr in ds_test:
        for nombre, modelo in modelos.items():
            sr = bicubica(lr, hr.shape[1:3]) if modelo is None else predecir(modelo, lr)
            res[nombre]["psnr"].append(float(tf.image.psnr(hr, sr, 1.0)[0]))
            res[nombre]["ssim"].append(float(tf.image.ssim(hr, sr, 1.0)[0]))
            if lpips_fn is not None:
                res[nombre]["lpips"].append(lpips_fn(hr[0].numpy(), sr[0].numpy()))
    return res
