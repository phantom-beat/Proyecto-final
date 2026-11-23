"""Entrenamiento en dos fases (como en el artículo SRGAN).

Fase 1 (SRResNet): el generador se entrena solo con MSE; se guarda la mejor época
según el PSNR de validación.
Fase 2 (SRGAN): parte de esa mejor época y entrena generador y discriminador; la tasa
de aprendizaje baja de 1e-4 a 1e-5 a mitad de la fase.
Los pesos y el historial se guardan al final de cada época para poder reanudar.
"""

import json
import os
import time

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import EPOCAS_GAN, EPOCAS_PRE, LR_GAN, LR_PRE
from .metricas import evaluar
from .perdidas import mse, perdida_discriminador, perdida_generador


class Checkpoints:
    """Rutas de pesos e historial dentro de una carpeta (local o Google Drive)."""

    def __init__(self, carpeta):
        os.makedirs(carpeta, exist_ok=True)
        self.carpeta = carpeta
        self.estado_json = os.path.join(carpeta, "estado.json")
        self.g_pre = os.path.join(carpeta, "generador_srresnet.weights.h5")
        self.g_mejor = os.path.join(carpeta, "generador_srresnet_mejor.weights.h5")
        self.g_gan = os.path.join(carpeta, "generador_srgan.weights.h5")
        self.d_gan = os.path.join(carpeta, "discriminador_srgan.weights.h5")

    def cargar_estado(self):
        if os.path.exists(self.estado_json):
            with open(self.estado_json) as f:
                return json.load(f)
        return {"epoca_pre": 0, "epoca_gan": 0, "hist_pre": [], "hist_gan": []}

    def guardar_estado(self, estado):
        with open(self.estado_json, "w") as f:
            json.dump(estado, f)
        for fase in ("pre", "gan"):
            if estado[f"hist_{fase}"]:
                pd.DataFrame(estado[f"hist_{fase}"]).to_csv(
                    os.path.join(self.carpeta, f"historial_{fase}.csv"), index=False)


def fase1(generador, ds_train, ds_val, pasos, ckpt, epocas=EPOCAS_PRE, lr=LR_PRE):
    """Pre-entrenamiento con MSE. Devuelve el historial por época."""
    opt = keras.optimizers.Adam(lr)

    @tf.function
    def paso(lr_b, hr_b):
        with tf.GradientTape() as tape:
            sr = generador(lr_b, training=True)
            perdida = mse(hr_b, sr)
        grads = tape.gradient(perdida, generador.trainable_variables)
        opt.apply_gradients(zip(grads, generador.trainable_variables))
        return perdida, tf.reduce_mean(tf.image.psnr(hr_b, sr, 1.0))

    estado = ckpt.cargar_estado()
    if estado["epoca_pre"] > 0 and os.path.exists(ckpt.g_pre):
        generador.load_weights(ckpt.g_pre)
    mejor = max(estado["hist_pre"], key=lambda h: h["val_psnr"], default=None)
    it = iter(ds_train)
    for epoca in range(estado["epoca_pre"] + 1, epocas + 1):
        print(f"Fase 1 - época {epoca}/{epocas}")
        barra = keras.utils.Progbar(pasos, stateful_metrics=["loss", "psnr"])
        t0, perdidas, psnrs = time.time(), [], []
        for i in range(pasos):
            p, ps = paso(*next(it))
            perdidas.append(float(p)); psnrs.append(float(ps))
            if (i + 1) % 20 == 0 or i + 1 == pasos:
                barra.update(i + 1, [("loss", np.mean(perdidas)), ("psnr", np.mean(psnrs))])
        val = evaluar(generador, ds_val)
        fila = {"epoca": epoca, "lr": lr, "loss": float(np.mean(perdidas)), "psnr": float(np.mean(psnrs)),
                "val_loss": val["mse"], "val_psnr": val["psnr"], "val_ssim": val["ssim"],
                "val_vgg": val["vgg"], "tiempo_s": time.time() - t0}
        estado["hist_pre"].append(fila)
        if mejor is None or fila["val_psnr"] > mejor["val_psnr"]:
            mejor = fila
            generador.save_weights(ckpt.g_mejor)
        generador.save_weights(ckpt.g_pre)
        estado["epoca_pre"] = epoca
        ckpt.guardar_estado(estado)
        print(f"  val_psnr {fila['val_psnr']:.2f} dB - val_ssim {fila['val_ssim']:.4f}")
    return estado["hist_pre"]


def fase2(generador, discriminador, ds_train, ds_val, pasos, ckpt, epocas=EPOCAS_GAN, lr=LR_GAN):
    """Entrenamiento adversarial. Devuelve el historial por época."""
    opt_g = keras.optimizers.Adam(lr)
    opt_d = keras.optimizers.Adam(lr)

    @tf.function
    def paso(lr_b, hr_b):
        with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
            sr = generador(lr_b, training=True)
            logit_real = discriminador(hr_b, training=True)
            logit_falso = discriminador(sr, training=True)
            p_g, p_vgg, p_adv, p_pix = perdida_generador(hr_b, sr, logit_falso)
            p_d = perdida_discriminador(logit_real, logit_falso)
        grads_g = tape_g.gradient(p_g, generador.trainable_variables)
        grads_d = tape_d.gradient(p_d, discriminador.trainable_variables)
        opt_g.apply_gradients(zip(grads_g, generador.trainable_variables))
        opt_d.apply_gradients(zip(grads_d, discriminador.trainable_variables))
        return {"loss_g": p_g, "loss_vgg": p_vgg, "loss_adv": p_adv, "loss_pix": p_pix, "loss_d": p_d,
                "d_real": tf.reduce_mean(tf.sigmoid(logit_real)),
                "d_falso": tf.reduce_mean(tf.sigmoid(logit_falso))}

    estado = ckpt.cargar_estado()
    if estado["epoca_gan"] > 0 and os.path.exists(ckpt.g_gan):
        generador.load_weights(ckpt.g_gan)
        discriminador.load_weights(ckpt.d_gan)
    else:
        generador.load_weights(ckpt.g_mejor)      # punto de partida: mejor época de la fase 1
    it = iter(ds_train)
    for epoca in range(estado["epoca_gan"] + 1, epocas + 1):
        tasa = lr if epoca <= epocas // 2 else lr / 10
        opt_g.learning_rate = tasa
        opt_d.learning_rate = tasa
        print(f"Fase 2 - época {epoca}/{epocas} (lr {tasa:.0e})")
        barra = keras.utils.Progbar(pasos, stateful_metrics=["L_G", "L_D"])
        t0, acum = time.time(), []
        for i in range(pasos):
            acum.append({k: float(v) for k, v in paso(*next(it)).items()})
            if (i + 1) % 20 == 0 or i + 1 == pasos:
                barra.update(i + 1, [("L_G", np.mean([a["loss_g"] for a in acum])),
                                     ("L_D", np.mean([a["loss_d"] for a in acum]))])
        val = evaluar(generador, ds_val)
        fila = {"epoca": epoca, "lr": tasa, **{k: float(np.mean([a[k] for a in acum])) for k in acum[0]},
                "val_loss": val["mse"], "val_psnr": val["psnr"], "val_ssim": val["ssim"],
                "val_vgg": val["vgg"], "tiempo_s": time.time() - t0}
        estado["hist_gan"].append(fila)
        generador.save_weights(ckpt.g_gan)
        discriminador.save_weights(ckpt.d_gan)
        estado["epoca_gan"] = epoca
        ckpt.guardar_estado(estado)
        print(f"  L_D {fila['loss_d']:.3f} - D(falso) {fila['d_falso']:.2f} - val_psnr {fila['val_psnr']:.2f} dB")
    return estado["hist_gan"]
