"""Hiperparámetros del proyecto (los mismos del notebook)."""

SEED = 42

# ---------------- Escala y tamaños ----------------
ESCALA = 4               # factor de super-resolución
HR_SIZE = 96             # parche de alta resolución para entrenar
LR_SIZE = HR_SIZE // ESCALA
BATCH = 16

# ---------------- Entrenamiento ----------------
EPOCAS_PRE = 18          # fase 1: generador solo (MSE)
EPOCAS_GAN = 18          # fase 2: entrenamiento adversarial
LR_PRE = 1e-4            # tasa de aprendizaje fase 1
LR_GAN = 1e-4            # tasa de aprendizaje fase 2 (baja a 1e-5 a mitad de la fase)
PESO_ADV = 1e-3          # peso de la pérdida adversarial (valor del artículo SRGAN)
PESO_PIXEL = 1.0         # pérdida MSE extra en la fase 2 (ancla de color)

# ---------------- Datos ----------------
FUENTES = ["div2k", "flores", "mascotas"]
RECORTES_POR_IMAGEN = 4
DEGRADACION_REALISTA = False  # True = además del downscale, agrega ruido y compresión JPEG
VAL_SIZE = 192                # recorte central HR para validar durante el entrenamiento
TEST_MAX = 512                # tamaño máximo del recorte HR en el conjunto de prueba
