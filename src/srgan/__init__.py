"""SRGAN ×4: super-resolución de imágenes con redes generativas antagónicas.

Módulos:
    config         hiperparámetros del proyecto
    modelos        generador (SRResNet modificado) y discriminador
    perdidas       pérdida de contenido VGG54, adversarial y MSE
    datos          carga del dataset, modelo de degradación y pipelines tf.data
    metricas       PSNR, SSIM, pérdida VGG y LPIPS
    entrenamiento  fase 1 (SRResNet, MSE) y fase 2 (SRGAN, adversarial)
    inferencia     super-resolución de imágenes nuevas por bloques
"""

__version__ = "1.0.0"
