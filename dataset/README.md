# Dataset personalizado

`srgan_dataset_personalizado.zip` (401.6 MB) contiene las 2 800 imágenes del proyecto y su `manifiesto.csv`:

| Fuente | Imágenes | Origen |
|---|---|---|
| `div2k` | 800 | DIV2K (Agustsson & Timofte, 2017) |
| `flores` | 1 000 | Oxford 102 Flowers (Nilsback & Zisserman, 2008) |
| `mascotas` | 1 000 | Oxford-IIIT Pet (Parkhi *et al.*, 2012) |

GitHub no acepta archivos de más de 100 MB, así que el zip está dividido en 5 partes (`.part00` a `.part04`). Para usarlo hay que **unirlas en orden**:

**Linux, macOS o Google Colab**

```bash
cat dataset/srgan_dataset_personalizado.zip.part* > srgan_dataset_personalizado.zip
```

**Windows (Símbolo del sistema)**

```bat
copy /b dataset\srgan_dataset_personalizado.zip.part00 + dataset\srgan_dataset_personalizado.zip.part01 + dataset\srgan_dataset_personalizado.zip.part02 + dataset\srgan_dataset_personalizado.zip.part03 + dataset\srgan_dataset_personalizado.zip.part04 srgan_dataset_personalizado.zip
```

**Verificación (opcional):** el SHA-256 del zip unido debe ser

```
6100fb7cffd04ce0b84652eb2e690fce3daa89e00304ef8a9e5db22103f30bde
```

## Usarlo en el notebook de Colab

Antes de ejecutar la sección 2 del notebook, en una celda:

```bash
!git clone https://github.com/phantom-beat/Proyecto-final.git
!cat Proyecto-final/dataset/srgan_dataset_personalizado.zip.part* > /content/srgan_dataset_personalizado.zip
```

El notebook busca el zip en `/content/` y lo descomprime automáticamente.

## Contenido del zip

```
div2k/      div2k_0000.jpg ...
flores/     flores_0000.jpg ...
mascotas/   mascotas_0000.jpg ...
manifiesto.csv   archivo, fuente, archivo de origen, ancho y alto de cada imagen
```

Cada imagen se corrigió según su orientación EXIF, se convirtió a RGB, se redujo si su lado mayor superaba 1 024 px y se guardó como JPEG de calidad 95. Las imágenes conservan las licencias de sus datasets de origen (uso académico).
