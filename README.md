# Sttable-Diffusion-con-imagenes-propias-
Creacion de imagenes con Sttable Diffusion a partir de una persona, mascota,cosa, etc...
![00015-1625625314-JOSEJAAM in a___](https://user-images.githubusercontent.com/92582462/201015637-a0d81858-e15c-4896-b86a-170f58c7cea1.png)


##  Pasos a seguir para la creacion de una imagen con sttable diffusion


###    **PASO 1**

CREAR DATASET:

- 3 o mas IMAGENES DE CUERPO COMPLETO
- 5 o mas IMAGENES DE MEDIO CUERPO
- 12 o mas  IMAGENES DE PRIMER PLANO ( en las que se vean bien los detalles del rostro)

**TIPS PARA OBTENER MEJORES RESULTADOS**
- Las fotos deber ser de diferentes fondos, ropa, perspectiva
- Usa imagenes de alta calidad, sin filtros, sombras extrañas, contraluces ni objetos o manos cerca del rostro
-Elimina con photoshop cualquier persona o rostro reconocible que no sea el tuyo (incluye cabello, manos, brazos de otras personas) que aparezca en la imagen porque puede interferir al algoritmo al momento del entrenamiento
-Evita usar demasiadas selfies, si no quieres que StableDiffusion te genere selfies montadas en cuerpos normales que se ven muy mal
-Evita los bordes oscuros o marcos en tus fotos 


###    **PASO 2** 
Redimensionar las fotos 512x512 en photoshop o en nuestro editor de confianza, al final tendremos una carpeta con + o - 20 imagenes.


#### Ten activada la Aceleracion por hardware con GPU en " Entorno de ejecucuion"

```
!nvidia-smi -L'

```
###  **PASO 3**   
CONECTAMOS CON GOOGLE DRIVE. Importante tener libres unos 4GB si no, el archivo de entrenamiento final no se guardará 

```
from google.colab import drive
drive.mount('/content/gdrive')

```
###  **Paso 4**    - Instalamos las librerías necesarias.
``` 
#@markdown # Dependencies
%%capture
%cd /content/
!git clone https://github.com/TheLastBen/diffusers
!pip install -q git+https://github.com/TheLastBen/diffusers
!pip install -q accelerate==0.12.0
!pip install -q OmegaConf
!wget https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dreambooth/Deps
!mv Deps Deps.7z
!7z x Deps.7z
!cp -r /content/usr/local/lib/python3.7/dist-packages /usr/local/lib/python3.7/
!rm Deps.7z
!rm -r /content/usr

```



