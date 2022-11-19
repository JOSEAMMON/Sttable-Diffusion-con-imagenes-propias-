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

**DEPENDENCIAS** 
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
**XFORMERS**

```
#@markdown # xformers

from subprocess import getoutput
from IPython.display import HTML
from IPython.display import clear_output
import time

s = getoutput('nvidia-smi')
if 'T4' in s:
  gpu = 'T4'
elif 'P100' in s:
  gpu = 'P100'
elif 'V100' in s:
  gpu = 'V100'
elif 'A100' in s:
  gpu = 'A100'

while True:
    try: 
        gpu=='T4'or gpu=='P100'or gpu=='V100'or gpu=='A100'
        break
    except:
        pass
    print('[1;31mit seems that your GPU is not supported at the moment')
    time.sleep(5)

if (gpu=='T4'):
  %pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/T4/xformers-0.0.13.dev0-py3-none-any.whl
  
elif (gpu=='P100'):
  %pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/P100/xformers-0.0.13.dev0-py3-none-any.whl

elif (gpu=='V100'):
  %pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/V100/xformers-0.0.13.dev0-py3-none-any.whl

elif (gpu=='A100'):
  %pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/A100/xformers-0.0.13.dev0-py3-none-any.whl  

clear_output()
print('[1;32mDONE !')

```
### **Paso 5** 
Descargamos el modelo .ckpt de Stable Diffusion original.

AQUI LO QUE TENEMOS QUE HACER ES PONER UN TOKEN DE CLAVE DE HUGGING FACE.

*o lo ponemos abajo dentro del cuadrito del codigo o lo ponemos a la derecha, en la linea de Huggingface_Token*

Entramos en Hugging Face, con el enlace que nos marca para pedir la clave de acceso:

- pedimos un token nuevo


![image](https://user-images.githubusercontent.com/92582462/202844196-67d0f62d-5cdf-489d-89a5-1ec9d4dc4987.png)


```
import os
import time
from IPython.display import clear_output
from IPython.utils import capture

with capture.capture_output() as cap: 
  %cd /content/
#@markdown ---
Huggingface_Token = "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" #@param {type:"string"}
token=Huggingface_Token

#@markdown *(Make sure you've accepted the terms in https://huggingface.co/runwayml/stable-diffusion-v1-5)*

#@markdown ---

CKPT_Path = "" #@param {type:"string"}

#@markdown Or

CKPT_gdrive_Link = "" #@param {type:"string"}


if CKPT_Path !="":
  if os.path.exists('/content/stable-diffusion-v1-5'):
    !rm -r /content/stable-diffusion-v1-5
  if os.path.exists(str(CKPT_Path)):
    !mkdir /content/stable-diffusion-v1-5
    with capture.capture_output() as cap: 
      !wget https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_original_stable_diffusion_to_diffusers.py
    !python /content/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "$CKPT_Path" --dump_path /content/stable-diffusion-v1-5
    if os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
      !rm /content/convert_original_stable_diffusion_to_diffusers.py
      !rm /content/v1-inference.yaml
      clear_output()
      print('[1;32mDONE !')
    else:
      !rm /content/convert_original_stable_diffusion_to_diffusers.py
      !rm /content/v1-inference.yaml
      !rm -r /content/stable-diffusion-v1-5
      while not os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
        print('[1;31mConversion error, check your CKPT and try again')
        time.sleep(5)
  else:
    while not os.path.exists(str(CKPT_Path)):
       print('[1;31mWrong path, use the colab file explorer to copy the path')
       time.sleep(5)


elif CKPT_gdrive_Link !="":   
    if os.path.exists('/content/stable-diffusion-v1-5'):
      !rm -r /content/stable-diffusion-v1-5     
    !gdown --fuzzy $CKPT_gdrive_Link -O model.ckpt    
    if os.path.exists('/content/model.ckpt'):
      if os.path.getsize("/content/model.ckpt") > 1810671599:
        !mkdir /content/stable-diffusion-v1-5
        with capture.capture_output() as cap: 
          !wget https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_original_stable_diffusion_to_diffusers.py
        !python /content/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path /content/model.ckpt --dump_path /content/stable-diffusion-v1-5
        if os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
          clear_output()
          print('[1;32mDONE !')
          !rm /content/convert_original_stable_diffusion_to_diffusers.py
          !rm /content/v1-inference.yaml
          !rm /content/model.ckpt
        else:
          if os.path.exists('/content/v1-inference.yaml'):
            !rm /content/v1-inference.yaml
          !rm /content/convert_original_stable_diffusion_to_diffusers.py
          !rm -r /content/stable-diffusion-v1-5
          !rm /content/model.ckpt
          while not os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
            print('[1;31mConversion error, check your CKPT and try again')
            time.sleep(5)
      else:
        while os.path.getsize('/content/model.ckpt') < 1810671599:
           print('[1;31mWrong link, check that the link is valid')
           time.sleep(5)


elif token =="":
  if os.path.exists('/content/stable-diffusion-v1-5'):
    !rm -r /content/stable-diffusion-v1-5
  clear_output()
  token=input("Insert your huggingface token :")
  %cd /content/
  clear_output()
  !mkdir /content/stable-diffusion-v1-5
  %cd /content/stable-diffusion-v1-5
  !git init
  !git lfs install --system --skip-repo
  !git remote add -f origin  "https://USER:{token}@huggingface.co/runwayml/stable-diffusion-v1-5"
  !git config core.sparsecheckout true
  !echo -e "feature_extractor\nsafety_checker\nscheduler\ntext_encoder\ntokenizer\nunet\nmodel_index.json" > .git/info/sparse-checkout
  !git pull origin main
  if os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
    !git clone "https://USER:{token}@huggingface.co/stabilityai/sd-vae-ft-mse"
    !mv /content/stable-diffusion-v1-5/sd-vae-ft-mse /content/stable-diffusion-v1-5/vae
    !rm -r /content/stable-diffusion-v1-5/.git
    %cd /content/    
    clear_output()
    print('[1;32mDONE !')
  else:
    while not os.path.exists('/content/stable-diffusion-v1-5'):
         print('[1;31mMake sure you accepted the terms in https://huggingface.co/runwayml/stable-diffusion-v1-5')
         time.sleep(5)
         
elif token !="":
  if os.path.exists('/content/stable-diffusion-v1-5'):
    !rm -r /content/stable-diffusion-v1-5   
  clear_output()
  %cd /content/
  clear_output()
  !mkdir /content/stable-diffusion-v1-5
  %cd /content/stable-diffusion-v1-5
  !git init
  !git lfs install --system --skip-repo
  !git remote add -f origin  "https://USER:{token}@huggingface.co/runwayml/stable-diffusion-v1-5"
  !git config core.sparsecheckout true
  !echo -e "feature_extractor\nsafety_checker\nscheduler\ntext_encoder\ntokenizer\nunet\nmodel_index.json" > .git/info/sparse-checkout
  !git pull origin main
  if os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
    !git clone "https://USER:{token}@huggingface.co/stabilityai/sd-vae-ft-mse"
    !mv /content/stable-diffusion-v1-5/sd-vae-ft-mse /content/stable-diffusion-v1-5/vae
    !rm -r /content/stable-diffusion-v1-5/.git
    %cd /content/    
    clear_output()
    print('[1;32mDONE !')
  else:
    while not os.path.exists('/content/stable-diffusion-v1-5'):
         print('[1;31mMake sure you accepted the terms in https://huggingface.co/runwayml/stable-diffusion-v1-5')
         time.sleep(5)
         
   ```

### **Paso 6**  - Configuramos el entrenamiento de Dreambooth.

**TRAINING SUBJECT**
En este apartado le debemos decir a drembooth para que entienda la tarea que haga.
 Que queremos entrenar:
 - a un personaje , 
 - un objeto ,
 - un estilo artistico , 
 - un artista  

Como queremos entrenarlo con imagenes nuestras marcamos " **character** " ( personaje) 

 **SUBJECT_TYPE**: Cual es la categoria del concepto con la que estas entrenando la inteligencia artificial ,
 - aqui lo estamos utilizando con fotos nuestras, podriamos poner la palabra "person " o " man " por ejemplo , 
 - si fuera una chica podrias utilizar la palabra " person " o " woman ", 
 - si son niños, hijos, etc podriamos poner " boy " , " girl" , "baby " etc.. 
 - si es tu mascota " dog " , " cat "

 o sea , seria la clase que representaria el concepto con el que vas a entrenar a sttable diffusion.

**INSTANCE_NAME:**  Vamos a poner el token que identifique al concepto que estamos entrenando o sea un (nombre identificativo,unico y corto) como si fuera un NIK, "KENYGUARHOLL", "JAMM34" ,ETC... 
***IMPORTANTE*** : que sea una palabra que stabble diffusion no la conozca por otro concepto.

**Es interesante que sea corto porque hay que escribirlo muchas veces.**
El resto de opciones las dejamos tal y como estan.



Al ejecutarlo , nos va a salir abajo un menu para elegir archivos , nos interesa subir una imagen solo y luego cuando haya creado la carpeta ( veremos la carpeta a la izquierda  dentro de DATA) dentro veremos una carpeta con el nombre de nuestro TOKEN y dentro la imagen, entonces desde ahí subimos el resto de las imagenes es porque es mucho mas rapido.


```
import os
import shutil
from google.colab import files
from IPython.display import clear_output
from IPython.utils import capture
#@markdown ---
Training_Subject = "Character" #@param ["Character", "Object", "Style", "Artist", "Movie", "TV Show"] 

With_Prior_Preservation = "Yes" #@param ["Yes", "No"] 
#@markdown - With the prior reservation method, the results are better, you will either have to upload around 200 pictures of the class you're training (dog, person, car, house ...) or let Dreambooth generate them.

MODEL_NAME="/content/stable-diffusion-v1-5"

Captionned_instance_images = False #@param {type:"boolean"}

#@markdown - Use the keywords included in each instance images as unique instance prompt, this allows to train on multiple subjects at the same time, example : 
#@markdown - An instance image named fat_dog_doginstancename_in_a_pool.jpg
#@markdown - another instance image named a_cat_catinstancename_in_the_woods.png
#@markdown - the unique training instance prompts would be : fat dog doginstancename in a pool, a cat doginstancename in the woods
#@markdown - at inference you can generate the dog by simply using doginstancename (a random unique identifier) or the cat by catinstancename

#@markdown - Also you can enhance the training of a simple subject by simply describing the image using keywords like : smiling, outdoor, sad, lether jacket ...etc

#@markdown - If you enable this feature, and want to train on multiple subjects, use the AUTOMATIC1111 colab to generate good quality 512x512 100-200 Class images for each subject (dog and a cat and a cow), then put them all in the same folder and entrer the folder's path in the cell below.

#@markdown - If you enable this feature, you must add an instance name and a subject type (dog, man, car) to all the images, separate keywords by an underscore (_).



SUBJECT_TYPE = "person" #@param{type: 'string'}
while SUBJECT_TYPE=="":
   SUBJECT_TYPE=input('Input the subject type:')

#@markdown - If you're training on a character or an object, the subject type would be : Man, Woman, Shirt, Car, Dog, Baby ...etc
#@markdown - If you're training on a Style, the subject type would be : impressionist, brutalist, abstract, use "beautiful" for a general style...etc
#@markdown - If you're training on a Movie/Show, the subject type would be : Action, Drama, Science-fiction, Comedy ...etc
#@markdown - If you're training on an Artist, the subject type would be : Painting, sketch, drawing, photography, art ...etc


INSTANCE_NAME= "JOSEJAAM" #@param{type: 'string'}
while INSTANCE_NAME=="":
   INSTANCE_NAME=input('Input the instance name (identifier) :')

#@markdown - The instance is an identifier, choose a unique identifier unknown by stable diffusion. 

INSTANCE_DIR_OPTIONAL="" #@param{type: 'string'}
INSTANCE_DIR=INSTANCE_DIR_OPTIONAL
while INSTANCE_DIR_OPTIONAL!="" and not os.path.exists(str(INSTANCE_DIR)):
    INSTANCE_DIR=input('[1;31mThe instance folder specified does not exist, use the colab file explorer to copy the path :')

#@markdown - If the number of instance pictures is large, it is preferable to specify directly the folder instead of uploading, leave EMPTY to upload.

CLASS_DIR="/content/data/"+ SUBJECT_TYPE
Number_of_subject_images=500#@param{type: 'number'}
while Number_of_subject_images==None:
     Number_of_subject_images=input('Input the number of subject images :')
SUBJECT_IMAGES=Number_of_subject_images

Save_class_images_to_gdrive = False #@param {type:"boolean"}
#@markdown - Save time in case you're training multiple instances of the same class

if Training_Subject=="Character" or Training_Subject=="Object":
  PT="photo of "+INSTANCE_NAME+" "+SUBJECT_TYPE
  CPT="a photo of a "+SUBJECT_TYPE+", ultra detailed"
  if Captionned_instance_images:
    PT="photo of"
elif Training_Subject=="Style":
  With_Prior_Preservation = "No"
  PT="in the "+SUBJECT_TYPE+" style of "+INSTANCE_NAME
  if Captionned_instance_images:
    PT="in the style of"  
elif Training_Subject=="Artist":
  With_Prior_Preservation = "No"
  PT=SUBJECT_TYPE+" By "+INSTANCE_NAME
  if Captionned_instance_images:
    PT="by the artist"  
elif Training_Subject=="Movie":
  PT="from the "+SUBJECT_TYPE+" movie "+ INSTANCE_NAME
  CPT="still frame from "+SUBJECT_TYPE+" movie, ultra detailed, 4k uhd"
  if Captionned_instance_images:
    PT="from the movie"  
elif Training_Subject=="TV Show":
  CPT="still frame from "+SUBJECT_TYPE+" tv show, ultra detailed, 4k uhd"
  PT="from the "+SUBJECT_TYPE+" tv show "+ INSTANCE_NAME
  if Captionned_instance_images:
    PT="from the tv show"    
  
OUTPUT_DIR="/content/models/"+ INSTANCE_NAME

if INSTANCE_DIR_OPTIONAL=="":
  INSTANCE_DIR="/content/data/"+INSTANCE_NAME
  !mkdir -p "$INSTANCE_DIR"
  uploaded = files.upload()
  for filename in uploaded.keys():
    shutil.move(filename, INSTANCE_DIR)
    clear_output()

with capture.capture_output() as cap:
   %cd "$INSTANCE_DIR"
   !find . -name "* *" -type f | rename 's/ /_/g'
   %cd /content
print('[1;32mOK')

```
### **Paso 7** - (Opcional) Descargamos imágenes de regularización.  💖 Gracias [Joe Penna](https://github.com/JoePenna/Dreambooth-Stable-Diffusion)!

Esto es opcional, pero es mejor marcarlo, esto va asociado a la opcion de antes, en el paso 6 " **With_Prior_Preservation** " que hemos puesto " yes " , esto lo que significa es que cuando nosotros vamos a entrenar a Dreamboth para que aprenda el concepto de nuestra cara vamos a estar controlando al mismo tiempo de que Dreamboth no se olvide de como eran las caras de las otras personas que puede generar , si no marcara esta opcion como " yes " todas las personas serian el token que hemos puesto o todos los perros serian tu mascota y no queremos eso.

Entonces la estrategia en este caso, es generar unas cuantas imagenes ( aconseja sobre unas 200) que representen a la clase " perro "  o a la clase " persona " para estar entrenando en paralelo a Dreambooth con imagenes de tu cara pero tambien con imagenes de lo que son personas .

Y donde podemos conseguir un conjunto de datos de personas ? 
Podriamos generaralas ,pero tardariamos bastante , asi que en este caso vamos a utilizar esta opcion que son unos datasets ya creados de diferentes tipos de personas( suelen tener cada dataset sobre 1500 imagenes) 

Como hemos puesto arriba la clase " person " vamos a elegir el dataset " person_ddim " en caso de haber puesto otro tipo de clase podemos probar con alguno de los otros datasets.



En caso de que no coincida con ninguno lo podemos dejar sin ejecutar pero la opcion será bastante mas lenta 

```
#@markdown We’ve created the following image sets
#@markdown - `man_euler` - provided by Niko Pueringer (Corridor Digital) - euler @ 40 steps, CFG 7.5
#@markdown - `man_unsplash` - pictures from various photographers
#@markdown - `person_ddim`
#@markdown - `woman_ddim` - provided by David Bielejeski - ddim @ 50 steps, CFG 10.0 <br />
#@markdown - `blonde_woman` - provided by David Bielejeski - ddim @ 50 steps, CFG 10.0 <br />

dataset="person_ddim" #@param ["man_euler", "man_unsplash", "person_ddim", "woman_ddim", "blonde_woman"]
!git clone https://github.com/djbielejeski/Stable-Diffusion-Regularization-Images-{dataset}.git

!mkdir -p regularization_images/{dataset}
!mv -v Stable-Diffusion-Regularization-Images-{dataset}/{dataset}/*.* regularization_images/{dataset}
CLASS_DIR="/content/regularization_images/" + dataset

```

### **Paso 8** - ...y ahora **¡A ENTRENAR!** 💪

Le damos a ejecutar , sin cambiar nada , si todo esta bien saldra un panel que pondra " TRAINING " , tardará mas o menos 40 minutos 1 hora depende de la tarjeta grafica que nos hayan asignado , SI TODO HA ESTADO BIEN " IMPORTANTE " SI HEMOS DEJADO ESPACIO EN DRIVE ( 2GB) , NOS GUARDARA AL FINAL UN ARCHIVO DEL ENTRENAMIENTO ". CKPT"  EN EL DRIVE.

```
#@markdown ---
import os
from subprocess import getoutput
from IPython.display import HTML

fp16 = True #@param {type:"boolean"}
if fp16:
  prec="fp16"
else:
  prec="no"

#@markdown  - fp16 or half precision meaning slightly lower quality but double the speed.
s = getoutput('nvidia-smi')
if 'A100' in s:
  precision="no"
else:
  precision=prec

Training_Steps="1600" #@param{type: 'string'}
#@markdown - Keep it around 1600 to avoid overtraining.

Seed=75576 #@param{type: 'number'}

#@markdown ---------------------------
Save_Checkpoint_Every_n_Steps = False #@param {type:"boolean"}
Save_Checkpoint_Every=500 #@param{type: 'number'}
if Save_Checkpoint_Every==None:
  Save_Checkpoint_Every=1
#@markdown - Minimum 200 steps between each save.
stp=0
Start_saving_from_the_step=500 #@param{type: 'number'}
if Start_saving_from_the_step==None:
  Start_saving_from_the_step=0
if (Start_saving_from_the_step < 200):
  Start_saving_from_the_step=Save_Checkpoint_Every
stpsv=Start_saving_from_the_step
if Save_Checkpoint_Every_n_Steps:
  stp=Save_Checkpoint_Every
#@markdown - Start saving intermediary checkpoints from this step.

Caption=''
if Captionned_instance_images:
  Caption='--image_captions_filename'

if With_Prior_Preservation=='No':
  !accelerate launch /content/diffusers/examples/dreambooth/train_dreambooth.py \
    $Caption \
    --save_starting_step=$stpsv \
    --save_n_steps=$stp \
    --train_text_encoder \
    --pretrained_model_name_or_path="$MODEL_NAME" \
    --instance_data_dir="$INSTANCE_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --instance_prompt="$PT" \
    --seed=$Seed \
    --resolution=512 \
    --mixed_precision=$precision \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --learning_rate=1e-6 \
    --lr_scheduler="constant" \
    --center_crop \
    --lr_warmup_steps=0 \
    --max_train_steps=$Training_Steps 

else:

  !accelerate launch /content/diffusers/examples/dreambooth/train_dreambooth.py \
    $Caption \
    --save_starting_step=$stpsv \
    --save_n_steps=$stp \
    --train_text_encoder \
    --pretrained_model_name_or_path="$MODEL_NAME" \
    --instance_data_dir="$INSTANCE_DIR" \
    --class_data_dir="$CLASS_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="$PT"\
    --class_prompt="$CPT" \
    --seed=$Seed \
    --resolution=512 \
    --mixed_precision=$precision \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 --gradient_checkpointing \
    --use_8bit_adam \
    --learning_rate=1e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --center_crop \
    --max_train_steps=$Training_Steps \
    --num_class_images=$SUBJECT_IMAGES

if Save_class_images_to_gdrive:
  if os.path.exists(str(CLASS_DIR)):
    if not os.path.exists('/content/gdrive/MyDrive/Class_images'):
      !mkdir /content/gdrive/MyDrive/Class_images
    Class_gdir= '/content/gdrive/MyDrive/Class_images/'+SUBJECT_TYPE
    if not os.path.exists(str(Class_gdir)):
      !cp -r "$CLASS_DIR" /content/gdrive/MyDrive/Class_images

if os.path.exists('/content/models/'+INSTANCE_NAME+'/unet/diffusion_pytorch_model.bin'):
  print("Almost done ...")
  %cd /content    
  !wget -O convertosd.py https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dreambooth/convertosd.py
  clear_output()
  if precision=="no":
    !sed -i '226s@.*@@' /content/convertosd.py
  !sed -i '201s@.*@    model_path = "{OUTPUT_DIR}"@' /content/convertosd.py
  !sed -i '202s@.*@    checkpoint_path= "/content/gdrive/MyDrive/{INSTANCE_NAME}.ckpt"@' /content/convertosd.py
  !python /content/convertosd.py
  clear_output()
  if os.path.exists('/content/gdrive/MyDrive/'+INSTANCE_NAME+'.ckpt'):
    print("[1;32mDONE, the CKPT model is in your Gdrive")
  else:
    print("[1;31mSomething went wrong")
else:
  print("[1;31mSomething went wrong")
  
  ```
  En caso de que despues del entrenamiento no acabe de parecerse a ti, puede ser porque las imagenes no acaban de estar bien definidas, ( probar con cambiar imagenes ) o darle mas trianing steps de entrenamiento ( por ejemplo 3000) pero puede ser que tengas un overfitting y te de mal las imagenes , por lo que habria que bajarlo, por ejemplo (2500) hay que encontrar el numero que le vaya bien.
  
  Despues del entrenamiento en google drive nos aparecera el archivo del entrenamiento que nos podemos llevar a cualquier programa para utilizarlo
  
  ### **Paso 9** (Opcional) - **Prueba el modelo**
  
  Esta es una opcion para probar el modelo , damos a ejecutar **sin tocar nada**,

Lo que hace es crearnos una interfaz para poder utilizar stabble diffusion desde collab .. ( Esto tarda un poco )

Una vez este todo ejecutado abajo nos pone ," Running on local URL " y te da un enlace, que ten lleva a una aplicacion para utlizar stabble diffusion desde collab.

**COMO ESCRIBIR EL TEXTO " PROMPT INPUT "**  
Donde escribes el texto , tienes que escribirlo sobre el token que tu has puesto: por ejemplo si mi token es " JAAAM"  seria  "a photagraph of JAAM", debemos construir inputs complejos para que defina mejor el dibujo, si no , no llega a hacerlos bien. 

imputs de ejemplo que va bien

SUBIR SAMPLING STEPS A 50 ( VA MEJOR LA DEFINICION ) 

🡆 Modelos con traje:   
Close portrait of elegant [tu_token] person in tailored suit- futurist style, intricate baroque detial, elegant, glowing lights, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by wlop, mars ravelo and greg rutkowski

🡆  Ilustración de fantasía:   
Highly detailed portrait of [tu_token], stephen bliss, unreal engine, fantasy art by greg rutkowski, loish, rhads, ferdinand knab, makoto shinkai and lois van baarle, ilya kuvshinov, rossdraws, tom bagshaw, alphonse mucha, global illumination, radiant light, detailed and intricate environment

🡆 Tu funko pop:   
A funko pop of [tu_token]

🡆 Tu personaje de Pixar:    
highly detailed still of [tu_token] as a Pixar movie 3d character, renderman engine


*no marcar la casilla **update_repo** " ya que queremos utilizar el archivo que hemos entrenado antes, esta opcion es para generear desde 0 con una archivo ya preentrenado*" 


```
import os
from IPython.display import clear_output
from subprocess import getoutput
from IPython.utils import capture
import time

Update_repo = False #@param {type:"boolean"}

INSTANCE__NAME="" #@param{type: 'string'}

#@markdown - Leave empty if you want to use the current trained model

if INSTANCE__NAME!="":
  INSTANCE_NAME=INSTANCE__NAME

Use_Custom_Path = False #@param {type:"boolean"}

try:
  INSTANCE_NAME
  if Use_Custom_Path:
    del INSTANCE_NAME
except:
  pass
#@markdown - if checked, an input box will ask the full path to a desired model

try:
  INSTANCE_NAME
  path_to_trained_model='/content/gdrive/MyDrive/'+INSTANCE_NAME+'.ckpt'
except:
  print('[1;31mIt seems that you did not perform training during this session [1;32mor you chose to use a custom path,\nprovide the full path to the model (including the name of the model):\n')
  path_to_trained_model=input()
     
while not os.path.exists(path_to_trained_model):
   print("[1;31mThe model doesn't exist on you Gdrive, use the file explorer to get the path : ")
   path_to_trained_model=input()

         
with capture.capture_output() as cap:
    %cd /content/gdrive/MyDrive/
    %mkdir sd
    %cd sd
    !git clone https://github.com/CompVis/stable-diffusion
    !git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
    %cd /content/gdrive/MyDrive/sd/stable-diffusion-webui/
    !mkdir -p cache/{huggingface,torch}
    %cd /content/
    !ln -s /content/gdrive/MyDrive/sd/stable-diffusion-webui/cache/huggingface ../root/.cache/
    !ln -s /content/gdrive/MyDrive/sd/stable-diffusion-webui/cache/torch ../root/.cache/

if Update_repo:
  !rm /content/gdrive/MyDrive/sd/stable-diffusion-webui/webui.sh  
  !rm /content/gdrive/MyDrive/sd/stable-diffusion-webui/modules/paths.py
  !rm /content/gdrive/MyDrive/sd/stable-diffusion-webui/webui.py 
  !rm /content/gdrive/MyDrive/sd/stable-diffusion-webui/modules/ui.py
  !rm /content/gdrive/MyDrive/sd/stable-diffusion-webui/style.css
  %cd /content/gdrive/MyDrive/sd/stable-diffusion-webui/
  clear_output()
  print('[1;32m')
  !git pull


with capture.capture_output() as cap:  
  if not os.path.exists('/content/gdrive/MyDrive/sd/stable-diffusion/src/k-diffusion/k_diffusion'):
    !mkdir /content/gdrive/MyDrive/sd/stable-diffusion/src
    %cd /content/gdrive/MyDrive/sd/stable-diffusion/src
    !git clone https://github.com/CompVis/taming-transformers
    !git clone https://github.com/openai/CLIP
    !mv /content/gdrive/MyDrive/sd/stable-diffusion/src/CLIP /content/gdrive/MyDrive/sd/stable-diffusion/src/clip
    !git clone https://github.com/TencentARC/GFPGAN
    !mv  /content/gdrive/MyDrive/sd/stable-diffusion/src/GFPGAN/gfpgan /content/gdrive/MyDrive/sd/stable-diffusion-webui
    !git clone https://github.com/salesforce/BLIP
    !mv  /content/gdrive/MyDrive/sd/stable-diffusion/src/BLIP /content/gdrive/MyDrive/sd/stable-diffusion/src/blip
    !git clone https://github.com/sczhou/CodeFormer
    !mv  /content/gdrive/MyDrive/sd/stable-diffusion/src/CodeFormer /content/gdrive/MyDrive/sd/stable-diffusion/src/codeformer
    !git clone https://github.com/xinntao/Real-ESRGAN
    !mv  /content/gdrive/MyDrive/sd/stable-diffusion/src/Real-ESRGAN/ /content/gdrive/MyDrive/sd/stable-diffusion/src/realesrgan
    !git clone https://github.com/crowsonkb/k-diffusion.git
    !cp -r /content/gdrive/MyDrive/sd/stable-diffusion/src/k-diffusion/k_diffusion /content/gdrive/MyDrive/sd/stable-diffusion-webui
    !git clone https://github.com/Hafiidz/latent-diffusion
    !cp -r  /content/gdrive/MyDrive/sd/stable-diffusion/ldm /content/gdrive/MyDrive/sd/stable-diffusion-webui/


with capture.capture_output() as cap:
  if not os.path.exists('/usr/local/lib/python3.7/dist-packages/gradio-3.4b3.dist-info'):
    %cd /content/
    !wget https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dependencies/Dependencies_AUT.1
    !wget https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dependencies/Dependencies_AUT.2
    %mv Dependencies_AUT.1 Dependencies_AUT.7z.001
    %mv Dependencies_AUT.2 Dependencies_AUT.7z.002
    !7z x Dependencies_AUT.7z.001
    time.sleep(2)
    !rm -r /content/usr/local/lib/python3.7/dist-packages/transformers
    !rm -r /content/usr/local/lib/python3.7/dist-packages/transformers-4.19.2.dist-info
    !rm -r /content/usr/local/lib/python3.7/dist-packages/diffusers
    !rm -r /content/usr/local/lib/python3.7/dist-packages/diffusers-0.3.0.dist-info
    !rm -r /content/usr/local/lib/python3.7/dist-packages/accelerate
    !rm -r /content/usr/local/lib/python3.7/dist-packages/accelerate-0.12.0.dist-info    
    !cp -r /content/usr/local/lib/python3.7/dist-packages /usr/local/lib/python3.7/
    !rm -r /content/usr
    !rm Dependencies_AUT.7z.001
    !rm Dependencies_AUT.7z.002
    %cd /content/gdrive/MyDrive/sd/stable-diffusion-webui/ldm/modules
    !wget -O attention.py https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/precompiled/attention.py
    


with capture.capture_output() as cap:
  %cd /content/gdrive/MyDrive/sd/stable-diffusion-webui/modules
  !wget -O paths.py https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/AUTOMATIC1111_files/paths.py
  if not os.path.exists('/tools/node/bin/lt'):
    !npm install -g localtunnel

with capture.capture_output() as cap: 
  %cd /content/gdrive/MyDrive/sd/stable-diffusion-webui/
  time.sleep(1)
  !wget -O webui.py https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.py
  !sed -i 's@gpu_call).*@gpu_call) \n        demo.queue(concurrency_count=111500)@' /content/gdrive/MyDrive/sd/stable-diffusion-webui/webui.py
  %cd /content/gdrive/MyDrive/sd/stable-diffusion-webui/modules/
  !wget -O ui.py https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/modules/ui.py
  !sed -i 's@css = "".*@with open(os.path.join(script_path, "style.css"), "r", encoding="utf8") as file:\n        css = file.read()@' /content/gdrive/MyDrive/sd/stable-diffusion-webui/modules/ui.py  
  %cd /content/gdrive/MyDrive/sd/stable-diffusion-webui
  !wget -O style.css https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/style.css
  !sed -i 's@min-height: 4.*@min-height: 5.5em;@g' /content/gdrive/MyDrive/sd/stable-diffusion-webui/style.css  
  %cd /content


Use_Gradio_Server = False #@param {type:"boolean"}
#@markdown  - Only if you have trouble connecting to the local server


share=''
if Use_Gradio_Server:
  share='--share'
  !sed -i '1037s@.*@            self.server_name = server_name@' /usr/local/lib/python3.7/dist-packages/gradio/blocks.py
  !sed -i '1039s@.*@            self.server_port = server_port@' /usr/local/lib/python3.7/dist-packages/gradio/blocks.py  
  !sed -i '1043s@.*@            self.protocol = "https" if self.local_url.startswith("https") else "http"@' /usr/local/lib/python3.7/dist-packages/gradio/blocks.py  
  clear_output()
  
else:
  share=''

  !nohup lt --port 7860 > srv.txt 2>&1 &
  time.sleep(2)
  !grep -o 'https[^ ]*' /content/srv.txt >srvr.txt
  time.sleep(2)
  srv= getoutput('cat /content/srvr.txt')

  !sed -i '1037s@.*@            self.server_name = "{srv[8:]}"@' /usr/local/lib/python3.7/dist-packages/gradio/blocks.py
  !sed -i '1039s@.*@            self.server_port = 443@' /usr/local/lib/python3.7/dist-packages/gradio/blocks.py
  !sed -i '1043s@.*@            self.protocol = "https"@' /usr/local/lib/python3.7/dist-packages/gradio/blocks.py  
          
  !sed -i '13s@.*@    "PUBLIC_SHARE_TRUE": "[32mConnected",@' /usr/local/lib/python3.7/dist-packages/gradio/strings.py
  
  !rm /content/srv.txt
  !rm /content/srvr.txt
  clear_output()

with capture.capture_output() as cap:
  %cd /content/gdrive/MyDrive/sd/stable-diffusion/

!python /content/gdrive/MyDrive/sd/stable-diffusion-webui/webui.py $share --disable-safe-unpickle --ckpt "$path_to_trained_model"

```


  





