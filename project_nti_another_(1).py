from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import cv2
import numpy as np
from PIL import Image
import io
import os # Import os module

def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = '📸 Capture';
            div.appendChild(capture);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            // Resize video to square
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            await new Promise((resolve) => capture.onclick = resolve);

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getTracks().forEach(track => track.stop());
            div.remove();

            return canvas.toDataURL('image/jpeg', quality);
        }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

# Test capturing
from IPython.display import Image as IPyImage
image_path = take_photo()
print(f"Image saved to: {image_path}") # Added print statement

# Check if the file exists after saving
if os.path.exists(image_path):
    print(f"File {image_path} found.")
else:
    print(f"File {image_path} not found.")


IPyImage(filename=image_path)

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/TachibanaYoshino/AnimeGANv2.git
# %cd AnimeGANv2

import torch
from PIL import Image
from torchvision import transforms

# Download model and style transfer function
generator = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512)

# Load your webcam photo
input_img = Image.open("/content/photo.jpg").convert("RGB")

# Apply style transfer
output_img = face2paint(generator, input_img)

# Save and show result
output_img.save("anime_result.jpg")

from IPython.display import Image as IPyImage
IPyImage("anime_result.jpg")

config_text = """
dataset: my_dataset
light: False
init_lr: 1e-4
epoch: 100
batch_size: 4
img_size: 256
lambda_gray: 0
lambda_color: 10
lambda_tv: 1
lambda_adv: 300
lambda_content: 1
g_adv_weight: 1
style: my_style
"""
with open("train_my_dataset.yaml", "w") as f:
    f.write(config_text)

import yaml

# Load the YAML config
with open("train_my_dataset.yaml", "r") as f:
    config = yaml.safe_load(f)

# Config values from YAML
dataset_name = config['dataset']
light = config['light']
init_lr = config['init_lr']
epochs = config['epoch']
batch_size = config['batch_size']
img_size = config['img_size']
lambda_gray = config['lambda_gray']
lambda_color = config['lambda_color']
lambda_tv = config['lambda_tv']
lambda_adv = config['lambda_adv']
lambda_content = config['lambda_content']
g_adv_weight = config['g_adv_weight']
style = config['style']

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d splcher/animefacedataset
!unzip animefacedataset.zip

!kaggle datasets download -d jessicali9530/celeba-dataset
!unzip celeba-dataset.zip -d celeba_dataset

import os
from glob import glob

anime_paths = glob('/content/images/*.jpg')
human_paths = glob('/content/celeba_dataset/img_align_celeba/img_align_celeba/*.jpg')

print("Anime images:", len(anime_paths))
print("Human images:", len(human_paths))

import torch
from torchvision import transforms, datasets # Import datasets
from torch.utils.data import DataLoader # Import DataLoader
import os # Import os module

# Use batch_size, img_size, etc.
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# Assuming you already organized dataset under data/trainA and data/trainB
trainA_path = "/content/AnimeGANv2/animegan2-pytorch/data/trainA"
trainB_path = "/content/AnimeGANv2/animegan2-pytorch/data/trainB"

# Check if directories exist and contain files
if not os.path.exists(trainA_path):
    print(f"Error: Directory not found: {trainA_path}")
elif not os.listdir(trainA_path):
    print(f"Error: Directory is empty: {trainA_path}")
else:
    dataset_real = datasets.ImageFolder(trainA_path, transform=transform)
    loader_real = DataLoader(dataset_real, batch_size=batch_size, shuffle=True)
    print(f"Successfully loaded dataset from: {trainA_path}")


if not os.path.exists(trainB_path):
    print(f"Error: Directory not found: {trainB_path}")
elif not os.listdir(trainB_path):
    print(f"Error: Directory is empty: {trainB_path}")
else:
    dataset_anime = datasets.ImageFolder(trainB_path, transform=transform)
    loader_anime = DataLoader(dataset_anime, batch_size=batch_size, shuffle=True)
    print(f"Successfully loaded dataset from: {trainB_path}")

!pip install opencv-python

import cv2
from IPython.display import display, clear_output
import time
from PIL import Image
import numpy as np

# Load model again (if session restarted)
generator = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512)

# Setup webcam (for local use, replace with cv2.VideoCapture(0))
from google.colab.output import eval_js
from IPython.display import Javascript
from base64 import b64decode
import io

def take_webcam_frame():
    display(Javascript('''
        async function captureWebcamFrame() {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = '🎥 Capture Frame';
            div.appendChild(capture);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            await new Promise((resolve) => capture.onclick = resolve);

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getTracks().forEach(track => track.stop());
            div.remove();
            return canvas.toDataURL('image/jpeg', 0.8);
        }
        '''))

    data = eval_js('captureWebcamFrame()')
    binary = b64decode(data.split(',')[1])
    return Image.open(io.BytesIO(binary)).convert("RGB")

# Simulate live processing loop
for i in range(10):  # You can increase this to 10+ for testing more frames
    print(f"⏳ Capturing frame {i+1}...")
    frame = take_webcam_frame()

    print("🎨 Applying anime style...")
    anime_frame = face2paint(generator, frame)

    anime_frame.save(f"anime_frame_{i+1}.jpg")
    clear_output(wait=True)
    display(anime_frame)

    time.sleep(1)

!pip install -q kaggle
from google.colab import files
files.upload()  # Upload your kaggle.json file here
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d splcher/animefacedataset
!unzip -q animefacedataset.zip -d anime_faces

!kaggle datasets download -d jessicali9530/celeba-dataset
!unzip -q celeba-dataset.zip -d celeba_faces

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/AnimeGANv2

!mkdir -p AnimeGANv2/dataset/my_dataset/trainA
!mkdir -p AnimeGANv2/dataset/my_dataset/trainB

!find /content/AnimeGANv2/celeba_faces/img_align_celeba/img_align_celeba -name "*.jpg" | head -n 5000 | xargs -I {} cp {} /content/AnimeGANv2/dataset/my_dataset/trainA/

!ls /content/AnimeGANv2/dataset/my_dataset/trainA

!find /content/AnimeGANv2/anime_faces/images -name "*.jpg" | head -n 5000 | xargs -I {} cp {} /content/AnimeGANv2/dataset/my_dataset/trainB

!ls /content/AnimeGANv2/dataset/my_dataset/trainB

config_text = """
dataset: my_dataset
light: False
init_lr: 1e-4
epoch: 20
batch_size: 4
img_size: 256
lambda_gray: 0
lambda_color: 10
lambda_tv: 1
lambda_adv: 300
lambda_content: 1
g_adv_weight: 1
style: my_style
"""
with open("/content/AnimeGANv2/train_my_dataset.yaml", "w") as f:
    f.write(config_text)

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/AnimeGANv2

!python train.py --yaml /content/AnimeGANv2/train_my_dataset.yaml

# Commented out IPython magic to ensure Python compatibility.
# Clone the repository
!git clone https://github.com/bryandlee/animegan2-pytorch.git
# %cd animegan2-pytorch

# Copy your Kaggle datasets over
!mkdir -p data/trainA data/trainB
!cp -r /content/AnimeGANv2/dataset/my_dataset/trainA/* data/trainA/
!cp -r /content/AnimeGANv2/dataset/my_dataset/trainB/* data/trainB/

import os
import shutil

os.makedirs("data/trainA", exist_ok=True)
os.makedirs("data/trainB", exist_ok=True)

# Move or copy data (adjust paths if needed)
shutil.copytree("/content/AnimeGANv2/celeba_faces/img_align_celeba", "data/trainA", dirs_exist_ok=True)
shutil.copytree("/content/AnimeGANv2/anime_faces/images", "data/trainB", dirs_exist_ok=True)

!pip install -r requirements.txt

import tensorflow as tf
import numpy as np
from PIL import Image
import os

IMG_SIZE = 256
BATCH_SIZE = 1

def load_and_preprocess(path):
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32)
    img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
    return img

# Generator function
def image_gen(image_paths):
    for path in image_paths:
        try:
            yield load_and_preprocess(path)
        except:
            continue  # Skip corrupted files

# Create TensorFlow datasets
anime_dataset = tf.data.Dataset.from_generator(
    lambda: image_gen(anime_paths),
    output_signature=tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
)

human_dataset = tf.data.Dataset.from_generator(
    lambda: image_gen(human_paths),
    output_signature=tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
)

# Shuffle, repeat, and batch
anime_dataset = anime_dataset.shuffle(1000).repeat().batch(BATCH_SIZE)
human_dataset = human_dataset.shuffle(1000).repeat().batch(BATCH_SIZE)

# Generators
generator_g = pix2pix.unet_generator(3, norm_type='instancenorm')  # human → anime
generator_f = pix2pix.unet_generator(3, norm_type='instancenorm')  # anime → human
def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result

def build_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    x = downsample(64, 4, False)(inp)
    x = downsample(128, 4)(x)
    x = downsample(256, 4)(x)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(x)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)

    norm1 = tfa.layers.InstanceNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=inp, outputs=last)

# Discriminators
discriminator_x = build_discriminator()  # human
discriminator_y = build_discriminator()  # anime

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA = 10

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    return (real_loss + generated_loss) * 0.5

def calc_cycle_loss(real, cycled):
    return LAMBDA * tf.reduce_mean(tf.abs(real - cycled))

def identity_loss(real, same):
    return LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real - same))

# Optimizers
gen_g_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
gen_f_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_x_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_y_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # Losses
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)
        cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
        id_loss = identity_loss(real_x, same_x) + identity_loss(real_y, same_y)

        total_g_loss = gen_g_loss + cycle_loss + id_loss
        total_f_loss = gen_f_loss + cycle_loss + id_loss

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Apply gradients
    gen_g_opt.apply_gradients(zip(tape.gradient(total_g_loss, generator_g.trainable_variables), generator_g.trainable_variables))
    gen_f_opt.apply_gradients(zip(tape.gradient(total_f_loss, generator_f.trainable_variables), generator_f.trainable_variables))
    disc_x_opt.apply_gradients(zip(tape.gradient(disc_x_loss, discriminator_x.trainable_variables), discriminator_x.trainable_variables))
    disc_y_opt.apply_gradients(zip(tape.gradient(disc_y_loss, discriminator_y.trainable_variables), discriminator_y.trainable_variables))

import time

EPOCHS = 3

train_human_iter = iter(human_dataset)
train_anime_iter = iter(anime_dataset)

for epoch in range(EPOCHS):
    start = time.time()
    print(f"Epoch {epoch+1}/{EPOCHS}")

    for _ in range(1000):  # 1000 steps per epoch (tune later)
        real_human = next(train_human_iter)
        real_anime = next(train_anime_iter)
        train_step(real_human, real_anime)

    print(f" Epoch {epoch+1} completed in {time.time()-start:.2f} seconds\n")
