import tensorflow as tf
import matplotlib.pyplot as plt

file_path = "aurora/competition01_gray_128x128/test/c01_20090607201012.jpg"
# test_dir = "aurora/competition01_gray_128x128/test"

# 画像の読み込み
img = tf.io.read_file(file_path)
img = tf.io.decode_jpeg(img, channels=3)
# 解像度を変更し，値の正規化を実施する
# img = tf.image.resize(img, [85, 128])
# img = tf.cast(img, tf.float32) / 255.0

IMG_HEIGHT = 85
IMG_WIDTH = 85

offset_height = 0
offset_width = 21
target_height = IMG_HEIGHT
target_width = IMG_WIDTH
img = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)

print(img)
# 特定の範囲の画素値を変更
img = tf.Variable(img)
for i in range(IMG_HEIGHT):
    for j in range(IMG_WIDTH):
        img[i, j].assign(tf.constant([0, 0, 50], dtype=tf.uint8))

img = tf.io.encode_jpeg(img)
tf.io.write_file('output.jpg', img)
