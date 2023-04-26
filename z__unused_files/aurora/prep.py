import tensorflow as tf

class Prep:
    def __init__(self, data_dir, class_names, batch_size, num_train, shape):
        self.data_dir = data_dir
        self.class_names = class_names
        self.batch_size = batch_size
        self.num_train = num_train
        self.shape = shape
        
        
    @classmethod
    def instance_with_info(cls, batch_size, num_train, shape):
        data_dir_path = "./aurora/competition01_gray_128x128/"
        train_val_dir = data_dir_path + "train_val"      # 訓練・検証データが格納されているフォルダを指定します
        class_names = ["aurora", "clearsky", "cloud", "milkyway"]
        test_dir = data_dir_path + "test"                # テストデータが格納されているフォルダを指定します
        data_dir = {"train_val":train_val_dir, "test":test_dir}
        return cls(data_dir, class_names, batch_size, num_train, shape)
        

    def get_classes(self):
        return len(self.class_names)


    # 画像データのサイズ変更と，画素値の正規化を実施する。正規化：0～255 → 0.0～1.0
    def normalize_img(self, img, label):
        IMG_HEIGHT = self.shape[0]   # 画像のサイズ (縦幅) 単位：ピクセル
        IMG_WIDTH = self.shape[1]    # 画像のサイズ (横幅) 単位：ピクセル

        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])     # 画像のリサイズ 順番入れ替えた
        img = tf.cast(img, tf.float32) / 255.0                  # 画像のデータ型を浮動小数点型に変換する (32bitの浮動小数点)

        return img, label       # 正規化した画素値と，画像の正解ラベルを返す


    def preprocess(self, ds):
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


    # 画像ファイルを読み込んだり，モデルに入力するための形式に変換する処理
    def decode_image_from_path(self, file_path):
        # 画像の読み込み
        img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img, channels=3)
        return img, file_path


    def fetch_train_val(self):
        train_val_dir = self.data_dir["train_val"]
        # 訓練・検証データを読み込む    read, decode
        ds_train_val = tf.keras.utils.image_dataset_from_directory(train_val_dir, seed=0, batch_size=None)

        # 訓練データの準備
        ds_train = ds_train_val.take(self.num_train)
        ds_train = ds_train.map(self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = self.preprocess(ds_train)

        # 検証データの準備
        ds_val = ds_train_val.skip(self.num_train)
        ds_val = ds_val.map(self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_val = self.preprocess(ds_val)
        
        return ds_train, ds_val


    def fetch_test(self):
        test_dir = self.data_dir["test"]

        # テストデータの読み込みと，モデルに入力するための形式に変換する処理
        ds_test = tf.data.Dataset.list_files(test_dir + "/*.jpg", shuffle=False)
        ds_test = ds_test.map(self.decode_image_from_path, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = self.preprocess(ds_test)
        
        return ds_test
    

    def fetch_detaset(self):
        ds_train, ds_val = self.fetch_train_val()
        ds_test = self.fetch_test()
        
        return ds_train, ds_val, ds_test
        
    
    


            
            