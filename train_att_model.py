from att_model import make_att_model
import argparse
import tensorflow as tf
import keras


parser = argparse.ArgumentParser("model params")
parser.add_argument("--input_shape", '-sh', help="input shape: height, width",nargs=2, type=int, default=[256, 256])
parser.add_argument('--num_classes', '-c', help='number of classes: int', type=int, default=3)
parser.add_argument('--batch_size', '-b', help='batch size: int', type=int,default=32)
parser.add_argument('--epochs', '-e', help='number of epochs: int', type=int,default=20)
parser.add_argument('--path', '-p', help='data path', type=str)


def main():
    args = parser.parse_args()
    img_height, img_width = tuple(args.input_shape)
    batch_size = args.batch_size
    num_classes = args.num_classes
    data_path = args.path
    epochs = args.epochs
    seed = 42

    # data_path = 'cov_data'
    if data_path in ('cov_data', 'covid_radiography_dataset'):
        train_ds, val_ds = keras.utils.image_dataset_from_directory(
            data_path,
            validation_split=0.2,
            subset="both",
            label_mode='categorical',
            seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    elif data_path == 'eurosat':
        import tensorflow_datasets as tfds
        (train_ds, val_ds, _), _ = tfds.load(
            name='eurosat',
            split=['train[:60%]', 'train[60%:80%]','train[80%:]'],
            with_info=True,
            as_supervised=True,
            batch_size=32,
        )
        img_height, img_width = 64, 64
        num_classes = 10
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    model = make_att_model(img_height, img_width, num_classes, batch_size)

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        # callbacks=keras.callbacks.EarlyStopping(patience=4),
        batch_size=batch_size
    )
    
    weights_file = f'{data_path}/weights/att_model.weights.h5'

    print(f"Training finished, saving weights to {weights_file}")
    model.save_weights(weights_file)



if __name__ == "__main__":
    main()