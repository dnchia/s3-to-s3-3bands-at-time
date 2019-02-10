import argparse
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from imageio import imwrite

from models.ThreeLayerCNN import ThreeLayerCNN
from providers.BatchSubimageProvider import BatchSubimageProvider
from transforms.LinearTransform import LinearTransform
from transforms.LogTransform import LogTransform

operation_train = 'train'
operation_evaluate = 'evaluate'

log_dir = 'logs'
save_file = os.path.join('savedata', 'model.ckpt')
output_dir = 'output'

images_per_place = 196
images_per_row = 14

target_band = 0

s3_domain_min = 0
s3_domain_max = 65535

s2_transform_offset = 1
s2_domain_max = 65535

batch_size = 100
beta = 0.1
learning_rate = 0.0001
summary_every = 10
epochs = 10000


def main():
    parser = argparse.ArgumentParser(description='Encapsulates a CNN model to transform S3 -> S2 satellite images')
    parser.add_argument('--evaluate', dest='operation', action='store_const', const=operation_evaluate,
                        default=operation_train, help='apply S3 -> S2 transform to real images')
    args = parser.parse_args()

    bsp = BatchSubimageProvider('subimages', target_bands=[target_band])
    s3_transform = LinearTransform(s3_domain_min, s3_domain_max)
    s2_trasform = LogTransform(s2_transform_offset, s2_domain_max)
    three_layer_cnn = ThreeLayerCNN(bsp, s3_transform, s2_trasform, batch_size=batch_size, beta=beta,
                                    learning_rate=learning_rate, save_file=save_file)

    if args.operation is operation_train:
        train(three_layer_cnn)
    else:
        evaluate(three_layer_cnn, bsp)


def train(cnn):
    log_writer = tf.summary.FileWriter(create_folder(log_dir), graph=cnn.session.graph)
    training_loss_values, evaluation_loss_values = cnn.train_model(epochs=epochs, summary_every=summary_every,
                                                                   log_writer=log_writer)

    plt.plot(range(0, epochs, summary_every), training_loss_values, 'b-', label='Training MSE')
    plt.plot(range(0, epochs, summary_every), evaluation_loss_values, 'r-', label='Evaluation MSE')
    plt.legend(loc='upper right', prop={'size': 11})
    plt.show()


def evaluate(cnn, image_provider):
    output = cnn.evaluate(image_provider.evaluation_input_subimages)

    create_folder(output_dir)
    for i in range(image_provider.evaluation_subimage_count() // images_per_place):
        place_dir = create_folder(os.path.join(output_dir, 'place_{:02d}'.format(i)))
        for image_num, image_index in enumerate(range(i*images_per_place, i*images_per_place+images_per_place)):
            row = image_num // images_per_row + 1
            col = image_num % images_per_row + 1
            imwrite(os.path.join(place_dir, '{}_{}.tif'.format(row, col)), output[image_index, :, :, 0])


def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder


if __name__ == '__main__':
    main()
