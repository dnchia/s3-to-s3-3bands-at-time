import argparse
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from imageio import imwrite

from models.ThreeLayerCNN_4D_4layers import ThreeLayerCNN
from providers.BatchSubimageProvider import BatchSubimageProvider
from transforms.LinearTransform import LinearTransform
from transforms.LogTransform import LogTransform

operation_train = 'train'
operation_evaluate = 'evaluate'

log_dir = 'logs'

images_per_place = 196
images_per_row = 14

s3_domain_min = 0
s3_domain_max = 65535

s2_transform_offset = 1
s2_domain_max = 65535

batch_size = 100
beta = 0.1
learning_rate = 0.0001
summary_every = 1
epochs = 5

first_layer_value = 32
second_layer_value = 16
third_layer_value = 3
#fourth_layer_value = 3

output_dir = 'output'
save_file = 'savedata'


def main():
    parser = argparse.ArgumentParser(description='Encapsulates a CNN model to transform S3 -> S2 satellite images')
    parser.add_argument('--evaluate', dest='operation', action='store_const', const=operation_evaluate,
                        default=operation_train, help='apply S3 -> S2 transform to real images')
    #parser.add_argument('first_layer_value', type=int, default=64)
    #parser.add_argument('second_layer_value', type=int, default=32)
    #parser.add_argument('third_layer_value', type=int, default=20)
    #parser.add_argument('fourth_layer_value', type=int, default=3)
    parser.add_argument('first_layer_value', type=int, default=32)
    parser.add_argument('second_layer_value', type=int, default=16)
    parser.add_argument('third_layer_value', type=int, default=3)
    args = parser.parse_args()

    first_layer = (5, 5, args.first_layer_value)
    second_layer = (5, 5, args.second_layer_value)
    third_layer = (5, 5, args.third_layer_value)
    #fourth_layer = (5, 5, args.fourth_layer_value)

    #save_file = os.path.join('savedata_{0}_{1}_{2}_{3}'.format(args.first_layer_value, args.second_layer_value, args.third_layer_value, args.fourth_layer_value), 'model.ckpt')
    save_file = os.path.join(
        'savedata_{0}_{1}_{2}'.format(args.first_layer_value, args.second_layer_value, args.third_layer_value), 'model.ckpt')
    #log_dir = 'logs_{0}_{1}_{2}_{3}'.format(args.first_layer_value, args.second_layer_value, args.third_layer_value, args.fourth_layer_value)
    log_dir = 'logs_{0}_{1}_{2}'.format(args.first_layer_value, args.second_layer_value, args.third_layer_value)
    print(save_file)
    print(first_layer)
    print(second_layer)
    print(third_layer)
    #print(fourth_layer)

    bsp = BatchSubimageProvider('/home/dnchia/TII/Mi versión/s3-to-s2-3bands-at-time/subimages/', target_bands=[0, 1, 2])
    s3_transform = LinearTransform(s3_domain_min, s3_domain_max)
    s2_trasform = LogTransform(s2_transform_offset, s2_domain_max)
    three_layer_cnn = ThreeLayerCNN(bsp, s3_transform, s2_trasform, first_layer_shape=first_layer, second_layer_shape=second_layer, 
    #third_layer_shape=third_layer, fourth_layer_shape=fourth_layer, batch_size=batch_size, beta=beta, learning_rate=learning_rate,
                                    third_layer_shape=third_layer,
                                    batch_size=batch_size, beta=beta, learning_rate=learning_rate,
    save_file=save_file)

    if args.operation is operation_train:
        train(three_layer_cnn)
    else:
        #evaluate(three_layer_cnn, bsp, [args.first_layer_value, args.second_layer_value, args.third_layer_value, args.fourth_layer_value])
        evaluate(three_layer_cnn, bsp,
                 [args.first_layer_value, args.second_layer_value, args.third_layer_value])

def train(cnn):
    log_writer = tf.summary.FileWriter(create_folder(log_dir), graph=cnn.session.graph)
    training_loss_values, evaluation_loss_values = cnn.train_model(epochs=epochs, summary_every=summary_every,
                                                                   log_writer=log_writer)

    plt.plot(range(0, epochs, summary_every), training_loss_values, 'b-', label='Training MSE')
    plt.plot(range(0, epochs, summary_every), evaluation_loss_values, 'r-', label='Evaluation MSE')
    plt.legend(loc='upper right', prop={'size': 11})
    plt.show()


def evaluate(cnn, image_provider, layers):
    output = cnn.evaluate(image_provider.evaluation_input_subimages)
    #output_dir = 'output_{0}_{1}_{2}_{3}'.format(layers[0], layers[1], layers[2], layers[3])
    output_dir = 'output_{0}_{1}_{2}'.format(layers[0], layers[1], layers[2])
    print(output_dir)
    create_folder(output_dir)
    print(image_provider.evaluation_subimage_count())
    print(images_per_place)

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
