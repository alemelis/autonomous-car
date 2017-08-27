import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your NN.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and
                     "saved_model.pb"
    :return: Tuple of Tensors from VGG model
             (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag],vgg_path)

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    graph = tf.get_default_graph()
    vgg_input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return (vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor,
            vgg_layer4_out_tensor, vgg_layer7_out_tensor)
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.
    Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # Complete encoder with a 1x1 convolution
    num_outputs = 512
    k_size = 1
    stride = (1, 1)
    one_by_one = tf.layers.conv2d(vgg_layer7_out, num_outputs, k_size, stride)

    # Decoder: transposed convolution - https://arxiv.org/pdf/1505.04366.pdf
    k_size = 2
    stride = (2, 2)
    t_conv1 = tf.layers.conv2d_transpose(one_by_one, num_outputs, k_size,
                                         stride)
    skip1 = tf.add(t_conv1, vgg_layer4_out)

    num_outputs = 256
    t_conv2 = tf.layers.conv2d_transpose(skip1, num_outputs, k_size, stride)
    skip2 = tf.add(t_conv2, vgg_layer3_out)

    k_size = 8
    stride = (8, 8)
    num_outputs = num_classes
    t_conv3 = tf.layers.conv2d_transpose(skip2, num_outputs, k_size, stride)

    return t_conv3
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                        labels=labels, logits=logits))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return (logits, train_op, cross_entropy_loss)
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image, correct_label, keep_prob,
             learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.
                           Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(epochs):
        print("Epoch: {}".format(i))
        for x_train, y_train in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                            feed_dict={input_image: x_train,
                                       correct_label: y_train,
                                       keep_prob: 1.0})
            print("--- loss: {}".format(loss))
    pass
get_batches_fn = helper.gen_batch_function('./data', (160, 576))
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir,
                                                   'data_road/training'),
                                                   image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        (vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor,
                vgg_layer4_out_tensor, vgg_layer7_out_tensor) = load_vgg(sess,
                                                                       vgg_path)

        nn_last_layer = layers(vgg_layer3_out_tensor, vgg_layer4_out_tensor,
                         vgg_layer7_out_tensor, num_classes)

        learning_rate = 1e-4
        correct_label = tf.placeholder(tf.float32, shape = [None,
                                                            image_shape[0],
                                                            image_shape[1],
                                                            num_classes])

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer,
                                                        correct_label,
                                                        learning_rate,
                                                        num_classes)

        # TODO: Train NN using the train_nn function
        epochs = 30
        batch_size = 5
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
                 cross_entropy_loss, vgg_input_tensor, correct_label,
                 vgg_keep_prob_tensor, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape,
                                      logits, vgg_keep_prob_tensor,
                                      vgg_input_tensor)


if __name__ == '__main__':
    run()
