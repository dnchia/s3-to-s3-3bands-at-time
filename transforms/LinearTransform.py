import tensorflow as tf


class LinearTransform:
    __slots__ = ('domain_min', 'domain_max')

    def __init__(self, domain_min, domain_max):
        self.domain_min = domain_min
        self.domain_max = domain_max

    def normalize(self, original_image):
        int_image = tf.cast(original_image, tf.int32)
        normalized = tf.truediv(tf.subtract(int_image, self.domain_min), self.domain_max - self.domain_min)
        return tf.cast(normalized, tf.float32)

    def denormalize(self, normalized_image):
        denormalized = tf.add(tf.multiply(normalized_image, self.domain_max - self.domain_min), self.domain_min)
        return tf.cast(denormalized, tf.uint16)
