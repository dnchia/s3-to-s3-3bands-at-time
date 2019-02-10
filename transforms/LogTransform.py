import tensorflow as tf


class LogTransform:
    __slots__ = ('bottom_offset', 'domain_max')

    def __init__(self, bottom_offset, domain_max):
        self.bottom_offset = bottom_offset
        self.domain_max = domain_max

    def normalize(self, original_image):
        int_image = tf.cast(original_image, tf.int32)
        normalized = tf.truediv(tf.log(tf.cast(tf.add(self.bottom_offset, int_image), dtype=tf.float32)),
                                tf.log(tf.constant(self.domain_max, dtype=tf.float32)))
        return tf.cast(normalized, tf.float32)

    def denormalize(self, normalized_image):
        denormalized = tf.subtract(tf.round(
            tf.pow(tf.constant(self.domain_max, dtype=tf.float32), normalized_image)), self.bottom_offset)
        return tf.cast(denormalized, dtype=tf.uint16)
