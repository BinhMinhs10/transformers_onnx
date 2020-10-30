import tensorflow as tf


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits
    return seq[:, tf.newaxis, tf.newaxis, :]


if __name__ == "__main__":
    x = tf.constant([[7, 6, 0, 0, 1],
                     [1, 2, 4, 0, 0],
                     [0, 0, 0, 4, 5]]
    )
    print(create_padding_mask(x))
