import tensorflow as tf


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


if __name__ == "__main__":
    x = tf.random.uniform((1, 3))
    temp = create_look_ahead_mask(x.shape[1])
    print(temp)
