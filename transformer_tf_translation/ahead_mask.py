import tensorflow as tf


# mask future tokens in seq, to predict third word only the first and second word will be used.
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


if __name__ == "__main__":
    x = tf.random.uniform((1, 3))
    temp = create_look_ahead_mask(x.shape[1])
    print(temp)
