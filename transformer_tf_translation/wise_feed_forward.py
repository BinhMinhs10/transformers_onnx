import tensorflow as tf


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


if __name__ == "__main__":
    sample_ffn = point_wise_feed_forward_network(512, 2048)
    print(
        sample_ffn(tf.random.uniform((64, 50, 512))).shape
    )