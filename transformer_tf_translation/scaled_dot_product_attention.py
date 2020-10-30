import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    """calculate the attention weight."""
    matmul_qk = tf.matpul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the score
    # add up to 1
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_v)
    return output, attention_weights
