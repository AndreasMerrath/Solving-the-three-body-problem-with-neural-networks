import tensorflow as tf
import TensorCode.kepler_solver as ks


def kepler_solver(vector):
    if tf.reduce_all(tf.equal(vector, 0)):
        return tf.zeros(6, dtype=tf.float64)
    else:
        tau = vector[0]
        mij = vector[1]
        r0 = vector[2:5]
        v0 = vector[5:8]
        keplerConstant = mij * (6.67418478 * 10 ** -11) * (24 * 60 * 60) ** 2 * 1988500 * 10 ** 24 * (
                1 / (1.496 * 10 ** 11) ** 3)  # In AU/M*d**2

        r0, v0 = ks.kepler_step(keplerConstant, tau, r0, v0)

        # Concatenate the results into a single vector
        return tf.concat([r0, v0], axis=0)

@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(None,), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64)
])
def do_step(tau, n, m, r, v):
    tauDiv2 = tf.multiply(tau, 0.5)
    r = tf.add(r, tf.multiply(v, tauDiv2))
    r, v = evolve_HW(tau, n, m, r, v)
    r = tf.add(r, tf.multiply(v, tauDiv2))
    return r, v


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(None,), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64)
])
def evolve_HW(tau, n, m, r, v):
    maskMatrix2D = 1 - tf.eye(n, dtype=tf.float64)
    maskMatrix3D = tf.expand_dims(maskMatrix2D, 2)

    m = tf.reshape(m, (n, 1))

    mij = m * maskMatrix2D + tf.transpose(m * maskMatrix2D)

    mij_with_1_on_diagonal_instead_of_0 = mij + tf.eye(n, dtype=tf.float64)

    mu = (m * maskMatrix2D) * (tf.transpose(m * maskMatrix2D)) / mij_with_1_on_diagonal_instead_of_0

    r_expanded = tf.expand_dims(r, 1) * maskMatrix3D
    v_expanded = tf.expand_dims(v, 1) * maskMatrix3D

    rr0 = r_expanded - tf.transpose(r_expanded, perm=[1, 0, 2])
    vv0 = v_expanded - tf.transpose(v_expanded, perm=[1, 0, 2])

    r0 = rr0 - vv0 * tau * 0.5

    tau = tf.broadcast_to(tau, (n, n, 1))
    mij = tf.expand_dims(mij, 2)
    concatenated = tf.concat([tau, mij, r0, vv0], axis=2)

    lower_triangular_1_matrix = tf.expand_dims(tf.linalg.band_part(maskMatrix2D, -1, 0), 2)

    concatenated = concatenated * lower_triangular_1_matrix
    concatenated = tf.reshape(concatenated, (-1, 8))

    result = tf.map_fn(kepler_solver, concatenated,
                       fn_output_signature=tf.TensorSpec(shape=None, dtype=tf.float64))

    result = tf.reshape(result, (n, n, 6))
    r1 = result[:, :, :3]
    v1 = result[:, :, 3:]

    r1 = r1 + tf.transpose(-r1, perm=[1, 0, 2])
    v1 = v1 + tf.transpose(-v1, perm=[1, 0, 2])

    rr1 = r1 - (v1 * (tau * 0.5))

    mu = tf.reshape(mu, (n, n, 1))

    dmr = tf.reduce_sum(mu * (rr1 - rr0), 1)
    dmv = tf.reduce_sum(mu * (v1 - vv0), 1)

    r = r + tf.divide(dmr, m)
    v = v + tf.divide(dmv, m)

    return r, v
