import tensorflow as tf


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64)
])
def solve_universal_newton(r0, beta, b, eta, zeta, h, X):
    xnew = X
    count = tf.constant(1)

    x = xnew
    arg = b * x / 2.0
    s2 = tf.sin(arg)
    c2 = tf.cos(arg)
    g1 = 2.0 * s2 * c2 / b
    g2 = 2.0 * s2 * s2 / beta
    g3 = (x - g1) / beta
    cc = eta * g1 + zeta * g2
    xnew = (h + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

    while count <= 10 and tf.abs((x - xnew) / xnew) > 1.e-8:
        x = xnew
        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        xnew = (h + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)
        count = count + 1
    if count > 10:
        return x, s2, c2, tf.constant(
            False, dtype=tf.bool)
    else:
        x = xnew
        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return x, s2, c2, tf.constant(
            True, dtype=tf.bool)


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64)
])
def solve_universal_laguerre(r0, beta, b, eta, zeta, h, X):
    xnew = X
    count = tf.constant(1)

    c5 = 5.0
    c16 = 16.0
    c20 = 20.0

    x = xnew
    arg = b * x / 2.0
    s2 = tf.sin(arg)
    c2 = tf.cos(arg)
    g1 = 2.0 * s2 * c2 / b
    g2 = 2.0 * s2 * s2 / beta
    g3 = (x - g1) / beta
    f = r0 * x + eta * g2 + zeta * g3 - h
    fp = r0 + eta * g1 + zeta * g2
    g0 = 1.0 - beta * g2
    fpp = eta * g0 + zeta * g1
    dx = -c5 * f / (fp + tf.sqrt(tf.abs(c16 * fp * fp - c20 * f * fpp)))
    xnew = x + dx

    while count <= 10 and tf.abs(dx) > 2.e-7 * tf.abs(xnew):
        x = xnew
        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        f = r0 * x + eta * g2 + zeta * g3 - h
        fp = r0 + eta * g1 + zeta * g2
        g0 = 1.0 - beta * g2
        fpp = eta * g0 + zeta * g1
        dx = -c5 * f / (fp + tf.sqrt(tf.abs(c16 * fp * fp - c20 * f * fpp)))
        xnew = x + dx
        count = count + 1
    if count > 10:
        return x, s2, c2, tf.constant(
            False, dtype=tf.bool)
    else:
        x = xnew
        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        xnew = (h + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

        x = xnew
        arg = b * x / 2.0
        s2 = tf.sin(arg)
        c2 = tf.cos(arg)
        return x, s2, c2, tf.constant(
            True, dtype=tf.bool)


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64)
])
def cubic1(a, b, c):
    Q = (a * a - 3.0 * b) / 9.0
    R = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0
    if R * R < Q * Q * Q:
        tf.print("cubic1 failed")
        return tf.constant(0.0, dtype=tf.float64)
    else:
        A = -tf.sign(R) * tf.pow(tf.abs(R) + tf.sqrt(R * R - Q * Q * Q), 1. / 3.)
        if A == 0.0:
            B = tf.constant(0.0, dtype=tf.float64)
        else:
            B = Q / A
        x1 = (A + B) - a / 3.0
        return x1


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64)
])
def solve_universal_parabolic(r0, eta, zeta, h):
    x = cubic1(3.0 * eta / zeta, 6.0 * r0 / zeta, -6.0 * h / zeta)
    s2 = tf.constant(0.0, dtype=tf.float64)
    c2 = tf.constant(1.0, dtype=tf.float64)
    return x, s2, c2, tf.constant(
        True, dtype=tf.bool)


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64)
])
def solve_universal_hyperbolic_newton(r0, minus_beta, b, eta, zeta, h, X):
    xnew = X
    count = tf.constant(1)

    x = xnew
    arg = b * x / 2.0
    if tf.abs(arg) > 200.0:
        return tf.constant(0.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64), tf.constant(0.0,
                                                                                                   dtype=tf.float64), tf.constant(
            False, dtype=tf.bool)
    else:
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / minus_beta
        g3 = -(x - g1) / minus_beta
        g = eta * g1 + zeta * g2
        xnew = (x * g - eta * g2 - zeta * g3 + h) / (r0 + g)

        while count <= 10 and tf.abs(x - xnew) > 1.e-9 * tf.abs(xnew):
            x = xnew
            arg = b * x / 2.0
            if tf.abs(arg) > 200.0:
                count = tf.constant(10)
            s2 = tf.sinh(arg)
            c2 = tf.cosh(arg)
            g1 = 2.0 * s2 * c2 / b
            g2 = 2.0 * s2 * s2 / minus_beta
            g3 = -(x - g1) / minus_beta
            g = eta * g1 + zeta * g2
            xnew = (x * g - eta * g2 - zeta * g3 + h) / (r0 + g)
            count = count + 1
        if count >= 10 or tf.abs(arg) > 200.0:
            return x, s2, c2, tf.constant(
                False, dtype=tf.bool)
        else:
            x = xnew
            arg = b * x / 2.0
            s2 = tf.sinh(arg)
            c2 = tf.cosh(arg)
            return x, s2, c2, tf.constant(True, dtype=tf.bool)


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64)
])
def solve_universal_hyperbolic_laguerre(r0, minus_beta, b, eta, zeta, h, X):
    xnew = X
    count = tf.constant(1)

    c5 = 5.0
    c16 = 16.0
    c20 = 20.0

    x = xnew

    arg = b * x / 2.0
    if tf.abs(arg) > 50.0:
        return tf.constant(0.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64), tf.constant(0.0,
                                                                                                   dtype=tf.float64), tf.constant(
            False, dtype=tf.bool)

    s2 = tf.sinh(arg)
    c2 = tf.cosh(arg)
    g1 = 2.0 * s2 * c2 / b
    g2 = 2.0 * s2 * s2 / minus_beta
    g3 = -(x - g1) / minus_beta
    f = r0 * x + eta * g2 + zeta * g3 - h
    fp = r0 + eta * g1 + zeta * g2
    g0 = 1.0 + minus_beta * g2
    fpp = eta * g0 + zeta * g1
    den = (fp + tf.sqrt(tf.abs(c16 * fp * fp - c20 * f * fpp)))
    if den == 0.0:
        return tf.constant(0.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64), tf.constant(0.0,
                                                                                                   dtype=tf.float64), tf.constant(
            False, dtype=tf.bool)
    dx = -c5 * f / den
    xnew = x + dx

    while count <= 20 and tf.abs(x - xnew) > 1.e-9 * tf.abs(xnew):
        c5 = 5.0
        c16 = 16.0
        c20 = 20.0

        x = xnew

        arg = b * x / 2.0
        if tf.abs(arg) > 50.0:
            count = tf.constant(20)
        s2 = tf.sinh(arg)
        c2 = tf.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / minus_beta
        g3 = -(x - g1) / minus_beta
        f = r0 * x + eta * g2 + zeta * g3 - h
        fp = r0 + eta * g1 + zeta * g2
        g0 = 1.0 + minus_beta * g2
        fpp = eta * g0 + zeta * g1
        den = (fp + tf.sqrt(tf.abs(c16 * fp * fp - c20 * f * fpp)))
        if den == 0.0:
            count = tf.constant(20)
        dx = -c5 * f / den
        xnew = x + dx
        count = count + 1

    if tf.abs(arg) > 50.0 or den == 0.0 or count > 20:
        return tf.constant(0.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64), tf.constant(0.0,
                                                                                                   dtype=tf.float64), tf.constant(
            False, dtype=tf.bool)

    x = xnew
    arg = b * x / 2.0
    if tf.abs(arg) > 200.0:
        return tf.constant(0.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64), tf.constant(0.0,
                                                                                                   dtype=tf.float64), tf.constant(
            False, dtype=tf.bool)
    s2 = tf.sinh(arg)
    c2 = tf.cosh(arg)
    g1 = 2.0 * s2 * c2 / b
    g2 = 2.0 * s2 * s2 / minus_beta
    g3 = -(x - g1) / minus_beta
    g = eta * g1 + zeta * g2
    xnew = (x * g - eta * g2 - zeta * g3 + h) / (r0 + g)

    x = xnew
    arg = b * x / 2.0
    s2 = tf.sinh(arg)
    c2 = tf.cosh(arg)
    return x, s2, c2, tf.constant(True, dtype=tf.bool)


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64)
])
def new_guess(r0, eta, zeta, dt):
    if zeta != 0.0:
        s = cubic1(3.0 * eta / zeta, 6.0 * r0 / zeta, -6.0 * dt / zeta)
    elif eta != 0.0:
        reta = r0 / eta
        disc = reta * reta + 8.0 * dt / eta
        if disc >= 0.0:
            s = -reta + tf.sqrt(disc)
        else:
            s = dt / r0
    else:
        s = dt / r0
    return s


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(3), dtype=tf.float64),
    tf.TensorSpec(shape=(3), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64)
])
def kepler_step_internal(kc, dt, beta, b, directionVector, velocityVector, r0, eta, zeta):
    G1 = tf.constant(0.0, dtype=tf.float64)
    G2 = tf.constant(0.0, dtype=tf.float64)
    bsa = tf.constant(0.0, dtype=tf.float64)
    ca = tf.constant(0.0, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)
    returnBool = tf.constant(True, dtype=tf.bool)
    if beta < 0.0:
        x0 = new_guess(r0, eta, zeta, dt)
        x = tf.identity(x0)
        x, s2, c2, did_it_work = solve_universal_hyperbolic_newton(r0, -beta, b, eta, zeta, dt, x)
        if tf.equal(did_it_work, False):
            x = tf.identity(x0)
            x, s2, c2, did_it_work = solve_universal_hyperbolic_laguerre(r0, -beta, b, eta, zeta, dt, x)
        if tf.equal(did_it_work, False):
            returnBool = tf.constant(False, dtype=tf.bool)
        else:
            a = kc / (-beta)
            G1 = 2.0 * s2 * c2 / b
            c = 2.0 * s2 * s2
            G2 = c / (-beta)
            ca = c * a
            r = r0 + eta * G1 + zeta * G2
            bsa = (a / r) * (b / r0) * 2.0 * s2 * c2
    elif beta > 0.0:
        x0 = dt / r0
        ff = zeta * x0 * x0 * x0 + eta * x0 * x0 * 3.0
        fp = zeta * x0 * x0 * 3.0 + eta * x0 + 6.0 * r0 * 6.0
        x0 = x0 - tf.divide(ff, fp)

        x = tf.identity(x0)
        x, s2, c2, did_it_work = solve_universal_newton(r0, beta, b, eta, zeta, dt, x)

        if tf.equal(did_it_work, False):
            x = tf.identity(x0)
            x, s2, c2, did_it_work = solve_universal_laguerre(r0, -beta, b, eta, zeta, dt, x)
        if tf.equal(did_it_work, False):
            returnBool = tf.constant(False, dtype=tf.bool)
        else:
            a = kc / beta
            G1 = 2.0 * s2 * c2 / b
            c = 2.0 * s2 * s2
            G2 = c / beta
            ca = c * a
            r = r0 + eta * G1 + zeta * G2
            bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    else:

        x, s2, c2, did_it_work = solve_universal_parabolic(r0, eta, zeta, dt)
        if tf.equal(did_it_work, False):
            tf.errors.UnknownError(node_def=None, op=None, message="kepler_step_internal error")
            tf.print("kepler_step_internal error")
        else:
            G1 = x
            G2 = x * x / 2.0
            ca = kc * G2
            r = r0 + eta * G1 + zeta * G2
            bsa = kc * x / (r * r0)

    if returnBool == tf.constant(False, dtype=tf.bool):
        return directionVector, velocityVector, returnBool
    else:
        fhat = -(ca / r0)
        g = eta * G2 + r0 * G1
        fdot = -bsa
        gdothat = -(ca / r)

        storage = tf.identity(directionVector)

        directionVector = directionVector + tf.multiply(fhat, directionVector) + tf.multiply(g, velocityVector)

        velocityVector = velocityVector + tf.multiply(fdot, storage) + tf.multiply(gdothat, velocityVector)

        return directionVector, velocityVector, returnBool


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(3), dtype=tf.float64),
    tf.TensorSpec(shape=(3), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64)
])
def kepler_step_depth_iterative(kc, dtGlobal, beta, b, directionVector, velocityVector, r, eta, zeta):
    # kc, beta, b is never changed
    # r, eta, zeta is dependent on dir and vel vectors
    # direction and velocity vectors are threw the computations changed.
    # => dt can be calculated with the depth and in what of the four calls I am right now has to be remembered

    stack = tf.constant(0, dtype=tf.int64)
    depth = tf.constant(0, dtype=tf.int64)

    while tf.less(depth, 30) and tf.not_equal(depth, -1):
        # if the depth is -1, that means, that either the computations were successful with stack = 1
        # ------------------------------------ or the stack as a whole could be devided by 5

        currentElement = tf.math.floormod(stack, 4)
        depth_float = tf.cast(depth, tf.float64)
        directionVector, velocityVector, did_it_work = tf.cond(tf.equal(currentElement, 0),
                                                               lambda: kepler_step_internal(kc, (dtGlobal / tf.pow(
                                                                   tf.constant(4, dtype=tf.float64), depth_float)),
                                                                                            beta,
                                                                                            b,
                                                                                            directionVector,
                                                                                            velocityVector, r,
                                                                                            eta, zeta),
                                                               lambda: (directionVector, velocityVector,
                                                                        tf.constant(False)))
        if tf.equal(currentElement, 0):
            if tf.equal(did_it_work, False):
                stack = (stack + 1) * 4  # ---------------- currentElement +=1 and 0 is pushed onto the stack
                depth = depth + 1
            else:  # ---------------- going back up
                stack = tf.math.floordiv(stack, 4)
                depth = depth - 1
                while tf.not_equal(depth, -1) and tf.equal(tf.math.floormod(stack, 4), 0):
                    stack = tf.math.floordiv(stack, 4)
                    depth = depth - 1
        else:
            r = tf.norm(directionVector)
            eta = tf.reduce_sum(directionVector * velocityVector)
            zeta = kc - beta * r
            if tf.equal(currentElement, 3):
                stack = (stack - 3) * 4  # if it is 0, it gets cleaned up by the way up
            else:
                stack = (stack + 1) * 4  # ---------------- currentElement +=1 and 0 is pushed onto the stack
            depth = depth + 1

    if tf.greater_equal(depth, 30):
        tf.print("kepler depth exceeded")

    return directionVector, velocityVector


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(3), dtype=tf.float64),
    tf.TensorSpec(shape=(3), dtype=tf.float64)
])
def kepler_step(kc, dt, directionVector, velocityVector):
    r0 = tf.norm(directionVector)
    v2 = tf.reduce_sum(velocityVector * velocityVector)
    eta = tf.reduce_sum(directionVector * velocityVector)
    beta = 2.0 * kc / r0 - v2
    zeta = kc - beta * r0
    b = tf.sqrt(tf.abs(beta))

    return kepler_step_depth_iterative(kc, dt, beta, b, directionVector, velocityVector, r0,
                                       eta, zeta)

# This is the original implementation of kepler_step_depth_iterative, but Tensorflow always had an issue with the calling of kepler_step_internal
# This issue was very confusing and to resolve this, I called kepler_step_internal a bit earlier and this resolved it.
# I can't explain this weird behavior of TensorFlow, but if you can, then please tell me: andreas.merrath@tum.de

# def kepler_step_depth_iterative(kc, dtGlobal, beta, b, directionVector, velocityVector, r, eta, zeta):
#     # kc, beta, b is never changed
#     # r, eta, zeta is dependent on dir and vel vectors
#     # direction and velocity vectors are threw the computations changed.
#     # => dt can be calculated with the depth and in what of the four calls I am right now has to be remembered
#
#     stack = tf.constant(0, dtype=tf.int64)
#     depth = tf.constant(0, dtype=tf.float64)
#
#     while tf.less(depth, 30) and tf.not_equal(depth, -1):
#         # if the depth is -1, that means, that either the computations were successful with stack = 1
#         # ------------------------------------ or the stack as a whole could be devided by 5
#
#         currentElement = tf.math.floormod(stack, 4)
#         if tf.equal(currentElement, 0):
#
#             dt = dtGlobal / tf.pow(tf.constant(4, dtype=tf.float64), depth)
#
#             directionVector, velocityVector, did_it_work = kepler_step_internal(kc, dt, beta, b, directionVector,
#                                                                                 velocityVector, r,
#                                                                                 eta, zeta)
#             if tf.equal(did_it_work, False):
#                 stack = (stack + 1) * 4  # ---------------- currentElement +=1 and 0 is pushed onto the stack
#                 depth = depth + 1
#             else:  # ---------------- going back up
#                 stack = tf.math.floordiv(stack, 4)
#                 depth = depth - 1
#                 while tf.not_equal(depth, -1) and tf.equal(tf.math.floormod(stack, 4), 0):
#                     stack = tf.math.floordiv(stack, 4)
#                     depth = depth - 1
#         else:
#             r = tf.norm(directionVector)
#             eta = tf.reduce_sum(directionVector * velocityVector)
#             zeta = kc - beta * r
#             if tf.equal(currentElement, 3):
#                 stack = (stack - 3) * 4  # if it is 0, it gets cleaned up by the way up
#             else:
#                 stack = (stack + 1) * 4  # ---------------- currentElement +=1 and 0 is pushed onto the stack
#             depth = depth + 1
#
#     if tf.greater_equal(depth, 30):
#         tf.print("kepler depth exceeded")
#
#     return directionVector, velocityVector
