import math


class State:
    def __init__(self, x, y, z, xd, yd, zd, xdd, ydd, zdd):
        self.x = x
        self.y = y
        self.z = z
        self.xd = xd
        self.yd = yd
        self.zd = zd
        self.xdd = xdd
        self.ydd = ydd
        self.zdd = zdd


# input: (double kc,
# double r0,
# double beta,
# double b,
# double eta,
# double zeta,
# double h,
# double *X,
# double *S2,
# double *C2)

# output: X, S2, C2, FAILURE/SUCCESS wird mit False/True zurückgegeben
def solve_universal_newton(kc, r0, beta, b, eta, zeta, h, X, S2, C2):
    xnew = X
    count = 0

    while True:
        x = xnew
        arg = b * x / 2.0
        s2 = math.sin(arg)
        c2 = math.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        cc = eta * g1 + zeta * g2
        xnew = (h + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)
        if count > 10:
            return 0.0, 0.0, 0.0, False
        if math.fabs((x - xnew) / xnew) <= 1.e-8:
            break
        # if math.fabs(xnew) > 1.e-10:
        #     if math.fabs((x - xnew) / xnew) <= 1.e-8:
        #         break
        count += 1

    x = xnew
    arg = b * x / 2.0
    s2 = math.sin(arg)
    c2 = math.cos(arg)
    return x, s2, c2, True


# input:double kc,
# double r0,
# double beta,
# double b,
# double eta,
# double zeta,
# double h,
# double *X,
# double *S2,
# double *C2
# outputs: x, s2, c2, True/False
def solve_universal_laguerre(kc, r0, beta, b, eta, zeta, h, X, S2, C2):
    xnew = X
    count = 0

    c5 = 5.0
    c16 = 16.0
    c20 = 20.0

    while True:
        x = xnew
        arg = b * x / 2.0
        s2 = math.sin(arg)
        c2 = math.cos(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / beta
        g3 = (x - g1) / beta
        f = r0 * x + eta * g2 + zeta * g3 - h
        fp = r0 + eta * g1 + zeta * g2
        g0 = 1.0 - beta * g2
        fpp = eta * g0 + zeta * g1
        dx = -c5 * f / (fp + math.sqrt(math.fabs(c16 * fp * fp - c20 * f * fpp)))
        xnew = x + dx
        if count > 10:
            return 0.0, 0.0, 0.0, False
        if math.fabs(dx) <= 2.e-7 * math.fabs(xnew):
            break
        count += 1

    x = xnew
    arg = b * x / 2.0
    s2 = math.sin(arg)
    c2 = math.cos(arg)
    g1 = 2.0 * s2 * c2 / b
    g2 = 2.0 * s2 * s2 / beta
    g3 = (x - g1) / beta
    cc = eta * g1 + zeta * g2
    xnew = (h + (x * cc - (eta * g2 + zeta * g3))) / (r0 + cc)

    x = xnew
    arg = b * x / 2.0
    s2 = math.sin(arg)
    c2 = math.cos(arg)
    return x, s2, c2, True


def cubic1(a, b, c):
    Q = (a * a - 3.0 * b) / 9.0
    R = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0
    if R * R < Q * Q * Q:
        theta = math.acos(R / math.sqrt(Q * Q * Q))
        x1 = -2.0 * math.sqrt(Q) * math.cos(theta / 3.0) - a / 3.0
        x2 = -2.0 * math.sqrt(Q) * math.cos((theta + 2.0 * math.pi) / 3.0) - a / 3.0
        x3 = -2.0 * math.sqrt(Q) * math.cos((theta - 2.0 * math.pi) / 3.0) - a / 3.0
        print(f"three cubic roots {x1:.16e} {x2:.16e} {x3:.16e}")
        exit(-1)
    else:
        A = -math.copysign(1.0, R) * math.pow(math.fabs(R) + math.sqrt(R * R - Q * Q * Q), 1. / 3.)
        if A == 0.0:
            B = 0.0
        else:
            B = Q / A
        x1 = (A + B) - a / 3.0
        return x1


# inputs: double kc,
# double r0,
# double beta,
# double b,
# double eta,
# double zeta,
# double h,
# double *X,
# double *S2,
# double *C2
# output: X, S2, C2, True/False
def solve_universal_parabolic(kc, r0, beta, b, eta, zeta, h, X, S2, C2):
    x = 0.0
    s2 = 0.0
    c2 = 1.0
    x = cubic1(3.0 * eta / zeta, 6.0 * r0 / zeta, -6.0 * h / zeta)
    s2 = 0.0
    c2 = 1.0
    return x, s2, c2, True


# inputs: double kc,
# double r0,
# double minus_beta,
# double b,
# double eta,
# double zeta,
# double h,
# double *X,
# double *S2,
# double *C2
# output: X, S2, C2, True/False
def solve_universal_hyperbolic_newton(kc, r0, minus_beta, b, eta, zeta, h, X, S2, C2):
    xnew = X
    count = 0
    while True:
        x = xnew
        arg = b * x / 2.0
        if math.fabs(arg) > 200.0:
            return 0.0, 0.0, 0.0, False  # TODO: Make shure, that the values aren't overwritten, if False is returned
        s2 = math.sinh(arg)
        c2 = math.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / minus_beta
        g3 = -(x - g1) / minus_beta
        g = eta * g1 + zeta * g2
        xnew = (x * g - eta * g2 - zeta * g3 + h) / (r0 + g)

        if count > 10:
            return 0.0, 0.0, 0.0, False
        if math.fabs(x - xnew) <= 1.e-9 * math.fabs(xnew):
            break
        count += 1
    x = xnew
    arg = b * x / 2.0
    s2 = math.sinh(arg)
    c2 = math.cosh(arg)
    return x, s2, c2, True


# inputs: double kc,
# double r0,
# double minus_beta,
# double b,
# double eta,
# double zeta,
# double h,
# double *X,
# double *S2,
# double *C2
# output: X, S2, C2, True/False
def solve_universal_hyperbolic_laguerre(kc, r0, minus_beta, b, eta, zeta, h, X, S2, C2):
    xnew = X
    count = 0
    while True:
        c5 = 5.0
        c16 = 16.0
        c20 = 20.0

        x = xnew

        arg = b * x / 2.0
        if math.fabs(arg) > 50.0:
            return 0.0, 0.0, 0.0, False
        s2 = math.sinh(arg)
        c2 = math.cosh(arg)
        g1 = 2.0 * s2 * c2 / b
        g2 = 2.0 * s2 * s2 / minus_beta
        g3 = -(x - g1) / minus_beta
        f = r0 * x + eta * g2 + zeta * g3 - h
        fp = r0 + eta * g1 + zeta * g2
        g0 = 1.0 + minus_beta * g2
        fpp = eta * g0 + zeta * g1
        den = (fp + math.sqrt(math.fabs(c16 * fp * fp - c20 * f * fpp)))
        if den == 0.0:
            return 0.0, 0.0, 0.0, False
        dx = -c5 * f / den
        xnew = x + dx
        if count > 20:
            return 0.0, 0.0, 0.0, False
        if math.fabs(x - xnew) <= 1.e-9 * math.fabs(xnew):
            break
        count += 1

    g = 0.0
    x = xnew
    arg = b * x / 2.0
    if abs(arg) > 200.0:
        return 0.0, 0.0, 0.0, False
    s2 = math.sinh(arg)
    c2 = math.cosh(arg)
    g1 = 2.0 * s2 * c2 / b
    g2 = 2.0 * s2 * s2 / minus_beta
    g3 = -(x - g1) / minus_beta
    g = eta * g1 + zeta * g2
    xnew = (x * g - eta * g2 - zeta * g3 + h) / (r0 + g)

    x = xnew
    arg = b * x / 2.0
    s2 = math.sinh(arg)
    c2 = math.cosh(arg)
    return x, s2, c2, True


# input double kc,
# double dt,
# double beta,
# double b,
# State *s0,
# State *s,
# double r0,
# double v2,
# double eta,
# double zeta
# output: True/False s and s0 are manipulated
def kepler_step_internal(kc, dt, beta, b, s0, s, r0, v2, eta, zeta):
    c2 = 0.0
    s2 = 0.0

    if beta < 0.0:
        x0 = new_guess(r0, eta, zeta, dt)
        x = x0
        x, s2, c2, mybool = solve_universal_hyperbolic_newton(kc, r0, -beta, b, eta, zeta, dt, x, s2, c2)
        if mybool == False:
            x = x0
            x, s2, c2, mybool = solve_universal_hyperbolic_laguerre(kc, r0, -beta, b, eta, zeta, dt, x, s2, c2)
        if mybool == False:
            return False

        a = kc / (-beta)
        G1 = 2.0 * s2 * c2 / b
        c = 2.0 * s2 * s2
        G2 = c / (-beta)
        ca = c * a
        r = r0 + eta * G1 + zeta * G2
        bsa = (a / r) * (b / r0) * 2.0 * s2 * c2
    elif beta > 0.0:
        x0 = dt / r0
        ff = zeta * x0 * x0 * x0 + 3.0 * eta * x0 * x0
        fp = 3.0 * zeta * x0 * x0 + 6.0 * eta * x0 + 6.0 * r0
        x0 -= ff / fp

        x = x0
        x, s2, c2, mybool = solve_universal_newton(kc, r0, beta, b, eta, zeta, dt, x, s2, c2)

        if mybool == False:
            x = x0
            x, s2, c2, mybool = solve_universal_laguerre(kc, r0, -beta, b, eta, zeta, dt, x, s2, c2)
        if mybool == False:
            return False

        a = kc / beta
        G1 = 2.0 * s2 * c2 / b
        c = 2.0 * s2 * s2
        G2 = c / beta
        ca = c * a
        r = r0 + eta * G1 + zeta * G2
        bsa = (a / r) * (b / r0) * 2.0 * s2 * c2

    else:
        x = dt / r0

        x, s2, c2, mybool = solve_universal_parabolic(kc, r0, beta, b, eta, zeta, dt, x, s2, c2)
        if mybool == False:
            exit(-1)

        G1 = x
        G2 = x * x / 2.0
        ca = kc * G2
        r = r0 + eta * G1 + zeta * G2
        bsa = kc * x / (r * r0)

    fhat = -(ca / r0)
    g = eta * G2 + r0 * G1
    fdot = -bsa
    gdothat = -(ca / r)

    s.x = s0.x + (fhat * s0.x + g * s0.xd)
    s.y = s0.y + (fhat * s0.y + g * s0.yd)
    s.z = s0.z + (fhat * s0.z + g * s0.zd)
    s.xd = s0.xd + (fdot * s0.x + gdothat * s0.xd)
    s.yd = s0.yd + (fdot * s0.y + gdothat * s0.yd)
    s.zd = s0.zd + (fdot * s0.z + gdothat * s0.zd)
    return True


def new_guess(r0, eta, zeta, dt):
    if zeta != 0.0:
        s = cubic1(3.0 * eta / zeta, 6.0 * r0 / zeta, -6.0 * dt / zeta)
    elif eta != 0.0:
        reta = r0 / eta
        disc = reta * reta + 8.0 * dt / eta
        if disc >= 0.0:
            s = -reta + math.sqrt(disc)
        else:
            s = dt / r0
    else:
        s = dt / r0
    return s


# input: double kc,
# double dt,
# double beta,
# double b,
# State *s0,
# State *s,
# int depth,
# double r0,
# double v2,
# double eta,
# double zeta
# output: void
def kepler_step_depth(kc, dt, beta, b, s0, s, depth, r0, v2, eta, zeta):
    if depth > 30:
        print("kepler depth exceeded")
        exit(-1)

    flag = kepler_step_internal(kc, dt, beta, b, s0, s, r0, v2, eta, zeta)

    if flag == False:
        ss = State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        kepler_step_depth(kc, dt / 4.0, beta, b, s0, ss, depth + 1, r0, v2, eta, zeta)

        r0 = math.sqrt(ss.x * ss.x + ss.y * ss.y + ss.z * ss.z)
        v2 = ss.xd * ss.xd + ss.yd * ss.yd + ss.zd * ss.zd
        eta = ss.x * ss.xd + ss.y * ss.yd + ss.z * ss.zd
        zeta = kc - beta * r0

        kepler_step_depth(kc, dt / 4.0, beta, b, ss, s, depth + 1, r0, v2, eta, zeta)

        r0 = math.sqrt(s.x * s.x + s.y * s.y + s.z * s.z)
        v2 = s.xd * s.xd + s.yd * s.yd + s.zd * s.zd
        eta = s.x * s.xd + s.y * s.yd + s.z * s.zd
        zeta = kc - beta * r0

        kepler_step_depth(kc, dt / 4.0, beta, b, s, ss, depth + 1, r0, v2, eta, zeta)

        r0 = math.sqrt(ss.x * ss.x + ss.y * ss.y + ss.z * ss.z)
        v2 = ss.xd * ss.xd + ss.yd * ss.yd + ss.zd * ss.zd
        eta = ss.x * ss.xd + ss.y * ss.yd + ss.z * ss.zd
        zeta = kc - beta * r0

        kepler_step_depth(kc, dt / 4.0, beta, b, ss, s, depth + 1, r0, v2, eta, zeta)


def kepler_step(kc, dt, s0, s):
    r0 = math.sqrt(s0.x * s0.x + s0.y * s0.y + s0.z * s0.z)
    v2 = s0.xd * s0.xd + s0.yd * s0.yd + s0.zd * s0.zd
    eta = s0.x * s0.xd + s0.y * s0.yd + s0.z * s0.zd
    beta = 2.0 * kc / r0 - v2
    zeta = kc - beta * r0
    b = math.sqrt(math.fabs(beta))

    kepler_step_depth(kc, dt, beta, b, s0, s, 0, r0, v2, eta, zeta)

# These methods aren't used for some reason:

# # inputs: double kc,
# # double r0,
# # double minus_beta,
# # double b,
# # double eta,
# # double zeta,
# # double h,
# # double *X,
# # double *S2,
# # double *C2
# # output: X, S2, C2, True/False
#
# def solve_universal_hyperbolic_bisection(r0, minus_beta, b, eta, zeta, h, X):
#     xnew = X
#
#     err = tf.constant(1.e-10 * tf.abs(xnew))
#
#     X_min = tf.constant(0.5 * xnew)
#     X_max = tf.constant(10.0 * xnew)
#
#     x = tf.constant(X_min)
#     arg = tf.constant(b * x / 2.0)
#     if tf.abs(arg) > tf.constant(200.0):
#         return tf.constant(0.0), tf.constant(0.0), tf.constant(0.0), False
#     s2 = tf.sinh(arg)
#     c2 = tf.cosh(arg)
#     g1 = tf.constant(2.0 * s2 * c2 / b)
#     g2 = tf.constant(2.0 * s2 * s2 / minus_beta)
#     g3 = tf.constant(-(x - g1) / minus_beta)
#     fmin = tf.constant(r0 * x + eta * g2 + zeta * g3 - h)
#
#     x = tf.constant(X_max)
#     arg = tf.constant(b * x / 2.0)
#     if tf.abs(arg) > tf.constant(200.0):
#         x = tf.constant(200.0 / (b / 2.0))
#         arg = tf.constant(200.0)
#     s2 = tf.sinh(arg)
#     c2 = tf.cosh(arg)
#     g1 = tf.constant(2.0 * s2 * c2 / b)
#     g2 = tf.constant(2.0 * s2 * s2 / minus_beta)
#     g3 = tf.constant(-(x - g1) / minus_beta)
#     fmax = tf.constant(r0 * x + eta * g2 + zeta * g3 - h)
#
#     if fmin * fmax > tf.constant(0.0):
#         return tf.constant(0.0), tf.constant(0.0), tf.constant(0.0), False
#     # ****++++++++++++++++****************************+
#     count = tf.constant(0)
#     while True:
#         x = tf.constant(xnew)
#         arg = tf.constant(b * x / 2.0)
#         if tf.abs(arg) > tf.constant(200.0):
#             return tf.constant(0.0), tf.constant(0.0), tf.constant(0.0), False
#         s2 = tf.sinh(arg)
#         c2 = tf.cosh(arg)
#         g1 = tf.constant(2.0 * s2 * c2 / b)
#         g2 = tf.constant(2.0 * s2 * s2 / minus_beta)
#         g3 = tf.constant(-(x - g1) / minus_beta)
#         f = tf.constant(r0 * x + eta * g2 + zeta * g3 - h)
#         if f >= tf.constant(0.):
#             X_max = tf.constant(x)
#         else:
#             X_min = tf.constant(x)
#         xnew = tf.constant((X_max + X_min) / 2.)
#         if count > tf.constant(100):
#             return tf.constant(0.0), tf.constant(0.0), tf.constant(0.0), False
#         if tf.abs(x - xnew) <= err:
#             break
#         count = tf.add(count, 1)
#
#     x = tf.constant(xnew)
#     arg = tf.constant(b * x / 2.0)
#     s2 = tf.sinh(arg)
#     c2 = tf.cosh(arg)
#     return x, s2, c2, True
#
# # input:double kc,
# # double r0,
# # double beta,
# # double b,
# # double eta,
# # double zeta,
# # double h,
# # double *X,
# # double *S2,
# # double *C2
# # outputs: x, s2, c2, True/False
# def solve_universal_bisection(kc, r0, beta, b, eta, zeta, h):
#     count = tf.constant(0)
#     xnew = tf.constant(0.0)
#     err = tf.constant(1.e-9 * tf.abs(xnew))
#
#     invperiod = tf.constant(b * beta / (2. * pi * kc))
#     X_per_period = tf.constant(2. * pi / b)
#     X_min = tf.constant(X_per_period * tf.floor(h * invperiod))
#     X_max = tf.constant(X_min + X_per_period)
#     xnew = tf.constant((X_max + X_min) / 2.)
#
#     while True:
#         x = tf.constant(xnew)
#         arg = tf.constant(b * x / 2.0)
#         s2 = tf.sin(arg)
#         c2 = tf.cos(arg)
#         g1 = tf.constant(2.0 * s2 * c2 / b)
#         g2 = tf.constant(2.0 * s2 * s2 / beta)
#         g3 = tf.constant((x - g1) / beta)
#         f = tf.constant(r0 * x + eta * g2 + zeta * g3 - h)
#         if f >= tf.constant(0.):
#             X_max = tf.constant(x)
#         else:
#             X_min = tf.constant(x)
#         xnew = tf.constant((X_max + X_min) / 2.)
#         if count > tf.constant(100):
#             return tf.constant(0.0), tf.constant(0.0), tf.constant(0.0), False
#         if tf.abs(x - xnew) <= err:
#             break
#         count = tf.add(count, 1)
#
#     x = tf.constant(xnew)
#
#     arg = tf.constant(b * x / 2.0)
#     s2 = tf.sin(arg)
#     c2 = tf.cos(arg)
#     return x, s2, c2, True


# # input:double kc,
# # double r0,
# # double beta,
# # double b,
# # double eta,
# # double zeta,
# # double h,
# # double *X,
# # double *S2,
# # double *C2
# # outputs: x, s2, c2, True/False
# def solve_universal_bisection(kc, r0, beta, b, eta, zeta, h, X, S2, C2):
#     x = 0.0
#     g1 = 0.0
#     g2 = 0.0
#     g3 = 0.0
#     arg = 0.0
#     s2 = 0.0
#     c2 = 0.0
#     f = 0.0
#     X_min = 0.0
#     X_max = 0.0
#     X_per_period = 0.0
#     invperiod = 0.0
#     count = 0
#     xnew = 0.0
#     err = 1.e-9 * math.fabs(xnew)
#
#     xnew = X
#
#     invperiod = b * beta / (2. * math.pi * kc)
#     X_per_period = 2. * math.pi / b
#     X_min = X_per_period * math.floor(h * invperiod)
#     X_max = X_min + X_per_period
#     xnew = (X_max + X_min) / 2.
#
#     while True:
#         x = xnew
#         arg = b * x / 2.0
#         s2 = math.sin(arg)
#         c2 = math.cos(arg)
#         g1 = 2.0 * s2 * c2 / b
#         g2 = 2.0 * s2 * s2 / beta
#         g3 = (x - g1) / beta
#         f = r0 * x + eta * g2 + zeta * g3 - h
#         if f >= 0.:
#             X_max = x
#         else:
#             X_min = x
#         xnew = (X_max + X_min) / 2.
#         if count > 100:
#             return 0.0, 0.0, 0.0, False
#         if math.fabs(x - xnew) <= err:
#             break
#         count += 1
#
#     x = xnew
#
#     arg = b * x / 2.0
#     s2 = math.sin(arg)
#     c2 = math.cos(arg)
#     return x, s2, c2, True


# # inputs: double kc,
# # double r0,
# # double minus_beta,
# # double b,
# # double eta,
# # double zeta,
# # double h,
# # double *X,
# # double *S2,
# # double *C2
# # output: X, S2, C2, True/False
#
# def solve_universal_hyperbolic_bisection(kc, r0, minus_beta, b, eta, zeta, h, X, S2, C2):
#     x = 0.0
#     g1 = 0.0
#     g2 = 0.0
#     g3 = 0.0
#     arg = 0.0
#     s2 = 0.0
#     c2 = 0.0
#     xnew = 0.0
#     count = 0
#     X_min = 0.0
#     X_max = 0.0
#     f = 0.0
#
#     xnew = X
#
#     err = 1.e-10 * math.fabs(xnew)
#
#     X_min = 0.5 * xnew
#     X_max = 10.0 * xnew
#     # ****++++++++++++++++****************************+
#     fmin = 0.0
#     fmax = 0.0
#
#     x = X_min
#     arg = b * x / 2.0
#     if math.fabs(arg) > 200.0:
#         return 0.0, 0.0, 0.0, False
#     s2 = math.sinh(arg)
#     c2 = math.cosh(arg)
#     g1 = 2.0 * s2 * c2 / b
#     g2 = 2.0 * s2 * s2 / minus_beta
#     g3 = -(x - g1) / minus_beta
#     fmin = r0 * x + eta * g2 + zeta * g3 - h
#
#     x = X_max
#     arg = b * x / 2.0
#     if math.fabs(arg) > 200.0:
#         x = 200.0 / (b / 2.0)
#         arg = 200.0
#     s2 = math.sinh(arg)
#     c2 = math.cosh(arg)
#     g1 = 2.0 * s2 * c2 / b
#     g2 = 2.0 * s2 * s2 / minus_beta
#     g3 = -(x - g1) / minus_beta
#     fmax = r0 * x + eta * g2 + zeta * g3 - h
#
#     if fmin * fmax > 0.0:
#         return 0.0, 0.0, 0.0, False
#     # ****++++++++++++++++****************************+
#
#     while True:
#         x = xnew
#         arg = b * x / 2.0
#         if math.fabs(arg) > 200.0:
#             return 0.0, 0.0, 0.0, False
#         s2 = math.sinh(arg)
#         c2 = math.cosh(arg)
#         g1 = 2.0 * s2 * c2 / b
#         g2 = 2.0 * s2 * s2 / minus_beta
#         g3 = -(x - g1) / minus_beta
#         f = r0 * x + eta * g2 + zeta * g3 - h
#         if f >= 0.:
#             X_max = x
#         else:
#             X_min = x
#         xnew = (X_max + X_min) / 2.
#         if count > 100:
#             return 0.0, 0.0, 0.0, False
#         if math.fabs(x - xnew) <= err:
#             break
#         count += 1
#
#     x = xnew
#     arg = b * x / 2.0
#     s2 = math.sinh(arg)
#     c2 = math.cosh(arg)
#     return x, s2, c2, True
