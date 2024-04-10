import matplotlib.pyplot as plt
import TensorCode.MainCode as mct
import NormalCode.MainCode as mc
import numpy as np
import tensorflow as tf

def plot(values, title, logarithmic=True):
    if logarithmic:
        plt.semilogy(values)
    else:
        plt.plot(values)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()


def instantiate_return_values(variable_to_learn, num_of_epochs):
    if variable_to_learn == "r" or variable_to_learn == "v":
        costs = np.zeros((num_of_epochs, 3))
        gradients = np.zeros((num_of_epochs, 3))
        values = np.zeros((num_of_epochs, 3))
    else:
        costs = np.zeros(num_of_epochs)
        gradients = np.zeros(num_of_epochs)
        values = np.zeros(num_of_epochs)
    return costs, gradients, values


def initTensors(tau, n, m, r, v, variable_to_learn, learned_value):
    if variable_to_learn == "r":
        r[0] = learned_value
        r = tf.Variable(r, dtype=tf.float64)
        v = tf.constant(v, dtype=tf.float64)
        m = tf.constant(m, dtype=tf.float64)
        tau = tf.constant(tau, tf.float64)
    elif variable_to_learn == "v":
        v[0] = learned_value
        r = tf.constant(r, dtype=tf.float64)
        v = tf.Variable(v, dtype=tf.float64)
        m = tf.constant(m, dtype=tf.float64)
        tau = tf.constant(tau, tf.float64)
    elif variable_to_learn == "all m":
        r = tf.constant(r, dtype=tf.float64)
        v = tf.constant(v, dtype=tf.float64)
        m = tf.Variable(learned_value, dtype=tf.float64)
        tau = tf.constant(tau, tf.float64)
    else:
        m[0] = learned_value
        r = tf.constant(r, dtype=tf.float64)
        v = tf.constant(v, dtype=tf.float64)
        m = tf.Variable(m, dtype=tf.float64)
        tau = tf.constant(tau, tf.float64)
    n = tf.constant(n, dtype=tf.int32)

    return tau, n, m, r, v


def execute_do_step(tau, n, m, r, v, using_the_tensorflow_code):
    if not using_the_tensorflow_code:
        r, v = mc.do_step(tau, n, m, r, v)
    else:
        r, v = mct.do_step(tau, n, m, r, v)
    return r, v


def execute_x_times_do_step_normal(tau, n, m, r, v, num_of_steps_in_do_step):
    for _ in range(num_of_steps_in_do_step):
        r, v = mc.do_step(tau, n, m, r, v)
    return r, v


def execute_x_times_do_step_Graph(tau, n, m, r, v, num_of_steps_in_do_step):
    for _ in range(num_of_steps_in_do_step):
        r, v = mct.do_step(tau, n, m, r, v)
    return r, v


def calculate_how_the_values_should_be(tau, n, m, r, v, num_of_steps_in_do_step):
    r1, v1 = execute_x_times_do_step_normal(tau, n, m, r, v, num_of_steps_in_do_step)
    r1 = tf.constant(r1, dtype=tf.float64)
    v1 = tf.constant(v1, dtype=tf.float64)
    return r1, v1


# ------------------------------------------------------------------------------------------------
def find_global_minimum(tauGlobal, nGlobal, mGlobal, rGlobal, vGlobal, learning_rate, startingPosition,
                        num_of_steps_in_do_step, accuracy, variable_to_learn):
    def iniValues():
        tau = tauGlobal
        n = nGlobal
        r = np.copy(rGlobal)
        v = np.copy(vGlobal)
        m = np.copy(mGlobal)
        return tau, n, m, r, v

    tau, n, m, r, v = iniValues()  # -----------initialisation------------
    r1, v1 = calculate_how_the_values_should_be(tau, n, m, r, v, num_of_steps_in_do_step)

    # divided by 8 because the learning_rate_multiplier starts with * 8
    learning_rate = tf.constant(learning_rate / 8, dtype=tf.float64)
    learned_value = tf.constant(startingPosition, dtype=tf.float64)
    learning_rate_multiplier = tf.ones(tf.shape(learned_value), dtype=tf.float64) * 8  # here, it starts with 8
    sign_of_the_last_gradient_step = tf.zeros(tf.shape(learned_value), dtype=tf.float64)

    costs = []
    gradients = []
    values = []

    while tf.reduce_any(tf.greater(learning_rate, accuracy)):

        # ------------------------------------initialisation------------
        tau, n, m, r, v = iniValues()
        tau, n, m, r, v = initTensors(tau, n, m, r, v, variable_to_learn, learned_value.numpy())

        # -------------------------------------gradient calculations------------
        with tf.GradientTape() as tape:
            r2, v2 = execute_x_times_do_step_Graph(tau, n, m, r, v, num_of_steps_in_do_step)
            cost = tf.reduce_sum(tf.pow(r1 - r2, 2)) + tf.reduce_sum(tf.pow(v1 - v2, 2))
        if variable_to_learn == "r":
            gradient = tape.gradient(cost, r)[0]
        elif variable_to_learn == "v":
            gradient = tape.gradient(cost, v)[0]
        else:
            gradient = tape.gradient(cost, m)[0]

        # --------------------------------------learning------------------------------
        # did the sign of the gradient change?
        # if the sign of the gradient changed, the multiplier is 0.5
        # if it didn't change, the multiplier stays the same
        learning_rate_multiplier = tf.where(tf.less(tf.sign(gradient * sign_of_the_last_gradient_step), 0),
                                            tf.ones(tf.shape(learned_value), dtype=tf.float64) * 0.5,
                                            learning_rate_multiplier)
        # if the value is correct, no more learning is needed
        learning_rate_multiplier = tf.where(tf.equal(gradient, 0), tf.zeros(tf.shape(learned_value), dtype=tf.float64),
                                            learning_rate_multiplier)

        learning_rate = learning_rate * learning_rate_multiplier

        learned_value = learned_value - learning_rate * tf.sign(gradient)

        sign_of_the_last_gradient_step = tf.sign(gradient)

        # -------------------------------------plotting------------------------------
        costs.append(cost.numpy())
        gradients.append(gradient.numpy())
        values.append(learned_value.numpy())

    # ------------------------------------------------------------------------------------------------
    return values, costs, gradients


def learn_all_masses_Simulated_Values(tauGlobal, nGlobal, mGlobal, rGlobal, vGlobal, learning_rate, starting_guess,
                                      num_of_steps_in_do_step, accuracy, convergence_rate, learning_rate_factor):
    def iniValues():
        tau = tauGlobal
        n = nGlobal
        r = np.copy(rGlobal)
        v = np.copy(vGlobal)
        m = np.copy(mGlobal)
        return tau, n, m, r, v

    tau, n, m, r, v = iniValues()  # -----------initialisation------------
    r1, v1 = calculate_how_the_values_should_be(tau, n, m, r, v, num_of_steps_in_do_step)

    # divided by learning_rate_factor because the learning_rate_multiplier starts with * learning_rate_factor
    learning_rate = tf.constant(learning_rate / learning_rate_factor, dtype=tf.float64)
    learned_masses = tf.constant(starting_guess, dtype=tf.float64)
    learning_rate_multiplier = tf.ones(tf.shape(learned_masses), dtype=tf.float64) * learning_rate_factor  # here, it starts with learning_rate_factor
    sign_of_the_last_gradient_step = tf.zeros(tf.shape(learned_masses), dtype=tf.float64)

    how_many_times_was_the_multiplier_1 = tf.zeros(tf.shape(learned_masses), dtype=tf.float64)

    costs = []
    gradients = []
    values = []

    while tf.reduce_any(tf.greater(learning_rate, accuracy)):
        # ------------------------------------initialisation------------
        tau, n, m, r, v = iniValues()
        tau, n, m, r, v = initTensors(tau, n, m, r, v, "all m", learned_masses.numpy())

        # -------------------------------------gradient calculations------------
        with tf.GradientTape() as tape:
            r2, v2 = execute_x_times_do_step_Graph(tau, n, m, r, v, num_of_steps_in_do_step)
            cost = tf.reduce_sum(tf.pow(r1 - r2, 2)) + tf.reduce_sum(tf.pow(v1 - v2, 2))

        gradient = tape.gradient(cost, m)

        learning_rate_multiplier = tf.where(tf.less(tf.sign(gradient * sign_of_the_last_gradient_step), 0),
                                            tf.ones(tf.shape(learned_masses), dtype=tf.float64) * 0.5,
                                            tf.where(tf.equal(learning_rate_multiplier, 0.5),
                                                     tf.ones(tf.shape(learned_masses), dtype=tf.float64),
                                                     tf.ones(tf.shape(learned_masses), dtype=tf.float64) * (
                                                             learning_rate_factor / convergence_rate ** how_many_times_was_the_multiplier_1)))

        how_many_times_was_the_multiplier_1 = tf.where(tf.equal(learning_rate_multiplier, 1),
                                                       how_many_times_was_the_multiplier_1 + 1,
                                                       how_many_times_was_the_multiplier_1)

        # if the value is correct, no more learning is needed
        learning_rate_multiplier = tf.where(tf.equal(gradient, 0), tf.zeros(tf.shape(learned_masses), dtype=tf.float64),
                                            learning_rate_multiplier)

        learning_rate = learning_rate * learning_rate_multiplier

        learned_masses = learned_masses - learning_rate * tf.sign(gradient)

        sign_of_the_last_gradient_step = tf.sign(gradient)

        # -------------------------------------plotting------------------------------
        costs.append(cost.numpy())
        gradients.append(gradient.numpy())
        values.append(learned_masses.numpy())

    # ------------------------------------------------------------------------------------------------
    return values, costs, gradients


def learn_all_masses(tau, n, r0, v0, r1_real, v1_real, start_value_learning_rate, startingPosition,
                     num_of_steps_in_do_step, accuracy, convergence_rate):
    learned_masses = tf.constant(startingPosition, dtype=tf.float64)
    learning_rate_multiplier = tf.ones(tf.shape(learned_masses), dtype=tf.float64) * 2  # here, it starts with 2
    # divided by 8 because the learning_rate_multiplier starts with * 2
    learning_rate = tf.multiply(tf.constant(start_value_learning_rate / 2, dtype=tf.float64), learning_rate_multiplier)
    sign_of_the_last_gradient_step = tf.zeros(tf.shape(learned_masses), dtype=tf.float64)
    how_many_times_was_the_multiplier_1 = tf.zeros(tf.shape(learned_masses), dtype=tf.float64)

    n = tf.constant(n, dtype=tf.int32)
    tau = tf.constant(tau, tf.float64)

    costs = []
    gradients = []
    values = []

    while tf.reduce_any(tf.greater(learning_rate, accuracy)):
        # ------------------------------------initialisation------------
        m = learned_masses.numpy()
        r = tf.constant(r0, dtype=tf.float64)
        v = tf.constant(v0, dtype=tf.float64)
        m = tf.Variable(m, dtype=tf.float64)

        # -------------------------------------gradient calculations------------
        with tf.GradientTape() as tape:
            r1_calculated, v1_calculated = execute_x_times_do_step_Graph(tau, n, m, r, v, num_of_steps_in_do_step)
            cost = tf.reduce_sum(tf.pow(r1_real - r1_calculated, 2)) + tf.reduce_sum(tf.pow(v1_real - v1_calculated, 2))

        gradient = tape.gradient(cost, m)

        learning_rate_multiplier = tf.where(tf.less(tf.sign(gradient * sign_of_the_last_gradient_step), 0),
                                            tf.ones(tf.shape(learned_masses), dtype=tf.float64) * 0.5,
                                            tf.where(tf.equal(learning_rate_multiplier, 0.5),
                                                     tf.ones(tf.shape(learned_masses), dtype=tf.float64),
                                                     tf.ones(tf.shape(learned_masses), dtype=tf.float64) * (
                                                             2 / convergence_rate ** how_many_times_was_the_multiplier_1)))

        how_many_times_was_the_multiplier_1 = tf.where(tf.equal(learning_rate_multiplier, 1),
                                                       how_many_times_was_the_multiplier_1 + 1,
                                                       how_many_times_was_the_multiplier_1)

        # if the value is correct, no more learning is needed
        learning_rate_multiplier = tf.where(tf.equal(gradient, 0), tf.zeros(tf.shape(learned_masses), dtype=tf.float64),
                                            learning_rate_multiplier)

        learning_rate = learning_rate * learning_rate_multiplier

        learned_masses = learned_masses - learning_rate * tf.sign(gradient)

        sign_of_the_last_gradient_step = tf.sign(gradient)

        # -------------------------------------plotting------------------------------
        costs.append(cost.numpy())
        gradients.append(gradient.numpy())
        values.append(learned_masses.numpy())

    # ------------------------------------------------------------------------------------------------
    return values, costs, gradients

