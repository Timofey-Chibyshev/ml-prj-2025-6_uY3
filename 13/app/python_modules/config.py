import tensorflow as tf

# Конфигурация модели
layers = [2] + [10]*10 + [1]

# Настройка оптимизатора
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=2000,
    decay_rate=0.95
)

tf_optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

# Параметры обучения
N_u = 1000  # точек для граничных условий
N_r = 10000  # точек для уравнения
tf_epochs = 5000