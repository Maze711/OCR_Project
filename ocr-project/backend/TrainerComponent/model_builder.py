import tensorflow as tf
from tensorflow.keras import layers, optimizers

def build_cnn_lstm_optimized(input_shape=(64, 160, 1), num_classes=69):
    inputs = layers.Input(shape=input_shape)

    # CNN feature extractor
    x = layers.SeparableConv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.SeparableConv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.SeparableConv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,1))(x)
    x = layers.Dropout(0.2)(x)

    # Prepare for RNN
    rnn_input = layers.Reshape((input_shape[1]//4, 64*8))(x)

    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(rnn_input)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)

    # Global pooling + Output
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

# Instantiate and compile
model = build_cnn_lstm_optimized()
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()