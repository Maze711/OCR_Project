import tensorflow as tf
from tensorflow.keras.layers import (
    Input, SeparableConv2D, BatchNormalization, MaxPooling2D,
    Dropout, Reshape, Bidirectional, GRU, Dense, Lambda
)
from tensorflow.keras import optimizers
from .data_utils import NUM_CHARS, IMAGE_WIDTH, IMAGE_HEIGHT, ctc_lambda_func

def build_ocr_model_tflite_compatible():
    input_img = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name='input_image')

    # CNN feature extractor (separable convs for smaller size)
    x = SeparableConv2D(32, (3,3), padding='same', activation='relu')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = SeparableConv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = SeparableConv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,1))(x)
    x = Dropout(0.2)(x)  # Training only

    # Prepare for RNN
    time_steps = IMAGE_WIDTH // 4
    rnn_input = Reshape((time_steps, 64*8))(x)  # Adjust channels as needed

    # Bidirectional GRU
    x = Bidirectional(GRU(64, return_sequences=True))(rnn_input)
    x = Bidirectional(GRU(32, return_sequences=True))(x)

    # Output for prediction
    output = Dense(NUM_CHARS + 1, activation='softmax', name='output')(x)

    # CTC loss inputs
    labels = Input(name='labels', shape=(None,), dtype='int32')
    input_length = Input(name='input_length', shape=(1,), dtype='int64')
    label_length = Input(name='label_length', shape=(1,), dtype='int64')
    loss_out = Lambda(ctc_lambda_func, name='ctc')([output, labels, input_length, label_length])

    # Training model (with CTC loss)
    train_model = tf.keras.models.Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=loss_out
    )
    train_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss={'ctc': lambda y_true, y_pred: y_pred}
    )

    # Prediction model (for inference/TFLite)
    pred_model = tf.keras.models.Model(inputs=input_img, outputs=output)

    return train_model, pred_model, time_steps