import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D,
    Dropout, Reshape, Bidirectional, LSTM, Dense, Lambda
)
from tensorflow.keras import optimizers
from .data_utils import NUM_CHARS, IMAGE_WIDTH, IMAGE_HEIGHT, ctc_lambda_func

def build_ocr_model_tflite_compatible():
    input_img = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name='input_image')
    
    # CNN Feature Extractor
    # Layer 1
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Layer 2
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Layer 3
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = MaxPooling2D((1, 2))(x)  # Only pool width
    x = Dropout(0.2)(x)
    
    # Prepare for RNN - CRITICAL: Calculate feature map dimensions correctly
    # After 3 maxpool layers: (64/2/2)=16 height, (160/2/2/2)=20 width
    # With 128 filters
    time_steps = 20  # This is CRITICAL - must match the actual feature map width!
    feature_dim = 16 * 128  # height * filters
    
    x = Reshape((time_steps, feature_dim))(x)
    
    # RNN layers for sequence modeling
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, 
                          recurrent_dropout=0.2))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, 
                          recurrent_dropout=0.2))(x)
    
    # Output layer
    output = Dense(NUM_CHARS + 1, activation='softmax', name='output')(x)
    
    # CTC loss inputs
    labels = Input(name='labels', shape=(None,), dtype='int32')
    input_length = Input(name='input_length', shape=(1,), dtype='int64')
    label_length = Input(name='label_length', shape=(1,), dtype='int64')
    loss_out = Lambda(ctc_lambda_func, name='ctc')([output, labels, input_length, label_length])
    
    # Training model
    train_model = tf.keras.models.Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=loss_out
    )
    
    # Use Adam with gradient clipping - CRITICAL for CTC
    optimizer = optimizers.Adam(
        learning_rate=0.0005,  # Start with lower learning rate
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=1.0
    )
    
    train_model.compile(
        optimizer=optimizer,
        loss={'ctc': lambda y_true, y_pred: y_pred}
    )
    
    # Prediction model
    pred_model = tf.keras.models.Model(inputs=input_img, outputs=output)
    
    print("✅ Built CTC model with proper dimensions")
    print(f"Time steps: {time_steps}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Number of classes: {NUM_CHARS + 1}")
    print(f"Expected max label length: {time_steps}")
    
    return train_model, pred_model, time_steps

def layer_summary_table(model):
    """
    Generate a summary table of the layers in the model using model.summary().
    Args:
        model: The Keras model object.
    Returns:
        A string containing the summary table in markdown format.
    """
    from io import StringIO
    summary_lines = []
    summary_lines.append("| Layer Name | Output Shape | Parameters |")
    summary_lines.append("|------------|--------------|------------|")

    # Capture model.summary() output
    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_output = stream.getvalue().split("\n")

    # Parse the summary output to extract layer details
    for line in summary_output:
        if "│" in line:  # Look for table rows in the summary
            parts = line.split("│")
            if len(parts) >= 4:
                layer_name = parts[1].strip()
                output_shape = parts[2].strip()
                num_params = parts[3].strip()
                summary_lines.append(f"| {layer_name} | {output_shape} | {num_params} |")

    return "\n".join(summary_lines)