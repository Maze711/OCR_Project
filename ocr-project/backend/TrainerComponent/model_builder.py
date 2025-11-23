import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Bidirectional, GRU, Dense, Lambda, Conv1D
from .data_utils import NUM_CHARS, IMAGE_WIDTH, IMAGE_HEIGHT, ctc_lambda_func

def build_ocr_model_tflite_compatible():
    input_img = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name='input_image')
    
    # Balanced CNN Backbone
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.1)(x)  # Light dropout
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # Moderate dropout
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)
    x = tf.keras.layers.Dropout(0.25)(x)  # Slightly higher dropout
    
    # Column-wise Max Pooling
    x = Lambda(lambda x: tf.reduce_max(x, axis=1), name='column_wise_max_pool')(x)
    
    # Balanced RNN Branch
    main_branch = Bidirectional(GRU(96, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    main_branch = Bidirectional(GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(main_branch)
    main_output = Dense(NUM_CHARS + 1, activation='softmax', name='main_output')(main_branch)
    
    # CTC Shortcut Branch (Training only helper)
    ctc_shortcut = Conv1D(48, 3, padding='same', activation='relu')(x)
    ctc_shortcut = Dense(NUM_CHARS + 1, activation='softmax', name='ctc_shortcut')(ctc_shortcut)
    
    # Training Model with Multi-task Loss
    labels = Input(name='labels', shape=(None,), dtype='int32')
    input_length = Input(name='input_length', shape=(1,), dtype='int64')
    label_length = Input(name='label_length', shape=(1,), dtype='int64')
    
    def combined_ctc_loss(args):
        main_pred, shortcut_pred, y_true, input_len, label_len = args
        main_loss = ctc_lambda_func([main_pred, y_true, input_len, label_len])
        shortcut_loss = ctc_lambda_func([shortcut_pred, y_true, input_len, label_len])
        return main_loss + 0.1 * shortcut_loss  # Balanced weight
    
    loss_out = Lambda(combined_ctc_loss, name='combined_ctc')([main_output, ctc_shortcut, labels, input_length, label_length])
    
    train_model = tf.keras.models.Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=loss_out
    )
    
    # Use Adam with lower learning rate and weight decay
    train_model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=0.001, 
            beta_1=0.9, 
            beta_2=0.999, 
            epsilon=1e-08,
            clipvalue=1.0  # Gradient clipping to prevent explosions
        ),
        loss={'combined_ctc': lambda y_true, y_pred: y_pred}
    )
    
    # Inference Model
    pred_model = tf.keras.models.Model(inputs=input_img, outputs=main_output)
    
    time_steps = IMAGE_WIDTH // 4
    
    return train_model, pred_model, time_steps