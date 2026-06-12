import numpy as np
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger("ModelTrainer")

def build_and_train_model(X, y, num_classes):
    """
    Builds and trains the LSTM model to learn normal log sequences.
    """
    logger.info("Building LSTM model with Embedding layer...")
    
    model = Sequential([
        # Embedding: maps integer IDs to dense vectors
        Embedding(input_dim=num_classes + 1, output_dim=32),
        
        # LSTM: learns temporal patterns in log sequences
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        
        # Final layer: Predicts the next likely template_id
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # Callbacks to prevent overfitting and save the best version
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint('models/best_log_model.keras', monitor='val_loss', save_best_only=True)
    
    logger.info("Starting training...")
    history = model.fit(
        X, y, 
        epochs=30, 
        batch_size=64, 
        validation_split=0.2, 
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    return model, history
    