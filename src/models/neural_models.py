"""
Implementation of the Hybrid GRU-LSTM architecture specified in Farhadi et. al.

Note: Model architecture and structure only.
Training, evaluation, and orchestration happen in pipeline.py.
"""

import tensorflow as tf

class GRUModel:
    def __init__(self, gru_units=50, dropout=0.2, learning_rate=0.001, window_size=30, n_features=138):
        self.gru_units=gru_units
        self.dropout=dropout
        self.learning_rate=learning_rate
        self.window_size = window_size
        self.n_features=n_features

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.window_size, self.n_features)),
            tf.keras.layers.GRU(units=self.gru_units, dropout=self.dropout, return_sequences=True),
            tf.keras.layers.GRU(units=self.gru_units, dropout=self.dropout, return_sequences=False),
            tf.keras.layers.Dense(units=128),
            tf.keras.layers.Dense(1)  # Linear activation for regression
        ])

        # Model compilation
        # Optimizer: Adam (adaptive learning rate)
        # Loss: Mean Squared Error (MSE) for regression
        # Metrics: Mean Absolute Error (MAE) for interpretability
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
    def train(self, X_train, val_data=None, epochs=10, callbacks=None):
        """Train the model on windowed time series data.
        
        Args:
            X_train: tf.data.Dataset containing (features, targets) tuples
        
        Returns:
            History object containing training metrics
        """        
        trained_model = self.model.fit(X_train,
                                       validation_data = val_data,
                                       epochs = epochs,
                                       callbacks = callbacks,
                                       verbose = 0
                                       )
        return trained_model
    def evaluate(self, X_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: tf.data.Dataset containing (features, targets) tuples
        
        Returns:
            List of [loss, metrics]
        """
        metrics = self.model.evaluate(X_test)
        return metrics
    
    def predict(self, X_new):

        predictions = self.model.predict(X_new)

        return predictions

class LSTMModel:
    def __init__(self, lstm_units=50, dropout=0.2, learning_rate=0.001, window_size=30, n_features=138):
        self.lstm_units=lstm_units
        self.dropout=dropout
        self.learning_rate=learning_rate
        self.window_size = window_size
        self.n_features=n_features

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.window_size, self.n_features)),
            tf.keras.layers.LSTM(units=self.lstm_units, dropout=self.dropout, return_sequences=True),
            tf.keras.layers.LSTM(units=self.lstm_units, dropout=self.dropout, return_sequences=False),
            tf.keras.layers.Dense(units=128),
            tf.keras.layers.Dense(1)  # Linear activation for regression
        ])

        # Model compilation
        # Optimizer: Adam (adaptive learning rate)
        # Loss: Mean Squared Error (MSE) for regression
        # Metrics: Mean Absolute Error (MAE) for interpretability
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
    def train(self, X_train, val_data=None, epochs=10, callbacks=None):
        """Train the model on windowed time series data.
        
        Args:
            X_train: tf.data.Dataset containing (features, targets) tuples
        
        Returns:
            History object containing training metrics
        """        
        trained_model = self.model.fit(X_train,
                                       validation_data = val_data,
                                       epochs = epochs,
                                       callbacks = callbacks,
                                       verbose = 0
                                       )
        return trained_model
    def evaluate(self, X_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: tf.data.Dataset containing (features, targets) tuples
        
        Returns:
            List of [loss, metrics]
        """
        metrics = self.model.evaluate(X_test)
        return metrics
    
    def predict(self, X_new):

        predictions = self.model.predict(X_new)

        return predictions

class HybridGRU_LSTM:
    def __init__(self, gru_units=50, lstm_units=50, dropout=0.2, learning_rate=0.001, window_size=30, n_features=138):
        self.gru_units=gru_units
        self.lstm_units=lstm_units
        self.dropout=dropout
        self.learning_rate=learning_rate
        self.window_size = window_size
        self.n_features=n_features

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.window_size, self.n_features)),
            tf.keras.layers.GRU(units=self.gru_units, dropout=self.dropout, return_sequences=True),
            tf.keras.layers.GRU(units=self.gru_units, dropout=self.dropout, return_sequences=True),
            tf.keras.layers.LSTM(units=self.lstm_units, dropout=self.dropout,return_sequences=False),
            #tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128),
            tf.keras.layers.Dense(1)  # Linear activation for regression
        ])

        # Model compilation
        # Optimizer: Adam (adaptive learning rate)
        # Loss: Mean Squared Error (MSE) for regression
        # Metrics: Mean Absolute Error (MAE) for interpretability
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

    def train(self, X_train, val_data=None, epochs=10, callbacks=None):
        """
        Train the model on windowed time series data.
        
        Args:
            X_train: tf.data.Dataset containing (features, targets) tuples
        
        Returns:
            History object containing training metrics
        """
        trained_model = self.model.fit(X_train,
                                       validation_data = val_data,
                                       epochs = epochs,
                                       callbacks = callbacks,
                                       verbose = 0
                                       )
        return trained_model
    
    def evaluate(self, X_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: tf.data.Dataset containing (features, targets) tuples
        
        Returns:
            List of [loss, metrics]
        """
        metrics = self.model.evaluate(X_test)
        return metrics
    
    def predict(self, X_new):

        predictions = self.model.predict(X_new)

        return predictions