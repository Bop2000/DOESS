"""
This module provides classes for training neural network models for various objective functions.
It includes an abstract base class and specific implementations for different objective functions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    Lambda,
    BatchNormalization,
    LayerNormalization,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from typing import Any, Set, Optional, Dict, List
from collections import defaultdict, namedtuple

@dataclass
class SurrogateModel(ABC):
    """
    Abstract base class for surrogate model implementations.

    Attributes:
        input_dims (tuple): The input dimensions for the model.
        learning_rate (float): The learning rate for the optimizer.
        path (str): Where to save the trained mdoels.
        batch_size (int): The number of samples per gradient update.
        epochs (int): The number of epochs to train the model.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        models (dict): Load all trained models and store in the dict "models".
    """
    indicator: int = 1
    input_dims: tuple = (24,9)
    conv_type: str = '2D'
    learning_rate: float = 0.001
    path: str = '/'
    check_point_path: Path = field(default_factory=lambda: Path("NN.keras"))
    batch_size: int = 64
    epochs: int = 5000
    patience: int = 100
    models: Dict[Any, int] = field(default_factory=lambda: defaultdict(int))
    verbose: bool = False

    x_scaler: Optional[StandardScaler] = field(default_factory=StandardScaler)
    y_scaler: Optional[StandardScaler] = field(default_factory=StandardScaler)

    @abstractmethod
    def create_model(self) -> keras.Model:
        """
        Create and return a Keras model.

        This method should be implemented by subclasses to define the specific
        architecture of the neural network model.

        Returns:
            keras.Model: The created Keras model.
        """
        pass

    def __call__(self, X,y):
        """
        Train the model on the given data.

        This method handles the entire training process, including data splitting,
        model creation, training, and evaluation.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            model_name (str): Name to save the model.

        """
        "set the seed for reproducibility"
        random.seed(1) # set random seed for reproducibility
        np.random.seed(1) #
        
        X_processed, y_processed = self.preprocess_data(X, y)
        # print(y_processed)
        X_train,X_test,y_train,y_test = train_test_split(
            X_processed,
            y_processed,
            test_size=0.2,
            random_state=1,
        )
        # print(y_test)
        model, y_test, y_pred, R2, MAE = self.model_training(
            X_train, 
            X_test, 
            y_train, 
            y_test,
            )
        model.summary()
        model.save(self.path + f'/indicator{self.indicator}.keras')
        self.evaluate_model(
            y_test, 
            y_pred, 
            filename = f'indicator{self.indicator}',
            save_file=True)
        # self.load_model()
        self.model = model
        return model
        
    def preprocess_data(self, X, y):
        """Standardize and reshape input/output data"""
        # Reshape for scaling
        original_shape = X.shape
        X_flat = X.reshape(len(y), -1)  # Flatten spatial dimensions
        
        # Standardize data
        X_scaled = self.x_scaler.fit_transform(X_flat)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1))

        # Reshape to appropriate dimensions
        if self.conv_type == '2D':
            # Reshape to (samples, height, width, channels)
            X_processed = X_scaled.reshape(
                (X.shape[0], *self.input_dims)
            )
        else:  # 1D case
            # Reshape to (samples, timesteps, features)
            X_processed = X_scaled.reshape(
                (X.shape[0], *self.input_dims)
            )

        return X_processed, y_scaled.flatten()
    
    
    def load_model(self):
        # load all models and store in the dict "models"
        self.models['model']= keras.models.load_model(self.path+f'/indicator{self.indicator}.keras')

    def pred(self, X):
        """prediction"""
        X_scaled=self.x_scaler.transform(X.reshape(len(X),self.input_dims[0] * self.input_dims[1]))
        temp=self.model.predict(
            X_scaled.reshape(len(X_scaled),*self.input_dims,1), verbose = 0)
        # print(temp.shape)
        pred_all = self.y_scaler.inverse_transform(temp.reshape(len(temp),1))
        return pred_all.flatten()

    def model_training(self, X1, X2, y1, y2):
        """
        Train the model on the given data.

        This method handles the entire training process, including 
        model creation, training, and evaluation.

        Args:
            X1 (np.ndarray): Input features of train-set.
            X2 (np.ndarray): Input features of test-set.
            y1 (np.ndarray): Target values of train-set.
            y2 (np.ndarray): Target values of test-set.

        Returns:
            keras.Model: The trained Keras model and its metrics.
        """
        self.model = self.create_model()
       
        mc = ModelCheckpoint(
            self.check_point_path,
            monitor="val_loss",
            mode="min",
            verbose=self.verbose,
            save_best_only=True,
        )
        early_stop = EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )

        self.model.fit(
            X1.reshape(len(X1), *self.input_dims, 1),
            y1.flatten(),
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X2.reshape(len(X2),*self.input_dims,1), y2.flatten()),
            callbacks=[early_stop, mc],
            verbose=self.verbose,
        )

        self.model = keras.models.load_model(self.check_point_path)
        y_pred = self.model.predict(
            X2.reshape(len(X2),*self.input_dims,1), 
            verbose=self.verbose
        )
        
        y_pred2 = self.y_scaler.inverse_transform(y_pred.reshape(len(y_pred), 1))
        y_test = self.y_scaler.inverse_transform(y2.reshape(len(y2), 1))
        # print(y_test.flatten())
        r_squared, mae = self.evaluate_model(y_test.flatten(), y_pred2.flatten())

        return self.model, y_test, y_pred2, r_squared, mae


    def evaluate_model(self, y_test, y_pred, filename=None, save_file=False):
        """
        Evaluate the model's performance and plot results.

        This method calculates various performance metrics and creates a regression plot.

        Args:
            y_test (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.
        """
        # print(y_test)
        # Calculate metrics
        metrics_dict = {
            'R': stats.pearsonr(y_pred.flatten(), y_test.flatten())[0],
            'R²': metrics.r2_score(y_test, y_pred),
            'MAE': metrics.mean_absolute_error(y_test, y_pred),
            'MSE': metrics.mean_squared_error(y_test, y_pred),
            'MAPE': metrics.mean_absolute_percentage_error(y_test, y_pred)
        }

        # Visualization
        ub = max(max(y_test.flatten()), max(y_pred.flatten()))
        lb = min(min(y_test.flatten()), min(y_pred.flatten()))
        u2l = ub - lb
        ub += 0.1 * u2l
        lb -= 0.1 * u2l
        label = {k: f"{v:.4f}" for k, v in metrics_dict.items()}
        print(f'{label}')
        
        if save_file:
        
            plt.figure(figsize=(6, 6))
            sns.regplot(x=y_pred.flatten(), y=y_test.flatten(), 
                        scatter_kws={'alpha':0.4}, line_kws={'color':'red'},
                        label = f'{label}')
            plt.title(f'{label}')
            plt.xlabel('Predicted Values')
            plt.ylabel('Actual Values')
            plt.xlim(lb,ub)
            plt.ylim(lb,ub)
            plt.grid(True)
            # plt.legend()
            plt.show()
            plt.savefig(self.path + f'/regplot_{filename}.png')
            plt.close()
            
            perform_list = pd.read_csv(self.path + f'/model_performance.csv')
            y_test = pd.DataFrame(y_test)
            y_test.columns= ['ground truth']
            y_pred = pd.DataFrame(y_pred)
            y_pred.columns= ['pred']
            metric = pd.DataFrame([metrics_dict['R'], 
                                   metrics_dict['R²'], 
                                   metrics_dict['MAE'],
                                   metrics_dict['MSE'],
                                   metrics_dict['MAPE']
                                   ])
            metric.columns= ['R&R2&MAE&MSE&MAPE']
            perform_list2=pd.concat((perform_list,y_test,y_pred,metric),axis=1)
            perform_list2.drop([perform_list2.columns[0]],axis=1, inplace=True)
            perform_list2.to_csv(self.path + f'/model_performance.csv')

        return metrics_dict['R²'], metrics_dict['MAE']
        


class IndicatorSurrogateModel(SurrogateModel):
    """
    Surrogate model implementation for indicator score prediction.
    """

    def create_model(self) -> keras.Model:

        # Model architecture
        if self.indicator == 1: # for performance indicator #1
            model = Sequential([
                layers.Conv2D(32,kernel_size=(5,5),padding='same', activation='elu', 
            	          input_shape=(24,9,1)),
                layers.Conv2D(32,kernel_size=(5,5),padding='same', activation='elu'),
                layers.Conv2D(32,kernel_size=(5,5),padding='same', activation='elu'),
                layers.Conv2D(32,kernel_size=(5,5),padding='same', activation='elu'),
                layers.Flatten(),
                layers.Dense(512, activation='elu'),
                layers.Dropout(0.2),
                layers.Dense(256, activation='elu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='linear')
            ])
        elif self.indicator == 2: # for performance indicator #2
            model = Sequential([
                layers.Conv2D(32,kernel_size=(5,5),padding='same', activation='elu', 
            	          input_shape=(24,9,1)),
                layers.Conv2D(32,kernel_size=(5,5),padding='same', activation='elu'),
                layers.Conv2D(16,kernel_size=(5,5),padding='same', activation='elu'),
                layers.Dropout(0.5),
                layers.Conv2D(8,kernel_size=(5,5),padding='same', activation='elu'),
                layers.Dropout(0.5),
                layers.Conv2D(4,kernel_size=(5,5),padding='same', activation='elu'),
                layers.Dropout(0.5),
                layers.Flatten(),
                layers.Dense(128, activation='elu'),
                layers.Dropout(0.5),
                layers.Dense(1, activation='linear')
            ])
        elif self.indicator == 3 or self.indicator == 4: # for performance indicator #3 and #4
            model = Sequential([
                layers.Conv2D(32,kernel_size=(2,2),padding='same', activation='elu', 
                              input_shape=(24,9,1)),
                layers.Conv2D(32,kernel_size=(2,2),padding='same', activation='elu'),
                layers.Conv2D(32,kernel_size=(2,2),padding='same', activation='elu'),
                layers.Dropout(0.5),
                layers.Conv2D(32,kernel_size=(2,2),padding='same', activation='elu'),
                layers.Dropout(0.5),
                layers.Conv2D(16,kernel_size=(2,2),padding='same', activation='elu'),
                layers.Dropout(0.5),
                layers.Conv2D(8,kernel_size=(2,2),padding='same', activation='elu'),
                layers.Dropout(0.5),
                layers.Conv2D(4,kernel_size=(2,2),padding='same', activation='elu'),
                layers.Dropout(0.5),
                layers.Flatten(),
                layers.Dense(128, activation='elu'),
                layers.Dropout(0.5),
                layers.Dense(1, activation='linear')
            ])

        elif self.indicator == 5: # for performance indicator #5
            model = Sequential([
                layers.Conv2D(32,kernel_size=(2,2),padding='same', activation='elu', 
                              input_shape=(24,9,1)),
                layers.Dropout(0.2),
                layers.Conv2D(16,kernel_size=(2,2),padding='same', activation='elu'),
                layers.Dropout(0.2),
                layers.Conv2D(8,kernel_size=(2,2),padding='same', activation='elu'),
                layers.Dropout(0.2),
                layers.Flatten(),
                layers.Dense(128, activation='elu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='linear')
            ])
        
              
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), 
            loss='mse', metrics=["mean_squared_error"]
        )
        model.summary()
        return model



class DefaultSurrogateModel(SurrogateModel):
    """
    Default surrogate model implementation.
    """

    def create_model(self) -> keras.Model:
        model = Sequential(
            [
                Conv1D(
                    128,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation="relu",
                    input_shape=(self.input_dims, 1),
                ),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu"),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(32, kernel_size=3, strides=1, padding="same", activation="relu"),
                MaxPooling1D(pool_size=2, strides=1),
                Conv1D(16, kernel_size=3, strides=1, padding="same", activation="relu"),
                Conv1D(8, kernel_size=3, strides=1, padding="same", activation="relu"),
                Conv1D(4, kernel_size=3, strides=1, padding="same", activation="relu"),
                Flatten(),
                Dense(64, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error"
        )
        return model
