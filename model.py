import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer(object):
    def __init__(self, path, target_column, test_size=0.2, random_state=42):
        self.path = path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None

    def load_data(self):
        data = pd.read_csv(self.path)

        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]

        X = pd.get_dummies(X)

        if y.dtype == "object":
            y = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def create_model(self, input_dim, num_classes):
        model = models.Sequential([
            layers.Dense(128, activation="relu", input_shape=(input_dim,)),  
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),  
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),  
            layers.Dense(num_classes, activation="softmax")  
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, epochs=50, batch_size=32):
        X_train, X_test, y_train, y_test = self.load_data()
        
        num_classes = len(np.unique(y_train))
        self.model = self.create_model(X_train.shape[1], num_classes)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True,
                monitor="val_loss"
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.2,
                patience=3,
                min_lr=1e-6
            )
        ]

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        return {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "history": self.history.history
        }

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model doesnt trained!")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model doesnt trained!")
        
        self.model.save(filepath)

    def plot_target_distribution(self, save_path: str = None):
        data = pd.read_csv(self.path)
        
        plt.figure(figsize=(10, 6))
        churn_counts = data['Churn'].value_counts()
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(churn_counts.index, churn_counts.values, 
                    color=['lightblue', 'lightcoral'], alpha=0.8)
        plt.title('Распределение Churn', fontsize=14, fontweight='bold')
        plt.xlabel('Churn')
        plt.ylabel('Количество клиентов')
        
        for bar, count in zip(bars, churn_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    str(count), ha='center', va='bottom')
        
        plt.subplot(1, 2, 2)
        plt.pie(churn_counts.values, labels=churn_counts.index, 
            autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        plt.title('Процентное соотношение Churn', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"Всего клиентов: {len(data)}")
        print(f"Остались: {churn_counts.get('No', 0)} ({churn_counts.get('No', 0)/len(data)*100:.1f}%)")
        print(f"Ушли: {churn_counts.get('Yes', 0)} ({churn_counts.get('Yes', 0)/len(data)*100:.1f}%)")

    def plot_numerical_features(self, save_path: str = None):
        data = pd.read_csv(self.path)
        
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data = data.dropna(subset=['TotalCharges'])
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        for i, feature in enumerate(numerical_features):
            data.boxplot(column=feature, by='Churn', ax=axes[0, i])
            axes[0, i].set_title(f'{feature} по Churn')
            axes[0, i].set_ylabel(feature)
            
            for churn_value in ['Yes', 'No']:
                subset = data[data['Churn'] == churn_value]
                axes[1, i].hist(subset[feature], alpha=0.7, 
                            label=churn_value, bins=30, density=True)
            axes[1, i].set_title(f'Распределение {feature}')
            axes[1, i].legend()
            axes[1, i].set_xlabel(feature)
            axes[1, i].set_ylabel('Плотность')
        
        plt.suptitle('Анализ числовых признаков', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    def plot_categorical_features(self, save_path: str = None):
        data = pd.read_csv(self.path)
        
        categorical_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'InternetService', 'Contract', 'PaymentMethod'
        ]
        
        n_cols = 3
        n_rows = (len(categorical_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten()
        
        for i, feature in enumerate(categorical_features):
            cross_tab = pd.crosstab(data[feature], data['Churn'], normalize='index') * 100
            
            # Столбчатая диаграмма
            bars = cross_tab.plot(kind='bar', ax=axes[i], 
                                color=['lightblue', 'lightcoral'],
                                alpha=0.8)
            
            axes[i].set_title(f'Churn rate по {feature}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Процент (%)')
            axes[i].legend(['No', 'Yes'])
            axes[i].tick_params(axis='x', rotation=45)
            
            for p in bars.patches:
                if p.get_height() > 5:  
                    axes[i].annotate(f'{p.get_height():.1f}%', 
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', fontsize=9)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.suptitle('Анализ категориальных признаков', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    def plot_correlation_matrix(self, save_path: str = None):
        data = pd.read_csv(self.path)

        data_numeric = data.copy()
        data_numeric['Churn'] = data_numeric['Churn'].map({'Yes': 1, 'No': 0})
        data_numeric['TotalCharges'] = pd.to_numeric(data_numeric['TotalCharges'], errors='coerce')
        data_numeric = data_numeric[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].dropna()
        
        corr_matrix = data_numeric.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Матрица корреляций числовых признаков', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    def plot_predictions_analysis(self, save_path: str = None):
        if self.model is None:
            raise ValueError("Модель не обучена!")
        
        X_train, X_test, y_train, y_test = self.load_data()
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        axes[0, 0].hist(y_pred_proba[:, 1], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Распределение вероятностей Churn')
        axes[0, 0].set_xlabel('Вероятность Churn')
        axes[0, 0].set_ylabel('Количество')
        
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (area = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
        
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        
        axes[1, 0].plot(recall, precision, color='green', lw=2)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        
        if hasattr(self.model, 'coef_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.coef_[0]
            }).sort_values('importance', ascending=True)
            
            axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'])
            axes[1, 1].set_title('Важность признаков')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    def complete_analysis(self):
        import os
        os.makedirs('analysis_results', exist_ok=True)
        
        print("🔍 Начинаем анализ датасета...")
        
        print("1. Анализ распределения Churn...")
        self.plot_target_distribution('analysis_results/churn_distribution.png')
        
        print("2. Анализ числовых признаков...")
        self.plot_numerical_features('analysis_results/numerical_features.png')
        
        print("3. Анализ категориальных признаков...")
        self.plot_categorical_features('analysis_results/categorical_features.png')
        
        print("4. Анализ корреляций...")
        self.plot_correlation_matrix('analysis_results/correlation_matrix.png')
        
        print("✅ Анализ завершен! Все графики сохранены в папке 'analysis_results'")