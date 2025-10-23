from data_pre_processing.e_data_pre_process import main as train_test
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score, make_scorer, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class DatasetContext:
    def __init__(self, nome_col_target='CLASSIFICACAO_FIM'):
        conj_treino_teste = train_test()
        self.X_train = conj_treino_teste[0].drop(columns=[nome_col_target])
        self.X_test = conj_treino_teste[1].drop(columns=[nome_col_target])
        self.y_train = conj_treino_teste[0][nome_col_target]
        self.y_test = conj_treino_teste[1][nome_col_target]
        self.train_sample = None
        self.test_sample = None

class ModelType:
    KNN = 1
    RANDOM_FOREST = 2
    NEURAL_NETWORK = 3
    LIGHTGBM = 4
    knn_obj = KNeighborsClassifier()
    rf_obj = RandomForestClassifier()
    lgb_obj = LGBMClassifier()

class ModelingTools:
    def __init__(self):
        self.context = DatasetContext()
    
    def aplicarOversamplingEmTreinoTeste(self):
        print('Cardinalidade original:', self.context.y_train.value_counts())
        oversample = SMOTE(sampling_strategy='not majority', random_state=42)
        X_train_os, y_train_os = oversample.fit_resample(self.context.X_train, self.context.y_train)
        print('Cardinalidade oversampling:', y_train_os.value_counts())
        self.context.X_train = X_train_os
        self.context.y_train = y_train_os

    def configuracaoGridSearchCV(self, model_obj: ModelType, param_grid: dict, n_kfold: int = 3) -> dict:
        gs_metric = make_scorer(
            lambda y_true, y_pred: recall_score(
                y_true, y_pred,
                average='weighted',
                sample_weight=np.where(y_true==3, 3, 1)
            ),
            greater_is_better=True
        )
        # Configuração de KFold.
        kfold  = KFold(n_splits=n_kfold, shuffle=True, random_state=42) 
        grid = GridSearchCV(
            model_obj,
            param_grid=param_grid,
            scoring=gs_metric,
            cv=kfold, n_jobs=1, 
            verbose=3
        )
        grid.fit(self.context.X_train, self.context.y_train)
        print('Melhores parâmetros:', grid.best_params_)
        best_p = grid.best_params_
        return best_p

    def plotarCurvaROC_AUC(self, y_test: np.array, y_pred_prob: np.array):
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        # AUC média (multiclasse)
        roc_auc = roc_auc_score(y_test_bin, y_pred_prob, average='macro', multi_class='ovr')
        print(f"AUC macro: {roc_auc:.4f}")
        # Curva ROC por classe
        plt.figure(figsize=(8, 6))
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
            plt.plot(fpr, tpr, label=f'Classe {c} (AUC = {auc(fpr, tpr):.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.5)')
        plt.xlabel('Falso Positivo (FPR)')
        plt.ylabel('Verdadeiro Positivo (TPR)')
        plt.title('Curvas ROC por Classe')
        plt.legend(loc='lower right')
        plt.show()

    def avaliarPerfomanceDoModelo(self, model_obj_and_odds: tuple[object, np.array]):
        modelo = model_obj_and_odds[0]
        y_pred_prob: np.array = model_obj_and_odds[1]

        if isinstance(modelo, KNeighborsClassifier):
            y_test = self.context.y_test.sample(1000, random_state=42)
            X_test = self.context.X_test.sample(1000, random_state=42)
        else:
            y_test = self.context.y_test
            X_test = self.context.X_test
            
        # Relatório de métricas
        y_pred = np.argmax(y_pred_prob, axis=1)
        print("\n📋 Relatório de Classificação:")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0.0))
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot(cmap='Blues', xticks_rotation=45)
        plt.title("Matriz de Confusão - Classificação Dengue")
        plt.show()
        self.plotarCurvaROC_AUC(y_test, y_pred_prob)

        print("\n📊 Calculando o índice de Permutation Importance...")
        if isinstance(modelo, (KNeighborsClassifier, LGBMClassifier, RandomForestClassifier)):
            result = permutation_importance(
                modelo, X_test, y_test, 
                n_repeats=2, random_state=42
            )
            permutations = pd.Series(
                result.importances_mean, index=self.context.X_test.columns
            ).sort_values(ascending=False)
            permutations.plot.bar(figsize=(8, 4), title="Permutation Importance")
            plt.tight_layout()
            plt.show()
            if isinstance(modelo, (LGBMClassifier, RandomForestClassifier)):
                importances = pd.Series(modelo.feature_importances_, index=self.context.X_train.columns)
                importances.nlargest(30).plot.barh(title="Tree-based Feature Importance")
                plt.tight_layout()
                plt.show()
                plt.figure(figsize=(12,6))

        self.context.train_sample = None
        self.context.test_sample = None
    
