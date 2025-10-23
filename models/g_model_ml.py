import numpy as np
from time import sleep

class PipelineForModelSelection:
    CLASS_WEIGHT = [{0: 1, 1: 3, 2: 3, 3: 5}]

    def __init__(self, model_tools: object, model_types: object):
        self.GET_TOOL = model_tools
        self.TYPE_MODEL = model_types
        self.var = self.GET_TOOL.context
  
    def executarModelagemKNN(self, range_inicio: int, range_fim: int, train_sample: int, test_sample: int) -> tuple[object, np.array]:
        self.var.train_sample = train_sample
        self.var.test_sample = test_sample
        error = []
        neighbor = []
        knn_predicts = []
        knn_model = []
        for i in range(range_inicio, (range_fim+1)):
            print(f"Avaliando K-Value '{i}' para KNN")
            knn = self.TYPE_MODEL.knn_obj.set_params(n_neighbors=i, metric='manhattan', algorithm='ball_tree')
            knn.fit(self.var.X_train.sample(self.var.train_sample, random_state=42), self.var.y_train.sample(self.var.train_sample, random_state=42))
            pred_i = knn.predict(self.var.X_test.sample(self.var.test_sample, random_state=42))
            pred_i_proba = knn.predict_proba(self.var.X_test.sample(self.var.test_sample, random_state=42))
            error.append(np.mean(pred_i != self.var.y_test.sample(self.var.test_sample, random_state=42)))
            neighbor.append(i)
            knn_predicts.append(pred_i_proba)
            knn_model.append(knn)

        best_n = neighbor[error.index(min(error))]
        best_proba = knn_predicts[error.index(min(error))]
        best_model = knn_model[error.index(min(error))]
        print(f'Melhor N-Value: {best_n}\nMenor erro encontrado: {min(error)}')
        return best_model, best_proba

    def executarGridSearchNoModelo(self, n_model: object) -> tuple[object, np.array]:
        if n_model == self.TYPE_MODEL.RANDOM_FOREST:
            param_grid = {
                'n_estimators': [133, 253], 'max_depth': [85, 133],
                'min_samples_split': [4], 'criterion': ['entropy', 'gini'],
                'class_weight': self.CLASS_WEIGHT
            }
            best_p = self.GET_TOOL.configuracaoGridSearchCV(
                self.TYPE_MODEL.rf_obj, 
                param_grid
            )
            rf = self.TYPE_MODEL.rf_obj.set_params(
                n_estimators=best_p['n_estimators'], max_depth=best_p['max_depth'],
                min_samples_split=best_p['min_samples_split'], criterion=best_p['criterion'],
                class_weight=best_p['class_weight']
            )
            rf.fit(self.var.X_train, self.var.y_train)
            return rf, rf.predict_proba(self.var.X_test)
        
        elif n_model == self.TYPE_MODEL.LIGHTGBM:
            param_grid = {
                'n_estimators': [133, 255], 'num_leaves': [55, 85],
                'max_depth': [85, 133], 'min_child_samples': [35],
                'subsample': [0.6], 'reg_alpha': [0.3],
                'reg_lambda': [0.3], 'class_weight': self.CLASS_WEIGHT,
                'boosting_type': ['gbdt'], 'verbose': [-1]
            }
            best_p = self.GET_TOOL.configuracaoGridSearchCV(
                self.TYPE_MODEL.lgb_obj,
                param_grid
            )
            lgbm = self.TYPE_MODEL.lgb_obj.set_params(
                n_estimators=best_p['n_estimators'], num_leaves=best_p['num_leaves'], max_depth=best_p['max_depth'],
                min_child_samples=best_p['min_child_samples'], subsample=best_p['subsample'],
                reg_alpha=best_p['reg_alpha'], reg_lambda=best_p['reg_lambda'],
                class_weight=best_p['class_weight'], objective='multiclass', random_state=42
            )
            lgbm.fit(self.var.X_train, self.var.y_train)
            return lgbm, lgbm.predict_proba(self.var.X_test)

def main(model_tools: object, model_types: object):
    pipeline = PipelineForModelSelection(model_tools, model_types)
    print("\nAplicando oversampling nos conjuntos de treino e teste...")
    pipeline.GET_TOOL.aplicarOversamplingEmTreinoTeste()
    sleep(5)
    print("\nIniciando modelagem KNN...")
    knn: tuple = pipeline.executarModelagemKNN(range_inicio=8, range_fim=16, train_sample=25000, test_sample=6500)
    pipeline.GET_TOOL.avaliarPerfomanceDoModelo(knn)
    sleep(10)
    print("\nIniciando Grid Search para Random Forest e LightGBM...")
    rf: tuple = pipeline.executarGridSearchNoModelo(pipeline.TYPE_MODEL.RANDOM_FOREST)
    pipeline.GET_TOOL.avaliarPerfomanceDoModelo(rf)
    sleep(10)
    lgbm: tuple = pipeline.executarGridSearchNoModelo(pipeline.TYPE_MODEL.LIGHTGBM)
    pipeline.GET_TOOL.avaliarPerfomanceDoModelo(lgbm)
    sleep(10)
    
