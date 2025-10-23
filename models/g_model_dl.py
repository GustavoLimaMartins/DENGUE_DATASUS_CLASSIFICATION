from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

def main(model_tools: object):
    GET_TOOL = model_tools
    context = GET_TOOL.context

    num_classes = len(np.unique(context.y_train))
    y_train_cat = to_categorical(context.y_train, num_classes)
    y_test_cat = to_categorical(context.y_test, num_classes)

    classes = np.unique(context.y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=context.y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))

    input_dim = context.X_train.shape[1] 
    model = Sequential([
        Input(shape=(input_dim,)), 
        Dense(256, activation='relu'), # Reduzido
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu'),  # Reduzido
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),  # Reduzido
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        Dropout(0.2),

        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=1e-4)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    model.fit(
        context.X_train, y_train_cat,
        validation_data=(context.X_test, y_test_cat),
        epochs=25,
        batch_size=128,
        callbacks=[es],
        verbose=1,
        class_weight=class_weight_dict
    )

    loss, acc = model.evaluate(context.X_test, y_test_cat, verbose=0)
    print(f"\n✅ Acurácia no conjunto de teste: {acc:.4f}")
    #y_pred_prob = model.predict(context.X_test)
    #GET_TOOL.avaliarPerfomanceDoModelo((model, y_pred_prob))

    def keras_recall_scorer(model, X, y_true):
        y_proba = model.predict(X, verbose=0) 
        y_pred_classes = np.argmax(y_proba, axis=1)
        return recall_score(y_true, y_pred_classes, average='weighted')

    r = permutation_importance(
        estimator=model,         
        X=context.X_test,         # Dados de validação
        y=context.y_test,         # Variável alvo (target)
        scoring=keras_recall_scorer, # O scorer personalizado
        n_repeats=1,            # Número de vezes que a permutação é repetida
        random_state=42,
        n_jobs=1                # Use todos os cores da CPU
    )

    # 1. Criar uma Série Pandas com as importâncias médias e os nomes das colunas
    importances_df = pd.Series(
        r.importances_mean, 
        index=context.X_test.columns 
    )

    # 2. Ordenar a série para identificar as mais importantes
    # e filtrar as top 20 para uma visualização clara
    sorted_importances = importances_df.sort_values(ascending=False)
    top_n = 20 

    # 3. Plotar o gráfico de barras
    plt.figure(figsize=(10, 8))
    sorted_importances.nlargest(top_n).plot.bar(color='skyblue')

    plt.title(f"Permutation Importance (Top {top_n} Features) 📊")
    plt.ylabel("Queda na Acurácia (Média da Importância)")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha='right') # Rotação para os nomes longos
    plt.tight_layout()
    plt.show()

    print("\n📋 Relatório de Permutation Importance (Top 10):")
    feature_names = context.X_test.columns

    # Filtro: Importância média > 2 * desvio padrão (para focar em features significativas)
    count = 0
    for i in r.importances_mean.argsort()[::-1]:
        mean = r.importances_mean[i]
        std = r.importances_std[i]
        
        if mean - 2 * std > 0:
            name = feature_names[i]
            print(f"[{count+1:2d}] {name:<25}: {mean:.4f} +/- {std:.4f}")
            count += 1
            if count >= 10: # Limite o print a 10 para concisão
                break
