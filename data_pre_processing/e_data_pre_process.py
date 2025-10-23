from data_pre_processing.c_data_formatting import main as dataframe
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import polars as pl
import pandas as pd
import numpy as np

class DataPreProcessing:
    ONE_HOT_COLUMNS = [
        'SEXO', 'GESTANTE', 'ESCOLARIDADE', 'CRITERIO', 'TPAUTOCTO',
        'HOSPITALIZ', 'CS_RACA', 'UF', 'EVOLUCAO_FIM'
    ]
    
    BINARY_COLUMNS = [
        'FEBRE', 'MIALGIA', 'CEFALEIA', 'EXANTEMA', 
        'VOMITO', 'NAUSEA', 'PETEQUIA_N', 'DOR_COSTAS',
        'CONJUNTVIT', 'ARTRITE', 'ARTRALGIA', 'LEUCOPENIA',
        'LACO', 'DOR_RETRO'
    ]

    MONTH_COLUMNS = ['MES_INVEST', 'MES_ENCERRA']
    WEEK_COLUMNS = ['SEMANA_ISO_INVEST', 'SEMANA_ISO_ENCERRA']

    def __init__(self, df_pl: pl.DataFrame):
        self.df_pl: pl.DataFrame = df_pl
        self.df_pd: pd.DataFrame = self.df_pl.to_pandas()

    def separarConjuntosDeTreinoTeste(self, tamanho_teste: float = 0.3, semente: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Separar treino e teste com shuffle
        train_df, test_df = train_test_split(self.df_pd, test_size=tamanho_teste, random_state=semente, shuffle=True)
        return train_df, test_df

    def converterColunasBinarias(self):
        # Converte 1→1 e 2→0, tipo int8
        self.df_pl = self.df_pl.with_columns([
            pl.col(c).replace({2: 0}).cast(pl.UInt8).alias(c) for c in self.BINARY_COLUMNS
        ])
        self.df_pd = self.df_pl.to_pandas()

    def transporColunasPorCadinalidade(self):
        self.df_pd = pd.get_dummies(self.df_pl.to_pandas(), columns=self.ONE_HOT_COLUMNS, dtype='uint8')

    def codificarTempoEmSenoCosseno(self):
        # Codificação cíclica
        for month in self.MONTH_COLUMNS:
            self.df_pd[f'{month}_sin'] = np.sin(2 * np.pi * self.df_pd[month] / 12)
            self.df_pd[f'{month}_cos'] = np.cos(2 * np.pi * self.df_pd[month] / 12)
            
        for week in self.WEEK_COLUMNS:
            self.df_pd[f'{week}_sin'] = np.sin(2 * np.pi * self.df_pd[week] / 52)
            self.df_pd[f'{week}_cos'] = np.cos(2 * np.pi * self.df_pd[week] / 52)
        
        self.df_pd.drop(columns=self.WEEK_COLUMNS + self.MONTH_COLUMNS, inplace=True)

    def codificarColunaComTargetEncoding(self, tupla_treino_teste: tuple[pd.DataFrame, pd.DataFrame], nome_coluna: str, nome_col_target: str = 'CLASSIFICACAO_FIM') -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df = tupla_treino_teste[0]
        test_df = tupla_treino_teste[1]
        # Média global (com base no treino!)
        global_mean = train_df[nome_col_target].mean()
        # Suavização opcional
        k = 5
        # Cálculo do target encoding no treino
        counts = train_df.groupby(nome_coluna)[nome_col_target].count()
        means = train_df.groupby(nome_coluna)[nome_col_target].mean()
        smooth: float = ((counts * means + k * global_mean) / (counts + k))
        # Mapear para treino e teste
        train_df[f'{nome_coluna}_TE'] = train_df[nome_coluna].map(smooth)
        test_df[f'{nome_coluna}_TE'] = test_df[nome_coluna].map(smooth)
        # Preencher valores desconhecidos no teste
        test_df[f'{nome_coluna}_TE'] = test_df[f'{nome_coluna}_TE'].fillna(global_mean)
        train_df = train_df.drop(columns=[nome_coluna])
        test_df = test_df.drop(columns=[nome_coluna])
        
        return train_df, test_df
    
    def normalizarColunaNaoCategorica(self, tupla_treino_teste: tuple[pd.DataFrame, pd.DataFrame], nome_coluna: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df = tupla_treino_teste[0]
        test_df = tupla_treino_teste[1]
        scaler = MinMaxScaler()
        scaler.fit(train_df[[nome_coluna]])
        train_df[[nome_coluna]] = scaler.transform(train_df[[nome_coluna]])
        test_df[[nome_coluna]] = scaler.transform(test_df[[nome_coluna]])

        return train_df, test_df

    def validarDataframeDeSaida(self, tupla_treino_teste: tuple[pd.DataFrame, pd.DataFrame]):
        df_train = tupla_treino_teste[0]
        df_test = tupla_treino_teste[1]
        print(df_train.info())
        print('Dimensões no dataframe original:\n', self.df_pl.shape)
        print('Dimensões no dataframe de treino:\n', df_train.shape)
        print('Dimensões no dataframe de teste:\n', df_test.shape)

        print('=' * 40)
        if self.df_pl.shape[0] == (sum([df_train.shape[0], df_test.shape[0]])):
            print('QTDE DE LINHAS FOI VALIDADA!')
        else:
            print('HÁ LINHAS AUSENTES!')
        print('=' * 40)

        cols_ok = list()
        for col_origin in self.df_pl.collect_schema().names():
            for col_deriv in df_train.columns.to_list():
                if col_origin in col_deriv:
                    cols_ok.append(col_deriv)
                    break
        
        print('=' * 40)
        if len(cols_ok) == len(self.df_pl.collect_schema().names()):
            print('AS COLUNAS FORAM VALIDADAS!')
        else:
            print('HÁ COLUNAS AUSENTES!')
        print('=' * 40)

    def verificarCardinalidadeDasColunas(self, tupla_treino_teste: tuple[pd.DataFrame, pd.DataFrame]):
        # Itera por cada coluna
        for i, item in enumerate(tupla_treino_teste):
            if str(input("Deseja avaliar a cardinalidade do conjunto de treino?\nDigite 1 para SIM: ")) != '1':
                break

            for col in item.columns:
                print("*" * 30)
                if i == 0:
                    print("Conjunto de Treino")
                else:
                    print("Conjunto de Teste")
                
                print(f"Coluna: {col}")
                print(item[col].value_counts())
                print("*" * 30)
                input("Pressione Enter para continuar...") 
            
            if str(input("Deseja avaliar a cardinalidade do conjunto de teste também?\nDigite 1 para SIM: ")) != '1':
                break

        
def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    dpp = DataPreProcessing(dataframe())
    dpp.converterColunasBinarias()
    dpp.transporColunasPorCadinalidade()
    dpp.codificarTempoEmSenoCosseno()
    conjuntos = dpp.separarConjuntosDeTreinoTeste(tamanho_teste=0.2)
    conjuntos = dpp.codificarColunaComTargetEncoding(conjuntos, 'MUNICIPIO')
    conjuntos = dpp.normalizarColunaNaoCategorica(conjuntos, 'FAIXA_ETARIA')
    conjuntos = dpp.normalizarColunaNaoCategorica(conjuntos, 'MUNICIPIO_TE')
    dpp.validarDataframeDeSaida(conjuntos)
    #dpp.verificarCardinalidadeDasColunas(conjuntos)
    return conjuntos

if __name__ == '__main__':
    main()
