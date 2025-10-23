import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from data_pre_processing.c_data_formatting import main as dataframe

def analisarDadosViaGraficos():
    lista_perfil = [
        'FAIXA_ETARIA', 'SEXO', 'ESCOLARIDADE', 'GESTANTE', 
        'EVOLUCAO_FIM', 'CS_RACA', 'UF', 'MUNICIPIO', 'MES_INVEST',
        'MES_ENCERRA', 'SEMANA_ISO_INVEST', 'SEMANA_ISO_ENCERRA',
        'CLASSIFICACAO_FIM'
    ]

    lista_sintomas = [
        'FEBRE', 'MIALGIA', 'CEFALEIA', 'EXANTEMA', 
        'VOMITO', 'NAUSEA', 'PETEQUIA_N', 'DOR_COSTAS',
        'CONJUNTVIT', 'ARTRITE', 'ARTRALGIA', 'LEUCOPENIA',
        'LACO', 'DOR_RETRO', 'HOSPITALIZ'
    ]

    with pl.Config(tbl_cols=20, tbl_rows=-1):
        df_pl = dataframe()
        for column in df_pl.drop('CLASSIFICACAO_FIM').collect_schema().names():
            resultado_cruzado = (
                df_pl.group_by(['CLASSIFICACAO_FIM', column])
                .agg(pl.len().alias("qtd_ocorrencias"))
                .sort(['qtd_ocorrencias'], descending=True)
                .limit(30)
            )
            print(f"\n--- Coluna: {column} ---")
            print(resultado_cruzado)

        df_perfil = df_pl.select(lista_perfil)
        df_sintomas = df_pl.select(lista_sintomas)
        print('Estatísticas:\n', df_perfil.group_by('CLASSIFICACAO_FIM').mean())
        print('Dimensão:\n', df_pl.shape)

        df_pd_perfil = df_perfil.drop('CLASSIFICACAO_FIM').to_pandas()
        df_pd_sintomas = df_sintomas.to_pandas()
        
        df_perfil_corr = df_pd_perfil.corr()
        sns.heatmap(df_perfil_corr, annot=True, fmt='.2f')
        plt.show()
        
        df_sintomas_corr = df_pd_sintomas.corr()
        sns.heatmap(df_sintomas_corr, annot=True, fmt='.2f')
        plt.show()
        
        df_pd_perfil.hist()
        plt.show()
        
        for var in lista_perfil:
            if var != 'CLASSIFICACAO_FIM':
                sns.boxplot(df_pd_perfil[var])
                plt.show()

if __name__ == '__main__':
    analisarDadosViaGraficos()
    