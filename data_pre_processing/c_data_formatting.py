from data_extract.b_convert_in_dataframe import main as dataframe
import polars as pl
from typing import Dict

class DataframeFormatting:
    # Constantes para o cálculo de idade SINAN
    DIAS_POR_ANO = 365
    MESES_POR_ANO = 12
    ANO = 4
    MES = 5
    DIAS = 3
    IGNORADO = 0

    def __init__(self, df: pl.DataFrame):
        self.df = df
    
    def criarFaixaEtaria(self) -> pl.DataFrame:
        """Calcula a idade em anos a partir do código NU_IDADE_N do SINAN e cria a FAUXA_ETARIA."""
        # 1️⃣ Extrair unidade e valor
        idade_codigo = self.df["NU_IDADE_N"].cast(pl.UInt32)
        unidade = pl.when(idade_codigo.is_null()).then(0).otherwise(
            idade_codigo.cast(pl.String).str.slice(0,1).cast(pl.UInt8)
        )
        valor = pl.when(idade_codigo.is_null()).then(None).otherwise(
            idade_codigo % 1000
        )

        self.df = self.df.with_columns([
            unidade.alias("UNIDADE"),
            valor.alias("VALOR")
        ])

        # 2️⃣ Calcular idade em anos como float
        self.df = self.df.with_columns([
            pl.when(pl.col("UNIDADE") == self.ANO).then(pl.col("VALOR").cast(pl.Float32))
            .when(pl.col("UNIDADE") == self.MES).then(pl.col("VALOR").cast(pl.Float32) / self.MESES_POR_ANO)
            .when(pl.col("UNIDADE") == self.DIAS).then(pl.col("VALOR").cast(pl.Float32) / self.DIAS_POR_ANO)
            .otherwise(None)
            .alias("IDADE_ANOS")
        ])

        # 3️⃣ Criar faixa etária (todas as branches float)
        self.df = self.df.with_columns([
            pl.when(pl.col("IDADE_ANOS").is_null()).then(0.0)
            .when(pl.col("IDADE_ANOS") < 1).then(1.0)
            .when(pl.col("IDADE_ANOS") < 5).then(2.0)
            .when(pl.col("IDADE_ANOS") < 10).then(3.0)
            .when(pl.col("IDADE_ANOS") < 20).then(4.0)
            .when(pl.col("IDADE_ANOS") < 30).then(5.0)
            .when(pl.col("IDADE_ANOS") < 40).then(6.0)
            .when(pl.col("IDADE_ANOS") < 50).then(7.0)
            .when(pl.col("IDADE_ANOS") < 60).then(8.0)
            .when(pl.col("IDADE_ANOS") < 70).then(9.0)
            .when(pl.col("IDADE_ANOS") < 80).then(10.0)
            .otherwise(11.0)
            .alias("FAIXA_ETARIA")
        ])

        self.df = self.df.drop(["NU_IDADE_N", "VALOR", "IDADE_ANOS", "UNIDADE"])
        return self.df
    
    def criarMesSemana(self) -> pl.DataFrame:
        # A coluna DT_INVEST deve ser do tipo pl.Date ou pl.Datetime para usar .dt.
        self.df = self.df.with_columns([
            # 1. Extrair o Mês (1=Janeiro, 12=Dezembro)
            pl.col("DT_INVEST").dt.month().alias("MES_INVEST"),
            pl.col("DT_ENCERRA").dt.month().alias("MES_ENCERRA"),
            # 2. Extrair a Semana do Ano (ISO)
            pl.col("DT_INVEST").dt.week().alias("SEMANA_ISO_INVEST"),
            pl.col("DT_ENCERRA").dt.week().alias("SEMANA_ISO_ENCERRA")
        ]).drop(["DT_INVEST", "DT_ENCERRA"])
        return self.df
    
    def converterSexoEmNumeros(self) -> pl.DataFrame:
        sexo = pl.col("CS_SEXO")

        # Expressão para calcular a idade em ANOS
        sexo_expr = (
            pl.when(sexo == "F").then(1)
            .when(sexo == "M").then(2)
            .when(sexo == "I").then(self.IGNORADO)
            .otherwise(None)
            .cast(pl.UInt8)
            .alias("SEXO")
        )
        self.df = self.df.with_columns([sexo_expr]).drop(["CS_SEXO"])
        return self.df
    
    def reduzirIntervaloEscolaridade(self) -> pl.DataFrame:
        MAP_ESCOLARIDADE_CODIFICADA: Dict[int, int] = {
            9: self.IGNORADO, 10: self.IGNORADO,    # 0: Ignorado / Não se Aplica
            0: 1, 11: 1,    # 1: Analfabeto/Sem Escolaridade
            1: 2, 2: 2, 3: 2, # 2: Ensino Fundamental Incompleto
            4: 3,             # 3: Ensino Fundamental Completo
            5: 4,             # 4: Ensino Médio Incompleto
            6: 5,             # 5: Ensino Médio Completo
            7: 6,             # 6: Educação Superior Incompleta
            8: 7,             # 7: Educação Superior Completa
        }

        self.df = self.df.with_columns(
            pl.col("CS_ESCOL_N")
            .replace_strict(MAP_ESCOLARIDADE_CODIFICADA, default=pl.lit(None))
            .cast(pl.UInt8)
            .alias("ESCOLARIDADE")
        ).drop("CS_ESCOL_N")
        return self.df
    
    def reduzirIntervaloGestante(self) -> pl.DataFrame:
        MAP_GESTANT_CONSOLIDADA: Dict[int, int] = {
            # 1: Sim (Gestante em qualquer trimestre)
            1: 1, 2: 1, 3: 1, 4: 1,
            # 0: Não (Não gestante ou Não se aplica)
            5: 2, 6: 2, 
            # 9: Ignorado
            9: self.IGNORADO
        }

        self.df = self.df.with_columns(
            pl.col("CS_GESTANT")
            .replace_strict(MAP_GESTANT_CONSOLIDADA, default=pl.lit(None))
            .cast(pl.UInt8)
            .alias("GESTANTE")
        ).drop("CS_GESTANT")
        return self.df
    
    def reduzirIntervaloClassificacaoFinal(self) -> pl.DataFrame:
        MAP_CLASSIFICACAO: Dict[int, int] = {
            # 0: Não (descartado)
            5: self.IGNORADO, 8: self.IGNORADO,
            # 1: Sim (confirmado)
            10: 1,  # Dengue
            11: 2,  # Dengue com sinais de alarme
            12: 3   # Dengue grave
        }

        self.df = self.df.with_columns(
            pl.col("CLASSI_FIN")
            .replace_strict(MAP_CLASSIFICACAO, default=pl.lit(None))
            .cast(pl.UInt8)
            .alias("CLASSIFICACAO_FIM")
        ).drop("CLASSI_FIN")
        return self.df
    
    def reduzirIntervaloEvolucao(self) -> pl.DataFrame:
        MAP_EVOLUCAO: Dict[int, int] = {
            9: self.IGNORADO,   # Ignorado
            1: 1,   # Cura
            2: 2,   # Óbito pelo agravo
            3: 2,   # Óbito por causas
            4: 2    # Óbito em investigação
        }

        self.df = self.df.with_columns(
            pl.col("EVOLUCAO")
            .replace_strict(MAP_EVOLUCAO, default=pl.lit(None))
            .cast(pl.UInt8)
            .alias("EVOLUCAO_FIM")
        ).drop("EVOLUCAO")
        return self.df
    

def main() -> pl.DataFrame:
    pp = DataframeFormatting(dataframe())
    pp.criarFaixaEtaria()
    pp.criarMesSemana()
    pp.converterSexoEmNumeros()
    pp.reduzirIntervaloEscolaridade()
    pp.reduzirIntervaloGestante()
    pp.reduzirIntervaloEvolucao()
    df = pp.reduzirIntervaloClassificacaoFinal()
    print(df.describe())
    return df

if __name__ == '__main__':
    main()
