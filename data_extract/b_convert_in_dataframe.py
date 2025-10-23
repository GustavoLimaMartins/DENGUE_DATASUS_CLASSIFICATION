import polars as pl
from pathlib import Path

class LimpadorDeDadosBrutos:
    OUTPUT = 'files/.parquet_normalizado'
    NULL_STRINGS : list = ["", " ", "NA", "None", "nan"]
    SCHEMA_OVERRIDES = {
        "DT_INVEST": pl.Date, "DT_ENCERRA": pl.Date, "NU_IDADE_N": pl.Float64, 
        "CS_SEXO": pl.String, "CS_ESCOL_N": pl.Int64, "CS_GESTANT": pl.Int64, "CS_RACA": pl.Int64, 
        "FEBRE": pl.Int64, "MIALGIA": pl.Int64, "CEFALEIA": pl.Int64, "EXANTEMA": pl.Int64, "VOMITO": pl.Int64, 
        "NAUSEA": pl.Int64, "PETEQUIA_N": pl.Int64, "DOR_COSTAS": pl.Int64, "CONJUNTVIT": pl.Int64, "ARTRITE": pl.Int64,
        "ARTRALGIA": pl.Int64, "LEUCOPENIA": pl.Int64, "LACO": pl.Int64, "DOR_RETRO": pl.Int64, 
        "HOSPITALIZ": pl.Int64, "UF": pl.Int64, "MUNICIPIO": pl.Int64, "TPAUTOCTO": pl.Int64, 
        "EVOLUCAO": pl.Int64, "CRITERIO": pl.Int64, "CLASSI_FIN": pl.Int64
    }
    
    def __init__(self):
        pass

    def normalizarArquivosParquet(self):
        """Varrer e normalizar todos os arquivos da pasta"""
        arquivos = list(Path(self.OUTPUT).glob('*.parquet'))
        print(f"🔍 {len(arquivos)} arquivos encontrados em {self.OUTPUT}")
        
        dfs = []
        for arquivo in arquivos:
            try:
                df : pl.DataFrame = self._normalizarArquivoIndividual(arquivo)
                dfs.append(df)
                print(f"\n📂 OK!: {arquivo.name}")
            except Exception as e:
                print(f"⚠️ Erro ao processar {arquivo.name}: {e}")

        for arquivo, df in zip(arquivos, dfs):
            arquivo = Path(arquivo)
            df.write_parquet(arquivo, compression='snappy')
            #print(f"✅ Arquivo salvo em subpasta: {arquivo.name}, linhas restantes: {df.shape[0]}")
                
    def _normalizarArquivoIndividual(self, arquivo: Path) -> pl.DataFrame:
        """Normaliza um único arquivo e retorna um DataFrame com linhas completas"""
        self.df_lazy = pl.scan_parquet(str(arquivo), extra_columns='ignore')
        for coluna, tipo in self.SCHEMA_OVERRIDES.items():
            if coluna in self.df_lazy.collect_schema().names():
                self.df_lazy = self.df_lazy.with_columns(pl.col(coluna).cast(tipo, strict=False))

        # Substitui valores considerados nulos por None
        colunas_string = [c for c, t in self.SCHEMA_OVERRIDES.items() if t == pl.String and c in self.df_lazy.collect_schema().names()]

        for coluna in colunas_string:
            self.df_lazy = self.df_lazy.with_columns(
                pl.when(pl.col(coluna).is_in(self.NULL_STRINGS))
                .then(None)
                .otherwise(pl.col(coluna))
                .alias(coluna)
            )

        # Agora remove linhas que têm nulo em qualquer coluna do schema
        return self.df_lazy.select(self.SCHEMA_OVERRIDES.keys()).collect().drop_nulls(subset=self.SCHEMA_OVERRIDES.keys())

    def return_dataset(self, dfs: pl.DataFrame) -> pl.DataFrame:
        with pl.Config(tbl_cols=-1):
            df = dfs.select(list(self.SCHEMA_OVERRIDES.keys()))
            df_final = df.collect()
            print(df_final.shape)
        return df_final

    def dataframeDoDataset(self, caminho_arquivo: str = None) -> pl.DataFrame:
        if caminho_arquivo is None:
            caminho_arquivo = self.OUTPUT

        dfs = pl.scan_parquet(caminho_arquivo, extra_columns='ignore')
        return self.return_dataset(dfs)

def main() -> pl.DataFrame:
    limpador = LimpadorDeDadosBrutos()
    limpador.normalizarArquivosParquet()
    return limpador.dataframeDoDataset()

if __name__ == '__main__':
    main()
