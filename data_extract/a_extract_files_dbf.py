from dbfread import DBF
import pandas as pd

class ExtractTransformLoadDBF:
    def __init__(self):    
        self.prefix_path = 'files/.dbc_dbf'
        # Arquivos .dbc expandidos para .dbf via TABWIN (aplicativo do DATASUS)
        self.entrada = ['/DENGBR19.dbf', '/DENGBR20.dbf', '/DENGBR21.dbf', '/DENGBR22.dbf', '/DENGBR23.dbf', '/DENGBR24.dbf']
        self.saida = 'files/.parquet'
    
    def converterDbfEmParquet(self):
        dados = []
        for item in range(len(self.entrada)):
            tabela = DBF(fr'{self.prefix_path}{self.entrada[item]}', encoding='latin1', load=False)
            for i, rec in enumerate(tabela, 1):
                dados.append(rec)
                if i % 100000 == 0:
                    print(f"{i} registros processados...")
                    pd.DataFrame(dados).to_parquet(fr'{self.saida}\{i}_dengue_{2018+(item+1)}_datasus.parquet', compression='snappy')
                    dados = []
                    #break
            
            print(f"✅ Conversão completa: {self.entrada[item]}", self.saida)

def main():
    etl = ExtractTransformLoadDBF()
    etl.converterDbfEmParquet()

if __name__ == '__main__':
    main()
