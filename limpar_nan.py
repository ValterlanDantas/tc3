"""
Script para limpar valores NaN dos arquivos já gerados
"""
import pandas as pd

print("="*80)
print("LIMPEZA DE VALORES NaN DOS DATASETS")
print("="*80)

arquivos = [
    'dados_ml_limpos.csv',
    'dados_ml_normalizados.csv',
    'sample_10000_normalizado.csv',
    'sample_50000_normalizado.csv',
    'sample_100000_normalizado.csv'
]

for arquivo in arquivos:
    try:
        print(f"\nProcessando: {arquivo}")
        
        # Carregar
        df = pd.read_csv(arquivo)
        print(f"   Tamanho original: {df.shape}")
        print(f"   NaN encontrados: {df.isnull().sum().sum()}")
        
        # Remover NaN
        df_limpo = df.dropna()
        linhas_removidas = len(df) - len(df_limpo)
        
        # Salvar
        df_limpo.to_csv(arquivo, index=False)
        
        print(f"   [OK] Linhas removidas: {linhas_removidas:,}")
        print(f"   [OK] Tamanho final: {df_limpo.shape}")
        print(f"   [OK] NaN restantes: {df_limpo.isnull().sum().sum()}")
        
    except FileNotFoundError:
        print(f"   [AVISO] Arquivo nao encontrado: {arquivo}")
    except Exception as e:
        print(f"   [ERRO] {e}")

print("\n" + "="*80)
print("[OK] LIMPEZA CONCLUIDA!")
print("="*80)
