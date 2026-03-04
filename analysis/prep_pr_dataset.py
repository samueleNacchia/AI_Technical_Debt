from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


def process_dataset(type_path, meta_path, is_human=False):

    # 1. Caricamento e integrazione dei dati base
    df_type = pd.read_parquet(type_path)
    df_meta = pd.read_parquet(meta_path)
    df = pd.merge(df_type, df_meta, on='id', how='inner', suffixes=('', '_y'))

    # 2. Logica di filtraggio per lo strato AI
    if not is_human:
        # Verifica della 'massa tecnica': inclusione solo di PR con modifiche file
        commit_path = "hf://datasets/hao-li/AIDev/pr_commit_details.parquet"
        df_commits = pd.read_parquet(commit_path, columns=['pr_id'])
        valid_pr_ids = df_commits['pr_id'].unique()

        # Selezione dei record con evidenza di contribuzione strutturale
        df = df[df['id'].isin(valid_pr_ids)].copy()

        # Filtro di qualità: soglia minima di confidenza (Confidence >= 9)
        if 'confidence' in df.columns:
            df = df[df['confidence'] >= 9].copy()
    else:
        # Etichettatura del gruppo di controllo umano
        df['agent'] = 'Human'

    # 3. Pulizia e standardizzazione del dataset
    # Rimozione di metadati non rilevanti ai fini del calcolo delle metriche
    cols_to_drop = ['title', 'body', 'reason', 'repo_url']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    df.drop(columns=[c for c in df.columns if c.endswith('_y')], inplace=True)

    # Selezione delle tipologie di task correlate allo sviluppo produttivo
    if 'type' in df.columns:
        # Esclusione di 'other' e 'revert' per minimizzare il rumore statistico
        df = df[~df['type'].isin(['other', 'revert'])].copy()

    # Finalizzazione del dataframe
    if 'confidence' in df.columns:
        df.drop(columns=['confidence'], inplace=True)

    return df


def analyze_distributions(df, output_name="pr_sample_distribution.png"):
    # 1. Creazione cartella figs se non esiste
    output_dir = Path("figs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerazione Matrice e salvataggio in {output_dir}/{output_name}...")

    # 2. Creazione della tabella di contingenza
    distribution = pd.crosstab(df['agent'], df['type'])

    # 3. Creazione della Matrice Grafica
    plt.figure(figsize=(14, 8))
    sns.heatmap(distribution, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)

    plt.title('Matrice di Distribuzione PR: Conteggi per Agente e Tipo', fontsize=16, pad=20)
    plt.ylabel('Agente / Origine', fontsize=12)
    plt.xlabel('Tipo di Pull Request', fontsize=12)

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # --- MODIFICA QUI: Salva invece di mostrare ---
    save_path = output_dir / output_name
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Chiude la figura per liberare memoria

    print(f"Immagine salvata correttamente.")
    return distribution


def get_robust_sample(df, target_moe, confidence_level, min_per_stratum):
    N = len(df)

    alpha = 1 - confidence_level
    # Calcoliamo lo Z-score esatto
    z = stats.norm.ppf(1 - alpha / 2)

    p = 0.5  # Massima variabilità per un calcolo conservativo

    # 1. Calcolo dimensione del campione (n) con correzione per popolazione finita
    numerator = (z ** 2 * p * (1 - p)) / (target_moe ** 2)
    n_required = int(np.ceil(numerator / (1 + (numerator - 1) / N)))

    print(f"Analisi popolazione di {N} PR...")
    print(f"Obiettivo scientifico: Confidenza {confidence_level:.0%}, Errore {target_moe:.0%}")
    print(f"Campioni necessari calcolati: {n_required}")

    # 2. Base Stratificata Obbligatoria (Minimo 10 o tutto ciò che c'è)
    strata = df.groupby(['agent', 'type'])
    sample_indices = []

    for _, group in strata:
        n_to_take = min(len(group), min_per_stratum)
        sample_indices.extend(group.sample(n=n_to_take, random_state=42).index)

    df_min = df.loc[sample_indices]

    # 3. Completamento fino al numero richiesto (n_required)
    remaining_n = n_required - len(df_min)
    if remaining_n > 0:
        pool = df.drop(df_min.index)
        extra = pool.sample(n=remaining_n, random_state=42)
        df_final = pd.concat([df_min, extra])
    else:
        # Se i minimi superano già il n richiesto, teniamo i minimi
        df_final = df_min

    # --- VALIDAZIONE STATISTICA FINALE ---
    n_final = len(df_final)
    # Calcolo MoE reale ottenuto
    moe_real = z * np.sqrt((0.25 / n_final) * (N - n_final) / (N - 1))

    print("\n" + "=" * 40)
    print(f"CAMPIONE ESTRATTO: {n_final} PR")
    print(f"Margine d'Errore Reale: {moe_real:.2%}")
    print(f"Confidenza Effettiva: {confidence_level:.0%}")
    print("=" * 40)

    # Ritorna sia il dataframe che il numero totale di campioni
    return df_final, n_final

if __name__ == "__main__":
    paths = {
        "agent_type": "hf://datasets/hao-li/AIDev/pr_task_type.parquet",
        "agent_meta": "hf://datasets/hao-li/AIDev/pull_request.parquet",
        "human_type": "hf://datasets/hao-li/AIDev/human_pr_task_type.parquet",
        "human_meta": "hf://datasets/hao-li/AIDev/human_pull_request.parquet"
    }

    try:
        # Elaborazione
        df_agents = process_dataset(paths["agent_type"], paths["agent_meta"], is_human=False)
        df_human = process_dataset(paths["human_type"], paths["human_meta"], is_human=True)

        # Unione
        all_pr_with_type = pd.concat([df_agents, df_human], ignore_index=True)

        print("\n" + "=" * 30)
        print(f"CONTEGGIO FINALE: {len(all_pr_with_type)} righe")
        print(f"Elenco colonne: {list(all_pr_with_type.columns)}")
        print("=" * 30)

        # Salvataggio
        all_pr_with_type.to_csv("dataset/all_pr_type.csv", index=False)

        df_studio, n_campioni = get_robust_sample(all_pr_with_type, target_moe=0.02, confidence_level=0.95, min_per_stratum=20)

        dist_counts = analyze_distributions(df_studio)

        # Salvataggio del dataset definitivo
        df_studio.to_csv("dataset/pr_study_sample.csv", index=False)

    except Exception as e:
        print(f"Errore: {e}")

