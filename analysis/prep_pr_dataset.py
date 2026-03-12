import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# --- UTILS ---

def clean_id(series: pd.Series) -> pd.Series:
    """Standardizza gli ID rimuovendo decimali e spazi."""
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()


def extract_repo_id(df: pd.DataFrame) -> pd.Series:
    """Estrae il path 'owner/repo' dagli URL di GitHub."""
    if 'repo_url' in df.columns:
        return df['repo_url'].str.replace('https://github.com/', '', regex=False).str.strip('/')
    elif 'html_url' in df.columns:
        return df['html_url'].str.extract(r'github\.com/([^/]+/[^/]+)')[0]
    return pd.Series("unknown_repository", index=df.index)


# --- CORE LOGIC ---

def process_dataset(type_path: str, meta_path: str, is_human: bool = False,
                    human_metrics_path: str = "human_pr_commit_details.csv") -> pd.DataFrame:
    """
    Carica, pulisce e integra i dati delle Pull Request.
    """
    tag = "HUMAN" if is_human else "AI-AGENT"
    print(f"--- Processamento dataset: [{tag}] ---")

    # 1. Caricamento e Join Iniziale
    df_type = pd.read_parquet(type_path).rename(columns={'pr_id': 'id'})
    df_meta = pd.read_parquet(meta_path)

    df_type['id'] = clean_id(df_type['id'])
    df_meta['id'] = clean_id(df_meta['id'])

    # Evitiamo duplicati di colonne prima del merge
    cols_to_drop = {'additions', 'deletions', 'changed_files', 'dir_count',
                    'comments', 'review_comments', 'n_comments', 'agent'}
    df_meta = df_meta.drop(columns=[c for c in cols_to_drop if c in df_meta.columns])

    df = pd.merge(df_type, df_meta, on='id', how='inner')

    # 2. Identificazione Repository
    if 'repo_id' not in df.columns or df['repo_id'].isna().all():
        df['repo_id'] = extract_repo_id(df)

    # 3. Arricchimento Dati (AI vs Human)
    if not is_human:
        # Statistiche Commit AI
        commit_path = "hf://datasets/hao-li/AIDev/pr_commit_details.parquet"
        try:
            df_det = pd.read_parquet(commit_path, columns=['pr_id', 'additions', 'deletions', 'filename'])
            df_det['pr_id'] = clean_id(df_det['pr_id'])

            # Calcolo directory e file unici
            df_det['directory'] = df_det['filename'].apply(lambda x: os.path.dirname(str(x)) if pd.notnull(x) else "")

            pr_stats = df_det.groupby('pr_id').agg({
                'additions': 'sum',
                'deletions': 'sum',
                'filename': 'nunique',
                'directory': 'nunique'
            }).rename(columns={'filename': 'changed_files', 'directory': 'dir_count'}).reset_index()

            df = pd.merge(df, pr_stats, left_on='id', right_on='pr_id', how='inner')
        except Exception as e:
            print(f" [!] Errore dettagli commit AI: {e}")

        # Commenti AI
        try:
            comm_path = "hf://datasets/hao-li/AIDev/pr_comments.parquet"
            df_comm = pd.read_parquet(comm_path, columns=['pr_id'])
            df_comm['pr_id'] = clean_id(df_comm['pr_id'])
            comm_stats = df_comm.groupby('pr_id').size().reset_index(name='n_comments')
            df = pd.merge(df, comm_stats, left_on='id', right_on='pr_id', how='left')
        except Exception as e:
            print(f" [!] Errore commenti AI: {e}")
    else:
        # Metriche Umane da CSV esterno
        if Path(human_metrics_path).exists():
            df_hu_metrics = pd.read_csv(human_metrics_path)
            df_hu_metrics['id'] = clean_id(df_hu_metrics['id'])
            # Teniamo solo colonne numeriche utili o non presenti
            cols_to_keep = ['id', 'additions', 'deletions', 'changed_files', 'dir_count', 'n_comments']
            df = pd.merge(df, df_hu_metrics[[c for c in cols_to_keep if c in df_hu_metrics.columns]], on='id',
                          how='left')
        else:
            print(f" [!] File metriche umane non trovato in: {human_metrics_path}")

    # 4. Pulizia e Filtri Strategici
    df['n_comments'] = df['n_comments'].fillna(0).astype(int)
    df['dir_count'] = df['dir_count'].replace(0, 1).fillna(1)

    # Filtro: Solo PR con contenuto
    if 'additions' in df.columns:
        df = df[(df['additions'] + df['deletions']) > 0].copy()

    # Filtro: Confidenza (Solo AI) e tipi non rilevanti
    if not is_human and 'confidence' in df.columns:
        df = df[df['confidence'] >= 9]

    df = df[~df['type'].isin(['other', 'revert'])].copy()

    # Uniformiamo la colonna 'agent' (usa agent_x se presente dal merge, altrimenti assegna manuale)
    if 'agent_x' in df.columns:
        df = df.rename(columns={'agent_x': 'agent'})
    elif 'agent' not in df.columns:
        df['agent'] = tag

    # Drop colonne di servizio finali
    final_keep = ['id', 'number', 'repo_id', 'agent', 'type', 'additions', 'deletions', 'changed_files', 'dir_count',
                  'n_comments', 'created_at', 'merged_at', 'html_url']
    df = df[[c for c in final_keep if c in df.columns]]

    print(f" [+] PR totali elaborate: {len(df)}")
    return df


def analyze_distributions(df: pd.DataFrame, output_name: str):
    """Genera e salva una heatmap della distribuzione dei dati."""
    output_dir = Path("../figs")
    output_dir.mkdir(parents=True, exist_ok=True)

    distribution = pd.crosstab(df['agent'], df['type'])

    plt.figure(figsize=(12, 6))
    sns.heatmap(distribution, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Numero di PR'})

    plt.title('Distribuzione Tipologie Pull Request per Agente', fontsize=14, fontweight='bold')
    plt.xlabel('Tipo Task', fontsize=12)
    plt.ylabel('Soggetto', fontsize=12)
    plt.tight_layout()

    save_path = output_dir / output_name
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" [OK] Matrice salvata in: {save_path}")
    return distribution


def get_robust_sample(df: pd.DataFrame, target_moe: float = 0.05,
                      confidence_level: float = 0.95, min_per_stratum: int = 20) -> Tuple[pd.DataFrame, int]:
    N = len(df)
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    p = 0.5
    num = (z ** 2 * p * (1 - p)) / (target_moe ** 2)
    n_required = int(np.ceil(num / (1 + (num - 1) / N)))

    print(f"\n--- Analisi Statistica Campionamento ---")
    print(f" Target MoE: {target_moe:.1%} | N richiesto: {n_required}")

    # 1. Stratificazione: estraiamo solo gli INDICI delle righe
    strata = df.groupby(['agent', 'type'], group_keys=False)

    # Estraiamo gli indici riga per lo strato minimo
    sampled_indices = strata.apply(
        lambda x: x.sample(n=min(len(x), min_per_stratum), random_state=42).index.to_series(),
        include_groups=False
    ).values

    # Flatten della lista di indici (nel caso apply restituisca una serie di liste)
    if len(sampled_indices) > 0 and isinstance(sampled_indices[0], (pd.Series, list, np.ndarray)):
        sampled_indices = np.concatenate(sampled_indices)

    # Creiamo il primo blocco usando gli indici trovati
    df_min = df.loc[sampled_indices].copy()

    # 2. Riempimento: aggiungiamo PR casuali dal resto del dataset
    remaining_n = n_required - len(df_min)
    if remaining_n > 0:
        pool = df.drop(df_min.index)
        extra = pool.sample(n=min(len(pool), remaining_n), random_state=42)
        df_final = pd.concat([df_min, extra], ignore_index=True)
    else:
        df_final = df_min.reset_index(drop=True)

    n_final = len(df_final)
    moe_real = z * np.sqrt((0.25 / n_final) * (N - n_final) / (N - 1)) if N > 1 else 0
    print(f" Risultato: {n_final} PR (MoE Reale: {moe_real:.2%})")

    return df_final, n_final


# --- MAIN ---

if __name__ == "__main__":
    paths = {
        "agent_type": "hf://datasets/hao-li/AIDev/pr_task_type.parquet",
        "agent_meta": "hf://datasets/hao-li/AIDev/pull_request.parquet",
        "human_type": "hf://datasets/hao-li/AIDev/human_pr_task_type.parquet",
        "human_meta": "hf://datasets/hao-li/AIDev/human_pull_request.parquet"
    }

    try:
        # Processamento
        df_agents = process_dataset(paths["agent_type"], paths["agent_meta"], is_human=False)
        df_human = process_dataset(paths["human_type"], paths["human_meta"],
                                   is_human=True, human_metrics_path=r"reports/metrics/human_pr_commit_details.csv")

        # Unione e salvataggio dataset globale
        all_pr = pd.concat([df_agents, df_human], ignore_index=True)
        os.makedirs("dataset", exist_ok=True)
        all_pr.to_csv("dataset/all_pr_type.csv", index=False)

        # Analisi Distribuzione Totale
        analyze_distributions(all_pr, "all_pr_distribution.png")

        # Campionamento per studio qualitativo/approfondito
        df_sample, _ = get_robust_sample(all_pr, target_moe=0.03, confidence_level=0.98, min_per_stratum=20)

        # Analisi Distribuzione Campione
        analyze_distributions(df_sample, "pr_sample_distribution.png")
        df_sample.to_csv("dataset/pr_study_sample.csv", index=False)

        print("\n[SUCCESS] Pipeline completata con successo.")

    except Exception as e:
        print(f"\n[ERROR] Errore critico durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()

