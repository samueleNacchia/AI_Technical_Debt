import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def process_dataset(type_path: str, meta_path: str, is_human_dataset: bool = False,
                    human_metrics_path: str = "reports/metrics/human_pr_commit_details_v2.csv",
                    human_reviews_path: str = "dataset/human_reviews_cleaned.csv") -> pd.DataFrame:
    """
    Versione di PREPARAZIONE:
    Esclude i commenti dell'autore e traccia lo stato delle interazioni.
    """
    tag = "HUMAN" if is_human_dataset else "AI-AGENT"
    print(f"\n--- 🚀 Processamento dataset: [{tag}] ---")

    def robust_id(series):
        return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

    # --- 1. CARICAMENTO E JOIN INIZIALE ---
    df_type = pd.read_parquet(type_path).rename(columns={'pr_id': 'id'})
    df_meta = pd.read_parquet(meta_path)

    df_type['id'] = robust_id(df_type['id'])
    df_meta['id'] = robust_id(df_meta['id'])

    # Mappatura Autore della PR (fondamentale per filtrare i commenti dopo)
    df_authors = df_meta[['id', 'user']].rename(columns={'id': 'pr_id', 'user': 'pr_author'})

    cols_to_drop = {'comments', 'w_comments', 'agent', 'n_comments', 'n_total_comments'}
    df_meta = df_meta.drop(columns=[c for c in cols_to_drop if c in df_meta.columns])

    df = pd.merge(df_type, df_meta, on='id', how='inner')

    # --- 2. IDENTIFICAZIONE REPOSITORY ---
    if 'repo_id' not in df.columns or df['repo_id'].isna().all():
        df['repo_id'] = extract_repo_id(df)

    bot_keywords = [
        r'sonar', r'codacy', r'codecov', r'coveralls', r'linter', r'snyk',
        r'whitesource', r'jenkins', r'travis', r'circleci', r'appveyor',
        r'github-actions', r'vercel', r'netlify', r'houndci',
        r'mend-bolt', r'mend-bot', r'robot', r'commenter', r'mergebot'
    ]
    bot_pattern = '|'.join(bot_keywords)

    # --- 3. ARRICCHIMENTO DATI ---
    if not is_human_dataset:
        try:
            print(f"📡 Recupero interazioni UMANE da Hugging Face...")

            # A. REVIEWS & APPROVAL & INTERACTION STATES
            df_revs = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_reviews.parquet",
                                      columns=['id', 'pr_id', 'state', 'user_type', 'user'])
            df_revs['pr_id'] = robust_id(df_revs['pr_id'])
            df_revs['id'] = robust_id(df_revs['id'])
            df_revs_hu = df_revs[df_revs['user_type'].str.lower().str.strip() == 'user'].copy()
            df_revs_hu = df_revs_hu[~df_revs_hu['user'].str.lower().str.contains(bot_pattern, na=False)]

            # Tracciamo Approved e la presenza di interazioni critiche (Changes Requested o Commented)
            review_stats = df_revs_hu.groupby('pr_id')['state'].agg(
                has_approved_review=lambda x: (x == 'APPROVED').any(),
                has_interaction_review=lambda x: (x.isin(['CHANGES_REQUESTED', 'COMMENTED'])).any()
            ).reset_index()

            participant_stats = df_revs_hu.groupby('pr_id')['user'].nunique().reset_index(name='n_participants')

            # B. COMMENTI INLINE (Filtrando l'autore)
            df_v1 = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_review_comments.parquet",
                                    columns=['pull_request_review_id', 'user_type', 'user'])
            df_v2 = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_review_comments_v2.parquet",
                                    columns=['pull_request_review_id', 'user_type', 'user'])
            df_inline_all = pd.concat([df_v1, df_v2], ignore_index=True)
            df_inline_all['pull_request_review_id'] = robust_id(df_inline_all['pull_request_review_id'])

            df_inline_hu = df_inline_all[df_inline_all['user_type'].str.lower().str.strip() == 'user'].copy()
            df_inline_mapped = pd.merge(df_inline_hu, df_revs_hu[['id', 'pr_id']], left_on='pull_request_review_id',
                                        right_on='id', how='inner')

            # Filtro: contiamo solo se chi commenta NON è l'autore della PR
            df_inline_mapped = pd.merge(df_inline_mapped, df_authors, on='pr_id', how='left')
            inline_stats = df_inline_mapped[df_inline_mapped['user'] != df_inline_mapped['pr_author']].groupby(
                'pr_id').size().reset_index(name='n_inline_comments')

            # C. COMMENTI GENERALI (Filtrando l'autore)
            df_gen = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_comments.parquet",
                                     columns=['pr_id', 'user_type', 'user'])
            df_gen['pr_id'] = robust_id(df_gen['pr_id'])
            df_gen_with_author = pd.merge(df_gen, df_authors, on='pr_id', how='left')

            # Filtro: contiamo solo i commenti fatti da altri utenti (i revisori)
            gen_stats = df_gen_with_author[
                (df_gen_with_author['user_type'].str.lower().str.strip() == 'user') &
                (~df_gen_with_author['user'].str.lower().str.contains(bot_pattern, na=False)) &
                (df_gen_with_author['user'] != df_gen_with_author['pr_author'])
                ].groupby('pr_id').size().reset_index(name='n_general_comments')

            # --- MERGE AI ---
            for stats in [gen_stats, inline_stats, review_stats, participant_stats]:
                df = pd.merge(df, stats, left_on='id', right_on='pr_id', how='left').drop(columns=['pr_id'],
                                                                                          errors='ignore')

            # D. COMMIT METRICS
            df_det = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commit_details.parquet",
                                     columns=['pr_id', 'additions', 'deletions', 'filename'])
            df_det['pr_id'] = robust_id(df_det['pr_id'])
            pr_commit_stats = df_det.groupby('pr_id').agg(
                {'additions': 'sum', 'deletions': 'sum', 'filename': 'nunique'}).reset_index().rename(
                columns={'filename': 'changed_files'})
            df = pd.merge(df, pr_commit_stats, left_on='id', right_on='pr_id', how='left').drop(columns=['pr_id'],
                                                                                                errors='ignore')
        except Exception as e:
            print(f" [!] Errore AI: {e}")

    else:
        # --- 👨‍💻 LOGICA HUMAN ---
        # (Qui si assume che i file CSV siano già stati pre-filtrati o contengano i dati corretti)
        if Path(human_metrics_path).exists():
            df_hu_metrics = pd.read_csv(human_metrics_path).astype({'id': str})
            df_hu_metrics['id'] = robust_id(df_hu_metrics['id'])
            df = pd.merge(df, df_hu_metrics[['id', 'n_comments', 'additions', 'deletions', 'changed_files']], on='id',
                          how='left').rename(columns={'n_comments': 'n_general_comments'})

        if Path(human_reviews_path).exists():
            df_hu_revs = pd.read_csv(human_reviews_path).astype({'id': str})
            df_hu_revs['id'] = robust_id(df_hu_revs['id'])
            mask = (df_hu_revs['is_bot'] == False) & (~df_hu_revs['user'].str.lower().str.contains(bot_pattern, na=False))
            hu_review_stats = df_hu_revs[mask].groupby('id').agg(
                n_inline_comments=('tech_comments', 'sum'),
                has_approved_review=('state', lambda x: (x == 'APPROVED').any()),
                has_interaction_review=('state', lambda x: (x.isin(['CHANGES_REQUESTED', 'COMMENTED'])).any()),
                n_participants=('user', 'nunique')
            ).reset_index()
            df = pd.merge(df, hu_review_stats, on='id', how='left')

    # --- 4. PULIZIA FINALE ---
    cols_to_fix = ['n_general_comments', 'n_inline_comments', 'n_participants', 'additions', 'deletions',
                   'changed_files']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    df['has_approved_review'] = df['has_approved_review'].fillna(False).astype(bool)
    df['has_interaction_review'] = df['has_interaction_review'].fillna(False).astype(bool)

    if 'additions' in df.columns:
        df = df[(df['additions'] + df['deletions']) > 0].copy()
    if not is_human_dataset and 'confidence' in df.columns:
        df = df[df['confidence'] >= 9]
    df = df[~df['type'].isin(['other', 'revert'])].copy()

    final_keep = ['id', 'number', 'repo_id', 'agent', 'type', 'additions', 'deletions',
                  'changed_files', 'n_general_comments', 'n_inline_comments',
                  'has_approved_review', 'has_interaction_review', 'n_participants',
                  'created_at', 'merged_at', 'html_url']

    df = df[[c for c in final_keep if c in df.columns]]
    print(f" [+] PR {tag} elaborate: {len(df)}")
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


# --- MAIN ---
if __name__ == "__main__":
    pd.set_option('future.no_silent_downcasting', True)
    paths = {
        "agent_type": "hf://datasets/hao-li/AIDev/pr_task_type.parquet",
        "agent_meta": "hf://datasets/hao-li/AIDev/pull_request.parquet",
        "human_type": "hf://datasets/hao-li/AIDev/human_pr_task_type.parquet",
        "human_meta": "hf://datasets/hao-li/AIDev/human_pull_request.parquet"
    }

    try:
        # Processamento
        df_agents = process_dataset(paths["agent_type"], paths["agent_meta"], is_human_dataset=False)
        df_human = process_dataset(paths["human_type"], paths["human_meta"],
                                   is_human_dataset=True, human_metrics_path=r"reports/metrics/human_pr_commit_details_v2.csv")

        # Unione e salvataggio dataset globale
        all_pr = pd.concat([df_agents, df_human], ignore_index=True)
        os.makedirs("dataset", exist_ok=True)
        all_pr.to_csv("dataset/all_pr_type.csv", index=False)

        # Analisi Distribuzione Totale
        analyze_distributions(all_pr, "all_pr_distribution.png")

        print("\n[SUCCESS] Pipeline completata con successo.")

    except Exception as e:
        print(f"\n[ERROR] Errore critico durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()

