import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. CONFIGURAZIONE ---
REPORTS_DIR = Path("reports/metrics")
BASE_CSV = Path("dataset/all_pr_type.csv")
HUMAN_FILES_CSV = Path("reports/metrics/human_pr_commit_details.csv")
AI_DATASET_PATH = "hf://datasets/hao-li/AIDev/pr_commit_details.parquet"
FIGS_DIR = Path("../figs")

USE_OVERLAP = True
OVERLAP_MODE = 'filename'
MIN_SAMPLE_SIZE = 10
TOLERANCE_HOURS = 72

# --- 2. MAPPATURA FILE ---
def build_unified_file_map(ai_path, human_path, mode='filename'):
    file_map = {}
    try:
        df_ai = pd.read_parquet(ai_path, columns=['pr_id', 'filename'])
        ai_groups = df_ai.groupby('pr_id')['filename'].apply(set).to_dict()
        file_map.update({str(k): v for k, v in ai_groups.items()})
    except Exception: pass

    if human_path.exists():
        df_hu = pd.read_csv(human_path)
        if 'file_list_api' in df_hu.columns:
            hu_data = df_hu.set_index('id')['file_list_api'].dropna().to_dict()
            for pr_id, file_string in hu_data.items():
                if file_string == "error": continue
                file_map[str(pr_id)] = set(str(file_string).split('|'))

    if mode == 'directory':
        for pr_id in file_map:
            file_map[pr_id] = {os.path.dirname(f) for f in file_map[pr_id] if f}
            if not file_map[pr_id]: file_map[pr_id] = {"root"}
    return file_map

def get_failed_pr_ids(df, join_keys, file_map, tolerance_hours=72):
    # Forza consistenza tipi per join_keys
    df_c = df.copy()
    for k in join_keys: df_c[k] = df_c[k].astype(str)

    tasks = df_c[df_c['merged_at'].notna()].copy().sort_values('merged_at')
    fixes = df_c[df_c['type'] == 'fix'].copy().sort_values('created_at')

    if tasks.empty or fixes.empty: return []

    check = pd.merge_asof(
        tasks, fixes, left_on='merged_at', right_on='created_at',
        by=join_keys, direction='forward',
        tolerance=pd.Timedelta(hours=tolerance_hours), suffixes=('_t', '_f')
    )

    candidates = check[check['id_f'].notna()].copy()
    if candidates.empty: return []

    def verify_overlap(row):
        set_t = file_map.get(str(row['id_t']), set())
        set_f = file_map.get(str(row['id_f']), set())
        return not set_t.isdisjoint(set_f)

    candidates['has_overlap'] = candidates.apply(verify_overlap, axis=1)
    return candidates[candidates['has_overlap'] == True]['id_t'].unique()

# --- 3. CORE ANALYTICS ---
def finalize_report(df, grouping_cols, global_stats=None, return_raw=False):
    """
    Esegue l'aggregazione e il calcolo delle metriche di qualità.
    Se return_raw=True, restituisce i valori di instabilità non normalizzati.
    """
    # Aggregazione iniziale
    res = df.groupby(grouping_cols).agg({
        'merged_at': lambda x: x.notna().mean(),
        'time_hrs': ['median', 'mean'],
        'SFI_pr': 'median',
        'ACE_pr': 'median',
        'ASI_pr': 'median',
        'PCD_pr': 'median',
        'filename': ['median', 'mean'],
        'failed_scr': 'mean',
        'id': 'count',
        'n_comments': 'mean',
        'additions': 'mean',
        'deletions': 'mean'
    }).reset_index()

    # Appiattimento colonne
    res.columns = grouping_cols + ['acc_rate', 't_med', 't_mean', 'SFI', 'ACE', 'ASI', 'PCD', 'f_med', 'f_avg', 'SCR', 'sample', 'avg_comments', 'avg_add', 'avg_del']

    # 1. Calcolo instabilità logaritmica (valori grezzi)
    # Queste colonne servono per calcolare la media/std globale
    res['i_struct_raw'] = np.log1p(res['f_avg'] / (res['f_med'] + 0.001))
    res['i_proc_raw'] = np.log1p(res['t_mean'] / (res['t_med'] + 0.001))

    # Se stiamo solo estraendo i riferimenti globali, ci fermiamo qui
    if return_raw:
        return res

    # 2. Normalizzazione (Z-Score)
    for c in ['i_struct', 'i_proc']:
        raw_col = f"{c}_raw"
        if global_stats and f"{c}_mean" in global_stats:
            # ANCORAGGIO GLOBALE: usa parametri passati dall'esterno
            mu = global_stats[f"{c}_mean"]
            sigma = global_stats[f"{c}_std"]
            res[c] = (res[raw_col] - mu) / (sigma + 1e-6)
        else:
            # NORMALIZZAZIONE LOCALE: usa i dati del gruppo corrente
            res[c] = (res[raw_col] - res[raw_col].mean()) / (res[raw_col].std() + 1e-6)

    # 3. Metriche Avanzate e Sintetiche
    res['CDI'] = np.sqrt(np.square(res['i_struct']) + np.square(res['i_proc']))
    res['GRS'] = 1 / (1 + res['SCR'] * 2.5 + res['CDI'] * 0.4)
    res['ASR'] = (res['acc_rate'] * (1 - res['SCR'])) / (np.log1p(res['t_med']) + 1)

    # 4. Rating System
    def get_rating(row):
        if row['SCR'] > 0.18 or row['CDI'] > 2.8: return 'E-CRITICAL'
        if row['GRS'] > 0.75 and row['SCR'] < 0.07 and row['CDI'] < 1.1: return 'A-EXCELLENT'
        if row['GRS'] > 0.60 and row['SCR'] < 0.12 and row['CDI'] < 1.6: return 'B-GOOD'
        if row['GRS'] > 0.45 and row['CDI'] < 2.2: return 'C-AVERAGE'
        return 'D-RISKY'

    res['Rating'] = res.apply(get_rating, axis=1)

    final_cols = grouping_cols + ['Rating', 'GRS', 'CDI', 'SCR', 'ASR', 'SFI', 'ACE', 'ASI', 'PCD', 'sample', 'acc_rate', 't_med', 'avg_comments', 'f_avg', 'avg_add', 'avg_del']
    return res[final_cols].sort_values(by='GRS', ascending=False).round(2)

# --- 4. VISUALIZZAZIONE ---
def save_comparison_plot(df, path):
    # Ordinamento per il grafico
    df_plot = df.sort_values('GRS', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor('white')
    ax.set_facecolor('white')

    # Palette: Grigio scuro per l'umano, Grigio neutro per le AI
    colors = ['#455a64' if 'HUMAN' in str(a).upper() else '#cfd8dc' for a in df_plot['agent']]

    bars = ax.barh(df_plot['agent'], df_plot['GRS'], color=colors, height=0.5)

    # Label dei valori GRS a fine barra
    for bar, grs_val in zip(bars, df_plot['GRS']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{grs_val:.2f}', va='center', color='#607d8b', fontsize=9)

    # Linea Baseline Umana (sottile)
    h_mask = df_plot['agent'].str.contains('human', case=False)
    if h_mask.any():
        h_val = df_plot[h_mask]['GRS'].iloc[0]
        plt.axvline(x=h_val, color='#455a64', linestyle='-', linewidth=0.8, alpha=0.2)

    # Titolo e pulizia estetica
    plt.title('Agent Robustness Index (GRS)', loc='left', fontsize=12, color='#263238', pad=20)

    # Rimuove tutti i bordi (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Rimuove i tick (trattini) e imposta colori neutri
    ax.tick_params(axis='both', which='both', length=0, labelsize=9, colors='#546e7a')

    # Griglia solo verticale chiarissima
    ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='#eceff1', zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

# --- 5. ESECUZIONE ---
def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BASE_CSV)
    df['id'] = df['id'].astype(str)

    # Pre-processing
    df['n_comments'] = df.get('n_comments', 0).fillna(0)
    df['additions'] = df.get('additions', 0).fillna(0)
    df['deletions'] = df.get('deletions', 0).fillna(0)
    df['merged_at'] = pd.to_datetime(df['merged_at']).dt.tz_localize(None)
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)
    df['time_hrs'] = (df['merged_at'] - df['created_at']).dt.total_seconds() / 3600
    df = df.rename(columns={'changed_files': 'filename'})

    # 1. SCR
    print("⚖️ Calcolo SCR...")
    file_map = build_unified_file_map(AI_DATASET_PATH, HUMAN_FILES_CSV, mode=OVERLAP_MODE)
    failed_ids = get_failed_pr_ids(df, ['repo_id'], file_map)
    df['failed_scr'] = df['id'].isin([str(i) for i in failed_ids])

    # 2. Metriche Individuali
    f_safe = df['filename'].replace(0, 1)
    df['ASI_pr'] = df['dir_count'] / f_safe
    df['ACE_pr'] = df['deletions'] / (df['additions'] + 1)
    df['PCD_pr'] = (df['additions'] + df['deletions']) / f_safe
    df['SFI_pr'] = df.get('n_comments', 0) / f_safe

    # --- 3. CALCOLO RIFERIMENTI GLOBALI ---
    # Chiamiamo finalize_report senza global_stats per ottenere i valori medi del dataset
    full_view = finalize_report(df, ['agent', 'type'], return_raw=True)

    stats_rif = {
        'i_struct_mean': full_view['i_struct_raw'].mean(),
        'i_struct_std': full_view['i_struct_raw'].std(),
        'i_proc_mean': full_view['i_proc_raw'].mean(),
        'i_proc_std': full_view['i_proc_raw'].std()
    }

    # 3. Report Agenti con Baseline Umana fissa in cima
    print("📊 Generazione Report...")
    agent_report = finalize_report(df, ['agent'], global_stats=stats_rif)

    # Identificazione e rinomina Human
    is_human = agent_report['agent'].str.contains('human', case=False)
    if is_human.any():
        human_row = agent_report[is_human].copy()
        human_row['agent'] = 'HUMAN_REFERENCE'
        other_agents = agent_report[~is_human & (agent_report['sample'] >= MIN_SAMPLE_SIZE)]
        # Unione: Baseline sempre prima, poi gli altri ordinati per GRS
        final_agent_report = pd.concat([human_row, other_agents], ignore_index=True)
    else:
        final_agent_report = agent_report[agent_report['sample'] >= MIN_SAMPLE_SIZE]

    final_agent_report.to_csv(REPORTS_DIR / "REPORT_AGENTS.csv", index=False)
    save_comparison_plot(final_agent_report, FIGS_DIR / "GRS_COMPARISON.png")

    # 4. Global & Appendix
    full_report = finalize_report(df, ['type', 'agent'], global_stats=stats_rif)
    full_report[full_report['sample'] >= MIN_SAMPLE_SIZE].to_csv(REPORTS_DIR / "REPORT_GLOBAL.csv", index=False)
    full_report[full_report['sample'] < MIN_SAMPLE_SIZE].to_csv(REPORTS_DIR / "REPORT_APPENDIX.csv", index=False)

    # Report Meso: Solo Umani (per tipo)
    report_type_human = finalize_report(df[df['agent'] == 'Human'], ['type'], global_stats=stats_rif)
    report_type_human.to_csv(REPORTS_DIR / "REPORT_TYPE_HUMAN.csv", index=False)

    # Report Meso: Solo AI Aggregati (per tipo)
    report_type_ai = finalize_report(df[df['agent'] != 'Human'], ['type'], global_stats=stats_rif)
    report_type_ai.to_csv(REPORTS_DIR / "REPORT_TYPE_AI.csv", index=False)

    print(f"✅ Completato. Fallimenti SCR: {df['failed_scr'].sum()}. Grafico salvato.")

if __name__ == "__main__":
    main()