"""
クラスタリング結果の可視化と分析
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import seaborn as sns

# 日本語フォントの設定
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_results(results_path: str):
    """
    クラスタリング結果を読み込む
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def visualize_cluster_distribution(results: dict, output_path: str):
    """
    クラスター別の観光スポットとトリップの分布を可視化
    """
    n_clusters = len([k for k in results['spots'].keys()])

    cluster_ids = []
    spot_counts = []
    visitor_counts = []

    for cluster_id in range(n_clusters):
        spots = results['spots'].get(str(cluster_id), [])
        visitors = results['visitors'].get(str(cluster_id), [])

        cluster_ids.append(f"クラスター {cluster_id + 1}")
        spot_counts.append(len(spots))
        visitor_counts.append(len(visitors))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 観光スポット数
    ax1.bar(cluster_ids, spot_counts, color='steelblue', alpha=0.7)
    ax1.set_xlabel('クラスター')
    ax1.set_ylabel('観光スポット数')
    ax1.set_title('クラスター別の観光スポット数')
    ax1.grid(axis='y', alpha=0.3)

    # トリップ数
    ax2.bar(cluster_ids, visitor_counts, color='coral', alpha=0.7)
    ax2.set_xlabel('クラスター')
    ax2.set_ylabel('トリップ数')
    ax2.set_title('クラスター別のトリップ数')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"クラスター分布を {output_path} に保存しました。")
    plt.close()


def visualize_category_distribution(results: dict, output_path: str):
    """
    各クラスターのカテゴリ分布を可視化
    """
    n_clusters = len([k for k in results['spots'].keys()])

    # 全カテゴリを収集
    all_categories = set()
    for cluster_id in range(n_clusters):
        spots = results['spots'].get(str(cluster_id), [])
        for spot in spots:
            all_categories.add(spot['category'])

    all_categories = sorted(list(all_categories))

    # 各クラスターのカテゴリ分布を計算
    category_matrix = np.zeros((n_clusters, len(all_categories)))

    for cluster_id in range(n_clusters):
        spots = results['spots'].get(str(cluster_id), [])
        category_counts = Counter([spot['category'] for spot in spots])

        for i, category in enumerate(all_categories):
            category_matrix[cluster_id, i] = category_counts.get(category, 0)

    # ヒートマップの作成
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(category_matrix,
                xticklabels=all_categories,
                yticklabels=[f"クラスター {i+1}" for i in range(n_clusters)],
                annot=True,
                fmt='.0f',
                cmap='YlOrRd',
                cbar_kws={'label': 'スポット数'},
                ax=ax)

    ax.set_xlabel('カテゴリ')
    ax.set_ylabel('クラスター')
    ax.set_title('クラスター別のカテゴリ分布')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"カテゴリ分布を {output_path} に保存しました。")
    plt.close()


def extract_secret_spots(results: dict, cluster_id: int):
    """
    個人的穴場を抽出（論文の第5章）

    Parameters:
    -----------
    results : dict
        クラスタリング結果
    cluster_id : int
        対象のクラスターID

    Returns:
    --------
    dict
        穴場スポットの情報
    """
    # 対象クラスターのスポット
    target_spots = results['spots'].get(str(cluster_id), [])
    target_visitors = set(results['visitors'].get(str(cluster_id), []))

    # 全体のスポットと訪問者数を計算
    all_spot_visitor_counts = {}  # {(lat, lon): 訪問者数}
    cluster_spot_visitor_counts = {}  # {(lat, lon): クラスター内訪問者数}

    n_clusters = len([k for k in results['spots'].keys()])

    for cid in range(n_clusters):
        cluster_visitors = set(results['visitors'].get(str(cid), []))
        spots = results['spots'].get(str(cid), [])

        for spot in spots:
            spot_key = (spot['lat'], spot['lon'])

            # 全体のカウント
            if spot_key not in all_spot_visitor_counts:
                all_spot_visitor_counts[spot_key] = 0
            all_spot_visitor_counts[spot_key] += len(cluster_visitors)

            # クラスター内のカウント
            if cid == cluster_id:
                if spot_key not in cluster_spot_visitor_counts:
                    cluster_spot_visitor_counts[spot_key] = 0
                cluster_spot_visitor_counts[spot_key] += len(cluster_visitors)

    # 平均値の計算
    all_counts = list(all_spot_visitor_counts.values())
    cluster_counts = list(cluster_spot_visitor_counts.values())

    avg_all = np.mean(all_counts) if all_counts else 0
    avg_cluster = np.mean(cluster_counts) if cluster_counts else 0

    # 穴場の抽出
    exclusive_spots = []  # 独占的な穴場（右下）
    discovery_spots = []  # 発見的な穴場（左上）

    for spot in target_spots:
        spot_key = (spot['lat'], spot['lon'])
        x = cluster_spot_visitor_counts.get(spot_key, 0)
        y = all_spot_visitor_counts.get(spot_key, 0)

        spot_info = {
            **spot,
            'cluster_visitors': x,
            'all_visitors': y
        }

        # 独占的な穴場: クラスター内で人気、全体では不人気
        if x > avg_cluster and y < avg_all:
            exclusive_spots.append(spot_info)

        # 発見的な穴場: クラスター内で不人気、全体では人気
        if x < avg_cluster and y > avg_all:
            discovery_spots.append(spot_info)

    return {
        'exclusive_spots': exclusive_spots,
        'discovery_spots': discovery_spots,
        'avg_cluster': avg_cluster,
        'avg_all': avg_all
    }


def visualize_secret_spots(results: dict, cluster_id: int, output_path: str):
    """
    個人的穴場の可視化
    """
    secret_spots_data = extract_secret_spots(results, cluster_id)

    # 散布図の作成
    fig, ax = plt.subplots(figsize=(10, 8))

    # 全スポットのプロット
    all_x = []
    all_y = []

    target_spots = results['spots'].get(str(cluster_id), [])
    target_visitors = set(results['visitors'].get(str(cluster_id), []))

    n_clusters = len([k for k in results['spots'].keys()])

    all_spot_visitor_counts = {}
    cluster_spot_visitor_counts = {}

    for cid in range(n_clusters):
        cluster_visitors = set(results['visitors'].get(str(cid), []))
        spots = results['spots'].get(str(cid), [])

        for spot in spots:
            spot_key = (spot['lat'], spot['lon'])

            if spot_key not in all_spot_visitor_counts:
                all_spot_visitor_counts[spot_key] = 0
            all_spot_visitor_counts[spot_key] += len(cluster_visitors)

            if cid == cluster_id:
                if spot_key not in cluster_spot_visitor_counts:
                    cluster_spot_visitor_counts[spot_key] = 0
                cluster_spot_visitor_counts[spot_key] += len(cluster_visitors)

    for spot in target_spots:
        spot_key = (spot['lat'], spot['lon'])
        x = cluster_spot_visitor_counts.get(spot_key, 0)
        y = all_spot_visitor_counts.get(spot_key, 0)
        all_x.append(x)
        all_y.append(y)

    # 通常のスポット
    ax.scatter(all_x, all_y, alpha=0.5, s=50, color='gray', label='通常のスポット')

    # 独占的な穴場
    exclusive_x = [s['cluster_visitors'] for s in secret_spots_data['exclusive_spots']]
    exclusive_y = [s['all_visitors'] for s in secret_spots_data['exclusive_spots']]
    if exclusive_x:
        ax.scatter(exclusive_x, exclusive_y, alpha=0.8, s=100, color='green',
                  marker='o', label='独占的な穴場', edgecolors='black', linewidths=1.5)

    # 発見的な穴場
    discovery_x = [s['cluster_visitors'] for s in secret_spots_data['discovery_spots']]
    discovery_y = [s['all_visitors'] for s in secret_spots_data['discovery_spots']]
    if discovery_x:
        ax.scatter(discovery_x, discovery_y, alpha=0.8, s=100, color='orange',
                  marker='^', label='発見的な穴場', edgecolors='black', linewidths=1.5)

    # 平均線
    avg_cluster = secret_spots_data['avg_cluster']
    avg_all = secret_spots_data['avg_all']

    ax.axvline(avg_cluster, color='blue', linestyle='--', alpha=0.5, label=f'クラスター平均 ({avg_cluster:.1f})')
    ax.axhline(avg_all, color='red', linestyle='--', alpha=0.5, label=f'全体平均 ({avg_all:.1f})')

    ax.set_xlabel('クラスター内訪問者数')
    ax.set_ylabel('全体訪問者数')
    ax.set_title(f'クラスター {cluster_id + 1} の個人的穴場')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"個人的穴場の可視化を {output_path} に保存しました。")
    plt.close()

    # 穴場の詳細を表示
    print(f"\nクラスター {cluster_id + 1} の個人的穴場:")
    print("="*80)

    print(f"\n独占的な穴場 ({len(secret_spots_data['exclusive_spots'])}個):")
    for spot in secret_spots_data['exclusive_spots']:
        print(f"  - {spot['category']} (緯度: {spot['lat']}, 経度: {spot['lon']})")
        print(f"    クラスター内訪問者: {spot['cluster_visitors']}, 全体訪問者: {spot['all_visitors']}")

    print(f"\n発見的な穴場 ({len(secret_spots_data['discovery_spots'])}個):")
    for spot in secret_spots_data['discovery_spots']:
        print(f"  - {spot['category']} (緯度: {spot['lat']}, 経度: {spot['lon']})")
        print(f"    クラスター内訪問者: {spot['cluster_visitors']}, 全体訪問者: {spot['all_visitors']}")


def main():
    """
    メイン関数
    """
    import os

    # ディレクトリのパス
    base_dir = "/Users/kantacky/Developer/walking-experiment-2512"
    output_dir = os.path.join(base_dir, "output")

    results_path = os.path.join(output_dir, "clustering_results.json")
    results = load_results(results_path)

    # クラスター分布の可視化
    visualize_cluster_distribution(
        results,
        os.path.join(output_dir, "cluster_distribution.png")
    )

    # カテゴリ分布の可視化
    visualize_category_distribution(
        results,
        os.path.join(output_dir, "category_distribution.png")
    )

    # 個人的穴場の可視化（クラスター3を対象）
    visualize_secret_spots(
        results,
        cluster_id=2,  # クラスター3（インデックスは2）
        output_path=os.path.join(output_dir, "secret_spots_cluster3.png")
    )


if __name__ == "__main__":
    main()
