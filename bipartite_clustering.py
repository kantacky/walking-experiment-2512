"""
モバイル位置情報データに基づく観光スポットと訪問者の同時クラスタリング
論文: 根本ら (2023) の手法を Python で実装
"""

import json
import numpy as np
import pandas as pd
from itertools import combinations, chain
from typing import List, Set, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class BipartiteClusteringModel:
    """
    二部グラフの重複を考慮したクラスタリングモデル
    """

    def __init__(self, n_clusters: int = 4):
        """
        Parameters:
        -----------
        n_clusters : int
            クラスター数
        """
        self.n_clusters = n_clusters
        self.spots = None
        self.visitors = None
        self.edge_matrix = None
        self.spot_clusters = None
        self.visitor_clusters = None

    def load_data(self, data_path: str, use_trips: bool = True):
        """
        データを読み込み、二部グラフを構築

        Parameters:
        -----------
        data_path : str
            JSONデータファイルのパス
        use_trips : bool
            Trueの場合、sessionをトリップとして扱う（G2に相当）
            Falseの場合、user_idを訪問者として扱う（G1に相当）
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 観光スポットを位置情報で識別
        spots_dict = {}  # {(lat, lon): {category, index}}
        visitor_spot_pairs = []

        for session in data['sessions']:
            session_id = session['id']
            user_id = session['user_id']

            # トリップとして扱うか、ユーザーとして扱うか
            visitor_id = session_id if use_trips else user_id

            visited_spots_in_session = set()

            for record in session['records']:
                lat = round(record['latitude'], 4)  # 精度を丸める
                lon = round(record['longitude'], 4)
                category = record['category']

                # 観光スポットを位置情報で識別
                spot_key = (lat, lon)
                if spot_key not in spots_dict:
                    spots_dict[spot_key] = {
                        'category': category,
                        'lat': lat,
                        'lon': lon,
                        'index': len(spots_dict)
                    }

                # 同一session内での重複訪問を避ける
                if spot_key not in visited_spots_in_session:
                    visitor_spot_pairs.append((visitor_id, spot_key))
                    visited_spots_in_session.add(spot_key)

        # インデックスの作成
        self.spots = sorted(list(spots_dict.keys()))
        self.spot_info = {spot: spots_dict[spot] for spot in self.spots}
        self.visitors = sorted(list(set([pair[0] for pair in visitor_spot_pairs])))

        spot_to_idx = {spot: idx for idx, spot in enumerate(self.spots)}
        visitor_to_idx = {visitor: idx for idx, visitor in enumerate(self.visitors)}

        # エッジ行列の構築 (訪問者 x 観光スポット)
        n_visitors = len(self.visitors)
        n_spots = len(self.spots)
        self.edge_matrix = np.zeros((n_visitors, n_spots), dtype=int)

        for visitor, spot in visitor_spot_pairs:
            v_idx = visitor_to_idx[visitor]
            s_idx = spot_to_idx[spot]
            self.edge_matrix[v_idx, s_idx] = 1

        print(f"データ読み込み完了:")
        print(f"  観光スポット数: {n_spots}")
        print(f"  訪問者/トリップ数: {n_visitors}")
        print(f"  エッジ数: {np.sum(self.edge_matrix)}")
        print(f"  エッジ密度: {np.mean(self.edge_matrix):.4f}")

        # 各ノードの次数統計
        spot_degrees = np.sum(self.edge_matrix, axis=0)
        visitor_degrees = np.sum(self.edge_matrix, axis=1)
        print(f"  観光スポットの平均次数: {np.mean(spot_degrees):.2f}")
        print(f"  訪問者の平均次数: {np.mean(visitor_degrees):.2f}")

    def powerset(self, iterable):
        """
        冪集合を生成（空集合を除く）
        """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

    def cluster_similarity(self, p: Set[int], q: Set[int]) -> float:
        """
        クラスター組み合わせの類似度を計算 (式4)

        Parameters:
        -----------
        p, q : Set[int]
            クラスター組み合わせ

        Returns:
        --------
        float
            類似度 g_pq
        """
        if len(p) == 0 or len(q) == 0:
            return 0.0

        intersection = len(p & q)
        denominator = np.sqrt(len(p) * len(q))

        if denominator == 0:
            return 0.0

        return intersection / denominator

    def compute_edge_densities(self, spot_assignment: np.ndarray,
                               visitor_assignment: np.ndarray) -> Tuple[float, float]:
        """
        エッジ密度 f_in と f_out を計算 (式5, 6)

        Parameters:
        -----------
        spot_assignment : np.ndarray
            観光スポットのクラスター割り当て (n_spots,)
        visitor_assignment : np.ndarray
            訪問者のクラスター割り当て (n_visitors,)

        Returns:
        --------
        Tuple[float, float]
            (f_in, f_out)
        """
        n_visitors, n_spots = self.edge_matrix.shape

        # クラスター組み合わせのユニークなパターンを取得
        unique_spot_clusters = np.unique(spot_assignment)
        unique_visitor_clusters = np.unique(visitor_assignment)

        # 各ノードのクラスター組み合わせを集合として表現
        spot_cluster_sets = [frozenset([spot_assignment[i]]) for i in range(n_spots)]
        visitor_cluster_sets = [frozenset([visitor_assignment[j]]) for j in range(n_visitors)]

        # 類似度行列を事前計算
        n_spot_patterns = len(spot_cluster_sets)
        n_visitor_patterns = len(visitor_cluster_sets)

        # f_in と f_out の計算
        numerator_in = 0.0
        denominator_in = 0.0
        numerator_out = 0.0
        denominator_out = 0.0

        for i in range(n_spots):
            for j in range(n_visitors):
                p = spot_cluster_sets[i]
                q = visitor_cluster_sets[j]

                g_pq = self.cluster_similarity(p, q)
                e_ij = self.edge_matrix[j, i]

                # f_in の計算
                numerator_in += e_ij * g_pq
                denominator_in += g_pq

                # f_out の計算
                numerator_out += e_ij * (1 - g_pq)
                denominator_out += (1 - g_pq)

        f_in = numerator_in / denominator_in if denominator_in > 0 else 0.0
        f_out = numerator_out / denominator_out if denominator_out > 0 else 0.0

        return f_in, f_out

    def objective_function(self, spot_assignment: np.ndarray,
                          visitor_assignment: np.ndarray) -> float:
        """
        目的関数を計算 (式7)

        Parameters:
        -----------
        spot_assignment : np.ndarray
            観光スポットのクラスター割り当て
        visitor_assignment : np.ndarray
            訪問者のクラスター割り当て

        Returns:
        --------
        float
            目的関数値 ln(f_in) - ln(f_out)
        """
        f_in, f_out = self.compute_edge_densities(spot_assignment, visitor_assignment)

        # ログの計算（0の場合は小さな値で置き換え）
        eps = 1e-10
        f_in = max(f_in, eps)
        f_out = max(f_out, eps)

        return np.log(f_in) - np.log(f_out)

    def fit(self, max_iter: int = 100, random_state: int = 42):
        """
        クラスタリングを実行

        貪欲アルゴリズムとK-meansベースのアプローチを組み合わせて実装

        Parameters:
        -----------
        max_iter : int
            最大反復回数
        random_state : int
            乱数シード
        """
        np.random.seed(random_state)

        n_visitors, n_spots = self.edge_matrix.shape

        # 初期化: ランダムにクラスターを割り当て
        spot_assignment = np.random.randint(0, self.n_clusters, size=n_spots)
        visitor_assignment = np.random.randint(0, self.n_clusters, size=n_visitors)

        best_objective = self.objective_function(spot_assignment, visitor_assignment)
        best_spot_assignment = spot_assignment.copy()
        best_visitor_assignment = visitor_assignment.copy()

        print(f"\n初期目的関数値: {best_objective:.4f}")

        # 反復最適化
        for iteration in range(max_iter):
            improved = False

            # 観光スポットのクラスター割り当てを更新
            for i in range(n_spots):
                current_cluster = spot_assignment[i]
                best_cluster = current_cluster
                best_obj = best_objective

                # 各クラスターへの割り当てを試す
                for c in range(self.n_clusters):
                    spot_assignment[i] = c
                    obj = self.objective_function(spot_assignment, visitor_assignment)

                    if obj > best_obj:
                        best_obj = obj
                        best_cluster = c
                        improved = True

                spot_assignment[i] = best_cluster
                if improved:
                    best_objective = best_obj

            # 訪問者のクラスター割り当てを更新
            for j in range(n_visitors):
                current_cluster = visitor_assignment[j]
                best_cluster = current_cluster
                best_obj = best_objective

                # 各クラスターへの割り当てを試す
                for c in range(self.n_clusters):
                    visitor_assignment[j] = c
                    obj = self.objective_function(spot_assignment, visitor_assignment)

                    if obj > best_obj:
                        best_obj = obj
                        best_cluster = c
                        improved = True

                visitor_assignment[j] = best_cluster
                if improved:
                    best_objective = best_obj

            if iteration % 10 == 0:
                f_in, f_out = self.compute_edge_densities(spot_assignment, visitor_assignment)
                print(f"Iteration {iteration}: Objective = {best_objective:.4f}, "
                      f"f_in = {f_in:.4f}, f_out = {f_out:.4f}")

            # 改善がなければ終了
            if not improved:
                print(f"収束しました (Iteration {iteration})")
                break

        self.spot_clusters = spot_assignment
        self.visitor_clusters = visitor_assignment

        # 最終結果の表示
        f_in, f_out = self.compute_edge_densities(spot_assignment, visitor_assignment)
        print(f"\n最終結果:")
        print(f"  目的関数値: {best_objective:.4f}")
        print(f"  f_in (同一クラスター内): {f_in:.4f}")
        print(f"  f_out (異なるクラスター間): {f_out:.4f}")

        return self

    def get_results(self) -> Dict:
        """
        クラスタリング結果を取得

        Returns:
        --------
        Dict
            クラスタリング結果の辞書
        """
        results = {
            'spots': {},
            'visitors': {},
            'spot_details': {}
        }

        # 観光スポットのクラスター
        for i, spot in enumerate(self.spots):
            cluster = int(self.spot_clusters[i])
            if cluster not in results['spots']:
                results['spots'][cluster] = []

            spot_info = self.spot_info.get(spot, {})
            spot_data = {
                'lat': spot_info.get('lat'),
                'lon': spot_info.get('lon'),
                'category': spot_info.get('category')
            }
            results['spots'][cluster].append(spot_data)

        # 訪問者のクラスター
        for j, visitor in enumerate(self.visitors):
            cluster = int(self.visitor_clusters[j])
            if cluster not in results['visitors']:
                results['visitors'][cluster] = []
            results['visitors'][cluster].append(visitor)

        return results

    def print_results(self):
        """
        クラスタリング結果を表示
        """
        results = self.get_results()

        print("\n" + "="*80)
        print("クラスタリング結果")
        print("="*80)

        for cluster_id in range(self.n_clusters):
            print(f"\nクラスター {cluster_id + 1}:")
            print("-" * 80)

            # 観光スポット
            spots = results['spots'].get(cluster_id, [])
            print(f"  観光スポット ({len(spots)}個):")

            # カテゴリごとに集計
            category_counts = {}
            for spot_data in spots:
                category = spot_data['category']
                category_counts[category] = category_counts.get(category, 0) + 1

            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {category}: {count}個")

            # 訪問者
            visitors = results['visitors'].get(cluster_id, [])
            print(f"\n  訪問者/トリップ ({len(visitors)}個):")
            for visitor in visitors[:10]:  # 最初の10個のみ表示
                print(f"    - {visitor}")
            if len(visitors) > 10:
                print(f"    ... 他 {len(visitors) - 10} 個")


def main():
    """
    メイン関数
    """
    import os

    # 出力ディレクトリの作成
    output_dir = "/Users/kantacky/Developer/walking-experiment-2512/output"
    os.makedirs(output_dir, exist_ok=True)

    # データファイルのパス
    data_path = "/Users/kantacky/Developer/walking-experiment-2512/data/20251231T042013Z.json"

    # モデルの初期化
    model = BipartiteClusteringModel(n_clusters=4)

    # データの読み込み
    model.load_data(data_path)

    # クラスタリングの実行
    model.fit(max_iter=50, random_state=42)

    # 結果の表示
    model.print_results()

    # 結果をJSONファイルに保存
    results = model.get_results()
    output_path = os.path.join(output_dir, "clustering_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n結果を {output_path} に保存しました。")


if __name__ == "__main__":
    main()
