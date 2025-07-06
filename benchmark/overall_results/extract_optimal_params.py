import json
from pathlib import Path

import pandas as pd


def extract_optimal_params():
    """Extract optimal parameters for each baseline on each dataset"""

    baseline_info = [
        {
            "name": "Per-Label HNSW",
            "result_path": {
                "yfcc100m": None,  # too large
                "arxiv": "output/overall_results_v2/per_label_hnsw/arxiv-large-10_test0.005/results/ef128_m16.csv",
            },
            "construction_params": ["construction_ef", "m"],
            "search_params": ["search_ef"],
        },
        {
            "name": "Per-Label IVF",
            "result_path": {
                "yfcc100m": None,  # too large
                "arxiv": "output/overall_results_v2/per_label_ivf/arxiv-large-10_test0.005/results/nlist40.csv",
            },
            "construction_params": ["nlist"],
            "search_params": ["nprobe"],
        },
        {
            "name": "Parlay IVF",
            "result_path": {
                "yfcc100m": "output/overall_results_v2/parlay_ivf/yfcc100m-10m_test0.001/results/cutoff1000c40r16i10.csv",
                "arxiv": "output/overall_results_v2/parlay_ivf/arxiv-large-10_test0.005/results/cutoff10000c1000r16i10.csv",
            },
            "construction_params": [
                "cutoff",
                "ivf_cluster_size",
                "graph_degree",
                "ivf_max_iter",
            ],
            "search_params": ["ivf_search_radius", "graph_search_L"],
        },
        {
            "name": "Filtered DiskANN",
            "result_path": {
                "yfcc100m": "output/overall_results_v2/filtered_diskann/yfcc100m-10m_test0.001/results/R512_L600_a1.2.csv",
                "arxiv": "output/overall_results_v2/filtered_diskann/arxiv-large-10_test0.005/results/r256ef256a1.2.csv",
            },
            "construction_params": ["graph_degree", "ef_construct", "alpha"],
            "search_params": ["ef_search"],
            # Handle both old and new parameter naming conventions
            "param_mapping": {
                "max_degree": "graph_degree",
                "Lbuild": "ef_construct",
                "Lsearch": "ef_search",
            },
        },
        {
            "name": "Shared HNSW",
            "result_path": {
                "yfcc100m": "output/overall_results_v2/shared_hnsw/yfcc100m-10m_test0.001/results/ef128_m32.csv",
                "arxiv": "output/overall_results_v2/shared_hnsw/arxiv-large-10_test0.005/results/ef128_m16.csv",
            },
            "construction_params": ["construction_ef", "m"],
            "search_params": ["search_ef"],
        },
        {
            "name": "Shared IVF",
            "result_path": {
                "yfcc100m": "output/overall_results_v2/shared_ivf/yfcc100m-10m_test0.001/results/nlist32768.csv",
                "arxiv": "output/overall_results_v2/shared_ivf/arxiv-large-10_test0.005/results/nlist1600_nprobe16.csv",
            },
            "construction_params": ["nlist"],
            "search_params": ["nprobe"],
        },
        {
            "name": "ACORN-1",
            "result_path": {
                "yfcc100m": "output/overall_results_v2/acorn/yfcc100m-10m_test0.001/results/m64_g1_b128.csv",
                "arxiv": "output/overall_results_v2/acorn/arxiv-large-10_test0.005/results/m64_g1_b128.csv",
            },
            "construction_params": ["m", "gamma", "m_beta"],
            "search_params": ["search_ef"],
        },
        {
            "name": "ACORN-gamma",
            "result_path": {
                "yfcc100m": "output/overall_results_v2/acorn/yfcc100m-10m_test0.001/results/m64_g20_b128.csv",
                "arxiv": "output/overall_results_v2/acorn/arxiv-large-10_test0.005/results/m32_gamma20_m_beta64.csv",
            },
            "construction_params": ["m", "gamma", "m_beta"],
            "search_params": ["search_ef"],
        },
        {
            "name": "Curator",
            "result_path": {
                "yfcc100m": "output/overall_results_v2/curator/yfcc100m-10m_test0.001/results/nlist32_sl256.csv",
                "arxiv": "output/overall_results_v2/curator/arxiv-large-10_test0.005/results/nlist32_sl256.csv",
            },
            "construction_params": ["nlist", "max_sl_size"],
            "search_params": ["search_ef", "beam_size", "variance_boost"],
        },
    ]

    optimal_params = {}

    for baseline in baseline_info:
        baseline_name = baseline["name"]
        optimal_params[baseline_name] = {}

        for dataset in ["yfcc100m", "arxiv"]:
            csv_path = baseline["result_path"][dataset]
            if csv_path is None:
                print(f"Skipping {baseline_name} on {dataset} - no result file")
                continue

            csv_path = Path(__file__).parent.parent.parent / csv_path
            if not csv_path.exists():
                print(
                    f"Warning: {csv_path} does not exist, skipping {baseline_name} on {dataset}"
                )
                continue

            try:
                # Read the CSV file
                df = pd.read_csv(csv_path)
                print(f"Processing {baseline_name} on {dataset} - {len(df)} rows")

                # Handle parameter mapping for different naming conventions
                param_mapping = baseline.get("param_mapping", {})

                # Extract construction parameters (should be constant across rows)
                construction_params = {}
                for param in baseline["construction_params"]:
                    # Check if parameter exists directly or through mapping
                    if param in df.columns:
                        construction_params[param] = df[param].iloc[0]
                    else:
                        # Check if there's an old name that maps to this parameter
                        old_param = None
                        for old_name, new_name in param_mapping.items():
                            if new_name == param and old_name in df.columns:
                                old_param = old_name
                                break
                        if old_param:
                            construction_params[param] = df[old_param].iloc[0]

                # Convert numpy types to native Python types for JSON serialization
                for param, value in construction_params.items():
                    if hasattr(value, 'item'):  # Check if it's a numpy type
                        construction_params[param] = value.item()

                # Extract all search parameter combinations with their performance
                search_param_combinations = []
                for _, row in df.iterrows():
                    search_params = {}
                    for param in baseline["search_params"]:
                        # Check if parameter exists directly or through mapping
                        if param in df.columns:
                            search_params[param] = row[param]
                        else:
                            # Check if there's an old name that maps to this parameter
                            old_param = None
                            for old_name, new_name in param_mapping.items():
                                if new_name == param and old_name in df.columns:
                                    old_param = old_name
                                    break
                            if old_param:
                                search_params[param] = row[old_param]

                    # Convert numpy types to native Python types for JSON serialization
                    for param, value in search_params.items():
                        if hasattr(value, 'item'):  # Check if it's a numpy type
                            search_params[param] = value.item()

                    search_param_combinations.append({"search_params": search_params})

                # Store results
                optimal_params[baseline_name][dataset] = {
                    "construction_params": construction_params,
                    "search_param_combinations": search_param_combinations,
                }

                print(f"  Construction params: {construction_params}")
                print(
                    f"  Found {len(search_param_combinations)} search parameter combinations"
                )

            except Exception as e:
                print(f"Error processing {baseline_name} on {dataset}: {e}")
                continue

    # Save to JSON file
    output_path = "benchmark/overall_results/optimal_baseline_params.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(optimal_params, f, indent=2)

    print(f"\nOptimal parameters saved to {output_path}")
    return optimal_params


if __name__ == "__main__":
    extract_optimal_params()
