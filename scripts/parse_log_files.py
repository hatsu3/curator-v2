import json
import re
from pathlib import Path

import fire


def extract_dict_from_line(line):
    line = re.sub(r": nan,", ': "nan",', line)
    line = re.sub(r": inf,", ': "inf",', line)
    ansi_escape_pattern = re.compile(r"\x1b\[([0-9]+)(;[0-9]+)*m")
    line = ansi_escape_pattern.sub("", line)
    dict_str = line.split("Aggregated results: ")[1].strip()
    dict_data = json.loads(dict_str.replace("'", '"'))
    return dict_data


def process_log_files(log_dir: str):
    results = []
    for file_path in Path(log_dir).glob("**/*.log"):
        fname = file_path.name.replace(".log", "")
        index_type, dataset_name = fname.split("_on_")

        with open(file_path, "r") as file:
            for line in file:
                if "Aggregated results: " in line:
                    dict_data = extract_dict_from_line(line)
                    if dict_data:
                        results.append(
                            {
                                "index_type": index_type,
                                "dataset_name": dataset_name,
                                **dict_data,
                            }
                        )

    return results


def collect_overall_results(log_dir: str, output_path: str = "aggregated_results.json"):
    results = process_log_files(log_dir)

    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)


def process_param_sweep_log_files(index_type: str, log_dir: str):
    results = []
    for file_path in Path(log_dir).glob("**/*.log"):
        fname = file_path.name.replace(".log", "")
        params = dict([param.split("=") for param in fname.split("-")])

        with file_path.open("r") as file:
            for line in file:
                if "Aggregated results: " in line:
                    dict_data = extract_dict_from_line(line)
                    if dict_data:
                        results.append(
                            {
                                "index_type": index_type,
                                **params,
                                **dict_data,
                            }
                        )

    return results


def collect_param_sweep_results(log_dir: str, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = list()
    for subdir in Path(log_dir).iterdir():
        index_type = subdir.name
        results.extend(process_param_sweep_log_files(index_type, str(subdir)))

    with output_path.open("w") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    fire.Fire()
