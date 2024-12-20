import csv
import glob
import os

import pandas as pd

LOG_DIR = "/home/josh/octo_digit/eval/eval_logs"


def read_logs_to_csv():
    model_names = {
        "josh_pod_final_combined_model_tvl_single_rephrase_full_combinations_b1024_20240910_220927": "Final model",
        "josh_pod_final_finetune_no_sensor_b1024_20240911_034103": "Finetuned, no sensors",
        "josh_pod_final_scratch_no_sensor_b1024_20240911_050827": "From scratch, all sensors",
        "josh_pod_final_bcz_b1024_20240911_185628": "Resnet",
    }

    def get_result_from_log_file(log_file_path):
        with open(log_file_path, "r") as file:
            last_log = [line for line in file][-1]
        result = last_log[last_log.find("result:") + len("result:") :]
        result = result[: result.find("video path")].strip()
        if result in {"success", "find", "failure"}:
            return result
        raise ValueError(f"Result was [{result}], in file {log_file_path}")

    def extract_result_dict(model_name, exp_name):
        log_path = os.path.join(LOG_DIR, model_name, "50000", exp_name)

        leaf_logs = [
            glob.glob(os.path.join(log_path, "*", str(i), "log.txt"))[0]
            for i in range(5)
        ]
        results = {"success": 0, "find": 0, "failure": 0}
        for leaf_log_path in leaf_logs:
            assert os.path.exists(leaf_log_path), leaf_log_path
            results[get_result_from_log_file(leaf_log_path)] += 1
        return results

    # column_names = [''] + list(model_names.values())
    # row_labels = [''] + ['bag']
    # df = pd.DataFrame()
    # exp_names = [f'bag_{i}' for i in range(10)] + [f'flat_{i}' for i in range(10)]
    # for model_name in model_names:
    #     for exp_name in exp_names:
    #         res = extract_result_dict(model_name, exp_name)
    # read into row representation

    def add_sep(rows, num_sep=3):
        for _ in range(num_sep):
            sep_row = ["" for _ in rows[-1]]
            rows.append(sep_row)
        return rows

    rows = []
    for j, exp_name in enumerate([f"bag_{i}" for i in range(5)]):  # bag test
        row_res = [f"bag test objects - {j+1}"]
        for model_name in model_names:
            res = extract_result_dict(model_name, exp_name)
            row_res.extend([res["success"], res["find"], res["success"] + res["find"]])
        rows.append(row_res)
    rows = add_sep(rows)

    for j, exp_name in enumerate([f"bag_{i}" for i in range(5, 10)]):  # bag train
        row_res = [f"bag train objects - {j+1}"]
        for model_name in model_names:
            res = extract_result_dict(model_name, exp_name)
            row_res.extend([res["success"], res["find"], res["success"] + res["find"]])
        rows.append(row_res)
    rows = add_sep(rows)

    for j, exp_name in enumerate([f"flat_{i}" for i in range(5)]):  # flattest
        row_res = [f"platter test objects - {j+1}"]
        for model_name in model_names:
            res = extract_result_dict(model_name, exp_name)
            row_res.extend([res["success"], res["find"], res["success"] + res["find"]])
        rows.append(row_res)
    rows = add_sep(rows)

    for j, exp_name in enumerate([f"flat_{i}" for i in range(5, 10)]):  # flat train
        row_res = [f"platter train objects - {j+1}"]
        for model_name in model_names:
            res = extract_result_dict(model_name, exp_name)
            row_res.extend([res["success"], res["find"], res["success"] + res["find"]])
        rows.append(row_res)

    def row_to_str(row):
        return [str(val) for val in row]

    csv_txt = "\n".join([",".join(row_to_str(row)) for row in rows])
    with open("./log_csv.csv", "w") as file:
        file.write(csv_txt)
    # print(csv_txt)
    # for row in rows:
    #     line = ','.join()


def read_multi_logs_to_csv():
    model_names = {
        "josh_pod_final_combined_model_tvl_single_rephrase_full_combinations_b1024_20240910_220927": "Final model",
        "josh_pod_final_finetune_no_sensor_b1024_20240911_034103": "Finetuned, no sensors",
    }

    def get_result_from_log_file(log_file_path):
        with open(log_file_path, "r") as file:
            last_log = [line for line in file][-1]
        result = last_log[last_log.find("result:") + len("result:") :]
        result = result[: result.find("video path")].strip()
        if result in {"success", "find", "failure"}:
            return result
        if result in {"correctcolorwrongtac", "wrongcolorcorrecttac"}:
            return "failure"
        raise ValueError(f"Result was [{result}], in file {log_file_path}")

    def extract_result_dict(model_name, exp_name, modality):
        log_path = os.path.join(LOG_DIR, model_name, "50000", exp_name)

        leaf_logs = [
            os.path.join(log_path, modality, str(i), "log.txt") for i in range(5)
        ]
        results = {"success": 0, "find": 0, "failure": 0}
        for leaf_log_path in leaf_logs:
            if not os.path.exists(leaf_log_path):
                print(leaf_log_path)
                continue
            # assert os.path.exists(leaf_log_path), leaf_log_path
            results[get_result_from_log_file(leaf_log_path)] += 1
        return results

    def add_sep(rows, num_sep=3):
        for _ in range(num_sep):
            sep_row = ["" for _ in rows[-1]]
            rows.append(sep_row)
        return rows

    MODALITIES = ["visual", "tactile", "visual,tactile"]
    rows = []
    for j, exp_name in enumerate([f"bag_{i}" for i in range(5)]):  # bag test
        row_res = [f"bag test objects - {j+1}"]
        for model_name in model_names:
            for modality in MODALITIES:
                res = extract_result_dict(model_name, exp_name, modality)
                row_res.extend(
                    [res["success"], res["find"], res["success"] + res["find"]]
                )
        rows.append(row_res)

    rows = add_sep(rows)

    for j, exp_name in enumerate([f"bag_{i}" for i in range(5, 10)]):  # bag train
        row_res = [f"bag train objects - {j+1}"]
        for model_name in model_names:
            for modality in MODALITIES:
                res = extract_result_dict(model_name, exp_name, modality)
                row_res.extend(
                    [res["success"], res["find"], res["success"] + res["find"]]
                )
        rows.append(row_res)
    rows = add_sep(rows)

    for j, exp_name in enumerate([f"flat_{i}" for i in range(5)]):  # flattest
        row_res = [f"platter test objects - {j+1}"]
        for model_name in model_names:
            for modality in MODALITIES:
                res = extract_result_dict(model_name, exp_name, modality)
                row_res.extend(
                    [res["success"], res["find"], res["success"] + res["find"]]
                )
        rows.append(row_res)
    rows = add_sep(rows)

    for j, exp_name in enumerate([f"flat_{i}" for i in range(5, 10)]):  # flat train
        row_res = [f"platter train objects - {j+1}"]
        for model_name in model_names:
            for modality in MODALITIES:
                res = extract_result_dict(model_name, exp_name, modality)
                row_res.extend(
                    [res["success"], res["find"], res["success"] + res["find"]]
                )
        rows.append(row_res)

    def row_to_str(row):
        return [str(val) for val in row]

    csv_txt = "\n".join([",".join(row_to_str(row)) for row in rows])
    with open("./multi_log_csv.csv", "w") as file:
        file.write(csv_txt)
    # print(csv_txt)
    # for row in rows:
    #     line = ','.join()


def read_button_logs_to_csv():
    model_names = {
        "josh_pod_final_combined_model_tvl_single_rephrase_full_combinations_b1024_20240910_220927": "Final model",
        "josh_pod_ours_no_sound_b1024_20240912_214503": "Finetuned, no sensors",
    }

    def get_result_from_log_file(log_file_path):
        with open(log_file_path, "r") as file:
            last_log = [line for line in file][-1]
        result = last_log[last_log.find("result:") + len("result:") :]
        result = result[: result.find("video path")].strip()
        if result in {"success", "find", "failure"}:
            return result
        if result in {"correctcolorwrongtac", "wrongcolorcorrecttac"}:
            return "failure"
        raise ValueError(f"Result was [{result}], in file {log_file_path}")

    def extract_result_dict(model_name, exp_name, modality):
        log_path = os.path.join(LOG_DIR, model_name, "50000", exp_name)

        leaf_logs = [
            os.path.join(log_path, modality, str(i), "log.txt") for i in range(5)
        ]
        results = {"success": 0, "find": 0, "failure": 0}
        for leaf_log_path in leaf_logs:
            if not os.path.exists(leaf_log_path):
                print(leaf_log_path)
                continue
            # assert os.path.exists(leaf_log_path), leaf_log_path
            results[get_result_from_log_file(leaf_log_path)] += 1
        return results

    def add_sep(rows, num_sep=3):
        for _ in range(num_sep):
            sep_row = ["" for _ in rows[-1]]
            rows.append(sep_row)
        return rows

    MODALITIES = ["visual", "audio", "visual,audio"]
    rows = []
    skip_modalities = ["visual", "visual,audio"]
    for j, exp_name in enumerate([f"button_{i}" for i in range(3)]):  # bag test
        row_res = [f"button train objects - {j+1}"]
        for model_name in model_names:
            for modality in MODALITIES:
                if modality in skip_modalities:
                    row_res.extend([""])
                else:
                    res = extract_result_dict(model_name, exp_name, modality)
                    row_res.extend([res["success"]])
        rows.append(row_res)

    rows = add_sep(rows)

    skip_modalities = ["audio", "visual,audio"]
    for j, exp_name in enumerate([f"button_{i}" for i in range(3, 5)]):  # bag train
        row_res = [f"button test objects - {j+1}"]
        for model_name in model_names:
            for modality in MODALITIES:
                if modality in skip_modalities:
                    row_res.extend([""])
                else:
                    res = extract_result_dict(model_name, exp_name, modality)
                    row_res.extend([res["success"]])
        rows.append(row_res)
    rows = add_sep(rows)

    # for j, exp_name in enumerate([f'flat_{i}' for i in range(5)]): # flattest
    #     row_res = [f'platter test objects - {j+1}']
    #     for model_name in model_names:
    #         for modality in MODALITIES:
    #             res = extract_result_dict(model_name, exp_name, modality)
    #             row_res.extend([res['success'], res['find'], res['success'] + res['find']])
    #     rows.append(row_res)
    # rows = add_sep(rows)

    # for j, exp_name in enumerate([f'flat_{i}' for i in range(5, 10)]): # flat train
    #     row_res = [f'platter train objects - {j+1}']
    #     for model_name in model_names:
    #         for modality in MODALITIES:
    #             res = extract_result_dict(model_name, exp_name, modality)
    #             row_res.extend([res['success'], res['find'], res['success'] + res['find']])
    #     rows.append(row_res)

    def row_to_str(row):
        return [str(val) for val in row]

    csv_txt = "\n".join([",".join(row_to_str(row)) for row in rows])
    with open("./button_log_csv.csv", "w") as file:
        file.write(csv_txt)
    # print(csv_txt)
    # for row in rows:
    #     line = ','.join()


if __name__ == "__main__":
    # read_multi_logs_to_csv()
    read_button_logs_to_csv()
