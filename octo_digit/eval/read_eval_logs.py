import os
import glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


FIGSIZE = 25
POSSIBLE_RESULTS = ('find', 'success', 'fail')

def read_log(fp): 
    with open (fp, 'r') as file:
        lines = [line.strip() for line in file]
    info_line = lines[-1]
    result = info_line[info_line.find('result:'):]
    result = result[:result.find('video path')]
    for res_str in POSSIBLE_RESULTS: 
        if res_str in result: 
            return res_str
    raise RuntimeError((result, fp))


POSSIBLE_RESULTS_TACDIS = {
    'success': 'success-correct', 
    'find': 'find-correct', 
    'success-correct': 'success-correct',
    'find-correct': 'find-correct',
    'success-correct_color_wrong_tactile': 'success-correct_color_wrong_tactile',
    'success-correct_color_wrong_tac': 'success-correct_color_wrong_tactile',
    'find-correct_color_wrong_tactile': 'find-correct_color_wrong_tactile',
    'find-correct_color_wrong_tac': 'find-correct_color_wrong_tactile', 
    'success-wrong_color_correct_tactile': 'success-wrong_color_correct_tactile',
    'success-wrong_color_correct_tac': 'success-wrong_color_correct_tactile',
    'find-wrong_color_correct_tactile': 'find-wrong_color_correct_tactile',
    'find-wrong_color_correct_tac': 'find-wrong_color_correct_tactile',
    'failure': 'failure',
    'fail': 'failure'
}
def read_log_dis1(fp): 
    
    with open (fp, 'r') as file:
        lines = [line.strip() for line in file]
    info_line = lines[-1]
    result = info_line[info_line.find('result:'):]
    result = result[len('result:'):].strip()
    result = result[:result.find('video path')].strip()
    for res_str in POSSIBLE_RESULTS_TACDIS: 
        # print(f'\n[{res_str}]\n[{result}]')
        if res_str == result: 
            return POSSIBLE_RESULTS_TACDIS[res_str]
    print('FAIL: result was ', f"[{result}]")
    raise RuntimeError(result)


POSSIBLE_RESULTS_SOUNDDIS = {
    'success': 'success-correct', 
    'find': 'find-correct', 
    'success-correct': 'success-correct',
    'find-correct': 'find-correct',
    'success-correct_color_wrong_sound': 'success-correct_color_wrong_sound',

    'find-correct_color_wrong_sound': 'find-correct_color_wrong_sound',

    'success-wrong_color_correct_sound': 'success-wrong_color_correct_sound',

    'find-wrong_color_correct_sound': 'find-wrong_color_correct_sound',

    'failure': 'failure',
    'fail': 'failure'
}

def read_log_dis3(fp): 
    
    with open (fp, 'r') as file:
        lines = [line.strip() for line in file]
    info_line = lines[-1]
    result = info_line[info_line.find('result:'):]
    result = result[len('result:'):].strip()
    result = result[:result.find('video path')].strip()
    for res_str in POSSIBLE_RESULTS_SOUNDDIS: 
        # print(f'\n[{res_str}]\n[{result}]')
        if res_str == result: 
            return POSSIBLE_RESULTS_SOUNDDIS[res_str]
    print('FAIL: result was ', f"[{result}]")
    raise RuntimeError(result)


def compute_result_logs(log):
    for key, results in log.items(): 
        num_successes = sum([1 for val in results.values() if val == 'success'])
        num_finds = sum([1 for val in results.values() if val == 'find'])
        num_fails = sum([1 for val in results.values() if val == 'fail'])
        # if num_successes + num_finds + num_fails != 10: 
        #     print(sorted(results.items()))
        # try:
        #     assert num_successes + num_finds + num_fails == 10,  (num_successes + num_finds + num_fails, key)
        # except Exception: 
        #     input(f'{}, pres enter:  ')
        num_fails = 10 - num_successes - num_finds
        log[key]['total'] = {
            'success': num_successes, 
            'find': num_finds, 
            'fail': num_fails
        }
    
    return log

def compute_result_logs_dis1(log):
    for key, results in log.items(): 
        total_dict = {}
        for result_type in POSSIBLE_RESULTS_TACDIS:
            if result_type not in POSSIBLE_RESULTS_TACDIS.values(): 
                continue 
            num_of_type = sum([1 for val in results.values() if val == result_type])
            total_dict[result_type] = num_of_type
        assert sum(total_dict.values()) == 10, (key, '\n\n', results, '\n\n', total_dict, sum(total_dict.values()), sorted(results.keys()))
            
        log[key]['total'] = total_dict
    return log

def compute_result_logs_dis3(log):
    for key, results in log.items(): 
        total_dict = {}
        for result_type in POSSIBLE_RESULTS_SOUNDDIS:
            if result_type not in POSSIBLE_RESULTS_SOUNDDIS.values(): 
                continue 
            num_of_type = sum([1 for val in results.values() if val == result_type])
            total_dict[result_type] = num_of_type
        assert sum(total_dict.values()) == 10, (key, '\n\n', results, '\n\n', total_dict, sum(total_dict.values()), sorted(results.keys()))
            
        log[key]['total'] = total_dict
    return log


def read_eval_logs(log_dir='/home/josh/octo_digit/eval/eval_logs', read_log_func=read_log, keep_models=None, keep_exps=None, keep_types=None): 
    log = defaultdict(dict)
    leaf_log_files = glob.glob(
        os.path.join(log_dir, *(['*'] * 5), 'log.txt')
    )
    for leaf_log in leaf_log_files:
        info = leaf_log.split('/')
        run_index = int(info[-2])
        command_type = info[-3]
        exp_name = info[-4]
        ckpt = info[-5]
        model_name = info[-6]
        # print(leaf_log)
        if keep_models is not None and model_name not in keep_models: 
            continue
        if keep_exps is not None and exp_name not in keep_exps: 
            continue
        if keep_types is not None and command_type not in keep_types: 
            continue
        try: 
            result = read_log_func(leaf_log)
        except Exception as e: 
            print('leaf log is', leaf_log)
            raise e
        key = (model_name, ckpt, exp_name, command_type)
        log[key][run_index] = result

    return log





def create_bar_graph(
    names, results, title, save_name
): 
    plt.figure()
    fig, ax = plt.subplots()
    bottom = np.zeros(len(names))
    x = np.arange(len(names))
    multiplier = 0
    width = 0.8
    for result in ['success', 'find']: 
        offset = width * multiplier
        print(names, result, results[result])
        p = ax.bar(x + offset, results[result], 0.5, label=result, bottom=bottom)
        bottom += results[result]
    # ax.bar_label(p, labels=names, rotation=90, padding=-100)
    ax.set_title(f'{title}\n')
    ax.legend(loc='upper center')# bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_yticks([i for i in range(12) if i % 2 == 0])
    ax.set_xticks(ticks=x, labels=[ '   ' + name for name in names], 
                  # rotation=90,
                  # verticalalignment='bottom',
                  )
    # ax.set_ylim((0, 15))
    plt.savefig(save_name)

def create_bar_graph_multi_labeled(
    model_names, results, title, save_name, # results is type to resdict
): 
    plt.figure()
    fig, ax = plt.subplots()

    total_bars = len(model_names) * len(results)

    width = 0.18
    x = np.arange(len(model_names))
    multiplier = 0
    type_to_display = { 
        'simple': 'simple', 
        'visual': 'viz', 
        'tactile': 'tac', 
        'visual,tactile,audio': 'all'
    }
    for i, ann in enumerate(type_to_display): 
        bottom = np.zeros(len(model_names))
        offset = width * multiplier
        rects = ax.bar(x + offset, results[ann]['success'], width, color=['tab:blue' for _ in range(len(model_names))], edgecolor='black')
        rects = ax.bar(x + offset, results[ann]['find'], width, bottom=results[ann]['success'], color=['tab:orange' for _ in model_names], edgecolor='black')
        ax.bar_label(rects, labels=[type_to_display[ann] for _ in range(len(model_names))], rotation=90, padding=3)
        multiplier += 1

    ax.set_title(f'{title}\n')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_xticks(x + width, model_names)
    ax.set_yticks([i for i in range(11) if i % 2 == 0])
    plt.savefig(save_name)

def create_bar_graph_multi(
    model_names, results, title, save_name, # results is type to resdict
): 
    plt.figure()
    fig, ax = plt.subplots()

    total_bars = len(model_names) * len(results)

    width = 0.18
    x = np.arange(len(model_names))
    multiplier = 0
    type_to_display = { 
        'tactile': 'tac', 
    }
    for i, ann in enumerate(type_to_display): 
        bottom = np.zeros(len(model_names))
        offset = width * multiplier
        rects = ax.bar(x + offset, results[ann]['success'], width, color=['tab:blue' for _ in range(len(model_names))], edgecolor='black')
        rects = ax.bar(x + offset, results[ann]['find'], width, bottom=results[ann]['success'], color=['tab:orange' for _ in model_names], edgecolor='black')
        ax.bar_label(rects, labels=[type_to_display[ann] for _ in range(len(model_names))], rotation=90, padding=3)
        multiplier += 1

    ax.set_title(f'{title}\n')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_xticks(x + width, model_names)
    ax.set_yticks([i for i in range(11) if i % 2 == 0])
    plt.savefig(save_name)
    
def create_bar_graph_tacs(
    display_names, model_names, results, title, save_name # results is type (tac/tac2/tac3) to resdict. Resdict: succ/find arr in order of model names
): 
    plt.figure()
    fig, ax = plt.subplots()

    # new residict: model_name -> success arr[tac, tac2, tac3], find...
    # reshape results to be model name to performance on each type
    # orig_results = results
    # mode_wise_results = {}
    # for i, model in enumerate(model_names): 
    #     model_perf = defaultdict(list) # succ to arr, find to arr
    #     for mode, r in results.items(): # tac, tac2, tac3
    #         for success_level, arr in r.items(): 
    #             model_performance_on_mode = arr[i]

    #             model_perf[success_level].append(model_performance_on_mode)
    #     for k, v in model_perf.items(): 
    #         model_perf[k] = np.array(v)
    #     mode_wise_results[model] = model_perf
    
    # results: list of (name, ckpt, exp, command) -> { total: }s 
    from functools import partial
    mode_wise_results = defaultdict(
        partial(defaultdict, list)
    )
    mode_wise_results = defaultdict(dict) # name -> {'succ': [], 'find': []}
    for model in model_names: 
        for exp_dic in results: # 
            for key in exp_dic: 
                if model != key[0]: 
                    continue
                for succ_key in ['success', 'find']: 
                    # print(exp_dic[key]['total'][succ_key])
                    if model not in mode_wise_results: 
                        mode_wise_results[model] = defaultdict(list)
                    # print(model in mode_wise_results)
                    # print(mode_wise_results[model])
                    mode_wise_results[model][succ_key].append(exp_dic[key]['total'][succ_key])
    
    for k, v in mode_wise_results.items(): 
        for k2, v2 in v.items(): 
            mode_wise_results[k][k2] = np.array(v2)
   
    
            
    total_bars = len(model_names) * len(results)

    width = 0.18
    x = np.arange(3)
    multiplier = 0
    
    for i, model in enumerate(model_names): 
        bottom = np.zeros(3,)
        offset = width * multiplier
        rects = ax.bar(x + offset, mode_wise_results[model]['success'], width, color=['tab:blue' for _ in range(3)], edgecolor='black')
        rects = ax.bar(x + offset, mode_wise_results[model]['find'], width, bottom=mode_wise_results[model]['success'], color=['tab:orange' for _ in model_names], edgecolor='black')
        ax.bar_label(rects, labels=[display_names[i] for _ in range(3)], rotation=90, padding=3)
        multiplier += 1 

    ax.set_title(f'{title}\n')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_xticks(
        x + width, 
        [
            'Wooden block', 
            'Stuffed animal\nNew annotations', 
            'Stuffed animal'
        ]
    )
    # ax.set_xticks(x + width, model_names)
    ax.set_yticks([i for i in range(11) if i % 2 == 0])
    plt.savefig(save_name)
    exit(0)

def create_bar_graph_dis1(
    display_names, model_names, results, title, save_name 
): 
    plt.figure()
    fig, ax = plt.subplots()
    
    # results: model name to total dict

    # new result: compute success_type to model name
    type_to_model_res = defaultdict(list)
    for model in model_names: 
        total_dict = results[model]
        for succ_type, val in total_dict.items(): 
            type_to_model_res[succ_type].append(val) 
    
    for k, v in type_to_model_res.items(): 
        type_to_model_res[k] = np.array(v)
    
    type_to_model_res.pop('failure')
    type_to_model_res['success-correct'] = type_to_model_res['success']
    type_to_model_res['find-correct'] = type_to_model_res['find']
    type_to_model_res.pop('success')
    type_to_model_res.pop('find')
    
    width = 0.18
    x = np.arange(len(model_names))
    multiplier = 0
    
    type_to_model_res = dict(type_to_model_res)
    print(type_to_model_res)
    display_types = {
        'correct': 'correct',
        'correct_color_wrong_tactile': '+visual, -tactile', 
        'wrong_color_correct_tactile': '-visual, +tactile'
    }
    
    for typ, disp in display_types.items(): 
        offset = width * multiplier
        typ_grasps = type_to_model_res[f'success-{typ}']
        typ_finds = type_to_model_res[f'find-{typ}']
        print(typ_finds, typ_grasps)
        rects = ax.bar(x + offset, typ_grasps, width, color=['tab:blue' for _ in range(len(model_names))], edgecolor='black')
        rects = ax.bar(x + offset, typ_finds, width, bottom=typ_grasps, color=['tab:orange' for _ in model_names], edgecolor='black')
        
        ax.bar_label(rects, labels=[disp for _ in range(len(model_names))], rotation=90, padding=3)
        multiplier += 1 

    ax.set_title(f'{title}\n')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_xticks(
        x + width, 
        display_names
    )
    # ax.set_xticks(x + width, model_names)
    ax.set_yticks([i for i in range(11) if i % 2 == 0])
    plt.savefig(save_name)

def create_bar_graph_dis2(
    display_names, results, title, save_name 
): 
    plt.figure()
    fig, ax = plt.subplots()
    
    # results: object name to to success counts

    bar_keys = {
        'simple': 'simple', 
        'correct': 'correct', 
        'correct_color_wrong_tactile': '+visual, -tactile', 
        'wrong_color_correct_tactile': '-visual, +tactile'
    }
    bar_colors = {
        'simple': {
            'success': 'lightblue', 
            'find': 'lightsalmon'
        },
        'default': {
            'success': 'tab:blue', 
            'find': 'tab:orange'
        }
    }
    width = 0.18
    x = np.arange(len(results))
    multiplier = 0
    
    for typ, disp in bar_keys.items(): 
        offset = width * multiplier
        grasps = np.array([results[obj][f'success-{typ}'] for obj in results]) 
        finds = np.array([results[obj][f'find-{typ}'] for obj in results])
        # print(typ, grasps, finds)
        grasp_colors = [bar_colors.get(typ, bar_colors['default'])['success'] for _ in range(len(grasps))] 
        find_colors = [bar_colors.get(typ, bar_colors['default'])['find'] for _ in range(len(grasps))] 


        rects = ax.bar(x + offset, grasps, width, color=grasp_colors, edgecolor='black')
        rects = ax.bar(x + offset, finds, width, bottom=grasps, color=find_colors, edgecolor='black')
        
        ax.bar_label(rects, labels=[disp for _ in range(len(grasps))], rotation=90, padding=3)
        multiplier += 1 

    ax.set_title(f'{title}\n')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_xticks(
        x + width, 
        display_names
    )
    # ax.axhline(10, linestyle='--')
    # ax.set_xticks(x + width, model_names)
    ax.set_yticks([i for i in range(11) if i % 2 == 0])
    plt.savefig(save_name)

  
def create_bar_graph_sound1(
    model_display_names, model_names, results, title, save_name 
): 
    plt.figure()
    fig, ax = plt.subplots()
    
    # results: model name to to success counts

    bar_keys = {
        'simple': 'simple', 
        'sound': 'sound', 
    }
    bar_colors = {
        'simple': {
            'success': 'lightblue', 
            'find': 'lightsalmon'
        },
        'default': {
            'success': 'tab:blue', 
            'find': 'tab:orange'
        }
    }
    width = 0.18
    x = np.arange(len(model_names))
    multiplier = 0
    
    for typ, disp in bar_keys.items(): 
        offset = width * multiplier
        grasps = np.array([results[model][f'success-{typ}'] for model in model_names]) 
        finds = np.array([results[model][f'find-{typ}'] for model in model_names])
        print(typ, grasps, finds)
        grasp_colors = [bar_colors.get(typ, bar_colors['default'])['success'] for _ in range(len(grasps))] 
        find_colors = [bar_colors.get(typ, bar_colors['default'])['find'] for _ in range(len(grasps))] 


        rects = ax.bar(x + offset, grasps, width, color=grasp_colors, edgecolor='black')
        rects = ax.bar(x + offset, finds, width, bottom=grasps, color=find_colors, edgecolor='black')
        
        ax.bar_label(rects, labels=[disp for _ in range(len(grasps))], rotation=90, padding=3)
        multiplier += 1 

    ax.set_title(f'{title}\n')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_xticks(
        x + width, 
        model_display_names
    )
    # ax.axhline(10, linestyle='--')
    # ax.set_xticks(x + width, model_names)
    ax.set_yticks([i for i in range(11) if i % 2 == 0])
    plt.savefig(save_name)  

def create_bar_graph_dis3(
    display_names, results, title, save_name 
): 
    plt.figure()
    fig, ax = plt.subplots()
    
    # results: object name to to success counts

    bar_keys = {
        'simple': 'simple', 
        'correct': 'correct', 
        'correct_color_wrong_sound': '+visual, -sound', 
        'wrong_color_correct_sound': '-visual, +sound'
    }
    bar_colors = {
        'simple': {
            'success': 'lightblue', 
            'find': 'lightsalmon'
        },
        'default': {
            'success': 'tab:blue', 
            'find': 'tab:orange'
        }
    }
    width = 0.18
    x = np.arange(len(results))
    multiplier = 0
    
    for typ, disp in bar_keys.items(): 
        offset = width * multiplier
        grasps = np.array([results[obj][f'success-{typ}'] for obj in results]) 
        finds = np.array([results[obj][f'find-{typ}'] for obj in results])
        # print(typ, grasps, finds)
        grasp_colors = [bar_colors.get(typ, bar_colors['default'])['success'] for _ in range(len(grasps))] 
        find_colors = [bar_colors.get(typ, bar_colors['default'])['find'] for _ in range(len(grasps))] 


        rects = ax.bar(x + offset, grasps, width, color=grasp_colors, edgecolor='black')
        rects = ax.bar(x + offset, finds, width, bottom=grasps, color=find_colors, edgecolor='black')
        
        ax.bar_label(rects, labels=[disp for _ in range(len(grasps))], rotation=90, padding=3)
        multiplier += 1 

    ax.set_title(f'{title}\n')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_xticks(
        x + width, 
        display_names
    )
    # ax.axhline(10, linestyle='--')
    # ax.set_xticks(x + width, model_names)
    ax.set_yticks([i for i in range(11) if i % 2 == 0])
    plt.savefig(save_name)
    
    
def create_bar_graph_tac_test(
    display_names, results, title, save_name 
): 
    plt.figure()
    fig, ax = plt.subplots()
    
    # results: object name to to success counts

    bar_keys = {
        'simple': 'simple', 
        'tac': 'tac', 
    }
    bar_colors = {
        'simple': {
            'success': 'lightblue', 
            'find': 'lightsalmon'
        },
        'default': {
            'success': 'tab:blue', 
            'find': 'tab:orange'
        }
    }
    width = 0.18
    x = np.arange(len(results))
    multiplier = 0
    
    for typ, disp in bar_keys.items(): 
        offset = width * multiplier
        grasps = np.array([results[model][f'success-{typ}'] for model in results]) 
        finds = np.array([results[model][f'find-{typ}'] for model in results])
        # print(typ, grasps, finds)
        grasp_colors = [bar_colors.get(typ, bar_colors['default'])['success'] for _ in range(len(grasps))] 
        find_colors = [bar_colors.get(typ, bar_colors['default'])['find'] for _ in range(len(grasps))] 


        rects = ax.bar(x + offset, grasps, width, color=grasp_colors, edgecolor='black')
        rects = ax.bar(x + offset, finds, width, bottom=grasps, color=find_colors, edgecolor='black')
        
        ax.bar_label(rects, labels=[disp for _ in range(len(grasps))], rotation=90, padding=3)
        multiplier += 1 

    ax.set_title(f'{title}\n')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_xticks(
        x + width, 
        display_names
    )
    # ax.axhline(10, linestyle='--')
    # ax.set_xticks(x + width, model_names)
    ax.set_yticks([i for i in range(11) if i % 2 == 0])
    plt.savefig(save_name)
    
def create_bar_graph_batch_test(
    display_names, results, title, save_name 
): 
    plt.figure()
    fig, ax = plt.subplots()
    
    # results: object name to to success counts

    bar_keys = {
        'simple': 'simple', 
        'tac': 'tac', 
        # 'tac,viz': 'tac,viz',
    }
    bar_colors = {
        'simple': {
            'success': 'lightblue', 
            'find': 'lightsalmon'
        },
        'default': {
            'success': 'tab:blue', 
            'find': 'tab:orange'
        }
    }
    width = 0.18
    x = np.arange(len(results))
    multiplier = 0
    
    for typ, disp in bar_keys.items(): 
        offset = width * multiplier
        grasps = np.array([results[model][f'success-{typ}'] for model in results]) 
        finds = np.array([results[model][f'find-{typ}'] for model in results])
        # print(typ, grasps, finds)
        grasp_colors = [bar_colors.get(typ, bar_colors['default'])['success'] for _ in range(len(grasps))] 
        find_colors = [bar_colors.get(typ, bar_colors['default'])['find'] for _ in range(len(grasps))] 


        rects = ax.bar(x + offset, grasps, width, color=grasp_colors, edgecolor='black')
        rects = ax.bar(x + offset, finds, width, bottom=grasps, color=find_colors, edgecolor='black')
        
        ax.bar_label(rects, labels=[disp for _ in range(len(grasps))], rotation=90, padding=3)
        multiplier += 1 

    ax.set_title(f'{title}\n')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_xticks(
        x + width, 
        display_names
    )
    # ax.axhline(10, linestyle='--')
    # ax.set_xticks(x + width, model_names)
    ax.set_yticks([i for i in range(11) if i % 2 == 0])
    plt.savefig(save_name)

def key_field(key, field_name): 
        # (model_name, ckpt, exp_name, command_type)
        if field_name == 'model_name' or field_name == 'model': 
            return key[0]
        elif field_name == 'ckpt': 
            return key[1]
        elif field_name == 'exp_name': 
            return key[2]
        elif field_name == 'command_type': 
            return key[3]
        else: 
            raise RuntimeError()

def plot_tacs(log_dir='/home/josh/octo_digit/eval/eval_logs'): 
    # filter fn takes item
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    keep_keys = {
        'multi_base_20k_20240806_224758',
        'multi_cont_20k_20240806_072717',
    }
    logs = read_eval_logs(log_dir)
    # print(logs.keys())
    logs = {k: v for k, v in logs.items() if k[0] in keep_keys}
    
    logs = filter_dict(logs, lambda k, v: key_field(k, 'command_type') == 'tactile')
    
    keep_mode = 'tac2'
    keep_modes = [
        'tac', 'tac2', 'tac3'
    ]
    log_modes = [] 
    for mode in keep_modes: 
        log_mode = {k: v for k, v in logs.items() if key_field(k, 'exp_name') == mode}
        log_modes.append(log_mode)
    
    
    # for l in log_modes: 
    #     for k, v in l.items(): 
    #         print(k, v)

    result_modes = []
    for l in log_modes: 
        result_modes.append(compute_result_logs(l))
    
    # for l in result_modes: 
    #     for k, v in l.items(): 
    #         print(k, v)
    #     print('\n')
    # exit(0)
    
    
    # logs = compute_result_logs(logs)
    # for k, v in logs.items(): 
    #     print(k, v)
    
    # def get_result_dict(keys): 
    #     return { 
    #         'success': np.array([logs[key]['total']['success'] for key in keys]),
    #         'find': np.array([logs[key]['total']['find'] for key in keys])
    #     }

    # run_names = set([key[0] for key in logs])
    run_names = [
        ('multi_base_20k_20240806_224758','50000'),
        ('multi_cont_20k_20240806_072717', '50000')
    ]
    run_names = [
        'multi_base_20k_20240806_224758', 
        'multi_cont_20k_20240806_072717'
    ]
    simple_display_names = [
        'Base',
        'CLIP'
    ]
    
    create_bar_graph_tacs(simple_display_names, run_names, result_modes, 'Tactile prompting - test objects', 'figs/tactiles.png')
    # simple_display_names = [run_to_type(run) for run in simple_runs]

    # simple, train 
    # result_dict = get_result_dict(simple_train_runs)   
    # mod_types = ['simple', 'visual', 'tactile', 'visual,tactile,audio']
    mod_types = ['tactile']
    res_dict = { 
        key : get_result_dict([(*run_name, keep_mode, key) for run_name in run_names]) for key in mod_types
    }
    
    print(res_dict)

    create_bar_graph(simple_display_names, res_dict['tactile'], 'Tactile commands, targeting wooden block', f'figs/{keep_mode}.png')

def plot_dis1(
    log_dir='/home/josh/octo_digit/eval/eval_logs'
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    keep_keys = {
        'multi_base_20k_20240806_224758',
        'multi_cont_20k_20240806_072717',
    }
    logs = read_eval_logs(log_dir, read_log_func=read_log_dis1, keep_models=keep_keys, keep_exps={'dis1'})

    logs = {k: v for k, v in logs.items() if k[0] in keep_keys}
    
    logs = filter_dict(logs, lambda k, v: key_field(k, 'exp_name') == 'dis1')
    
    
    ball_logs = filter_dict(logs, lambda k, v: key_field(k, 'command_type') == 'ball')
    # for k, v in ball_logs.items(): 
    #     print(k, v)

    
    # print('\n\n\n RESULTS:   ')
    ball_logs = compute_result_logs_dis1(ball_logs)
    ball_results = {
        k[0]: v['total'] for k, v in ball_logs.items()
    }
    model_names = [
        'multi_base_20k_20240806_224758',
        'multi_cont_20k_20240806_072717',
    ]
    simple_display_names = [
        'base', 
        'CLIP'
    ]
    create_bar_graph_dis1(simple_display_names, model_names, ball_results, 'Visual/Tactile disambiguation: targeting foam ball', 'figs/dis1_ball.png')
    
    block_logs = filter_dict(logs, lambda k, v: key_field(k, 'command_type') == 'block')
    block_logs = compute_result_logs_dis1(block_logs)
    block_results = {
        k[0]: v['total'] for k, v in block_logs.items()
    }
    create_bar_graph_dis1(simple_display_names, model_names, block_results, 'Visual/Tactile disambiguation: targeting wood block', 'figs/dis1_block.png')

def plot_dis2(
    log_dir='/home/josh/octo_digit/eval/eval_logs'
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    keep_keys = {
        'multi_base_20k_20240806_224758',
        'multi_cont_20k_20240806_072717',
    }
    logs = read_eval_logs(log_dir, read_log_func=read_log_dis1, keep_models=keep_keys, keep_exps={'dis2'})

    logs = {k: v for k, v in logs.items() if k[0] in keep_keys}
    
    logs = filter_dict(logs, lambda k, v: key_field(k, 'exp_name') == 'dis2')
    
    compute_result_logs_dis1(logs)
    
    blue_animal_logs = filter_dict(logs, lambda k, v: 'blue_animal' in k[-1])
    for k, v in blue_animal_logs.items(): 
        print(k, v)
    
    experiment_name = ('multi_cont_20k_20240806_072717', '50000', 'dis2')
    
    objects = ['blue_animal', 'blue_duck', 'green_animal', 'green_duck']
    
    RESULTS = {}
    for obj in objects: 
        simple_key = (*experiment_name, f'{obj}_simple')
        res =  {
            'success-simple': logs[simple_key]['total']['success-correct'], 
            'find-simple': logs[simple_key]['total']['find-correct'], 
        }
        res.update(
            logs[(*experiment_name, f'{obj}_viztac')]['total']
        )
        RESULTS[obj] = res
        
    for k, v in RESULTS.items(): 
        print(k, v)
        print('\n\n')
    
    create_bar_graph_dis2(
        ['Blue stuffed\nanimal', 'Blue\nrubber duck', 'Green stuffed\nanimal', 'Green\nrubber duck'], 
        RESULTS, 
        'Visual/Tactile disambiguation setting #2', 
        'figs/dis2.png'
    )

def plot_sound1(
    log_dir='/home/josh/octo_digit/eval/eval_logs'
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    keep_keys = [
        'audio_cont_20k_20240807_220439',
        'mic_audio_cont_20k_d128_20240808_234005',
    ]
    logs = dict(read_eval_logs(log_dir, read_log_func=read_log_dis1, keep_models=keep_keys, keep_exps={'sound1'}))

    # logs = {k: v for k, v in logs.items() if k[0] in keep_keys}
    
    # logs = filter_dict(logs, lambda k, v: key_field(k, 'exp_name') == 'dis2')
    
    compute_result_logs_dis1(logs)

    key_logs = filter_dict(logs, lambda k, v: 'key' in k[-1])
    RESULTS = defaultdict(dict)
    for k, v in key_logs.items(): 
        model = k[0]
        if k[-1] == 'key_simple': 
            suffix = 'simple'
        elif k[-1] == 'key_sound': 
            suffix = 'sound'
        else: 
            raise ValueError(k[-1])
        RESULTS[model][f'success-{suffix}'] = v['total']['success-correct']
        RESULTS[model][f'find-{suffix}'] = v['total']['find-correct']
    create_bar_graph_sound1(['no mic', 'mic'], keep_keys, RESULTS, 'Sound prompting - test keys', 'figs/sound1_keys.png')
    
    
    duck_logs = filter_dict(logs, lambda k, v: 'duck' in k[-1])
    RESULTS = defaultdict(dict)
    for k, v in key_logs.items(): 
        model = k[0]
        if k[-1] == 'duck_simple': 
            suffix = 'simple'
        elif k[-1] == 'duck_sound': 
            suffix = 'sound'
        else: 
            raise ValueError(k[-1])
        RESULTS[model][f'success-{suffix}'] = v['total']['success-correct']
        RESULTS[model][f'find-{suffix}'] = v['total']['find-correct']
    create_bar_graph_sound1(['no mic', 'mic'], keep_keys, RESULTS, 'Sound prompting - test duck', 'figs/sound1_duck.png')
        
def plot_dis3(
    log_dir='/home/josh/octo_digit/eval/eval_logs'
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    keep_keys = [
        # 'audio_cont_20k_20240807_220439',
        'mic_audio_cont_20k_d128_20240808_234005',
    ]
    logs = read_eval_logs(log_dir, read_log_func=read_log_dis3, keep_models=keep_keys, keep_exps={'dis3'})
    
    compute_result_logs_dis3(logs)
    
    for k, v in logs.items():
        print(k, v['total'])
        print('\n')
    
    experiment_name = (keep_keys[0], '50000', 'dis3')
    
    objects = ['blue_duck', 'green_duck']
    
    RESULTS = {}
    for obj in objects: 
        simple_key = (*experiment_name, f'{obj}_simple')
        res =  {
            'success-simple': logs[simple_key]['total']['success-correct'], 
            'find-simple': logs[simple_key]['total']['find-correct'], 
        }
        res.update(
            logs[(*experiment_name, f'{obj}_vizsound')]['total']
        )
        RESULTS[obj] = res
        
    for k, v in RESULTS.items(): 
        print(k, v)
        print('\n\n')
    
    fsuffix = 'mic' if 'mic' in keep_keys[0] else 'nomic'
    wwo = 'w/' if 'mic' in keep_keys[0] else 'w/o'
    
    create_bar_graph_dis3(
        ['Blue\nrubber duck', 'Green rubber\nduck'], 
        RESULTS, 
        f'Visual/sound disambiguation, model {wwo} mic', 
        f'figs/dis3_{fsuffix}.png'
    )    
    
def plot_tac_test(
    log_dir='/home/josh/octo_digit/eval/eval_logs'
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    keep_keys = [
        'multi_cont_20k_20240806_072717',
        'tac_base_20240824_173914',
        'vit_20240824_173949',
        'resnet_imagenet_20240824_174202',

        'tvl_finetuned_20240824_070132',
        'tvl_frozen_20240824_123443',

        't3_frozen_20240824_205857',
    ]
    logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'test_tac'}, keep_types={'yellow_ball_simple', 'yellow_ball_tac'})
    
    compute_result_logs(logs)
    
    # for k, v in logs.items():
    #     print(k, v['total'])
    #     print('\n')
        
    model_names = set(log[0] for log in logs)
    
    display_names = {
        'multi_cont_20k_20240806_072717': 'CLIP/ViT',
        'tac_base_20240824_173914': 'Base',
        'vit_20240824_173949': 'ViT',
        'resnet_imagenet_20240824_174202': 'Resnet\nImagenet',

        'tvl_finetuned_20240824_070132': 'TVL\nfinetune',
        'tvl_frozen_20240824_123443': 'TVL\nfrozen',

        't3_frozen_20240824_205857': 't3\nfrozen'
    }
    
    def compute_dic(model_name): 
        rel_log_keys = [log for log in logs if log[0] == model_name]
        simple_key = [log for log in rel_log_keys if 'simple' in log[-1]][0]
        simple_res = logs[simple_key]['total']
        
        
        tac_key = [log for log in rel_log_keys if 'tac' in log[-1]][0]
        tac_res = logs[tac_key]['total']
        result = {
            'success-simple': simple_res['success'], 
            'find-simple': simple_res['find'], 
            'success-tac': tac_res['success'], 
            'find-tac': tac_res['find']
        }
        return result 
    
    
    full_results = {
        display_name: compute_dic(model_name) for model_name, display_name in display_names.items()
    }

    create_bar_graph_tac_test(display_names.values(), full_results, 'Tactile encoder test - ball', 'figs/test_tac_ball.png')
    

    # experiment_name = (keep_keys[0], '50000', 'dis3')
    
    # objects = ['blue_duck', 'green_duck']
    
    # RESULTS = {}
    # for obj in objects: 
    #     simple_key = (*experiment_name, f'{obj}_simple')
    #     res =  {
    #         'success-simple': logs[simple_key]['total']['success-correct'], 
    #         'find-simple': logs[simple_key]['total']['find-correct'], 
    #     }
    #     res.update(
    #         logs[(*experiment_name, f'{obj}_vizsound')]['total']
    #     )
    #     RESULTS[obj] = res
        
    # for k, v in RESULTS.items(): 
    #     print(k, v)
    #     print('\n\n')
    
    # fsuffix = 'mic' if 'mic' in keep_keys[0] else 'nomic'
    # wwo = 'w/' if 'mic' in keep_keys[0] else 'w/o'
    
    # create_bar_graph_dis3(
    #     ['Blue\nrubber duck', 'Green rubber\nduck'], 
    #     RESULTS, 
    #     f'Visual/sound disambiguation, model {wwo} mic', 
    #     f'figs/dis3_{fsuffix}.png'
    # )    

def plot_pod_test(
    log_dir='/home/josh/octo_digit/eval/eval_logs'
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    keep_keys = [
        "experiment_20240827_231310",
        "good_pod_512",
        "good_pod_1024",
        "good_pod_2048",
    ]
    logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'pod'}, keep_types={'yellow_ball_simple', 'yellow_block_tac', 'yellow_block_tacviz'})
    
    compute_result_logs(logs)

    
    # for k, v in logs.items():
    #     print(k, v['total'])
    #     print('\n')
        
    model_names = set(log[0] for log in logs)
    
    display_names = {
        "experiment_20240827_231310": 'b128',
        "good_pod_512": 'b512',
        "good_pod_1024": 'b1024',
        "good_pod_2048": 'b2048',
    }
    
    def compute_dic(model_name): 
        rel_log_keys = [log for log in logs if log[0] == model_name]
        simple_key = [log for log in rel_log_keys if 'simple' in log[-1]][0]
        simple_res = logs[simple_key]['total']
        
        tac_key = [log for log in rel_log_keys if log[-1].endswith('tac')][0]
        tac_res = logs[tac_key]['total']
        
        tacviz_key = [log for log in rel_log_keys if log[-1].endswith('tacviz')][0]
        tacviz_res = logs[tacviz_key]['total']
        # print(tacviz_res == tac_res)
        result = {
            'success-simple': simple_res['success'], 
            'find-simple': simple_res['find'], 
            'success-tac': tac_res['success'], 
            'find-tac': tac_res['find'],
            'success-tac,viz': tacviz_res['success'],
            'find-tac,viz': tacviz_res['find']
        }
        return result 
    
    
    full_results = {
        display_name: compute_dic(model_name) for model_name, display_name in display_names.items()
    }

    create_bar_graph_batch_test(display_names.values(), full_results, 'Batch size ablation - ball', 'figs/pods_ball.png')
    
def plot_gen_test(
    log_dir='/home/josh/octo_digit/eval/eval_logs'
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    keep_keys = [
        "good_pod_1024",
        "josh_pod_gen_lang_b1024_20240830_012615",
        "josh_pod_gen_lang_single_headb1024_20240830_060804",
    ]
    # logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'gen', 'pod'}, keep_types={'yellow_ball_simple', 'yellow_block_tac', 'yellow_block_tacviz'})
    logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'gen', 'pod'}, keep_types={'red_block_viz', 'red_block_tac'})
    compute_result_logs(logs)

    
    # for k, v in logs.items():
    #     print(k, v['total'])
    #     print('\n')
        
    model_names = set(log[0] for log in logs)
    
    display_names = {
        "good_pod_1024" : 'contrastive b1024',
        "josh_pod_gen_lang_b1024_20240830_012615": 'multi-head',
        "josh_pod_gen_lang_single_headb1024_20240830_060804": 'single head',
    }
    
    def compute_dic(model_name): 
        rel_log_keys = [log for log in logs if log[0] == model_name]
        
        simple_key = [log for log in rel_log_keys if 'viz' in log[-1]][0]
        simple_res = logs[simple_key]['total']
        print(simple_res)
        
        tac_key = [log for log in rel_log_keys if log[-1].endswith('tac')][0]
        tac_res = logs[tac_key]['total']
        
        # tacviz_key = [log for log in rel_log_keys if log[-1].endswith('tacviz')][0]
        # tacviz_res = logs[tacviz_key]['total']
        # print(tacviz_res == tac_res)
        result = {
            'success-simple': simple_res['success'], 
            'find-simple': simple_res['find'], 
            'success-tac': tac_res['success'], 
            'find-tac': tac_res['find'],
            # 'success-tac,viz': tacviz_res['success'],
            # 'find-tac,viz': tacviz_res['find']
        }
        return result 
    
    
    full_results = {
        display_name: compute_dic(model_name) for model_name, display_name in display_names.items()
    }

    create_bar_graph_batch_test(display_names.values(), full_results, 'Generative test - block', 'figs/gen_block2.png')
    
def create_bar_graph_ball_test(
    display_names, results, title, save_name 
): 
    plt.figure()
    fig, ax = plt.subplots()
    fig.set_figwidth(FIGSIZE)
    
    # results: object name to to success counts

    bar_keys = {
        'simple': 'simple', 
        'tac': 'tac', 
        'tac,viz': 'tac,viz',
    }
    bar_colors = {
        'simple': {
            'success': 'lightblue', 
            'find': 'lightsalmon'
        },
        'default': {
            'success': 'tab:blue', 
            'find': 'tab:orange'
        }
    }
    width = 0.18
    x = np.arange(len(results))
    multiplier = 0
    
    for typ, disp in bar_keys.items(): 
        offset = width * multiplier
        grasps = np.array([results[model][f'success-{typ}'] for model in results]) 
        finds = np.array([results[model][f'find-{typ}'] for model in results])
        # print(typ, grasps, finds)
        grasp_colors = [bar_colors.get(typ, bar_colors['default'])['success'] for _ in range(len(grasps))] 
        find_colors = [bar_colors.get(typ, bar_colors['default'])['find'] for _ in range(len(grasps))] 


        rects = ax.bar(x + offset, grasps, width, color=grasp_colors, edgecolor='black')
        rects = ax.bar(x + offset, finds, width, bottom=grasps, color=find_colors, edgecolor='black')
        
        ax.bar_label(rects, labels=[disp for _ in range(len(grasps))], rotation=90, padding=3)
        multiplier += 1 

    ax.set_title(f'{title}\n')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_xticks(
        x + width, 
        display_names
    )
    # ax.axhline(10, linestyle='--')
    # ax.set_xticks(x + width, model_names)
    ax.set_yticks([i for i in range(11) if i % 2 == 0])
    plt.savefig(save_name)
    
def plot_ball_test(
    log_dir='/home/josh/octo_digit/eval/eval_logs'
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    keep_keys = {
        "good_pod_1024" : 'contrastive',
        "josh_pod_gen_lang_b1024_20240830_012615": 'multi-head',
        "josh_pod_gen_lang_single_headb1024_20240830_060804": 'single head',
        'window3': 'window=3', 
        'combined_loss': 'combined loss',
        'rephrase_full_finetune': 'rephrase\nfull T5',
        'rephrase_finetune_small': 'rephrase\nT5 last layers',
        'rephrase_frozen': 'rephrase\nT5 frozen',
        'mae_asym_tac': 'mae tac-only\nasym',
        'mae_uniform_tac': 'mae tac-only\nuniform',
        # 'mae_viz_asym': 'mae combined\nasym',
        # 'mae_viz_uniform': 'mae combined\nuniform'
    }.keys()
    logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'gen', 'pod'}, keep_types={'yellow_ball_simple', 'yellow_block_tac', 'yellow_block_tacviz'})
    # logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'gen', 'pod'}, keep_types={'red_block_viz', 'red_block_tac'})
    compute_result_logs(logs)

    
    # for k, v in logs.items():
    #     print(k, v['total'])
    #     print('\n')
        
    model_names = set(log[0] for log in logs)
    
    display_names = {
        "good_pod_1024" : 'contrastive',
        "josh_pod_gen_lang_b1024_20240830_012615": 'multi-head',
        "josh_pod_gen_lang_single_headb1024_20240830_060804": 'single head',
        'window3': 'window=3', 
        'combined_loss': 'combined loss',
        'rephrase_full_finetune': 'rephrase\nfull T5',
        'rephrase_finetune_small': 'rephrase\nT5 last layers',
        'rephrase_frozen': 'rephrase\nT5 frozen',
        'mae_asym_tac': 'mae tac-only\nasym',
        'mae_uniform_tac': 'mae tac-only\nuniform',
        # 'mae_viz_asym': 'mae combined\nasym',
        # 'mae_viz_uniform': 'mae combined\nuniform'
    }
    
    def compute_dic(model_name): 
        rel_log_keys = [log for log in logs if log[0] == model_name]
        
        simple_key = [log for log in rel_log_keys if 'simple' in log[-1]][0]
        simple_res = logs[simple_key]['total']
        print(simple_res)
        
        tac_key = [log for log in rel_log_keys if log[-1].endswith('tac')][0]
        tac_res = logs[tac_key]['total']
        
        tacviz_key = [log for log in rel_log_keys if log[-1].endswith('tacviz')][0]
        tacviz_res = logs[tacviz_key]['total']
        # print(tacviz_res == tac_res)
        result = {
            'success-simple': simple_res['success'], 
            'find-simple': simple_res['find'], 
            'success-tac': tac_res['success'], 
            'find-tac': tac_res['find'],
            'success-tac,viz': tacviz_res['success'],
            'find-tac,viz': tacviz_res['find']
        }
        return result 
    
    
    full_results = {
        display_name: compute_dic(model_name) for model_name, display_name in display_names.items()
    }

    create_bar_graph_ball_test(display_names.values(), full_results, 'Generative test - ball', 'figs/gen_ball5.png')
    
    
    
def create_bar_graph_block_test(
    display_names, results, title, save_name 
): 
    plt.figure()
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    
    # results: object name to to success counts

    bar_keys = {
        'simple': 'simple', 
        'tac': 'tac', 
        # 'tac,viz': 'tac,viz',
    }
    bar_colors = {
        'simple': {
            'success': 'lightblue', 
            'find': 'lightsalmon'
        },
        'default': {
            'success': 'tab:blue', 
            'find': 'tab:orange'
        }
    }
    width = 0.18
    x = np.arange(len(results))
    multiplier = 0
    
    for typ, disp in bar_keys.items(): 
        offset = width * multiplier
        grasps = np.array([results[model][f'success-{typ}'] for model in results]) 
        finds = np.array([results[model][f'find-{typ}'] for model in results])
        # print(typ, grasps, finds)
        grasp_colors = [bar_colors.get(typ, bar_colors['default'])['success'] for _ in range(len(grasps))] 
        find_colors = [bar_colors.get(typ, bar_colors['default'])['find'] for _ in range(len(grasps))] 


        rects = ax.bar(x + offset, grasps, width, color=grasp_colors, edgecolor='black')
        rects = ax.bar(x + offset, finds, width, bottom=grasps, color=find_colors, edgecolor='black')
        
        ax.bar_label(rects, labels=[disp for _ in range(len(grasps))], rotation=90, padding=3)
        multiplier += 1 

    ax.set_title(f'{title}\n')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_xticks(
        x + width, 
        display_names
    )
    # ax.axhline(10, linestyle='--')
    # ax.set_xticks(x + width, model_names)
    ax.set_yticks([i for i in range(11) if i % 2 == 0])
    plt.savefig(save_name)
    
def plot_block_test(
    log_dir='/home/josh/octo_digit/eval/eval_logs'
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    keep_keys = {
        "good_pod_1024" : 'contrastive',
        "josh_pod_gen_lang_b1024_20240830_012615": 'multi-head',
        "josh_pod_gen_lang_single_headb1024_20240830_060804": 'single head',
        'window3': 'window=3', 
        'combined_loss': 'combined loss',
        'rephrase_full_finetune': 'rephrase\nfull T5',
        'rephrase_finetune_small': 'rephrase\nT5 last layers',
        'rephrase_frozen': 'rephrase\nT5 frozen',
        'mae_asym_tac': 'mae tac-only\nasym',
        'mae_uniform_tac': 'mae tac-only\nuniform',
        # 'mae_viz_asym': 'mae combined\nasym',
        # 'mae_viz_uniform': 'mae combined\nuniform'
    }
    # logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'gen', 'pod'}, keep_types={'yellow_ball_simple', 'yellow_block_tac', 'yellow_block_tacviz'})
    logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'gen', 'pod'}, keep_types={'red_block_viz', 'red_block_tac'})
    compute_result_logs(logs)

    
    # for k, v in logs.items():
    #     print(k, v['total'])
    #     print('\n')
        
    model_names = set(log[0] for log in logs)
    
    display_names = {
        "good_pod_1024" : 'contrastive',
        "josh_pod_gen_lang_b1024_20240830_012615": 'multi-head',
        "josh_pod_gen_lang_single_headb1024_20240830_060804": 'single head',
        'window3': 'window=3', 
        'combined_loss': 'combined loss',
        'rephrase_full_finetune': 'rephrase\nfull T5',
        'rephrase_finetune_small': 'rephrase\nT5 last layers',
        'rephrase_frozen': 'rephrase\nT5 frozen',
        'mae_asym_tac': 'mae tac-only\nasym',
        'mae_uniform_tac': 'mae tac-only\nuniform',
        # 'mae_viz_asym': 'mae combined\nasym',
        # 'mae_viz_uniform': 'mae combined\nuniform'
    }
    
    def compute_dic(model_name): 
        rel_log_keys = [log for log in logs if log[0] == model_name]
        
        simple_key = [log for log in rel_log_keys if 'viz' in log[-1]][0]
        simple_res = logs[simple_key]['total']
        print(simple_res)
        
        tac_key = [log for log in rel_log_keys if log[-1].endswith('tac')][0]
        tac_res = logs[tac_key]['total']
        
        # tacviz_key = [log for log in rel_log_keys if log[-1].endswith('tacviz')][0]
        # tacviz_res = logs[tacviz_key]['total']
        # print(tacviz_res == tac_res)
        result = {
            'success-simple': simple_res['success'], 
            'find-simple': simple_res['find'], 
            'success-tac': tac_res['success'], 
            'find-tac': tac_res['find'],
            # 'success-tac,viz': tacviz_res['success'],
            # 'find-tac,viz': tacviz_res['find']
        }
        return result 
    
    
    full_results = {
        display_name: compute_dic(model_name) for model_name, display_name in display_names.items()
    }

    create_bar_graph_block_test(display_names.values(), full_results, 'Generative test - block', 'figs/gen_block5.png')
    
def create_bar_graph_rope_test(
    display_names, results, title, save_name 
): 
    plt.figure()
    fig, ax = plt.subplots()
    fig.set_figwidth(20)
    
    # results: object name to to success counts

    bar_keys = {
        # 'simple': 'simple', 
        'tac': 'tac', 
        'tac,viz': 'tac,viz',
    }
    bar_colors = {
        'simple': {
            'success': 'lightblue', 
            'find': 'lightsalmon'
        },
        'default': {
            'success': 'tab:blue', 
            'find': 'tab:orange'
        }
    }
    width = 0.18
    x = np.arange(len(results))
    multiplier = 0
    
    for typ, disp in bar_keys.items(): 
        offset = width * multiplier
        grasps = np.array([results[model][f'success-{typ}'] for model in results]) 
        finds = np.array([results[model][f'find-{typ}'] for model in results])
        # print(typ, grasps, finds)
        grasp_colors = [bar_colors.get(typ, bar_colors['default'])['success'] for _ in range(len(grasps))] 
        find_colors = [bar_colors.get(typ, bar_colors['default'])['find'] for _ in range(len(grasps))] 


        rects = ax.bar(x + offset, grasps, width, color=grasp_colors, edgecolor='black')
        rects = ax.bar(x + offset, finds, width, bottom=grasps, color=find_colors, edgecolor='black')
        
        ax.bar_label(rects, labels=[disp for _ in range(len(grasps))], rotation=90, padding=3)
        multiplier += 1 

    ax.set_title(f'{title}\n')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_xticks(
        x + width, 
        display_names
    )
    # ax.axhline(10, linestyle='--')
    # ax.set_xticks(x + width, model_names)
    ax.set_yticks([i for i in range(11) if i % 2 == 0])
    plt.savefig(save_name)
    
def plot_rope_test(
    log_dir='/home/josh/octo_digit/eval/eval_logs'
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    keep_keys =  {
        "good_pod_1024" : 'contrastive',
        "josh_pod_gen_lang_b1024_20240830_012615": 'multi-head',
        "josh_pod_gen_lang_single_headb1024_20240830_060804": 'single head',
        'window3': 'window=3', 
        'combined_loss': 'combined loss',
        'rephrase_full_finetune': 'rephrase\nfull T5',
        'rephrase_finetune_small': 'rephrase\nT5 last layers',
        'rephrase_frozen': 'rephrase\nT5 frozen',
        'mae_asym_tac': 'mae tac-only\nasym',
        'mae_uniform_tac': 'mae tac-only\nuniform',
        'mae_viz_asym': 'mae combined\nasym',
        'mae_viz_uniform': 'mae combined\nuniform'
    }.keys()
    # logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'gen', 'pod'}, keep_types={'yellow_ball_simple', 'yellow_block_tac', 'yellow_block_tacviz'})
    logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'gen'}, keep_types={'green_rope_tac', 'green_rope_viztac'})
    compute_result_logs(logs)

    
    # for k, v in logs.items():
    #     print(k, v['total'])
    #     print('\n')
        
    model_names = set(log[0] for log in logs)
    
    display_names = {
        "good_pod_1024" : 'contrastive',
        "josh_pod_gen_lang_b1024_20240830_012615": 'multi-head',
        "josh_pod_gen_lang_single_headb1024_20240830_060804": 'single head',
        'window3': 'window=3', 
        'combined_loss': 'combined loss',
        'rephrase_full_finetune': 'rephrase\nfull T5',
        'rephrase_finetune_small': 'rephrase\nT5 last layers',
        'rephrase_frozen': 'rephrase\nT5 frozen',
        'mae_asym_tac': 'mae tac-only\nasym',
        'mae_uniform_tac': 'mae tac-only\nuniform',
        'mae_viz_asym': 'mae combined\nasym',
        'mae_viz_uniform': 'mae combined\nuniform'
    }
    
    def compute_dic(model_name): 
        rel_log_keys = [log for log in logs if log[0] == model_name]
        
        # simple_key = [log for log in rel_log_keys if 'viz' in log[-1]][0]
        # simple_res = logs[simple_key]['total']
        # print(simple_res)
        
        tac_key = [log for log in rel_log_keys if log[-1].endswith('tac') and not log[-1].endswith('viztac') ][0]
        tac_res = logs[tac_key]['total']
        
        tacviz_key = [log for log in rel_log_keys if log[-1].endswith('viztac')][0]
        tacviz_res = logs[tacviz_key]['total']
        # print(tacviz_res == tac_res)
        result = {
            # 'success-simple': simple_res['success'], 
            # 'find-simple': simple_res['find'], 
            'success-tac': tac_res['success'], 
            'find-tac': tac_res['find'],
            'success-tac,viz': tacviz_res['success'],
            'find-tac,viz': tacviz_res['find']
        }
        return result 
    
    
    full_results = {
        display_name: compute_dic(model_name) for model_name, display_name in display_names.items()
    }

    create_bar_graph_rope_test(display_names.values(), full_results, 'Generative test - rope', 'figs/gen_rope5.png')
    
def plot_duck_test(
    log_dir='/home/josh/octo_digit/eval/eval_logs'
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    keep_keys = {
        "good_pod_1024" : 'contrastive',
        "josh_pod_gen_lang_b1024_20240830_012615": 'multi-head',
        "josh_pod_gen_lang_single_headb1024_20240830_060804": 'single head',
        'window3': 'window=3', 
        'combined_loss': 'combined loss',
        'rephrase_full_finetune': 'rephrase\nfull T5',
        'rephrase_finetune_small': 'rephrase\nT5 last layers',
        'rephrase_frozen': 'rephrase\nT5 frozen',
        'mae_asym_tac': 'mae tac-only\nasym',
        'mae_uniform_tac': 'mae tac-only\nuniform',
        'mae_viz_asym': 'mae combined\nasym',
        'mae_viz_uniform': 'mae combined\nuniform'
    }.keys()
    # logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'gen', 'pod'}, keep_types={'yellow_ball_simple', 'yellow_block_tac', 'yellow_block_tacviz'})
    logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'gen'}, keep_types={'blue_duck_tac', 'blue_duck_viztac'})
    compute_result_logs(logs)

    
    # for k, v in logs.items():
    #     print(k, v['total'])
    #     print('\n')
        
    model_names = set(log[0] for log in logs)
    
    display_names = {
        "good_pod_1024" : 'contrastive',
        "josh_pod_gen_lang_b1024_20240830_012615": 'multi-head',
        "josh_pod_gen_lang_single_headb1024_20240830_060804": 'single head',
        'window3': 'window=3', 
        'combined_loss': 'combined loss',
        'rephrase_full_finetune': 'rephrase\nfull T5',
        'rephrase_finetune_small': 'rephrase\nT5 last layers',
        'rephrase_frozen': 'rephrase\nT5 frozen',
        'mae_asym_tac': 'mae tac-only\nasym',
        'mae_uniform_tac': 'mae tac-only\nuniform',
        'mae_viz_asym': 'mae combined\nasym',
        'mae_viz_uniform': 'mae combined\nuniform'
    }
    
    def compute_dic(model_name): 
        rel_log_keys = [log for log in logs if log[0] == model_name]
        
        # simple_key = [log for log in rel_log_keys if 'viz' in log[-1]][0]
        # simple_res = logs[simple_key]['total']
        # print(simple_res)
        
        tac_key = [log for log in rel_log_keys if log[-1].endswith('tac') and not log[-1].endswith('viztac')][0]
        tac_res = logs[tac_key]['total']
        
        tacviz_key = [log for log in rel_log_keys if log[-1].endswith('viztac')][0]
        tacviz_res = logs[tacviz_key]['total']
        # print(tacviz_res == tac_res)
        result = {
            # 'success-simple': simple_res['success'], 
            # 'find-simple': simple_res['find'], 
            'success-tac': tac_res['success'], 
            'find-tac': tac_res['find'],
            'success-tac,viz': tacviz_res['success'],
            'find-tac,viz': tacviz_res['find']
        }
        return result 
    
    
    full_results = {
        display_name: compute_dic(model_name) for model_name, display_name in display_names.items()
    }

    create_bar_graph_rope_test(display_names.values(), full_results, 'Generative test - duck', 'figs/gen_duck5.png')
    
    
    
def create_bar_graph_bag_test_1(
    display_names, results, title, save_name 
): 
    plt.figure()
    fig, ax = plt.subplots()
    fig.set_figwidth(FIGSIZE)
    
    # results: object name to to success counts

    bar_keys = {
        # 'simple': 'simple', 
        'tac': 'tac', 
        'tac,viz': 'tac,viz',
    }
    bar_colors = {
        'simple': {
            'success': 'lightblue', 
            'find': 'lightsalmon'
        },
        'default': {
            'success': 'tab:blue', 
            'find': 'tab:orange'
        }
    }
    width = 0.18
    x = np.arange(len(results))
    multiplier = 0
    
    for typ, disp in bar_keys.items(): 
        offset = width * multiplier
        grasps = np.array([results[model][f'success-{typ}'] for model in results]) 
        finds = np.array([results[model][f'find-{typ}'] for model in results])
        # print(typ, grasps, finds)
        grasp_colors = [bar_colors.get(typ, bar_colors['default'])['success'] for _ in range(len(grasps))] 
        find_colors = [bar_colors.get(typ, bar_colors['default'])['find'] for _ in range(len(grasps))] 


        rects = ax.bar(x + offset, grasps, width, color=grasp_colors, edgecolor='black')
        rects = ax.bar(x + offset, finds, width, bottom=grasps, color=find_colors, edgecolor='black')
        
        ax.bar_label(rects, labels=[disp for _ in range(len(grasps))], rotation=90, padding=3)
        multiplier += 1 

    ax.set_title(f'{title}\n')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_xticks(
        x + width, 
        display_names
    )
    # ax.axhline(10, linestyle='--')
    # ax.set_xticks(x + width, model_names)
    ax.set_yticks([i for i in range(11) if i % 2 == 0])
    plt.savefig(save_name)
    
def plot_bag_test_1(
    log_dir='/home/josh/octo_digit/eval/eval_logs', display_names=None, save_num=-1
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    if display_names is None:
        display_names = {
            'full_base_scratch': 'scratch',
            'full_base': 'base',
            'full_contrastive': 'contrastive',
            'full_generative': 'generative',
            'full_combined': 'combined', 
        }
    keep_keys = display_names.keys()
    # logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'gen', 'pod'}, keep_types={'yellow_ball_simple', 'yellow_block_tac', 'yellow_block_tacviz'})
    logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'bag_test_1'}, keep_types={'yellow_tac', 'yellow_viztac'})
    compute_result_logs(logs)

    
    # for k, v in logs.items():
    #     print(k, v['total'])
    #     print('\n')
        
    model_names = set(log[0] for log in logs)
    
    
    def compute_dic(model_name): 
        rel_log_keys = [log for log in logs if log[0] == model_name]
        
        # simple_key = [log for log in rel_log_keys if 'viz' in log[-1]][0]
        # simple_res = logs[simple_key]['total']
        # print(simple_res)
        try:
            tac_key = [log for log in rel_log_keys if log[-1].endswith('tac') and not log[-1].endswith('viztac') ][0]
        except:
            print(model_name)
            raise RuntimeError
        tac_res = logs[tac_key]['total']
        
        tacviz_key = [log for log in rel_log_keys if log[-1].endswith('viztac')][0]
        tacviz_res = logs[tacviz_key]['total']
        # print(tacviz_res == tac_res)
        result = {
            # 'success-simple': simple_res['success'], 
            # 'find-simple': simple_res['find'], 
            'success-tac': tac_res['success'], 
            'find-tac': tac_res['find'],
            'success-tac,viz': tacviz_res['success'],
            'find-tac,viz': tacviz_res['find']
        }
        return result 
    
    
    full_results = {
        display_name: compute_dic(model_name) for model_name, display_name in display_names.items()
    }

    if save_num == -1: 
        save_num += 1 
        while os.path.exists(f'figs/bag_test_1_vers_{save_num}.png'):
            save_num += 1 
    create_bar_graph_bag_test_1(display_names.values(), full_results, 'Bag test - stuffed animal', f'figs/bag_test_1_vers_{save_num}.png')
    
def plot_bag_test_2(
    log_dir='/home/josh/octo_digit/eval/eval_logs', display_names=None, save_num = -1, 
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    if display_names is None:
        display_names = {
            'full_base': 'base',
            'full_contrastive': 'contrastive',
            'full_generative': 'generative',
            'full_combined': 'combined', 
        }
    keep_keys = display_names.keys()
    # logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'gen', 'pod'}, keep_types={'yellow_ball_simple', 'yellow_block_tac', 'yellow_block_tacviz'})
    logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'bag_test_2'}, keep_types={'yellow_paper_tac', 'yellow_paper_viztac'})
    compute_result_logs(logs)

    
    # for k, v in logs.items():
    #     print(k, v['total'])
    #     print('\n')
        
    model_names = set(log[0] for log in logs)
    
    
    def compute_dic(model_name): 
        rel_log_keys = [log for log in logs if log[0] == model_name]
        
        # simple_key = [log for log in rel_log_keys if 'viz' in log[-1]][0]
        # simple_res = logs[simple_key]['total']
        # print(simple_res)
        try:
            tac_key = [log for log in rel_log_keys if log[-1].endswith('tac') and not log[-1].endswith('viztac') ][0]
        except:
            print(model_name)
            raise RuntimeError
        tac_res = logs[tac_key]['total']
        
        tacviz_key = [log for log in rel_log_keys if log[-1].endswith('viztac')][0]
        tacviz_res = logs[tacviz_key]['total']
        # print(tacviz_res == tac_res)
        result = {
            # 'success-simple': simple_res['success'], 
            # 'find-simple': simple_res['find'], 
            'success-tac': tac_res['success'], 
            'find-tac': tac_res['find'],
            'success-tac,viz': tacviz_res['success'],
            'find-tac,viz': tacviz_res['find']
        }
        return result 
    
    
    full_results = {
        display_name: compute_dic(model_name) for model_name, display_name in display_names.items()
    }

    if save_num == -1: 
        save_num += 1 
        while os.path.exists(f'figs/bag_test_2_vers_{save_num}.png'):
            save_num += 1 
            
    create_bar_graph_bag_test_1(display_names.values(), full_results, 'Bag test - paper', f'figs/bag_test_2_vers_{save_num}.png')



def create_bar_graph_bag_test_3(
    display_names, results, title, save_name 
): 
    plt.figure()
    fig, ax = plt.subplots()
    fig.set_figwidth(FIGSIZE)
    
    # results: object name to to success counts

    bar_keys = {
        'simple': 'simple', 
    }
    bar_colors = {
        'default': {
            'success': 'tab:blue', 
            'find': 'tab:orange'
        }
    }
    width = 0.18
    x = np.arange(len(results))
    multiplier = 0
    
    for typ, disp in bar_keys.items(): 
        offset = width * multiplier
        grasps = np.array([results[model][f'success-{typ}'] for model in results]) 
        finds = np.array([results[model][f'find-{typ}'] for model in results])
        # print(typ, grasps, finds)
        grasp_colors = [bar_colors.get(typ, bar_colors['default'])['success'] for _ in range(len(grasps))] 
        find_colors = [bar_colors.get(typ, bar_colors['default'])['find'] for _ in range(len(grasps))] 


        rects = ax.bar(x + offset, grasps, width, color=grasp_colors, edgecolor='black')
        rects = ax.bar(x + offset, finds, width, bottom=grasps, color=find_colors, edgecolor='black')
        
        ax.bar_label(rects, labels=[disp for _ in range(len(grasps))], rotation=90, padding=3)
        multiplier += 1 

    ax.set_title(f'{title}\n')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_xticks(
        x + width, 
        display_names
    )
    # ax.axhline(10, linestyle='--')
    # ax.set_xticks(x + width, model_names)
    ax.set_yticks([i for i in range(11) if i % 2 == 0])
    plt.savefig(save_name)
    
def plot_bag_test_3(
    log_dir='/home/josh/octo_digit/eval/eval_logs', display_names = None, save_num=-1,
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    if display_names is None:
        display_names = {
            'full_base': 'base',
            'full_contrastive': 'contrastive',
            'full_generative': 'generative',
            'full_combined': 'combined', 
        }
    keep_keys = display_names.keys()
    # logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'gen', 'pod'}, keep_types={'yellow_ball_simple', 'yellow_block_tac', 'yellow_block_tacviz'})
    logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'bag_test_3'}, keep_types={'red_block_simple'})
    compute_result_logs(logs)

    
    # for k, v in logs.items():
    #     print(k, v['total'])
    #     print('\n')
        
    model_names = set(log[0] for log in logs)
    
    
    def compute_dic(model_name): 
        rel_log_keys = [log for log in logs if log[0] == model_name]
        
        simple_key = [log for log in rel_log_keys if 'simple' in log[-1]][0]
        simple_res = logs[simple_key]['total']
        # print(simple_res)
       
        # print(tacviz_res == tac_res)
        result = {
            'success-simple': simple_res['success'], 
            'find-simple': simple_res['find'], 
            
        }
        return result 
    
    
    full_results = {
        display_name: compute_dic(model_name) for model_name, display_name in display_names.items()
    }
    if save_num == -1: 
        save_num += 1 
        while os.path.exists(f'figs/bag_test_3_vers_{save_num}.png'):
            save_num += 1 

    create_bar_graph_bag_test_3(display_names.values(), full_results, 'Bag test - block', f'figs/bag_test_3_vers_{save_num}.png')
    

def plot_bag_test_4(
    log_dir='/home/josh/octo_digit/eval/eval_logs', display_names=None, save_num=-1,
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    if display_names is None:
        display_names = {
            'full_base': 'base',
            'full_contrastive': 'contrastive',
            'full_generative': 'generative',
            'full_combined': 'combined', 
        }
    keep_keys = display_names.keys()
    # logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'gen', 'pod'}, keep_types={'yellow_ball_simple', 'yellow_block_tac', 'yellow_block_tacviz'})
    logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'bag_test_4'}, keep_types={'blue_duck_simple'})
    compute_result_logs(logs)

    
    # for k, v in logs.items():
    #     print(k, v['total'])
    #     print('\n')
        
    model_names = set(log[0] for log in logs)
    
    
    def compute_dic(model_name): 
        rel_log_keys = [log for log in logs if log[0] == model_name]
        
        simple_key = [log for log in rel_log_keys if 'simple' in log[-1]][0]
        simple_res = logs[simple_key]['total']
        # print(simple_res)
       
        # print(tacviz_res == tac_res)
        result = {
            'success-simple': simple_res['success'], 
            'find-simple': simple_res['find'], 
            
        }
        return result 
    
    
    full_results = {
        display_name: compute_dic(model_name) for model_name, display_name in display_names.items()
    }
    
    if save_num == -1: 
        save_num += 1 
        while os.path.exists(f'figs/bag_test_4_vers_{save_num}.png'):
            save_num += 1 

    create_bar_graph_bag_test_3(display_names.values(), full_results, 'Bag test - duck', f'figs/bag_test_4_vers_{save_num}.png')
    
    
    
def create_bar_graph_bag_average(
    display_names, results, title, save_name 
): 
    plt.figure()
    fig, ax = plt.subplots()
    fig.set_figwidth(FIGSIZE)
    
    # results: object name to to success counts

    bar_keys = {
        'average': ''
        # 'simple': 'simple', 
        # 'tac': 'tac', 
        # 'tac,viz': 'tac,viz',
    }
    bar_colors = {
        'simple': {
            'success': 'lightblue', 
            'find': 'lightsalmon'
        },
        'default': {
            'success': 'tab:blue', 
            'find': 'tab:orange'
        }
    }
    width = 0.18
    x = np.arange(len(results))
    multiplier = 0
    
    for typ, disp in bar_keys.items(): 
        offset = width * multiplier
        grasps = np.array([results[model][f'success-{typ}'] for model in results]) 
        finds = np.array([results[model][f'find-{typ}'] for model in results])
        # print(typ, grasps, finds)
        grasp_colors = [bar_colors.get(typ, bar_colors['default'])['success'] for _ in range(len(grasps))] 
        find_colors = [bar_colors.get(typ, bar_colors['default'])['find'] for _ in range(len(grasps))] 


        rects = ax.bar(x + offset, grasps, width, color=grasp_colors, edgecolor='black')
        rects = ax.bar(x + offset, finds, width, bottom=grasps, color=find_colors, edgecolor='black')
        
        ax.bar_label(rects, labels=[disp for _ in range(len(grasps))], rotation=90, padding=3)
        multiplier += 1 

    ax.set_title(f'{title}\n')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09))
    ax.set_ylim((0, 12))
    ax.set_xticks(
        x + width, 
        display_names
    )
    # ax.axhline(10, linestyle='--')
    # ax.set_xticks(x + width, model_names)
    ax.set_yticks([i for i in range(11) if i % 2 == 0])
    plt.savefig(save_name)
    
def plot_bag_average(
    log_dir='/home/josh/octo_digit/eval/eval_logs', display_names=None, save_num=-1
): 
    def filter_dict(dic, filter_fn): 
        filtered_dict = {}
        for k, v in dic.items(): 
            if filter_fn(k, v): 
                filtered_dict[k] = v
        return filtered_dict
    if display_names is None:
        display_names = {
            'full_base': 'base',
            'full_contrastive': 'contrastive',
            'full_generative': 'generative',
            'full_combined': 'combined', 
        }
    keep_keys = display_names.keys()
    # logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'gen', 'pod'}, keep_types={'yellow_ball_simple', 'yellow_block_tac', 'yellow_block_tacviz'})
    logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'bag_test_1'}, keep_types={'yellow_tac', 'yellow_viztac'})
    compute_result_logs(logs)
    
    def compute_dic_tacviz(model_name, logs=logs): 
        rel_log_keys = [log for log in logs if log[0] == model_name]
        
        # simple_key = [log for log in rel_log_keys if 'viz' in log[-1]][0]
        # simple_res = logs[simple_key]['total']
        # print(simple_res)
        
        tac_key = [log for log in rel_log_keys if log[-1].endswith('tac') and not log[-1].endswith('viztac') ][0]
        tac_res = logs[tac_key]['total']
        
        tacviz_key = [log for log in rel_log_keys if log[-1].endswith('viztac')][0]
        tacviz_res = logs[tacviz_key]['total']
        # print(tacviz_res == tac_res)
        result = {
            # 'success-simple': simple_res['success'], 
            # 'find-simple': simple_res['find'], 
            'success-tac': tac_res['success'], 
            'find-tac': tac_res['find'],
            'success-tac,viz': tacviz_res['success'],
            'find-tac,viz': tacviz_res['find']
        }
        return result 
    
    
    full_results = {
        display_name: compute_dic_tacviz(model_name) for model_name, display_name in display_names.items()
    }
    
    all_results = {1: full_results}
    
    logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'bag_test_2'})
    compute_result_logs(logs)
    full_results = {
        display_name: compute_dic_tacviz(model_name, logs=logs) for model_name, display_name in display_names.items()
    }
    all_results[2] = full_results
    
    
    def compute_dic_simple(model_name, logs=logs): 
        rel_log_keys = [log for log in logs if log[0] == model_name]
        
        simple_key = [log for log in rel_log_keys if 'simple' in log[-1]][0]
        simple_res = logs[simple_key]['total']
        # print(simple_res)
       
        # print(tacviz_res == tac_res)
        result = {
            'success-simple': simple_res['success'], 
            'find-simple': simple_res['find'], 
            
        }
        return result 
    
    logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'bag_test_3'})
    compute_result_logs(logs)
    full_results = {
        display_name: compute_dic_simple(model_name, logs=logs) for model_name, display_name in display_names.items()
    }
    all_results[3] = full_results
    
    
    logs = read_eval_logs(log_dir, read_log_func=read_log, keep_models=keep_keys, keep_exps={'bag_test_4'})
    compute_result_logs(logs)
    full_results = {
        display_name: compute_dic_simple(model_name, logs=logs) for model_name, display_name in display_names.items()
    }
    all_results[4] = full_results
    
    
    total_totals = {}
    for display in display_names.values():
        total_success = 0
        total_find = 0
        for exp in range(1, 5):
            rel_tot = all_results[exp][display]
            for k, v in rel_tot.items():
                if k.startswith('success'):
                    total_success += v
                elif k.startswith('find'):
                    total_find += v  
                else:
                    raise ValueError((v, display, exp))
        total_totals[display] = {'success-average': total_success * 1.0 / 6.0, 'find-average': total_find * 1.0/6.0 }
    
    # print(total_totals)
    
    
    
    
    # print(all_results)
    if save_num == -1:
        save_num = 0
        while os.path.exists(f'figs/bag_average_vers_{save_num}.png'): 
            save_num += 1
            

    create_bar_graph_bag_average(display_names.values(), total_totals, 'Bag test - average', f'figs/bag_average_vers_{save_num}.png')

if __name__ == '__main__': 
    # plot_dis3()
    # plot_dis2()
    # plot_pod_test()
    # plot_gen_test()
    # plot_rope_test()
    # plot_duck_test()
    # plot_ball_test()
    # plot_block_test()
    # plot_bag_test_4()
    display_names = {
            'full_base': 'base',
            'full_contrastive': 'contrastive',
            'full_generative': 'generative',
            'full_combined': 'combined', 
            'new_tac_uniform': 'mae',
            'frozen_combined': 'combined\nfrozen',
            'combined_channels': 'combined\nchannels',
            'josh_pod_final_combined_model_rephrase_cond_b1024_20240908_045441': 'rephrase cond\n50k',
            'josh_pod_final_combined_model_rephrase_full_b1024_20240908_220627': 'rephrase full\n50k',
            '100k_josh_pod_final_combined_model_rephrase_cond_b1024_20240908_045441': 'rephrase cond\n100k',
            '100k_josh_pod_final_combined_model_rephrase_full_b1024_20240908_220627': 'rephrase full\n100k',
            'josh_pod_final_combined_model_tvl_single_b1024_20240909_184554': 'final TVL\n50k',
            '100k_josh_pod_final_combined_model_tvl_single_b1024_20240909_184554': 'final TVL\n100k',
            'josh_pod_final_combined_model_tvl_single_rephrase_full_b1024_20240910_020337': 'rephrase 50%\n50k',
            '100k_josh_pod_final_combined_model_tvl_single_rephrase_full_b1024_20240910_020337': 'rephrase 50%\n100k',
            'josh_pod_final_combined_model_tvl_single_rephrase_full_combinations_b1024_20240910_220927': 'final\nfinal final'
            # 'tvl_finetuned_20240824_070132': 'TVL\nfinetune',
            # 'tvl_frozen_20240824_123443': 'TVL\nfrozen',
            # 't3_frozen_20240824_205857': 'T3\nfrozen',
            # 't3_finetune_20240826_000930': 'T3\nfinetuned',
            # 'vit_20240824_173949': 'ViT',
            # 'resnet_imagenet_20240824_174202': 'Resnet\nImagenet',
            # 'tac_base_20240824_173914': 'Base\nno tactile'
            
        }
    for i in range(4):
        eval(f'plot_bag_test_{i+1}')(display_names=display_names)
    plot_bag_average(display_names=display_names)
    