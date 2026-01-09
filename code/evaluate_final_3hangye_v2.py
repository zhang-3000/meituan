#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAB属性评测脚本 V2 - 两阶段评测
第一阶段：调用LLM进行判断，保存所有LLM判断结果
第二阶段：读取保存的LLM结果，计算精确率、召回率、分类准确率
"""

import pandas as pd
import json
import os
import time
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime

# ============================================================================
# 配置区域
# ============================================================================

# 输入文件路径（过滤模块输出的文件）
INPUT_FILE_PATH = "/home/sankuai/dolphinfs_zhangyuntao06/daily_January/1.6/code/results/3_hangye_sft_V3_filtered.xlsx"

# 输出目录
OUTPUT_DIR = "../results/evaluation"

# 输出文件前缀
OUTPUT_PREFIX = "fab_evaluation_3_hangye"

# 指定需要统计的行业
TARGET_INDUSTRIES = ["健身中心", "台球", "运动培训"]

# 是否跳过第一阶段（仅使用已保存的LLM结果计算指标）
SKIP_LLM_STAGE = False

# ============================================================================

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="1871844672277114930",
    base_url="https://aigc.sankuai.com/v1/openai/native"
)

# 频率控制
last_request_time = 0
request_interval = 1.0

# LLM 评判 prompt
JUDGE_PROMPT = """你是一名算法专家，你需要完成两个任务：
1. 判定模型预测的结果是否有与人工标注的结果中存在相同的含义的短语，若存在相同含义结果则输出"是"，否则输出"否"。
2. 判定模型预测的结果是主观属性还是客观属性，若为客观属性则输出"客观"，若为主观属性则输出"主观"。

【主客观属性定义】
1. 客观属性：是指产品具备的客观物理属性，是人们对商品进行辨识的信息因素，比如产品的品牌、材料、工艺、尺寸、颜色等。
2. 主观属性：是将客观物理属性提炼为产品优点或者作用，不同用户对于该属性的解读拥有千人千面的主观解读。

【输出格式】
输出为JSON格式，格式为{"judge_1": "是/否", "judge_2": "主观/客观"}

下面开始执行任务：
模型预测结果：%s
人工标注结果：%s
输出：
"""


def llm_check(predict_text, label_text, model="gpt-4.1", max_retries=10):
    """
    调用 LLM 进行相似度和属性分类判定
    如果遇到速率限制，会自动等待后重试

    Args:
        predict_text: 模型预测结果
        label_text: 人工标注结果
        model: 使用的模型
        max_retries: 最大重试次数

    Returns:
        LLM 的判定结果 (JSON字符串)
    """
    global last_request_time

    for retry in range(max_retries):
        try:
            # 频率控制
            current_time = time.time()
            time_since_last = current_time - last_request_time
            if time_since_last < request_interval:
                sleep_time = request_interval - time_since_last
                time.sleep(sleep_time)

            last_request_time = time.time()

            prompt = JUDGE_PROMPT % (predict_text, label_text)
            result = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                stream=False
            )
            return result.choices[0].message.content
        except Exception as e:
            error_msg = str(e)

            if retry < max_retries - 1:
                # 检查是否是速率限制错误
                if '429' in error_msg or '频率' in error_msg or '限制' in error_msg:
                    wait_time = 2  # 速率限制时等待2秒
                    tqdm.write(f"⚠️  【速率限制】等待 {wait_time}s 后重试... (第 {retry + 1}/{max_retries} 次)")
                else:
                    wait_time = 2   # 其他错误等待2秒
                    tqdm.write(f"⚠️  【调用失败】等待 {wait_time}s 后重试... (第 {retry + 1}/{max_retries} 次)")

                time.sleep(wait_time)
            else:
                tqdm.write(f"✗ 【调用失败】达到最大重试次数 ({max_retries})")
                return 'error'


def parse_llm_result(llm_response):
    """
    解析 LLM 的返回结果

    Args:
        llm_response: LLM 返回的字符串

    Returns:
        (judge_1, judge_2) 元组，其中 judge_1 为"是"或"否"，judge_2 为"主观"或"客观"
    """
    try:
        if llm_response == 'error' or not llm_response:
            return None, None

        # 尝试解析 JSON
        if '{' in llm_response and '}' in llm_response:
            json_str = llm_response[llm_response.find('{'):llm_response.rfind('}')+1]
            result = json.loads(json_str)
            judge_1 = result.get('judge_1', '')
            judge_2 = result.get('judge_2', '')
            return judge_1, judge_2
        else:
            # 如果不是JSON，直接检查关键词
            judge_1 = '是' if '是' in llm_response else '否'
            judge_2 = '主观' if '主观' in llm_response else '客观'
            return judge_1, judge_2
    except Exception as e:
        return None, None


def parse_f_attribute_string(f_str):
    """
    解析 F 属性（字符串格式 - 用于人工标注数据）

    Args:
        f_str: F 属性字符串

    Returns:
        属性值列表
    """
    try:
        if pd.isna(f_str) or not f_str or f_str == '无' or f_str == '' or f_str == 'nan':
            return []
        f_str = str(f_str).strip()
        items = [item.strip() for item in f_str.replace('，', ',').split(',')]
        result = []
        for item in items:
            if item and item != '无' and item != 'nan':
                if '：' in item:
                    result.append(item.split('：')[1].strip())
                elif ':' in item:
                    result.append(item.split(':')[1].strip())
                else:
                    result.append(item)
        return result
    except Exception as e:
        return []


def parse_pred_f_attribute(f_str):
    """
    解析预测的 F 属性（冒号分隔格式）

    Args:
        f_str: F 属性字符串（格式：key:value,key:value,...）

    Returns:
        属性值列表
    """
    try:
        if pd.isna(f_str) or not f_str or f_str == '无' or f_str == '' or f_str == 'nan':
            return []
        f_str = str(f_str).strip()
        items = [item.strip() for item in f_str.replace('，', ',').split(',')]
        result = []
        for item in items:
            if item and item != '无' and item != 'nan':
                if ':' in item:
                    result.append(item.split(':')[1].strip())
                else:
                    result.append(item)
        return result
    except Exception as e:
        return []


def parse_ab_attribute(ab_str):
    """
    解析 A/B 属性（逗号分隔格式）

    Args:
        ab_str: A/B 属性字符串

    Returns:
        属性值列表
    """
    try:
        if pd.isna(ab_str) or not ab_str or ab_str == '无' or ab_str == '' or ab_str == 'nan':
            return []
        ab_str = str(ab_str).replace('，', ',')
        return [item.strip() for item in ab_str.split(',') if item.strip() and item.strip() != 'nan']
    except Exception as e:
        return []


def load_data():
    """加载数据文件"""
    print("\n加载评测数据...")
    print("-" * 80)

    if not INPUT_FILE_PATH:
        print("✗ 错误: INPUT_FILE_PATH 未指定")
        return None

    if not os.path.exists(INPUT_FILE_PATH):
        print(f"✗ 错误: 文件不存在")
        print(f"  指定的路径: {INPUT_FILE_PATH}")
        return None

    try:
        print(f"加载文件: {INPUT_FILE_PATH}")
        data = pd.read_excel(INPUT_FILE_PATH)
        print(f"✓ 数据加载成功，共 {len(data)} 条记录")
        print(f"列名: {list(data.columns)}")
        return data
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return None


def stage_1_llm_judgment(data):
    """
    第一阶段：调用LLM进行判断，保存所有结果
    单条数据失败时，在原地等待后重发，直到成功才处理下一条
    """
    print("\n" + "=" * 80)
    print("【第一阶段】LLM 判断阶段（单条数据本地重试）")
    print("=" * 80)
    print("\n初始化LLM判断列...")
    print("-" * 80)

    # 初始化新列用于保存LLM结果
    data['llm_judge_results'] = ''  # 保存所有LLM调用的结果
    data['llm_call_count'] = 0      # 记录每行调用LLM的次数

    llm_call_total = 0  # 全局LLM调用次数

    print(f"✓ 已初始化LLM判断列")

    print("\n开始LLM判断...")
    print("-" * 80)

    for idx in tqdm(range(len(data)), desc="LLM判断进度"):
        # 获取行业信息
        industry = str(data.iloc[idx].get('category', '未知')).strip()

        # 只处理指定行业的数据
        if industry not in TARGET_INDUSTRIES:
            continue

        # 解析标注数据（F、A、B）
        label_f_list = parse_f_attribute_string(data.iloc[idx]['F'])
        label_a_list = parse_ab_attribute(data.iloc[idx]['A'])
        label_b_list = parse_ab_attribute(data.iloc[idx]['B'])

        # 解析预测数据（pred_F、pred_A、pred_B）
        predict_f_list = parse_pred_f_attribute(data.iloc[idx]['pred_F'])
        predict_a_list = parse_ab_attribute(data.iloc[idx]['pred_A'])
        predict_b_list = parse_ab_attribute(data.iloc[idx]['pred_B'])

        # 如果标注和预测都为空，则跳过这一行
        if (not label_f_list and not label_a_list and not label_b_list and
            not predict_f_list and not predict_a_list and not predict_b_list):
            continue

        # 组合标注和预测的主客观属性
        zhuguan_label_list = label_a_list + label_b_list
        keguan_label_list = label_f_list
        zhuguan_predict_list = predict_a_list + predict_b_list
        keguan_predict_list = predict_f_list

        llm_results = []  # 保存本行的所有LLM结果
        llm_call_count = 0

        # 处理预测的 A 属性
        for a in predict_a_list:
            if a not in zhuguan_label_list and a not in keguan_label_list:
                # 需要LLM判定
                res = llm_check(a, str(zhuguan_label_list + keguan_label_list))
                llm_call_count += 1
                llm_call_total += 1
                llm_results.append({
                    'type': 'predict_a',
                    'value': a,
                    'llm_response': res
                })

        # 处理预测的 B 属性
        for b in predict_b_list:
            if b not in zhuguan_label_list and b not in keguan_label_list:
                # 需要LLM判定
                res = llm_check(b, str(zhuguan_label_list + keguan_label_list))
                llm_call_count += 1
                llm_call_total += 1
                llm_results.append({
                    'type': 'predict_b',
                    'value': b,
                    'llm_response': res
                })

        # 处理预测的 F 属性
        for f in predict_f_list:
            if f not in zhuguan_label_list and f not in keguan_label_list:
                # 需要LLM判定
                res = llm_check(f, str(zhuguan_label_list + keguan_label_list))
                llm_call_count += 1
                llm_call_total += 1
                llm_results.append({
                    'type': 'predict_f',
                    'value': f,
                    'llm_response': res
                })

        # 处理标注的 A 属性（召回率）
        for a in label_a_list:
            if a not in zhuguan_predict_list and a not in keguan_predict_list:
                # 需要LLM判定
                res = llm_check(a, str(zhuguan_predict_list + keguan_predict_list))
                llm_call_count += 1
                llm_call_total += 1
                llm_results.append({
                    'type': 'label_a',
                    'value': a,
                    'llm_response': res
                })

        # 处理标注的 B 属性（召回率）
        for b in label_b_list:
            if b not in zhuguan_predict_list and b not in keguan_predict_list:
                # 需要LLM判定
                res = llm_check(b, str(zhuguan_predict_list + keguan_predict_list))
                llm_call_count += 1
                llm_call_total += 1
                llm_results.append({
                    'type': 'label_b',
                    'value': b,
                    'llm_response': res
                })

        # 处理标注的 F 属性（召回率）
        for f in label_f_list:
            if f not in zhuguan_predict_list and f not in keguan_predict_list:
                # 需要LLM判定
                res = llm_check(f, str(zhuguan_predict_list + keguan_predict_list))
                llm_call_count += 1
                llm_call_total += 1
                llm_results.append({
                    'type': 'label_f',
                    'value': f,
                    'llm_response': res
                })

        # 保存本行的LLM结果
        data.at[idx, 'llm_judge_results'] = json.dumps(llm_results, ensure_ascii=False)
        data.at[idx, 'llm_call_count'] = llm_call_count

    print("\n" + "=" * 80)
    print(f"✓ LLM判断完成！")
    print(f"  总调用次数: {llm_call_total}")
    print("=" * 80)

    return data, llm_call_total


def stage_2_calculate_metrics(data, llm_call_total):
    """
    第二阶段：读取保存的LLM结果，计算各项指标
    """
    print("\n" + "=" * 80)
    print("【第二阶段】指标计算阶段")
    print("=" * 80)

    print("\n初始化评测列...")
    print("-" * 80)

    data['precision_error'] = ''
    data['recall_error'] = ''
    data['classification_error'] = ''

    print(f"✓ 已初始化评测列")

    print("\n开始计算指标...")
    print("-" * 80)

    # 全局计数器
    jingque_c = 0        # 精确率分子：模型预测正确的数量
    jingque_c_all = 0    # 精确率分母：模型预测的总数量

    zhaohui_c = 0        # 召回率分子：模型预测正确的数量
    zhaohui_c_all = 0    # 召回率分母：人工标注的总数量

    fenlei_c = 0         # 分类准确率分子：分类正确的数量
    fenlei_c_all = 0     # 分类准确率分母：模型预测的总数量

    # 按行业统计的字典
    industry_stats = {}

    for idx in tqdm(range(len(data)), desc="指标计算进度"):
        jingque_error_list = []
        zhaohui_error_list = []
        fenlei_error_list = []

        # 获取行业信息
        industry = str(data.iloc[idx].get('category', '未知')).strip()

        # 只处理指定行业的数据
        if industry not in TARGET_INDUSTRIES:
            continue

        if industry not in industry_stats:
            industry_stats[industry] = {
                'jingque_c': 0,
                'jingque_c_all': 0,
                'zhaohui_c': 0,
                'zhaohui_c_all': 0,
                'fenlei_c': 0,
                'fenlei_c_all': 0,
                'count': 0
            }
        industry_stats[industry]['count'] += 1

        # 记录本行开始时的计数值
        row_jingque_c_start = jingque_c
        row_zhaohui_c_start = zhaohui_c
        row_fenlei_c_start = fenlei_c
        row_jingque_c_all_start = jingque_c_all
        row_zhaohui_c_all_start = zhaohui_c_all
        row_fenlei_c_all_start = fenlei_c_all

        # 解析标注数据（F、A、B）
        label_f_list = parse_f_attribute_string(data.iloc[idx]['F'])
        label_a_list = parse_ab_attribute(data.iloc[idx]['A'])
        label_b_list = parse_ab_attribute(data.iloc[idx]['B'])

        # 解析预测数据（pred_F、pred_A、pred_B）
        predict_f_list = parse_pred_f_attribute(data.iloc[idx]['pred_F'])
        predict_a_list = parse_ab_attribute(data.iloc[idx]['pred_A'])
        predict_b_list = parse_ab_attribute(data.iloc[idx]['pred_B'])

        # 如果标注和预测都为空，则跳过这一行
        if (not label_f_list and not label_a_list and not label_b_list and
            not predict_f_list and not predict_a_list and not predict_b_list):
            continue

        # 组合标注和预测的主客观属性
        zhuguan_label_list = label_a_list + label_b_list
        keguan_label_list = label_f_list
        zhuguan_predict_list = predict_a_list + predict_b_list
        keguan_predict_list = predict_f_list

        # 加载本行的LLM结果
        llm_results_json = data.iloc[idx]['llm_judge_results']
        llm_results = json.loads(llm_results_json) if llm_results_json else []
        llm_results_dict = {(r['type'], r['value']): r['llm_response'] for r in llm_results}

        # ===== 计算精确率 =====
        # 检查预测的 A 属性（精确率）
        for a in predict_a_list:
            if a in zhuguan_label_list:
                jingque_c += 1
                fenlei_c += 1
            elif a in keguan_label_list:
                jingque_c += 1
                fenlei_error_list.append(f"a,{a}")
            else:
                # 从已保存的LLM结果中获取
                res = llm_results_dict.get(('predict_a', a), 'error')
                if res != 'error' and '是' in res:
                    jingque_c += 1
                    if '主观' in res:
                        fenlei_c += 1
                    else:
                        fenlei_error_list.append(f"a,{a}")
                elif res == 'error':
                    jingque_error_list.append(f"a,{a}")

        # 检查预测的 B 属性（精确率）
        for b in predict_b_list:
            if b in zhuguan_label_list:
                jingque_c += 1
                fenlei_c += 1
            elif b in keguan_label_list:
                jingque_c += 1
                fenlei_error_list.append(f"b,{b}")
            else:
                res = llm_results_dict.get(('predict_b', b), 'error')
                if res != 'error' and '是' in res:
                    jingque_c += 1
                    if '主观' in res:
                        fenlei_c += 1
                    else:
                        fenlei_error_list.append(f"b,{b}")
                elif res == 'error':
                    jingque_error_list.append(f"b,{b}")

        # 检查预测的 F 属性（精确率）
        for f in predict_f_list:
            if f in zhuguan_label_list:
                jingque_c += 1
                fenlei_error_list.append(f"f,{f}")
            elif f in keguan_label_list:
                jingque_c += 1
                fenlei_c += 1
            else:
                res = llm_results_dict.get(('predict_f', f), 'error')
                if res != 'error' and '是' in res:
                    jingque_c += 1
                    if '客观' in res:
                        fenlei_c += 1
                    else:
                        fenlei_error_list.append(f"f,{f}")
                elif res == 'error':
                    jingque_error_list.append(f"f,{f}")

        # 精确率分母
        predict_total = len(predict_a_list) + len(predict_b_list) + len(predict_f_list)
        jingque_c_all += predict_total
        fenlei_c_all += predict_total

        # ===== 计算召回率 =====
        # 检查标注的 A 属性（召回率）
        for a in label_a_list:
            if a in zhuguan_predict_list:
                zhaohui_c += 1
            elif a in keguan_predict_list:
                zhaohui_c += 1
            else:
                res = llm_results_dict.get(('label_a', a), 'error')
                if res != 'error' and '是' in res:
                    zhaohui_c += 1
                else:
                    zhaohui_error_list.append(f"a,{a}")

        # 检查标注的 B 属性（召回率）
        for b in label_b_list:
            if b in zhuguan_predict_list:
                zhaohui_c += 1
            elif b in keguan_predict_list:
                zhaohui_c += 1
            else:
                res = llm_results_dict.get(('label_b', b), 'error')
                if res != 'error' and '是' in res:
                    zhaohui_c += 1
                else:
                    zhaohui_error_list.append(f"b,{b}")

        # 检查标注的 F 属性（召回率）
        for f in label_f_list:
            if f in zhuguan_predict_list:
                zhaohui_c += 1
            elif f in keguan_predict_list:
                zhaohui_c += 1
            else:
                res = llm_results_dict.get(('label_f', f), 'error')
                if res != 'error' and '是' in res:
                    zhaohui_c += 1
                else:
                    zhaohui_error_list.append(f"f,{f}")

        # 召回率分母
        label_total = len(label_a_list) + len(label_b_list) + len(label_f_list)
        zhaohui_c_all += label_total

        # 保存错误信息
        data.at[idx, 'precision_error'] = ';'.join(jingque_error_list)
        data.at[idx, 'recall_error'] = ';'.join(zhaohui_error_list)
        data.at[idx, 'classification_error'] = ';'.join(fenlei_error_list)

        # 统计该行对行业的贡献
        industry_stats[industry]['jingque_c'] += jingque_c - row_jingque_c_start
        industry_stats[industry]['jingque_c_all'] += jingque_c_all - row_jingque_c_all_start
        industry_stats[industry]['zhaohui_c'] += zhaohui_c - row_zhaohui_c_start
        industry_stats[industry]['zhaohui_c_all'] += zhaohui_c_all - row_zhaohui_c_all_start
        industry_stats[industry]['fenlei_c'] += fenlei_c - row_fenlei_c_start
        industry_stats[industry]['fenlei_c_all'] += fenlei_c_all - row_fenlei_c_all_start

    # 4. 计算最终指标
    print("\n" + "=" * 80)
    print("评测结果")
    print("=" * 80)

    # 按行业统计结果
    print("\n【按行业统计】")
    print("-" * 80)
    for industry in TARGET_INDUSTRIES:
        if industry in industry_stats:
            stats = industry_stats[industry]
            precision = stats['jingque_c'] / stats['jingque_c_all'] if stats['jingque_c_all'] > 0 else 0
            recall = stats['zhaohui_c'] / stats['zhaohui_c_all'] if stats['zhaohui_c_all'] > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            classification_acc = stats['fenlei_c'] / stats['fenlei_c_all'] if stats['fenlei_c_all'] > 0 else 0

            print(f"\n{industry} (共 {stats['count']} 条记录)")
            print(f"  精确率: {precision:.4f}  ({stats['jingque_c']}/{stats['jingque_c_all']})")
            print(f"  召回率: {recall:.4f}  ({stats['zhaohui_c']}/{stats['zhaohui_c_all']})")
            print(f"  F1分数: {f1:.4f}")
            print(f"  分类准确率: {classification_acc:.4f}  ({stats['fenlei_c']}/{stats['fenlei_c_all']})")

    # 全局统计结果（三行业总体）
    print("\n" + "=" * 80)
    print("【三行业总体统计】")
    print("-" * 80)

    precision = jingque_c / jingque_c_all if jingque_c_all > 0 else 0
    recall = zhaohui_c / zhaohui_c_all if zhaohui_c_all > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    classification_accuracy = fenlei_c / fenlei_c_all if fenlei_c_all > 0 else 0

    print(f"\n【精确率 (Precision)】")
    print(f"  数值: {precision:.4f}")
    print(f"  详情: {jingque_c} / {jingque_c_all}")

    print(f"\n【召回率 (Recall)】")
    print(f"  数值: {recall:.4f}")
    print(f"  详情: {zhaohui_c} / {zhaohui_c_all}")

    print(f"\n【F1 分数】")
    print(f"  数值: {f1_score:.4f}")

    print(f"\n【主客观属性分类准确率】")
    print(f"  数值: {classification_accuracy:.4f}")
    print(f"  详情: {fenlei_c} / {fenlei_c_all}")

    print(f"\n【LLM 调用统计】")
    print(f"  总调用次数: {llm_call_total}")

    return data, industry_stats, precision, recall, f1_score, classification_accuracy


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("FAB属性评测脚本 V2 - 两阶段评测（保存LLM结果）")
    print("=" * 80)

    # 1. 加载数据
    data = load_data()
    if data is None:
        return

    llm_call_total = 0

    # 2. 第一阶段：LLM 判断
    if not SKIP_LLM_STAGE:
        data, llm_call_total = stage_1_llm_judgment(data)
    else:
        print("\n⏭️  跳过第一阶段，使用已保存的LLM结果...")

    # 3. 第二阶段：计算指标
    data, industry_stats, precision, recall, f1_score, classification_accuracy = stage_2_calculate_metrics(data, llm_call_total)

    # 4. 保存评测结果
    print("\n" + "=" * 80)
    print("保存评测结果...")
    print("-" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 4.1 保存详细的 Excel 结果文件
    result_file = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{timestamp}.xlsx")
    data.to_excel(result_file, index=False)
    print(f"✓ 详细评测结果已保存到: {result_file}")

    # 4.2 保存 TXT 格式的简洁报告
    report_file = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FAB属性评测报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总记录数: {len(data)}\n")
        f.write(f"LLM调用总次数: {llm_call_total}\n\n")

        # 【按行业统计】
        f.write("=" * 80 + "\n")
        f.write("【按行业统计】\n")
        f.write("=" * 80 + "\n\n")

        for industry in TARGET_INDUSTRIES:
            if industry in industry_stats:
                stats = industry_stats[industry]
                precision_ind = stats['jingque_c'] / stats['jingque_c_all'] if stats['jingque_c_all'] > 0 else 0
                recall_ind = stats['zhaohui_c'] / stats['zhaohui_c_all'] if stats['zhaohui_c_all'] > 0 else 0
                f1_ind = 2 * (precision_ind * recall_ind) / (precision_ind + recall_ind) if (precision_ind + recall_ind) > 0 else 0
                classification_acc_ind = stats['fenlei_c'] / stats['fenlei_c_all'] if stats['fenlei_c_all'] > 0 else 0

                f.write(f"{industry} (共 {stats['count']} 条记录)\n\n")
                f.write(f"  精确率: {precision_ind:.4f}  ({stats['jingque_c']}/{stats['jingque_c_all']})\n")
                f.write(f"  召回率: {recall_ind:.4f}  ({stats['zhaohui_c']}/{stats['zhaohui_c_all']})\n")
                f.write(f"  F1分数: {f1_ind:.4f}\n")
                f.write(f"  分类准确率: {classification_acc_ind:.4f}  ({stats['fenlei_c']}/{stats['fenlei_c_all']})\n\n")

        # 【三行业总体统计】
        f.write("=" * 80 + "\n")
        f.write("【三行业总体统计】\n")
        f.write("=" * 80 + "\n\n")

        f.write("【精确率 (Precision)】\n")
        f.write(f"  数值: {precision:.4f}\n\n")

        f.write("【召回率 (Recall)】\n")
        f.write(f"  数值: {recall:.4f}\n\n")

        f.write("【F1 分数】\n")
        f.write(f"  数值: {f1_score:.4f}\n\n")

        f.write("【主客观属性分类准确率】\n")
        f.write(f"  数值: {classification_accuracy:.4f}\n\n")

        f.write("【LLM调用统计】\n")
        f.write(f"  总调用次数: {llm_call_total}\n\n")

        f.write("=" * 80 + "\n")

    print(f"✓ TXT 格式报告已保存到: {report_file}")

    # 5. 打印最终总结
    print("\n" + "=" * 80)
    print("✓✓✓ 评测完成! ✓✓✓")
    print("=" * 80)
    print(f"\n【输出文件位置】")
    print(f"  📊 详细结果: {result_file}")
    print(f"  📄 TXT 报告: {report_file}")
    print(f"  💾 LLM结果已保存在 Excel 的 'llm_judge_results' 列中")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
