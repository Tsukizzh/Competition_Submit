import pandas as pd  # 导入pandas库，用于数据处理和分析
import xml.etree.ElementTree as ET  # 导入ElementTree库，用于解析XML文件
import os  # 导入os库，用于文件和目录操作
import numpy as np  # 导入numpy库，用于高效的数组操作和数学计算
from tqdm import tqdm  # 导入tqdm库，用于在循环中显示进度条
from collections import defaultdict  # 导入defaultdict库，用于处理字典
import time

# 1. 读取并合并表格数据
def merge_tabular_data(data_dir):
    """
    这个函数的目的是读取并合并所有的表格数据。
    
    参数:
    data_dir: 包含表格数据的目录路径
    
    返回:
    merged_df: 合并后的DataFrame
    """
    print(f"尝试访问目录: {data_dir}")  # 打印正在尝试访问的目录，用于调试
    if not os.path.exists(data_dir):  # 检查目录是否存在
        print(f"错误：目录 {data_dir} 不存在")  # 如果目录不存在，打印错误信息
        return None  # 如果目录不存在，返回None

    try:  # 尝试执行以下代码块
        files = os.listdir(data_dir)  # 获取目录中的所有文件名
        print(f"目录 {data_dir} 中的文件: {files}")  # 打印目录中的文件列表，用于调试
    except Exception as e:  # 如果发生异常（例如权限问题）
        print(f"访问目录 {data_dir} 时出错: {str(e)}")  # 打印错误信息
        return None  # 如果出现错误，返回None

    # 检查是否存在预期的文件
    expected_files = ['NMR.xlsx', 'life_style.xlsx', 'ground_true.xlsx', 'Baseline_characteristics.xlsx']
    for file in expected_files:
        if file not in files:
            print(f"警告：未找到预期的文件 {file}")  # 如果缺少预期文件，打印警告
        else:
            print(f"成功读取文件：{file}")  # 如果找到预期文件，打印成功信息

    all_data = {}  # 初始化一个字典来存储所有数据
    for file in files:  # 遍历目录中的每个文件
        if file.endswith(".xlsx"):  # 如果文件是Excel文件
            file_path = os.path.join(data_dir, file)  # 构建完整的文件路径
            try:  # 尝试执行以下代码块
                df = pd.read_excel(file_path)  # 读取Excel文件到DataFrame
                all_data[file] = df  # 将数据框存储在字典中，以文件名为键
                print(f"文件 {file} 的形状: {df.shape}")  # 打印文件的行数和列数
                print(f"文件 {file} 的列: {df.columns.tolist()}")  # 打印文件的列名列表
            except Exception as e:  # 如果读取文件时发生异常
                print(f"读取文件 {file} 时出错: {str(e)}")  # 打印错误信息
    
    if not all_data:  # 如果没有成功读取任何数据
        print(f"警告：在 {data_dir} 中没有找到 XLSX 文件")  # 打印警告信息
        return None  # 返回None
    
    # 合并所有数据框
    merged_df = pd.concat(all_data.values(), axis=1, join='outer')
    
    # 检查并处理重复的 'f.eid' 列
    eid_columns = [col for col in merged_df.columns if col == 'f.eid']
    if len(eid_columns) > 1:
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated(keep='first')]
        print("注意：删除了重复的 'f.eid' 列")
    
    # 新增：重新排序列，将ground_true的内容放在f.eid后面
    if 'ground_true.xlsx' in all_data:
        ground_true_cols = [col for col in all_data['ground_true.xlsx'].columns if col != 'f.eid']
        other_cols = [col for col in merged_df.columns if col != 'f.eid' and col not in ground_true_cols]
        merged_df = merged_df[['f.eid'] + ground_true_cols + other_cols]
    
    merged_df['f.eid'] = merged_df['f.eid'].astype(int)
    print(f"合并后的DataFrame形状: {merged_df.shape}")  # 打印合并后的数据框的行数和列数
    print(f"合并后的DataFrame列: {merged_df.columns.tolist()}")  # 打印合并后的数据框的列名列表
    print(f"f.eid列的数据类型: {merged_df['f.eid'].dtype}")  # 打印 f.eid 列的数据类型
    return merged_df  # 返回合并后的DataFrame


# 2. 表格数据的预处理和特征工程
def preprocess_tabular_data(df):
    """处理合并数据集中的表格数据"""
    print("\n开始处理表格数据...")
    processed_df = df.copy()
    
    # 定义各数据源的特征列
    baseline_cols = ['f.31.0.0', 'f.34.0.0', 'f.21022.0.0', 'f.21001.0.0', 
                    'f.4079.0.0', 'f.4080.0.0']
    
    # 直接指定lifestyle特征列
    lifestyle_cols = [
        'f.2907.0.0', 'f.1478.0.0', 'f.1349.0.0', 'f.1100.0.0', 'f.2277.0.0',
        'f.2926.0.0', 'f.1628.0.0', 'f.1269.0.0', 'f.1558.0.0', 'f.1210.0.0',
        'f.1329.0.0', 'f.3466.0.0', 'f.1160.0.0', 'f.100024.0.0', 'f.2139.0.0',
        'f.1050.0.0', 'f.971.0.0', 'f.1110.0.0', 'f.2887.0.0', 'f.100025.0.0',
        'f.20160.0.0', 'f.1359.0.0', 'f.100009.0.0', 'f.924.0.0', 'f.1279.0.0',
        'f.1598.0.0', 'f.1548.0.0', 'f.943.0.0', 'f.1170.0.0', 'f.2110.0.0',
        'f.1200.0.0', 'f.1408.0.0', 'f.2644.0.0', 'f.1458.0.0', 'f.1369.0.0',
        'f.2867.0.0', 'f.3731.0.0', 'f.1120.0.0', 'f.100017.0.0', 'f.20077.0.0',
        'f.1060.0.0', 'f.874.0.0', 'f.1309.0.0', 'f.2237.0.0', 'f.1190.0.0',
        'f.1249.0.0', 'f.914.0.0', 'f.100015.0.0', 'f.1299.0.0', 'f.1578.0.0',
        'f.1130.0.0', 'f.100011.0.0', 'f.1239.0.0', 'f.981.0.0', 'f.1070.0.0',
        'f.100760.0.0', 'f.3456.0.0', 'f.100005.0.0', 'f.1220.0.0', 'f.1319.0.0',
        'f.3436.0.0', 'f.104400.0.0', 'f.2634.0.0', 'f.1289.0.0', 'f.1259.0.0',
        'f.1568.0.0', 'f.1618.0.0', 'f.2149.0.0'
    ]
    
    nmr_cols = [f'f.{i}.0.0' for i in range(23400, 23649)]
    
    print(f"开始处理前的特征数量:")
    print(f"- Baseline特征: {len(baseline_cols)}")
    print(f"- Lifestyle特征: {len(lifestyle_cols)}")
    print(f"- NMR特征: {len(nmr_cols)}")
    
    # 1. 处理Baseline特征
    print("\n处理Baseline特征...")
    for col in baseline_cols:
        if col in processed_df.columns:
            # 处理缺失值：用中位数填充
            if processed_df[col].isnull().any():
                median_val = processed_df[col].median()
                processed_df[col].fillna(median_val, inplace=True)
                print(f"- {col}: 用中位数 {median_val:.2f} 填充缺失值")
            
            # 处理异常值：IQR方法
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)
    
    # 2. 处理Lifestyle特征
    print("\n处理Lifestyle特征...")
    for col in lifestyle_cols:
        if col in processed_df.columns:
            # 将负值转换为绝对值
            processed_df[col] = processed_df[col].abs()
            
            # 计算缺失率
            missing_rate = processed_df[col].isnull().mean()
            print(f"特征 {col} 的缺失率: {missing_rate:.2%}")
            
            if missing_rate > 0.6:
                print(f"- 删除特征 {col}: 缺失率 {missing_rate:.2%}")
                processed_df.drop(columns=[col], inplace=True)
                continue
            
            # 对于缺失率<=60%的列，用该列的中位数填充缺失值
            if processed_df[col].isnull().any():
                median_val = processed_df[col].median()
                processed_df[col].fillna(median_val, inplace=True)
                print(f"- {col}: 用中位数 {median_val:.2f} 填充缺失值")
            
            # 处理异常值：IQR方法
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)
    
    # 3. 处理NMR特征
    print("\n处理NMR特征...")
    for col in nmr_cols:
        if col in processed_df.columns:
            if processed_df[col].isnull().any():
                # 获取25%到75%位置的值
                Q1 = processed_df[col].quantile(0.25)
                Q3 = processed_df[col].quantile(0.75)
                # 随机选择这个范围内的值来填充
                fill_values = np.random.uniform(Q1, Q3, size=processed_df[col].isnull().sum())
                processed_df.loc[processed_df[col].isnull(), col] = fill_values
                print(f"- {col}: 用[{Q1:.2f}, {Q3:.2f}]范围内的随机值填充缺失值")
            
            # 处理异常值：IQR方法
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)
    
    # 检查处理结果
    print("\n处理后的数据统计:")
    print(f"- 总样本数量: {len(processed_df)}")
    print(f"- 总特征数量: {len(processed_df.columns)}")
    
    return processed_df


# 3. 处理XML格式的心电数据
def extract_ecg_features(xml_file):
    """
    这个函数用于从单个XML文件中提取心电图特征。
    
    参数:
    xml_file: XML文件路径
    
    返回:
    features: 包含提取特征的字典
    """
    tree = ET.parse(xml_file)  # 解析XML文件
    root = tree.getroot()  # 获取XML的根元素
    
    features = {}  # 创建一个空字典，用于存储提取的特征
    
    # 提取波形数据
    waveform_data = []  # 创建一个空列表，用于存储波形数据
    for elem in root.iter('WaveformData'):  # 遍历所有WaveformData元素
        data = elem.text.split(',')  # 将波形数据字符串按逗号分割
        waveform_data.extend([int(x) for x in data])  # 将数据转换为整数并添加到列表中
    
    # 计算波形数据的统计特征
    if waveform_data:  # 如果波形数据不为空
        features['WaveformMean'] = np.mean(waveform_data)  # 计算波形数的平均值
        features['WaveformStd'] = np.std(waveform_data)  # 计算波形数据的标准差
        features['WaveformMax'] = np.max(waveform_data)  # 计算波形数据的最大值
        features['WaveformMin'] = np.min(waveform_data)  # 计算波形数据的最小值
    
    return features  # 返回包含特征的字典

def process_ecg_data(ecg_dir):
    """处理ECG数据，分别处理两组信号并进行基础预处理"""
    print(f"开始处理ECG文件夹：{ecg_dir}")
    print(f"ECG文件夹中的文件总数：{len(os.listdir(ecg_dir))}")
    ecg_data = []

    # 定义所有导联名称
    LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    # 定义特征名称
    FEATURE_NAMES = ['mean', 'std', 'max', 'min']
    # 定义信号类型
    SIGNAL_TYPES = ['filtered', 'dynamic']

    for file in tqdm(os.listdir(ecg_dir)):
        if file.endswith(".xml"):
            f_eid = file.split('_')[0]
            tree = ET.parse(os.path.join(ecg_dir, file))
            root = tree.getroot()
            
            # 存储两组信号
            leads_data = defaultdict(list)
            lead_count = defaultdict(int)
            
            # 分离两组信号
            for waveform in root.findall(".//WaveformData"):
                lead = waveform.get("lead")
                # 清理数据：移除空白字符，转换为数值
                data = np.array([int(x.strip()) for x in waveform.text.split(",")])
                leads_data[lead].append(data)
                lead_count[lead] += 1

            # 创建特征字典
            features = {"f.eid": int(f_eid)}
            
            # 处理每个导联的数据
            for lead in LEADS:
                if lead in leads_data and len(leads_data[lead]) == 2:
                    # 处理第一组信号（滤波信号）
                    signal1 = leads_data[lead][0]
                    features.update({
                        f"ECG_{lead}_filtered_mean": np.mean(signal1),
                        f"ECG_{lead}_filtered_std": np.std(signal1),
                        f"ECG_{lead}_filtered_max": np.max(signal1),
                        f"ECG_{lead}_filtered_min": np.min(signal1)
                    })
                    
                    # 处理第二组信号（动态信号）
                    signal2 = leads_data[lead][1]
                    features.update({
                        f"ECG_{lead}_dynamic_mean": np.mean(signal2),
                        f"ECG_{lead}_dynamic_std": np.std(signal2),
                        f"ECG_{lead}_dynamic_max": np.max(signal2),
                        f"ECG_{lead}_dynamic_min": np.min(signal2)
                    })
                else:
                    # 如果导联数据缺失，用0填充
                    for signal_type in SIGNAL_TYPES:
                        for feature in FEATURE_NAMES:
                            features[f"ECG_{lead}_{signal_type}_{feature}"] = 0

            ecg_data.append(features)

    print(f"处理完成。共处理了 {len(ecg_data)} 个ECG文件。")
    ecg_df = pd.DataFrame(ecg_data)

    # 处理异常值
    feature_cols = [col for col in ecg_df.columns if col.startswith('ECG_')]
    for col in feature_cols:
        # 将极端异常值替换为分位数值
        q1 = ecg_df[col].quantile(0.01)
        q3 = ecg_df[col].quantile(0.99)
        ecg_df[col] = ecg_df[col].clip(q1, q3)
        
        # 处理可能的无效值
        ecg_df[col] = ecg_df[col].replace([np.inf, -np.inf], np.nan)
        ecg_df[col] = ecg_df[col].fillna(ecg_df[col].median())

    # 打印处理后的信息
    print(f"\n处理后的ECG数据形状: {ecg_df.shape}")
    print(f"ECG数据中f.eid的唯一值数量: {ecg_df['f.eid'].nunique()}")
    
    # 打印列名示例
    print("\nECG特征列名示例:")
    for lead in LEADS[:2]:  # 只显示前两个导联的示例
        for signal_type in SIGNAL_TYPES:
            for feature in FEATURE_NAMES:
                print(f"ECG_{lead}_{signal_type}_{feature}")

    return ecg_df


# 4. 数据对齐和合并
def check_output_files(directory):
    # 根据目录是训练集还是测试集确定预期文件列
    expected_files = ['train_preprocessed_tabular.csv', 'train_preprocessed_ecg.csv'] if 'train' in directory else ['test_preprocessed_tabular.csv', 'test_preprocessed_ecg.csv']
    
    # 获取目录中所有以.csv结尾的文件列表
    found_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    
    # 检查是否所有预期文件都存在
    for file in expected_files:
        if file not in found_files:
            print(f"警告：未找到预期文件 {file} 在 {directory}")  # 如果预期文件不存在，打印警告
    
    # 检查是否有意外的文件
    for file in found_files:
        if file not in expected_files:
            print(f"警告：发现意外的文件 {file} 在 {directory}")  # 如果发现意外文件，打印警告
    
    # 如果找到的文件集合与预期文件集合相同，则打印成功信息
    if set(found_files) == set(expected_files):
        print(f"在 {directory} 中找到了所有预期的文件。")

def align_and_merge_data(tabular_file, ecg_file, output_file):
    """合并预处理后的表格数据和ECG数据，只保留有ECG数据的患者"""
    print("始数据对齐与合并...")
    
    # 读取数据
    tabular_data = pd.read_csv(tabular_file)
    ecg_data = pd.read_csv(ecg_file)
    
    print(f"原始表格数据形状: {tabular_data.shape}")
    print(f"ECG数据形状: {ecg_data.shape}")
    
    # 获取有ECG数据的患者ID列表
    ecg_patients = set(ecg_data['f.eid'])
    
    # 找出没有ECG数据的患者
    missing_ecg = set(tabular_data['f.eid']) - ecg_patients
    print(f"\n缺少ECG数据的患者数量: {len(missing_ecg)}")
    
    # 只保留有ECG数据的患者的表格数据
    tabular_data_filtered = tabular_data[tabular_data['f.eid'].isin(ecg_patients)]
    print(f"过滤后的表格数据形状: {tabular_data_filtered.shape}")
    
    # 合并数据
    merged_data = pd.merge(tabular_data_filtered, ecg_data, on='f.eid', how='inner')
    print(f"合并后的数据形状: {merged_data.shape}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存合并后的数据
    merged_data.to_csv(output_file, index=False)
    print(f"合并后的数据已保存到: {output_file}")
    
    return merged_data

# 主函数
def main():
    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的目录
    base_dir = os.path.abspath(os.path.join(script_dir, "..", "competition_data", "CG2405 2"))  # 设置基础目录
    train_dir = os.path.join(base_dir, 'train')  # 设置训练数据目录
    test_dir = os.path.join(base_dir, 'test')  # 设置测试数据目录

    # 删除多余的文件
    for dir_path in [train_dir, test_dir]:
        extra_file = os.path.join(dir_path, 'preprocessed_data.csv')
        if os.path.exists(extra_file):
            os.remove(extra_file)  # 如果存在多余的文件，删除它
            print(f"已删除多余文件：{extra_file}")

    # 检查目录是否存在
    if not os.path.exists(base_dir):
        print(f"错误：础目录 {base_dir} 不存在")
        return

    if not os.path.exists(train_dir):
        print(f"错误：训练数据目录 {train_dir} 不存在")
        return

    if not os.path.exists(test_dir):
        print(f"错误：测试数据目录 {test_dir} 不存在")
        return

    # 处理训练数据
    train_tabular_data = merge_tabular_data(train_dir)  # 合并训练集的表格数据
    if train_tabular_data is None:
        print("错误：无法处理训练表格数据")
        return

    train_ecg_dir = os.path.join(train_dir, 'ecg')  # 设置训练集ECG数据目录
    if not os.path.exists(train_ecg_dir):
        print(f"错误：训练ECG数据目录 {train_ecg_dir} 不存在")
        return

    train_ecg_data = process_ecg_data(train_ecg_dir)  # 处理训练集的ECG数据

    # 处理测试数据
    test_tabular_data = merge_tabular_data(test_dir)  # 合并测试集的表格数据
    if test_tabular_data is None:
        print("错误：无法处理测试表格数据")
        return

    test_ecg_dir = os.path.join(test_dir, 'ecg')  # 设置测试集ECG数据目录
    if not os.path.exists(test_ecg_dir):
        print(f"错误：测试ECG数据目录 {test_ecg_dir} 不存在")
        return

    test_ecg_data = process_ecg_data(test_ecg_dir)  # 处理测试集的ECG数据

    # 保存预处理后的训练数据
    train_tabular_data.to_csv(os.path.join(train_dir, 'train_preprocessed_tabular.csv'), index=False)
    train_ecg_data.to_csv(os.path.join(train_dir, 'train_preprocessed_ecg.csv'), index=False)

    # 保存预处理后的测试数据
    test_tabular_data.to_csv(os.path.join(test_dir, 'test_preprocessed_tabular.csv'), index=False)
    test_ecg_data.to_csv(os.path.join(test_dir, 'test_preprocessed_ecg.csv'), index=False)

    # 创建最终输出目录
    final_train_dir = os.path.join(script_dir, "..", "data", "train")
    final_test_dir = os.path.join(script_dir, "..", "data", "test")
    os.makedirs(final_train_dir, exist_ok=True)
    os.makedirs(final_test_dir, exist_ok=True)

    # 合并并保存最终的训练数据
    print("\n处理训练集数据...")
    train_merged = align_and_merge_data(
        os.path.join(train_dir, 'train_preprocessed_tabular.csv'),
        os.path.join(train_dir, 'train_preprocessed_ecg.csv'),
        os.path.join(final_train_dir, 'train_merged.csv')
    )
    
    # 合并并保存最终的测试数据
    print("\n处理测试集数据...")
    test_merged = align_and_merge_data(
        os.path.join(test_dir, 'test_preprocessed_tabular.csv'),
        os.path.join(test_dir, 'test_preprocessed_ecg.csv'),
        os.path.join(final_test_dir, 'test_merged.csv')
    )

    print("\n数据处理完成！")
    print(f"最终训练数据保存在: {os.path.join(final_train_dir, 'train_merged.csv')}")
    print(f"最终测试数据保存在: {os.path.join(final_test_dir, 'test_merged.csv')}")

    # 在最后添加表格数据的预处理步骤
    print("\n开始对合并后的数据进行表格数据预处理...")
    
    # 处理训练集
    train_merged_path = os.path.join(final_train_dir, 'train_merged.csv')
    train_merged = pd.read_csv(train_merged_path)
    train_processed = preprocess_tabular_data(train_merged)
    train_processed.to_csv(os.path.join(final_train_dir, 'train_merged.csv'), index=False)
    
    # 处理测试集
    test_merged_path = os.path.join(final_test_dir, 'test_merged.csv')
    test_merged = pd.read_csv(test_merged_path)
    test_processed = preprocess_tabular_data(test_merged)
    test_processed.to_csv(os.path.join(final_test_dir, 'test_merged.csv'), index=False)
    
    print("\n全部数据处理完成！")
    print(f"最终处理后的训练数据保存在: {os.path.join(final_train_dir, 'train_merged.csv')}")
    print(f"最终处理后的测试数据保存在: {os.path.join(final_test_dir, 'test_merged.csv')}")

if __name__ == '__main__':
    start_time = time.time()
    main()  # 如果这个脚本是直接运行的（而不是被导入的），则执行main函数
    end_time = time.time()
    print(f"脚本运行时间: {end_time - start_time:.2f}秒")
