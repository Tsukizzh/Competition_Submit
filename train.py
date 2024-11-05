##############################
# 导入必要的库
################################
import argparse  # argparse用于处理命令行参数，让我们可以在运行程序时传入配置文件

# 从我们自己开发的tabtransformers包中导入核心功能
# 这些功能都定义在项目的tabtransformers目录下
from tabtransformers import (
    # 用于读取YAML格式的配置文件
    # 配置文件模板在 templates/ 目录下：
    # - 如果是分类问题，使用 templates/config_classification.yaml
    # - 如果是回归问题，使用 templates/config_regression.yaml
    load_config,
    
    # 用于创建模型实例
    # 模型定义在 tabtransformers/models/ 目录下
    # 支持两种模型：
    # 1. TabTransformer：适用于大多数表格数据
    # 2. FTTransformer：特征标记化转换器
    get_model,
    
    # 获取损失函数
    # 定义在 tabtransformers/loss_functions.py
    # 分类问题通常使用交叉熵损失
    # 回归问题通常使用均方误差损失
    get_loss_function,
    
    # 学习率调度器，用于在训练过程中调整学习率
    # 可以帮助模型更好地收敛
    get_lr_scheduler_object,
    
    # 优化器，用于更新模型参数
    # 常用的有：Adam, SGD等
    get_optimizer_object,
    
    # 获取评估指标
    # 定义在 tabtransformers/metrics.py
    # 分类问题常用：准确率、F1分数
    # 回归问题常用：MSE、MAE
    get_custom_metric,
)

# 从工具模块导入辅助函数
# 这些函数定义在 tabtransformers/tools.py
from tabtransformers.tools import (
    # 设置随机种子，确保实验可重复
    seed_everything, 
    
    # 核心训练函数
    train, 
    
    # 模型预测函数
    inference, 
    
    # 数据加载函数，会从data/目录读取数据：
    # - data/train.csv：训练数据
    # - data/test.csv：测试数据
    get_data, 
    
    # 创建数据集对象，处理特征编码
    get_dataset, 
    
    # 创建数据加载器，用于批量训练
    get_data_loader, 
    
    # 绘制训练过程的学习曲线
    plot_learning_curve, 
    
    # 生成预测结果文件
    to_submssion_csv
)

def main(args):
    """
    主函数：包含完整的模型训练和预测流程
    
    使用步骤：
    1. 准备数据集：
       - 将训练数据 train.csv 放在 data/ 目录
       - 将测试数据 test.csv 放在 data/ 目录
       
    2. 准备配置文件：
       - 分类问题：使用 templates/config_classification.yaml
       - 回归问题：使用 templates/config_regression.yaml
       
    3. 运行训练：
       python train.py --config_path templates/config_classification.yaml
    
    数据格式要求：
    - CSV格式文件
    - 训练集必须包含目标变量列
    - 特征可以是分类特征或数值特征
    - 可以包含一个索引列（可选）
    
    Args:
        args: 命令行参数对象，包含配置文件路径
    """
    
    ################################
    # 第1步：加载配置和数据
    ################################
    
    # 加载YAML格式的配置文件
    # 配置文件包含所有训练相关的参数：
    # - 模型参数：层数、维度等
    # - 训练参数：批量大小、学习率等
    # - 特征配置：哪些是分类特征，哪些是数值特征
    # - 优化器设置：使用什么优化器，参数是什么
    config = load_config(args.config_path)

    # 设置随机种子
    # 这样可以确保每次运行得到相同的结果
    # 对于实验的可重复性很重要
    seed_everything(config['seed'])

    ################################
    # 第2步：数据集划分和标签处理
    ################################
    
    # 配置验证集划分参数
    # 验证集用于在训练过程中评估模型性能
    # 例如配置文件中可以这样设置：
    # train_test_split:
    #   test_size: 0.2  # 使用20%的数据作为验证集
    #   stratify: true  # 分层采样，保持标签分布一致
    val_params = config['train_test_split']
    val_params['random_state'] = config['seed']  # 使用相同的随机种子确保可重复性

    # 加载并划分数据集
    # get_data函数会：
    # 1. 从data/目录读取train.csv和test.csv
    # 2. 将训练集划分出一部分作为验证集
    # 3. 返回三个pandas DataFrame：训练集、测试集、验证集
    train_data, test_data, val_data = \
        get_data('data',  # 数据目录
                split_val=True,  # 需要划分验证集
                val_params=val_params,  # 验证集划分参数
                index_col=config['index_col'])  # 索引列名，例如'id'
    
    ################################
    # 第3步：标签编码（针对分类问题）
    ################################
    
    # 如果配置文件中定义了标签映射，就需要将分类标签转换为数值
    # 例如在配置文件中：
    # target_mapping:
    #   'cat': 0
    #   'dog': 1
    if config['target_mapping'] is not None:
        # 创建反向映射字典，用于最后将预测结果转换回原始标签
        # 例如：{'cat': 0, 'dog': 1} -> {0: 'cat', 1: 'dog'}
        reverse_target_mapping = {v: k for k, v in config['target_mapping'].items()}
        
        # 将训练集的目标值转换为数值标签
        # 例如：'cat' -> 0, 'dog' -> 1
        train_data[config['target_name']] = train_data[config['target_name']].replace(config['target_mapping'])
        
        # 对验证集做同样的转换
        val_data[config['target_name']] = val_data[config['target_name']].replace(config['target_mapping'])
    else:
        reverse_target_mapping = None  # 如果是回归问题，不需要标签映射

    ################################
    # 第4步：创建数据集对象
    ################################
    
    # get_dataset函数会处理所有特征工程：
    # 1. 对分类特征进行编码（如one-hot编码）
    # 2. 对数值特征进行标准化处理
    # 3. 构建特征词表（用于分类特征）
    train_dataset, test_dataset, val_dataset = \
        get_dataset(
            train_data,   # 训练数据
            test_data,    # 测试数据
            val_data,     # 验证数据
            config['target_name'],  # 目标变量的列名，例如'label'
            config['model_kwargs']['output_dim'],  # 模型输出维度（分类问题为类别数）
            config['categorical_features'],  # 分类特征列表，例如['color', 'size']
            config['continuous_features']    # 连续特征列表，例如['age', 'weight']
        )
    
    ################################
    # 第5步：创建数据加载器
    ################################
    
    # DataLoader用于批量加载数据，支持：
    # 1. 数据打乱（提高训练效果）
    # 2. 批量处理（加速训练）
    # 3. 多线程加载（提高效率）
    train_loader, test_loader, val_loader = \
        get_data_loader(
            train_dataset, test_dataset, val_dataset,
            # 训练时的批量大小，例如256
            train_batch_size=config['train_batch_size'],
            # 预测时的批量大小，通常可以设置大一些
            inference_batch_size=config['eval_batch_size']
        )

    ################################
    # 第6步：初始化模型和优化器
    ################################
    
    # 创建模型实例
    # 模型结构在tabtransformers/models/目录下定义
    model = get_model(
        config,  # 配置参数
        vocabulary=train_dataset.get_vocabulary(),  # 特征词表
        num_continuous_features=len(config['continuous_features'])  # 连续特征数量
    )
    
    # 创建优化器实例
    # 优化器负责更新模型参数
    # 常用的优化器有：
    # - Adam：适应性较好，通常是默认选择
    # - SGD：最基础的随机梯度下降
    optimizer = get_optimizer_object(config)(
        model.parameters(),  # 模型参数
        **config['optim_kwargs']  # 优化器参数，如学习率
    )
    
    # 创建学习率调度器
    # 用于动态调整学习率，帮助模型更好地收敛
    scheduler = get_lr_scheduler_object(config)(
        optimizer,  # 优化器
        **config['lr_scheduler_kwargs']  # 调度器参数
    )

    ################################
    # 第7步：模型训练
    ################################
    
    # train函数执行实际的训练过程
    # 返回训练历史记录，包含每个epoch的损失值和评估指标
    train_history, val_history = train(
        model,  # 待训练的模型
        
        config['epochs'],  # 训练轮数
        # 例如：epochs: 100 表示训练100轮
        
        config['model_kwargs']['output_dim'],  # 模型输出维度
        # 分类问题：类别数量（如二分类为2）
        # 回归问题：通常为1
        
        train_loader,  # 训练数据加载器
        val_loader,    # 验证数据加载器
        
        optimizer,     # 优化器（如Adam）
        
        get_loss_function(config),  # 损失函数
        # 分类问题：通常使用交叉熵损失
        # 回归问题：通常使用均方误差损失
        
        scheduler=scheduler,  # 学习率调度器
        
        custom_metric=get_custom_metric(config),  # 自定义评估指标
        # 分类问题：如准确率(accuracy)、F1分数
        # 回归问题：如均方误差(MSE)、平均绝对误差(MAE)
        
        maximize=config['is_greater_better'],     # 是否最大化评估指标
        # True: 对于准确率、F1分数等指标
        # False: 对于损失值、错误率等指标
        
        # 是否使用自定义指标来调整学习率
        scheduler_custom_metric=config['is_greater_better'] and config['lr_scheduler_by_custom_metric'],
        
        early_stopping=config['early_stopping'],  # 是否启用早停
        # 当模型性能不再提升时提前结束训练
        # 可以防止过拟合
        
        early_stopping_patience=config['early_stopping_patience'],  
        # 早停耐心值：
        # 例如设置为10，表示如果连续10个epoch性能没有提升，就停止训练
        
        early_stopping_start_from=config['early_stopping_start_from']
        # 从第几轮开始启用早停
        # 例如设置为20，表示前20轮不启用早停
    )
    
    ################################
    # 第8步：绘制学习曲线
    ################################
    
    # 绘制训练过程中的学习曲线
    # 可以直观地看到模型训练的效果：
    # - 损失值的变化趋势
    # - 评估指标的变化趋势
    # - 是否存在过拟合
    plot_learning_curve(train_history, val_history)
    
    ################################
    # 第9步：模型预测
    ################################
    
    # 使用训练好的模型在测试集上进行预测
    predictions = inference(
        model,  # 训练好的模型
        test_loader,  # 测试数据加载器
        config['model_kwargs']['output_dim']  # 输出维度
    )
    
    # 如果之前进行了标签映射，需要将数值预测结果转换回原始标签
    # 例如：将[0, 1, 0]转换回['cat', 'dog', 'cat']
    if reverse_target_mapping is not None:
        predictions = [reverse_target_mapping[p] for p in predictions]
    
    ################################
    # 第10步：保存预测结果
    ################################
    
    # 如果在配置文件中指定了提交文件路径，就保存预测结果
    # submission_file: 'outputs/predictions.csv'
    if config['submission_file']:
        to_submssion_csv(
            predictions,  # 预测结果列表
            test_data,   # 测试集数据
            index_name=None,  # 索引列名（如果有）
            target_name=config['target_name'],  # 目标变量列名
            submission_path=config['submission_file']  # 保存路径
        )

################################
# 程序入口
################################

# 当直接运行这个脚本时（而不是作为模块导入时）
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    
    # 添加配置文件路径参数
    # 使用方式：python train.py --config_path templates/config_classification.yaml
    parser.add_argument(
        "--config_path",  # 参数名
        type=str,         # 参数类型：字符串
        required=True     # 必须提供此参数
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 运行主程序
    main(args)