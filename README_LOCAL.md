# PriCDR 本地运行与论文导读

代码来源：https://github.com/TiliaceaeSU/PriCDR

检出提交：f2e657677f283333f4107107684275c7d5797453。
原作者五个 Python 文件保持原样；新增 run_local.py、运行说明和依赖记录。
当前阶段验证的是作者样例的执行流程。样例结果不能直接当作论文表格复现结果。

## 在本机运行

打开 PowerShell，执行：

```powershell
cd D:\毕设论文\PriCDR
python run_local.py
```

默认使用 CPU、2 个计算线程、样例数据、1 轮训练、batch=16、embedding=32、JLT。
本机 python 指向 Python 3.14.3，已有 PyTorch 2.12.0+cpu 及所需依赖。
不需要安装 CUDA，也不需要复制原 README 的 Linux 环境变量命令。

每次产生独立的 logs/local-时间戳/，包括 console.txt、run.json、参数子目录中的 tmp.txt 和三个最佳模型权重。
console.txt 保存完整输出，run.json 保存解释器、种子、参数和提交号。脚本会在导入原版代码后初始化随机种子；原 main.py 虽声明 --seed，却没有实际设置它。

增加训练轮数或试验另一条发布分支：

```powershell
python run_local.py --epochs 10
python run_local.py --mode FJLT --epochs 1
python run_local.py --help
```

10 轮也仅用于学习流程；收敛和论文复现需要后续评估。FJLT 是仓库自己的分支名，不能直接等同于论文完整 SJLT。

2026-10-31 本机验证：原 main.py 的 JLT 运行，以及 run_local.py 的 JLT/FJLT 运行，均完成1轮训练、验证和最终测试，退出码0。启动脚本的 JLT 测试 HR@10=0.2273，FJLT 测试 HR@10=0.2316；这些只是流程检查结果，不支持方法优劣结论。

若换电脑缺依赖，可使用隔离环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe run_local.py
```

requirements-local.txt 仅列直接依赖；本机实际版本在 LOCAL_ENVIRONMENT.txt。新环境安装最新版本仍须重新验证。
已有环境的用户直接运行即可。大样本训练前先测耗时和内存，原代码会构造稠密交互矩阵和执行 SVD。

## 看懂输出

- preparing data：读取仓库样例，划分训练交互。
- (924, 3916) 与 (924, 4228)：924 个共享用户，两个域各自的物品数。代码命名为 movie 和 book，不能据变量名断定原始样例的真实品类；需核对数据来源。
- subspace dim is 204：本次发布后的特征维度。每位用户从 3916 维变为 204 维。
- epoch:0, self loss：三个损失之和，代码中的名字不代表它只有自编码器损失。
- 训练循环中打印 test: book：实际上调用 mode='val'，是验证集结果。
- Test TopK:5 / 10：训练结束后的测试集 HR、NDCG、MRR。
- loss 数值较大与代码按元素求和有关，不可直接与其他数据规模或 batch 下的数值比较。

## 第一次读论文的顺序

先读摘要、Introduction 的 Problem definition、图 1、第 4.4 节、第 6.1 节。
再读第 4.3 节和 Algorithm 1，然后补第 3 节定义，最后读第 5 节与附录证明。

第一遍回答五个问题：

1. 谁向谁推荐？用源域的交互帮助目标域推荐。
2. 为什么需要源域？目标域行为稀疏，其他域能提供偏好线索。
3. 通过什么把人对应起来？论文假设两个域共享同一批用户，物品不同。
4. 为什么不直接共享数据？论文考虑源域交互隐私。
5. 为什么是异构网络？发布后的源域是稠密随机特征，目标域仍是原始稀疏交互。

论文聚焦这组假设，不能直接推断为解决了无重叠用户或完全没有行为的新用户推荐。

## 三个核心损失

源域重建：把发布后的用户向量压缩为表示，再还原该向量。最小化还原误差 L_rec。

目标域推荐：由目标域用户交互和物品交互生成表示，对用户可能喜欢的物品打分，形成 L_reg。

用户对齐：同一个人在两个域的表示应能共享部分偏好，拉近对应表示，形成 L_ali。

总损失：L = L_rec + L_reg + alpha * L_ali。
alpha 控制迁移强度。过强的对齐可能抑制域特有偏好，这需要实验验证。
论文式 (6) 的交叉熵书写没有显式负号；学习时应区分对数似然与需要最小化的负对数似然，并核对实现。

以本地默认 batch=16、隐维度32为例，预期表示是两个 [16,32] 张量，同一行对应同一用户。
源域输入 [16,204]，重建输出 [16,204]；目标域预测为 [16,4228]。

## 隐私发布在做什么

统一按“行是用户、列是物品”理解：原矩阵 R 是 [m,n]，发布矩阵是 [m,k]。
先中心化，再做 SVD：R=U Sigma V^T；将每个奇异值改为 sqrt(sigma_i^2+w^2)，再随机投影。
按这种行约定，可把发布写成 R_tilde=R_perturbed M^T/sqrt(k)，其中 M 为 [k,n]。
论文有转置与符号约定需要逐处核对；先用维度检查，不要机械复制矩阵乘法。

JLT 使用稠密高斯随机矩阵；论文 SJLT 使用 PHD，其中 D 随机翻转符号，H 混合坐标，P 稀疏投影。
目标是保护单个交互变化，同时保留有助于推荐的结构。普通降维本身不能直接保证差分隐私；保证依赖完整机制、参数和邻接定义。
论文 Definition 3.2 和 Theorem 5.2 讨论单个评分改变，不能据“User-level”字样直接宣称保护整个用户历史的删除。

## 文件阅读路线

1. Dataset_PP.py：load_data、get_test_data、get_part_train_indices，理解输入矩阵和划分。
2. model.py：只先看 __init__、forward_dmf。其他 forward 分支不属于默认训练调用。
3. main.py：prosvd、模型构造、loss_x/loss_y/align_loss、训练和模型选择。
4. utils.py：test_process、evaluation，理解99个负物品加1个正物品。
5. random_proj_svd.py：prosvd → get_parameters → graph_publish → random_projection/sparse_projection。

建议逐步打印上述张量的 shape，再把每行代码对应到论文第4节。

## 正式复现前必须核对的差异

以下来自对检出代码和论文 v1 的核对，启动脚本没有修改这些科学实现：

1. 对齐广播：forward_dmf 返回的 h_user_x 为 [B,H]，out_user_y 为 [B,1,H]。main.py 直接相减会广播为 [B,B,H]，把不同用户也配对。论文要求对应用户的表示对齐；修正版本应先恢复 [B,H]，再按论文核对范数和聚合方式。修正会改变训练行为，应单独记录实验。
2. FJLT 分支使用 sklearn SparseRandomProjection，未见论文 SJLT 的 H、D 步骤；稀疏随机矩阵分布和归一化也需逐项核对。
3. --eps 是 [0.25,0.5,1,2,4,8,16,32,64] 的下标。--eps 0 对应传入 get_parameters 的0.25，不是零隐私预算。get_parameters 内又将其当 eps0 并换算打印 eps，不能直接把命令行数字当论文总预算。
4. --et 是 eta，k=int(8*ln(2/0.2)/eta^2)。--et 0.3 得204维；默认0.05得7368维，可能比原始物品数还大。
5. main.py 使用 BCEWithLogitsLoss，而 forward_dmf 输出余弦分数；这与论文概率及交叉熵写法需核对。不要为了跑通就认定实现和公式等价。
6. 原日志目录重名分支调用 Linux 的 rm -rf。run_local.py 使用唯一目录并禁用手工 --log，避免进入该分支。
7. 原默认隐维100、启动脚本32，而论文实验描述隐维200及不同网络宽度。当前是低开销学习配置。
8. 每个指标分别选择验证集最佳权重，最终三个测试指标可能来自不同 checkpoint。比较模型时需统一报告规则。

先保留“作者原版样例运行”作为参考，再建立有明确差异记录的修正复现；新增修正本身不应算作论文创新。

## 学习验收

第一次：能从一个用户的两域行为，讲到两个表示和推荐列表。
第二次：能解释三个损失分别监督什么，以及去掉对齐会失去什么。
第三次：能用张量维度对应论文公式，指出上述代码差异。
第四次：统一数据和协议后，对比单域基线、原实现、修正实现，记录真实结果。

论文来源：../2202.04893v1.pdf。官方代码来源见本文开头。
