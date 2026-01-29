# 数据库缺陷检测模型
## 1. 引言
 本记录是记录我完成毕业设计“数据库缺陷检测模型”的全过程。本文档将详细描述项目的背景、目标、方法、实验结果以及结论。

## 2.实验环境配置
### 2.1 本机环境

第一步在本机上配置环境，安装CUDA、cuDNN和pytorch等深度学习框架。测试小数据量的运行情况，确保模型基本能够运行。

目前使用的环境如下：

CUDA 13.1

cuDNN 9.11

pytorch 2.10.0

安装教程：https://zhuanlan.zhihu.com/p/24884367657
https://blog.csdn.net/weixin_52677672/article/details/135853106
https://zhuanlan.zhihu.com/p/27577871722

## 2. 实验记录

### 2.1 数据源
 
第一层模型用SQL injection数据集，数据集来源于 https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset/data 。 

生成了一个数据集，在kaggle那个经典的sql injection数据集的基础上面，增加了角色标签，数据集生成脚本在data_generation文件夹下面，生成的数据集在data/custom文件夹下面)

数据集的生成逻辑是这样的：在**经典数据集sqli**的基础上进行改动，首先标签改成**三项**，分别是**query**、**role**、**label**。其中query是sql语句，role是角色标签，label是正常和攻击的标签。

我规定了用户的表级限制，列级限制，还有用户经常做的一些操作，基于这些规则生成了正常的SQL查询语句

角色标签和其对应能够查询的表如下：

|编号|角色|可查询的表|
|:------:|:------:|:------------:|
|0|财务|'finance', 'salary', 'budget', 'revenue'|
|1|人事|'employee', 'staff', 'hr', 'attendance'|
|2|销售|'customer', 'orders', 'sales', 'leads'|
|3|开发|'logs', 'config', 'dev', 'test_table'|

每个角色只能查询自己权限内的表，查询其他角色权限内的表就算违规。

而且每个表也有可以查询的正常列和不能查询的敏感列，查询敏感列也算违规。

对应如下：

|表名|正常列|敏感列|
|:------:|:------:|:------------:|
|finance| 'record_id', 'trans_date', 'amount', 'dept_id', 'project_code', 'category'| 'admin_signature', 'hidden_assets', 'audit_trace_log' |
|salary| 'emp_id', 'basic_salary', 'bonus', 'tax_deduction', 'net_pay', 'pay_date' | 'personal_bank_card_no', 'id_card', 'family_info'|
|budget|'year', 'quarter', 'dept_name', 'approved_amt', 'remain_amt', 'status'|'manager_approval_key', 'secret_reserve_fund'|
|revenue|'source', 'channel', 'gross_income', 'net_income', 'currency', 'region'|'partner_commission_rate', 'encrypted_token'|
|employee|'emp_name', 'emp_id', 'gender', 'position', 'department', 'entry_date', 'email'|'password_hash', 'social_security_number', 'political_status'|
|staff|'staff_code', 'work_loc', 'office_phone', 'direct_manager', 'job_level', 'emergency_contact_mobile'|'home_address'|
|hr|'policy_id', 'doc_name', 'publish_date', 'interview_score', 'resume_path'|'background_check_result'|
|attendance| 'check_in', 'check_out', 'date', 'leave_type', 'overtime_hours'|'biometric_data'|
|customer|'cust_name', 'cust_level', 'last_purchase', 'points', 'public_phone'|'cust_id_card', 'home_address', 'credit_card'|
|orders|'order_id', 'prod_name', 'qty', 'total_price', 'status', 'ship_date'|'payment_gateway_token', 'fraud_check_score'|
|sales|'rep_id', 'monthly_target', 'achieved', 'region_code'|无|
|leads|'company', 'contact', 'industry', 'intent', 'source_web'|'private_mobile', 'ceo_email'|
|logs|'trace_id', 'level', 'msg', 'timestamp', 'service', 'ip_addr'|'user_session_token', 'cookie_content'|
|config|'key', 'value', 'env', 'is_active', 'version'|'db_password', 'secret_key', 'aws_access_key'|
|dev|'feature', 'test_uid', 'deploy_id', 'branch'|'prod_admin_credential'|
|test_table|'id', 'field1', 'field2', 'desc'|无|

我规定了正常操作的三种类型：
1. 访问允许的表
2. 只查询 'normal' 列 (禁止 * 查询)
3. 动词限制：
   - HR (1): SELECT, INSERT, UPDATE, DELETE
   - 其他 (0,2,3): SELECT, INSERT, UPDATE (严禁 DELETE)
   - 所有人: 严禁 DROP

具体这些表的列名和角色权限可以在tests/generate_data.py中查看

依靠这些规则，我生成了一个包含13000多条SQL语句的数据集，存储在data/custom/custom_dataset.csv中

### 2.2 模型搭建情况

已经完成第一层模型的搭建，在feature.py文件上面用DistilBERT模型，利用LLM的语言理解能力，将特征提取后输入孤立森林，目前的训练方法是半监督学习，仅抽取数据集中标签为正常的样本进行训练。

已经完成第二层模型的搭建，将role角色信息编码成One-hot编码，与第一层获取的SQL向量拼合输入DistilBERT模型中，让模型生成每个角色的概率值，取最高那个概率值与标签对比，如果角色对应上了那就通过，没对应上就是有问题。

目前我的想法就是先分开两层模型分别进行训练和测试，把两层模型都调试好之后，再进行联合训练。

### 2.3 实验结果
所有实验结果全部存放在notebooks文件夹下面，命名方式为"exp_模型关键参数_时间戳",

1.第一层模型实验结果

目前第一层模型在测试集(我自己生成的那个）上的效果如下(最好一次)(20260128_2027)：
```chatinput
              precision    recall  f1-score   support

      Normal       0.87      0.95      0.91     12295
      Attack       0.88      0.71      0.78      6229

    accuracy                           0.87     18524
   macro avg       0.87      0.83      0.85     18524
weighted avg       0.87      0.87      0.87     18524
```
还有roc图片如下:
![roc](assets/roc_curve_layer1.png)

2.第二层模型实验结果
目前第二层模型的实验结果如下(最好一次)(20260129_1102)：
```chatinput
              precision    recall  f1-score   support

          R0       0.89      0.91      0.90      3036
          R1       0.90      0.91      0.90      3095
          R2       0.94      0.88      0.91      3111
          R3       0.89      0.92      0.91      3108

    accuracy                           0.91     12350
   macro avg       0.91      0.91      0.91     12350
weighted avg       0.91      0.91      0.91     12350
```
混淆矩阵：
![confusion_matrix](assets/confusion_matrix_layer2.png)

Loss曲线变化:
![LOSS](assets/training_loss_layer2.png)

## 3. 后续工作计划
数据集优化：尝试优化自己生成的数据集，加入更多细节和样本量，使其更加真实

数据处理优化：1.尝试使用AST对SQL语句展平，提取更多结构化特征 2.尝试使用其他的预训练模型进行特征提取。

第一层模型的优化：1.尝试调整孤立森林的参数，提高准确率。 2.查找其他的集成异常检测器进行对比实验。

第二层模型的优化: 1.加入RAG技术，增强大语言模型对上下文的理解能力 2.尝试用其他的大语言模型进行对比实验。

将第一层模型和第二层模型进行联合训练，提升整体的检测效果

## 参考文献

[1] 罗艺铭，谭玉波，李建平．基于BERT-GAN的SQL注入攻击检测方法研究[J]. 微电子学与计算机，2024，41（11）：39-47. DOI:  10.19304/J.ISSN1000-7180.2023.0721（这个文章讲的是对抗网络取检测SQL注入攻击，可以试试用在第一层模型上面）
https://mc.spacejournal.cn/article/doi/10.19304/J.ISSN1000-7180.2023.0721


[2] 张昊,张小雨,张振友,等.基于深度学习的入侵检测模型综述[J].计算机工程与应用,2022,58(06):17-28.（这个文章是一篇类似综述的，把别人做过的深度学习做入侵检测的整合了）（知网搜）

[3] Enhancing GraphQL Security by Detecting Malicious Queries Using Large Language Models, Sentence Transformers, and Convolutional Neural Networks(利用大型语言模型、句子转换器和卷积神经网络检测恶意查询，增强GraphQL安全性，虽然不是SQL，但是也有参考意义)
https://arxiv.org/abs/2508.11711

[4] 胡修闻.基于AST-LSTM和对抗训练的混淆SQL注入攻击检测研究[D].东南大学,2024.DOI:10.27014/d.cnki.gdnau.2024.004766.（这篇文章用LSTM做了个类似上下文增强的，第二层模型可以试试，不用RAG）（知网搜）
