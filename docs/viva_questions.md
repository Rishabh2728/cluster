# INT315 – CLUSTER COMPUTING VIVA EXAMINATION GUIDE
## Comprehensive Questions & Answers: Predicting Insurance Claim Approval Using Apache Spark

**Course Code**: INT315  
**Course Title**: Cluster Computing  
**Student Name**: Rishabh Kumar  
**Roll Number**: 49  
**Registration Number**: 12310491  
**Section**: K2P23PD  
**Project Title**: Predicting Insurance Claim Approval Using Apache Spark  

---

### Q1: What is Cluster Computing?
**Answer**: Cluster computing is a distributed computing paradigm where multiple independent computing units (nodes)—connected via local-area networks (LAN) or virtual clusters—work collectively as a single unified computing resource. It provides high availability, fault tolerance, and horizontal scalability to process large-scale datasets beyond the RAM and processing capacity of a single machine.

### Q2: What is Apache Spark and why was it chosen over Hadoop MapReduce?
**Answer**: Apache Spark is an open-source, distributed general-purpose computing engine designed for large-scale data analytics and machine learning. Unlike Hadoop MapReduce, which relies heavily on writing intermediate results to physical disks (HDFS) between map and reduce steps, Spark achieves up to 100x faster execution through in-memory Directed Acyclic Graph (DAG) computation, pipelined transformations, and memory-cached DataFrames.

### Q3: What is PySpark?
**Answer**: PySpark is the Python API for Apache Spark. It allows developers to write Python code that leverages Spark’s distributed computing engine. PySpark communicates with the underlying Java Virtual Machine (JVM) through the `Py4J` gateway, serializing data and sending execution plans to worker nodes while exposing Pythonic DataFrame, SQL, and MLlib APIs.

### Q4: What is the core difference between an RDD, a DataFrame, and a Dataset in Spark?
**Answer**:
- **RDD (Resilient Distributed Dataset)**: The lowest-level distributed abstraction in Spark. It provides immutable, lazily evaluated, partitioned collections of objects with fault tolerance, but lacks schema metadata and Catalyst optimization.
- **DataFrame**: A distributed collection of data organized into named columns (conceptually equivalent to a relational table). It is fully optimized by Spark's Catalyst Query Optimizer and Tungsten execution engine.
- **Dataset**: An extension of DataFrames with compile-time type safety (available in Scala and Java, but unified with DataFrames in Python).

### Q5: What is SparkSession?
**Answer**: Introduced in Spark 2.0, `SparkSession` provides a unified entry point to all Spark functionality. It encapsulates and replaces earlier disparate contexts, including `SparkContext` (for core RDDs), `SQLContext` (for SQL queries), `HiveContext` (for Hive queries), and `StreamingContext`.

### Q6: What does execution master `local[*]` mean?
**Answer**: In Apache Spark, `local[*]` designates a local execution mode where Spark runs inside a single JVM on the local workstation, dynamically creating worker executor threads matching the total number of logical CPU cores available on the host machine. While it enables parallel multi-threaded scheduling, it does not represent physical multi-node network distribution.

### Q7: Explain Lazy Evaluation in Spark.
**Answer**: Lazy evaluation means Spark does not compute intermediate operations immediately when defined. Instead, Spark records operations as a Directed Acyclic Graph (DAG) of logical transformations. Execution only triggers when an *Action* (e.g., `count()`, `collect()`, `write()`, `save()`) is called, enabling the Catalyst optimizer to optimize queries, prune unused columns, and fuse operations before data execution.

### Q8: What is the difference between a Transformation and an Action?
**Answer**:
- **Transformation**: An operation on a Spark DataFrame/RDD that returns a new DataFrame/RDD without executing computation immediately (e.g., `filter()`, `select()`, `withColumn()`, `groupBy()`, `repartition()`).
- **Action**: An operation that triggers the execution of the DAG and returns a non-DataFrame value to the driver program or persists data to storage (e.g., `count()`, `collect()`, `show()`, `save()`, `take()`).

### Q9: What is the difference between Narrow and Wide Transformations?
**Answer**:
- **Narrow Transformation**: Each input partition contributes to at most one output partition (e.g., `map()`, `filter()`, `withColumn()`). These execute within the same stage without network shuffling.
- **Wide Transformation**: Multiple input partitions contribute to multiple output partitions (e.g., `groupBy()`, `orderBy()`, `distinct()`, `join()`). Wide transformations introduce stage boundaries and require data shuffling across worker threads/nodes.

### Q10: What is a Partition in Spark?
**Answer**: A partition is an atomic, logical chunk of a distributed dataset. Partitions represent the fundamental unit of parallelism in Spark: one executor task processes exactly one partition at any given time. If a dataset has 8 partitions, Spark can process up to 8 tasks concurrently across available CPU cores.

### Q11: What is the difference between `repartition()` and `coalesce()`?
**Answer**:
- `repartition(k)`: Performs a full round-robin shuffle across all partitions to redistribute data evenly into $k$ partitions. It can both increase or decrease partition counts, but incurs network/thread shuffle latency.
- `coalesce(k)`: Reduces the partition count to $k$ ($k < \text{current}$) by merging adjacent partitions *without* inducing a full shuffle. It is significantly more efficient than repartition for down-scaling partition counts.

### Q12: What is Shuffling in Spark and why is it expensive?
**Answer**: Shuffling is the process of redistributing data across partitions, cores, or cluster nodes during wide transformations. It involves disk I/O, data serialization, network transmission, and deserialization. Minimizing shuffles is the primary strategy for optimizing Spark job performance.

### Q13: What are Stages and Tasks in Spark execution?
**Answer**:
- **Application**: A complete user program managed by a SparkSession.
- **Job**: Triggered whenever an action is called on a DataFrame.
- **Stage**: A job is decomposed into stages separated by wide transformations (shuffle boundaries).
- **Task**: The smallest individual unit of work executed on a single partition inside an executor thread. A stage consists of multiple tasks running in parallel.

### Q14: What is the Spark Catalyst Optimizer?
**Answer**: The Catalyst Optimizer is Spark SQL’s extensible optimization framework. It converts logical DataFrame plans into optimized physical plans using four phases: Analysis (resolving column names and types), Logical Optimization (predicate pushdown, constant folding, projection pruning), Physical Planning (selecting physical operators like SortMergeJoin vs BroadcastHashJoin), and Code Generation (generating Java bytecode using Whole-Stage Code Generation).

### Q15: What is Spark MLlib?
**Answer**: Spark MLlib is Apache Spark’s distributed machine learning library. It provides high-level APIs based on DataFrames for feature extraction, transformation, dimensionality reduction, classification, regression, clustering, model evaluation, and multi-stage ML pipelines that execute in parallel across partitions.

### Q16: Why use an ML Pipeline (`Pipeline`) in Spark MLlib?
**Answer**: An ML `Pipeline` chains multiple `Transformers` (e.g., `StringIndexer`, `OneHotEncoder`, `VectorAssembler`) and `Estimators` (e.g., `LogisticRegression`, `RandomForestClassifier`) into a single reproducible, leak-free workflow. When fitting the pipeline on the training partition, all learned parameters (such as string index mappings) are encapsulated into a `PipelineModel`, preventing data leakage during test evaluation or production inference.

### Q17: What is Data Leakage in Machine Learning and how did we prevent it?
**Answer**: Data leakage occurs when information from outside the training partition is inadvertently used to train a model, leading to artificially inflated evaluation metrics that fail to generalize. We strictly prevented data leakage by:
1. Splitting data into 80% Train and 20% Test before training.
2. Fitting all transformations (`StringIndexer`, encoders, normalizers) exclusively on the training partition.
3. Excluding non-predictive identifiers (`claim_id`, `policy_id`) and the text label (`claim_approval`) from feature vector assembly.

### Q18: Explain Logistic Regression in the context of Spark MLlib.
**Answer**: Logistic Regression is a linear classification algorithm that models the log-odds of a binary categorical outcome as a linear combination of input features:
$$\log\left(\frac{P(Y=1)}{1 - P(Y=1)}\right) = \beta_0 + \sum_{j=1}^{p} \beta_j X_j$$
Spark MLlib optimizes the weights using L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno), computing gradients in parallel across partitions.

### Q19: Explain Random Forest Classifier in the context of Spark MLlib.
**Answer**: Random Forest is an ensemble learning method that constructs a collection of decorrelated decision trees during training. Each tree is trained on a bootstrap sample of data, and at each split, only a random subset of features is evaluated. Predictions are aggregated by majority voting. In Spark MLlib, tree building is distributed by parallelizing histogram computation for candidate split points across partitions.

### Q20: What are the key features engineered for this project?
**Answer**:
1. `claim_to_premium_ratio` = $\frac{\text{claim\_amount}}{\text{annual\_premium}}$: Assesses financial claim leverage against annual premium paid.
2. `claim_to_estimate_ratio` = $\frac{\text{claim\_amount}}{\text{repair\_or\_treatment\_estimate}}$: Identifies potential inflation or discrepancies beyond professional repair appraisals.
3. `policy_tenure_category`: Discretizes tenure into 'New (< 12m)', 'Established (12-48m)', and 'Mature (> 48m)'.
4. `submission_delay_category`: Bins filing delays into 'Immediate (<= 3d)', 'Normal (4-14d)', and 'Delayed (> 14d)'.
5. `previous_claim_category`: Categorizes prior claims into 'None (0)', 'Few (1-2)', and 'Frequent (3+)'.

### Q21: What is Class Imbalance and why does it matter here?
**Answer**: Class imbalance occurs when one class significantly outnumbers another in the target variable. In our synthetic insurance dataset, approximately 70% of claims are `Approved` and 30% are `Rejected`. If a naive model predicts `Approved` for every record, it would achieve 70% accuracy while failing entirely to identify fraudulent or ineligible claims. Hence, evaluating Recall for the minority `Rejected` class and using PR-AUC and ROC-AUC is essential.

### Q22: Define Accuracy, Precision, Recall, and F1-Score.
**Answer**:
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$ — Overall proportion of correct predictions.
- **Precision**: $\frac{TP}{TP + FP}$ — Proportion of predicted positives that were truly positive.
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ — Proportion of actual positives correctly identified.
- **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ — Harmonic mean of Precision and Recall.

### Q23: What is the ROC Curve and ROC-AUC?
**Answer**: The Receiver Operating Characteristic (ROC) curve plots True Positive Rate (Sensitivity) against False Positive Rate ($1 - \text{Specificity}$) across all classification probability thresholds. The Area Under the ROC Curve (ROC-AUC) quantifies the model's ability to discriminate between classes regardless of threshold. A value of 1.0 represents perfect discrimination, while 0.5 represents random guessing.

### Q24: What is PR-AUC and when is it preferred over ROC-AUC?
**Answer**: Precision-Recall AUC calculates the area under the Precision-Recall curve. Unlike ROC-AUC, PR-AUC does not incorporate True Negatives ($TN$), making it particularly sensitive to performance on the minority class in heavily imbalanced classification scenarios.

### Q25: What is a Confusion Matrix?
**Answer**: A confusion matrix is a $2 \times 2$ contingency table summarizing model classification performance:
- **True Positive (TP)**: Approved claims correctly predicted as Approved.
- **False Positive (FP)**: Rejected claims incorrectly predicted as Approved (Type I error).
- **True Negative (TN)**: Rejected claims correctly predicted as Rejected.
- **False Negative (FN)**: Approved claims incorrectly predicted as Rejected (Type II error).

### Q26: How does Model Persistence work in Spark MLlib?
**Answer**: Spark MLlib models implement the `MLWritable` interface. The `model.write().overwrite().save(path)` method serializes both the pipeline metadata (JSON format describing stages and schema) and the trained model parameters (Parquet format containing weights, tree structures, and feature mappings). The model can later be loaded via `PipelineModel.load(path)` for batch scoring or real-time inference without retraining.

### Q27: Why did you generate synthetic data instead of downloading a real dataset?
**Answer**: Real insurance claim datasets contain proprietary customer records, protected health information (PHI), and strict privacy regulations (GDPR/HIPAA). Generating synthetic data with seed 42 allows complete reproducibility, known anomaly injection rates, exact mathematical feature relationships, and zero risk of copyright or privacy violations.

### Q28: How does Spark achieve Fault Tolerance?
**Answer**: Spark achieves fault tolerance through RDD lineage graphs. Every transformation records the deterministic steps required to build a partition from its parent. If a worker node crashes and a partition in memory is lost, Spark recomputes only the missing partition from upstream lineage without restarting the entire computation.

### Q29: What were the empirical performance observations across 2, 4, and 8 partitions?
**Answer**: In our local benchmark (`local[*]`), partitioning allows tasks to execute concurrently across multi-core CPUs. A moderate partition count (4 partitions) balanced CPU core saturation with scheduling overhead. Increasing to 8 partitions on smaller workloads introduced slight task scheduling and thread synchronization overhead, illustrating that more partitions do not automatically guarantee faster execution.

### Q30: What are the main technical limitations of this project?
**Answer**:
1. Execution was performed on a single multi-core workstation using `local[*]`, which models thread-level parallelism rather than multi-node network cluster bottlenecks.
2. The dataset is synthetic; real-world insurance claim approvals involve unstructured doctor notes, fraud rings, and qualitative human adjuster investigations.
3. Local disk I/O replaced distributed network storage systems such as HDFS, S3, or Google Cloud Storage.
