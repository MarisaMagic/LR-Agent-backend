# 标注执行路径（唯一）

## 流程概览

标注操作（批量标注、标注变更、数据分析）的**唯一用户入口**是 Agent 工具 `execute_batch_annotation`、`mutate_annotation`、`analyze_data`。

```
1. 用户请求 → Agent（后端 assist）
2. Agent tool call: execute_batch_annotation / mutate_annotation / analyze_data
3. tool_pending → 前端 agentJobRegistry.runClientTool(toolName)
4. 对应流水线:
   - execute_batch_annotation → annotationBatchJob.ts → batchOrchestrator.ts
   - mutate_annotation → annotationMutationBatchJob.ts → mutationOrchestrator.ts
   - analyze_data → analysisBatchJob.ts → dataAnalysisRunner.ts
5. 主进程推理/执行: inferenceProcess (YOLO) / analysisProcess (Python sandbox)
6. 后端 annotation/ 服务函数: batch_prepare, judge_labels, map_labels 等
7. 生成提案 → 用户确认 → 应用提案
```

## 后端 annotation/ 模块

`app/agent/annotation/` 中的函数为**内部服务**,供前端流水线调用:
- `batch_prepare_service` — 批量标注参数准备
- `judge_labels_service` — 检测标签评判
- `map_labels_service` — 标签映射
- `map_validation_service` — 映射验证
- `mutation_prepare_service` — 标注变更准备
- `heuristic_map_service` — 启发式标签映射

这些函数通过独立 API 暴露 (`api/v1/annotation_agent.py`、`api/v1/agent_analysis.py`),
但标注的**用户入口**必须通过 Agent 工具链 (`execute_batch_annotation`)。

## 前端流水线

`src/renderer/services/annotationAgent/`:
- `batchOrchestrator.ts` — 批量标注编排
- `mutationOrchestrator.ts` — 标注变更编排
- `pipelineStages.ts` — 流水线阶段定义
- `pipelineImageSteps.ts` — 单张图片检测步骤

`src/renderer/services/`:
- `annotationBatchJob.ts` — 批量标注入口
- `annotationMutationBatchJob.ts` — 标注变更入口
- `analysisBatchJob.ts` — 数据分析入口
- `agentJobRegistry.ts` — 客户端工具调度中心

`src/main/`:
- `preAnnot/inferenceProcess.ts` — YOLO 推理子进程
- `analysis/analysisProcess.ts` — Python 分析子进程
- `annotation/annotationStore.ts` — 标注持久化
