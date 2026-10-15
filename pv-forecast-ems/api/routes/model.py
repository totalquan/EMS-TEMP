"""
API层 - 模型管理路由
模型列表、训练、加载、切换
"""
import os
import json
import uuid
import threading
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from api.schemas import (
    ModelInfo, ModelListResponse,
    TrainRequest, TrainResponse,
)
from config.settings import settings

router = APIRouter(prefix="/models", tags=["模型管理"])

# 训练任务记录
_training_tasks = {}


@router.get("", response_model=ModelListResponse)
async def list_models():
    """获取所有可用模型列表"""
    storage_path = settings.model_storage_path
    models = []

    # 扫描模型存储目录
    if os.path.exists(storage_path):
        for f in os.listdir(storage_path):
            if f.endswith("_meta.json"):
                model_name = f.replace("_meta.json", "")
                meta_path = os.path.join(storage_path, f)
                try:
                    with open(meta_path, "r") as mf:
                        meta = json.load(mf)
                    models.append(ModelInfo(
                        model_name=meta.get("model_name", model_name),
                        model_type=meta.get("model_type", "unknown"),
                        is_trained=meta.get("is_trained", True),
                        feature_count=len(meta.get("feature_cols", [])),
                        metrics=meta.get("metrics", {}),
                        created_at=meta.get("created_at"),
                    ))
                except Exception as e:
                    logger.warning(f"读取模型元数据失败 {f}: {e}")

    # 物理模型始终可用
    if not any(m.model_name == "physical_baseline" for m in models):
        models.append(ModelInfo(
            model_name="physical_baseline",
            model_type="physical",
            is_trained=True,
            feature_count=2,
            metrics={"note": "纯物理模型，无需训练"},
            created_at=datetime.now().isoformat(),
        ))

    return ModelListResponse(data=models)


@router.get("/{model_name}")
async def get_model_info(model_name: str):
    """获取指定模型的详细信息"""
    storage_path = settings.model_storage_path
    meta_path = os.path.join(storage_path, f"{model_name}_meta.json")

    if model_name == "physical_baseline":
        return {
            "code": 200,
            "data": {
                "model_name": "physical_baseline",
                "model_type": "physical",
                "is_trained": True,
                "description": "基于辐照度-温度-效率公式的纯物理预测模型，适用于新建园区冷启动",
                "formula": "P = (GHI/1000) × 装机容量 × 温度修正 × 系统效率",
                "parameters": {
                    "capacity_kw": settings.pv.capacity_kw,
                    "system_efficiency": settings.pv.system_efficiency,
                    "temp_coefficient": settings.pv.temp_coefficient,
                },
            }
        }

    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return {"code": 200, "data": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取模型信息失败: {str(e)}")


@router.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    触发模型训练（异步执行）
    """
    task_id = str(uuid.uuid4())[:8]

    def _do_training():
        try:
            _training_tasks[task_id]["status"] = "running"
            _training_tasks[task_id]["started_at"] = datetime.now().isoformat()

            from training.train import TrainingPipeline
            pipeline = TrainingPipeline()
            report = pipeline.run(model_types=request.model_types)

            _training_tasks[task_id]["status"] = "completed"
            _training_tasks[task_id]["completed_at"] = datetime.now().isoformat()
            _training_tasks[task_id]["result"] = report
            logger.info(f"训练任务 {task_id} 完成")

        except Exception as e:
            _training_tasks[task_id]["status"] = "failed"
            _training_tasks[task_id]["error"] = str(e)
            logger.error(f"训练任务 {task_id} 失败: {e}", exc_info=True)

    _training_tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "model_types": request.model_types,
        "data_source": request.data_source,
        "created_at": datetime.now().isoformat(),
    }

    background_tasks.add_task(_do_training)

    return TrainResponse(
        task_id=task_id,
        data={
            "task_id": task_id,
            "status": "pending",
            "model_types": request.model_types,
            "message": "训练任务已提交，将在后台执行",
        }
    )


@router.get("/train/{task_id}")
async def get_training_status(task_id: str):
    """查询训练任务状态"""
    if task_id not in _training_tasks:
        raise HTTPException(status_code=404, detail=f"训练任务 {task_id} 不存在")

    task = _training_tasks[task_id].copy()
    # 移除大对象
    if "result" in task and task["result"]:
        task["result_summary"] = {
            "best_model": task["result"].get("best_model"),
            "comparison_count": len(task["result"].get("comparison", [])),
        }
        del task["result"]

    return {"code": 200, "data": task}


@router.post("/{model_name}/activate")
async def activate_model(model_name: str):
    """切换当前使用的预测模型"""
    storage_path = settings.model_storage_path
    meta_path = os.path.join(storage_path, f"{model_name}_meta.json")

    if model_name != "physical_baseline" and not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")

    # 写入当前激活模型配置
    active_path = os.path.join(storage_path, "active_model.txt")
    try:
        with open(active_path, "w") as f:
            f.write(model_name)
        return {
            "code": 200,
            "message": f"模型 {model_name} 已激活为当前预测模型",
            "data": {"active_model": model_name},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"切换模型失败: {str(e)}")


@router.get("/active/current")
async def get_active_model():
    """获取当前激活的模型"""
    storage_path = settings.model_storage_path
    active_path = os.path.join(storage_path, "active_model.txt")

    active_model = "physical_baseline"  # 默认
    if os.path.exists(active_path):
        try:
            with open(active_path, "r") as f:
                active_model = f.read().strip()
        except Exception:
            pass

    return {
        "code": 200,
        "data": {
            "active_model": active_model,
            "available_models": [f.replace("_meta.json", "")
                                  for f in os.listdir(storage_path)
                                  if f.endswith("_meta.json")] + ["physical_baseline"],
        }
    }
