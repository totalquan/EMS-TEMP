"""
数据接入层 - S3/MinIO原始数据湖存储
将Kafka接收的原始数据以Parquet/JSON格式归档到对象存储，支持数据回溯和重训练
"""
import json
import io
from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception

try:
    import pandas as pd
except ImportError:
    pd = None

from config.settings import settings


class S3RawDataStorage:
    """S3/MinIO原始数据湖存储"""

    def __init__(self):
        self.endpoint = settings.s3.endpoint
        self.access_key = settings.s3.access_key
        self.secret_key = settings.s3.secret_key
        self.bucket = settings.s3.bucket
        self.region = settings.s3.region
        self._client = None

    def _get_client(self):
        """获取S3客户端"""
        if boto3 is None:
            raise RuntimeError("boto3 未安装，请执行 pip install boto3")

        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region,
                config=Config(signature_version="s3v4"),
            )
            self._ensure_bucket()
            logger.info(f"S3客户端创建成功 | endpoint={self.endpoint} | bucket={self.bucket}")
        return self._client

    def _ensure_bucket(self):
        """确保bucket存在，不存在则创建"""
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                self._client.create_bucket(Bucket=self.bucket)
                logger.info(f"创建S3 Bucket: {self.bucket}")
            except Exception as e:
                logger.error(f"创建Bucket失败: {e}")

    def _build_object_key(self, data_type: str, timestamp: Optional[str] = None) -> str:
        """
        构建S3对象路径，按日期分区
        格式: raw/{data_type}/{year}/{month}/{day}/{timestamp}.json
        """
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                dt = datetime.utcnow()
        else:
            dt = datetime.utcnow()

        return (
            f"raw/{data_type}/"
            f"{dt.strftime('%Y')}/"
            f"{dt.strftime('%m')}/"
            f"{dt.strftime('%d')}/"
            f"{dt.strftime('%H%M%S')}_{dt.microsecond}.json"
        )

    def save_raw_data(self, data_type: str, data: Dict[str, Any]):
        """
        保存单条原始数据到S3
        :param data_type: 数据类型: weather / inverter / meter
        :param data: 原始数据字典
        """
        try:
            client = self._get_client()
            timestamp = data.get("timestamp")
            object_key = self._build_object_key(data_type, timestamp)

            # 序列化为JSON字节流
            json_bytes = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")

            client.put_object(
                Bucket=self.bucket,
                Key=object_key,
                Body=io.BytesIO(json_bytes),
                ContentType="application/json",
                Metadata={
                    "data_type": data_type,
                    "ingested_at": datetime.utcnow().isoformat(),
                },
            )
            logger.debug(f"原始数据已归档到S3 | key={object_key}")

        except Exception as e:
            logger.error(f"S3归档失败 | data_type={data_type} | error={e}")

    def save_batch_data(self, data_type: str, data_list: List[Dict[str, Any]]):
        """
        批量保存原始数据（合并为一个Parquet文件，效率更高）
        :param data_type: 数据类型
        :param data_list: 数据列表
        """
        if not data_list:
            return

        try:
            client = self._get_client()
            dt = datetime.utcnow()
            object_key = (
                f"raw/{data_type}/"
                f"{dt.strftime('%Y')}/{dt.strftime('%m')}/{dt.strftime('%d')}/"
                f"batch_{dt.strftime('%H%M%S')}.parquet"
            )

            if pd is not None:
                # 转为DataFrame并保存为Parquet
                df = pd.DataFrame(data_list)
                buffer = io.BytesIO()
                df.to_parquet(buffer, index=False, engine="pyarrow")
                buffer.seek(0)

                client.put_object(
                    Bucket=self.bucket,
                    Key=object_key,
                    Body=buffer,
                    ContentType="application/octet-stream",
                    Metadata={"data_type": data_type, "record_count": str(len(data_list))},
                )
                logger.info(f"批量数据已归档到S3 | key={object_key} | 条数={len(data_list)}")
            else:
                # 无pandas时退化为JSON批量
                json_bytes = json.dumps(data_list, ensure_ascii=False, default=str).encode("utf-8")
                object_key = object_key.replace(".parquet", ".json")
                client.put_object(
                    Bucket=self.bucket,
                    Key=object_key,
                    Body=io.BytesIO(json_bytes),
                    ContentType="application/json",
                )
                logger.info(f"批量数据(JSON)已归档到S3 | key={object_key} | 条数={len(data_list)}")

        except Exception as e:
            logger.error(f"S3批量归档失败 | data_type={data_type} | error={e}")

    def list_objects(self, prefix: str = "", max_keys: int = 100) -> List[str]:
        """列出指定前缀下的对象"""
        try:
            client = self._get_client()
            response = client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=max_keys,
            )
            return [obj["Key"] for obj in response.get("Contents", [])]
        except Exception as e:
            logger.error(f"列出S3对象失败: {e}")
            return []

    def get_object(self, object_key: str) -> Optional[Dict[str, Any]]:
        """读取单个对象内容"""
        try:
            client = self._get_client()
            response = client.get_object(Bucket=self.bucket, Key=object_key)
            content = response["Body"].read()
            if object_key.endswith(".json"):
                return json.loads(content.decode("utf-8"))
            elif object_key.endswith(".parquet") and pd is not None:
                df = pd.read_parquet(io.BytesIO(content))
                return df.to_dict("records")
            return None
        except Exception as e:
            logger.error(f"读取S3对象失败 | key={object_key} | error={e}")
            return None

    def close(self):
        """关闭客户端"""
        if self._client:
            self._client.close()
            self._client = None
