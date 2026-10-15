"""
数据接入层 - InfluxDB时序数据库写入器
将Kafka接收的实时数据写入InfluxDB，供查询和模型训练使用
"""
from datetime import datetime
from typing import Dict, Any, List, Optional
from loguru import logger

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:
    InfluxDBClient = None
    Point = None
    WritePrecision = None
    SYNCHRONOUS = None

from config.settings import settings


class InfluxDBWriter:
    """InfluxDB时序数据写入器"""

    def __init__(self):
        self.url = settings.influxdb.url
        self.token = settings.influxdb.token
        self.org = settings.influxdb.org
        self.bucket = settings.influxdb.bucket
        self._client: Optional[InfluxDBClient] = None
        self._write_api = None

    def _get_client(self) -> InfluxDBClient:
        """获取或创建InfluxDB客户端"""
        if InfluxDBClient is None:
            raise RuntimeError("influxdb-client 未安装，请执行 pip install influxdb-client")

        if self._client is None:
            self._client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org,
                timeout=30000,
            )
            self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
            logger.info(f"InfluxDB客户端创建成功 | url={self.url} | bucket={self.bucket}")
        return self._client

    def write_weather(self, data: Dict[str, Any]):
        """
        写入气象站数据
        :param data: 气象数据字典
        """
        try:
            client = self._get_client()
            timestamp = self._parse_timestamp(data.get("timestamp"))

            point = Point("weather") \
                .tag("station_id", str(data.get("station_id", "unknown"))) \
                .field("ghi", float(data.get("ghi", 0.0))) \
                .field("dhi", float(data.get("dhi", 0.0))) \
                .field("temp", float(data.get("temp", 0.0))) \
                .field("humidity", float(data.get("humidity", 0.0))) \
                .field("wind_speed", float(data.get("wind_speed", 0.0))) \
                .field("wind_dir", float(data.get("wind_dir", 0.0))) \
                .field("pressure", float(data.get("pressure", 0.0))) \
                .field("cloud_cover", float(data.get("cloud_cover", 0.0))) \
                .time(timestamp, WritePrecision.MS)

            self._write_api.write(bucket=self.bucket, org=self.org, record=point)
            logger.debug(f"气象数据写入成功 | station={data.get('station_id')} | time={timestamp}")

        except Exception as e:
            logger.error(f"气象数据写入失败: {e}")

    def write_inverter(self, data: Dict[str, Any]):
        """
        写入逆变器数据
        :param data: 逆变器数据字典
        """
        try:
            client = self._get_client()
            timestamp = self._parse_timestamp(data.get("timestamp"))

            point = Point("inverter") \
                .tag("inverter_id", str(data.get("inverter_id", "unknown"))) \
                .tag("status", str(data.get("status", "unknown"))) \
                .field("dc_voltage", float(data.get("dc_voltage", 0.0))) \
                .field("dc_current", float(data.get("dc_current", 0.0))) \
                .field("dc_power", float(data.get("dc_power", 0.0))) \
                .field("ac_voltage", float(data.get("ac_voltage", 0.0))) \
                .field("ac_current", float(data.get("ac_current", 0.0))) \
                .field("ac_power", float(data.get("ac_power", 0.0))) \
                .field("ac_frequency", float(data.get("ac_frequency", 0.0))) \
                .field("temperature", float(data.get("temperature", 0.0))) \
                .field("daily_energy", float(data.get("daily_energy", 0.0))) \
                .time(timestamp, WritePrecision.MS)

            self._write_api.write(bucket=self.bucket, org=self.org, record=point)
            logger.debug(f"逆变器数据写入成功 | inverter={data.get('inverter_id')}")

        except Exception as e:
            logger.error(f"逆变器数据写入失败: {e}")

    def write_meter(self, data: Dict[str, Any]):
        """
        写入电表数据
        :param data: 电表数据字典
        """
        try:
            client = self._get_client()
            timestamp = self._parse_timestamp(data.get("timestamp"))

            point = Point("meter") \
                .tag("meter_id", str(data.get("meter_id", "unknown"))) \
                .field("voltage_a", float(data.get("voltage_a", 0.0))) \
                .field("voltage_b", float(data.get("voltage_b", 0.0))) \
                .field("voltage_c", float(data.get("voltage_c", 0.0))) \
                .field("current_a", float(data.get("current_a", 0.0))) \
                .field("current_b", float(data.get("current_b", 0.0))) \
                .field("current_c", float(data.get("current_c", 0.0))) \
                .field("active_power", float(data.get("active_power", 0.0))) \
                .field("reactive_power", float(data.get("reactive_power", 0.0))) \
                .field("power_factor", float(data.get("power_factor", 0.0))) \
                .field("frequency", float(data.get("frequency", 0.0))) \
                .field("total_energy_import", float(data.get("total_energy_import", 0.0))) \
                .field("total_energy_export", float(data.get("total_energy_export", 0.0))) \
                .time(timestamp, WritePrecision.MS)

            self._write_api.write(bucket=self.bucket, org=self.org, record=point)
            logger.debug(f"电表数据写入成功 | meter={data.get('meter_id')}")

        except Exception as e:
            logger.error(f"电表数据写入失败: {e}")

    def write_forecast(self, forecast_data: List[Dict[str, Any]], model_name: str = "default"):
        """
        写入光伏预测结果
        :param forecast_data: 预测结果列表，每项包含 timestamp 和 power_kw
        :param model_name: 模型名称
        """
        try:
            client = self._get_client()
            points = []

            for item in forecast_data:
                timestamp = self._parse_timestamp(item.get("timestamp"))
                point = Point("pv_forecast") \
                    .tag("model", model_name) \
                    .tag("station", "pv_5mw_ordos") \
                    .field("power_kw", float(item.get("power_kw", 0.0))) \
                    .field("ghi_pred", float(item.get("ghi_pred", 0.0))) \
                    .field("temp_pred", float(item.get("temp_pred", 0.0))) \
                    .field("horizon_min", int(item.get("horizon_min", 0))) \
                    .time(timestamp, WritePrecision.MS)
                points.append(point)

            if points:
                self._write_api.write(bucket=self.bucket, org=self.org, record=points)
                logger.info(f"预测结果写入成功 | 条数={len(points)} | model={model_name}")

        except Exception as e:
            logger.error(f"预测结果写入失败: {e}")

    def query_actual_power(self, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """
        查询实际光伏功率（用于模型评估和重训练）
        :param start_time: 起始时间 RFC3339格式
        :param end_time: 结束时间 RFC3339格式
        :return: 实际功率列表
        """
        try:
            client = self._get_client()
            query = f'''
                from(bucket: "{self.bucket}")
                  |> range(start: {start_time}, stop: {end_time})
                  |> filter(fn: (r) => r._measurement == "inverter")
                  |> filter(fn: (r) => r._field == "ac_power")
                  |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
                  |> yield(name: "mean")
            '''
            result = client.query_api().query(query, org=self.org)
            records = []
            for table in result:
                for record in table.records:
                    records.append({
                        "timestamp": record.get_time().isoformat(),
                        "power_kw": float(record.get_value()) / 1000.0,  # W -> kW
                    })
            logger.info(f"查询实际功率成功 | 条数={len(records)}")
            return records

        except Exception as e:
            logger.error(f"查询实际功率失败: {e}")
            return []

    def query_weather(self, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """
        查询历史气象数据（用于模型训练）
        """
        try:
            client = self._get_client()
            query = f'''
                from(bucket: "{self.bucket}")
                  |> range(start: {start_time}, stop: {end_time})
                  |> filter(fn: (r) => r._measurement == "weather")
                  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                  |> yield(name: "weather")
            '''
            result = client.query_api().query(query, org=self.org)
            records = []
            for table in result:
                for record in table.records:
                    records.append({
                        "timestamp": record.get_time().isoformat(),
                        "ghi": float(record.values.get("ghi", 0.0)),
                        "temp": float(record.values.get("temp", 0.0)),
                        "humidity": float(record.values.get("humidity", 0.0)),
                        "wind_speed": float(record.values.get("wind_speed", 0.0)),
                        "cloud_cover": float(record.values.get("cloud_cover", 0.0)),
                    })
            logger.info(f"查询气象数据成功 | 条数={len(records)}")
            return records

        except Exception as e:
            logger.error(f"查询气象数据失败: {e}")
            return []

    @staticmethod
    def _parse_timestamp(ts) -> int:
        """解析时间戳为毫秒级整数"""
        if ts is None:
            return int(datetime.utcnow().timestamp() * 1000)
        if isinstance(ts, (int, float)):
            return int(ts) if ts > 1e12 else int(ts * 1000)
        if isinstance(ts, str):
            try:
                # 处理ISO格式
                ts_clean = ts.replace("Z", "+00:00")
                dt = datetime.fromisoformat(ts_clean)
                return int(dt.timestamp() * 1000)
            except ValueError:
                return int(datetime.utcnow().timestamp() * 1000)
        return int(datetime.utcnow().timestamp() * 1000)

    def close(self):
        """关闭连接"""
        if self._write_api:
            self._write_api.close()
        if self._client:
            self._client.close()
            self._client = None
            self._write_api = None
        logger.info("InfluxDB连接已关闭")
