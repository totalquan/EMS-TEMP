"""
数据接入层 - Kafka消费者
接收气象站、逆变器、电表的实时数据流
支持多Topic并行消费，数据写入InfluxDB + S3原始数据湖
"""
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from loguru import logger

try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError
except ImportError:
    KafkaConsumer = None
    KafkaError = Exception

from config.settings import settings


class KafkaDataConsumer:
    """
    Kafka多主题数据消费者
    统一接收气象站、逆变器、电表实时数据
    """

    def __init__(self, on_message_callback: Optional[Callable] = None):
        """
        :param on_message_callback: 消息处理回调函数，签名: callback(topic, data_dict)
        """
        self.bootstrap_servers = settings.kafka.bootstrap_servers
        self.group_id = settings.kafka.group_id
        self.topics = [
            settings.kafka.topic_weather,
            settings.kafka.topic_inverter,
            settings.kafka.topic_meter,
        ]
        self.callback = on_message_callback
        self._consumer: Optional[KafkaConsumer] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _create_consumer(self) -> KafkaConsumer:
        """创建Kafka消费者实例"""
        if KafkaConsumer is None:
            raise RuntimeError("kafka-python 未安装，请执行 pip install kafka-python")

        consumer = KafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset=settings.kafka.auto_offset_reset,
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            max_poll_records=500,
            consumer_timeout_ms=1000,
        )
        logger.info(f"Kafka消费者创建成功 | 服务器={self.bootstrap_servers} | 主题={self.topics}")
        return consumer

    def _process_message(self, topic: str, value: Dict[str, Any], key: Optional[str] = None):
        """
        处理单条Kafka消息
        :param topic: 消息主题
        :param value: 消息内容（已解析为dict）
        :param key: 消息key
        """
        try:
            # 统一添加接收时间戳
            value["_received_at"] = datetime.utcnow().isoformat() + "Z"
            value["_source_topic"] = topic

            # 调用用户回调
            if self.callback:
                self.callback(topic, value)

            logger.debug(f"消息处理完成 | topic={topic} | key={key}")

        except Exception as e:
            logger.error(f"消息处理异常 | topic={topic} | error={e} | value={value}")

    def _consume_loop(self):
        """消费主循环（在线程中运行）"""
        logger.info("Kafka消费循环启动")
        consecutive_errors = 0

        while self._running:
            try:
                if self._consumer is None:
                    self._consumer = self._create_consumer()

                # 拉取消息
                messages = self._consumer.poll(timeout_ms=1000)

                if messages:
                    for topic_partition, records in messages.items():
                        for record in records:
                            self._process_message(
                                topic=record.topic,
                                value=record.value,
                                key=record.key,
                            )
                    consecutive_errors = 0

            except KafkaError as e:
                consecutive_errors += 1
                logger.error(f"Kafka连接错误 (第{consecutive_errors}次): {e}")
                # 关闭旧连接，准备重连
                if self._consumer:
                    try:
                        self._consumer.close()
                    except Exception:
                        pass
                    self._consumer = None
                # 指数退避重连
                wait_time = min(2 ** consecutive_errors, 60)
                logger.warning(f"{wait_time}秒后尝试重连Kafka...")
                time.sleep(wait_time)

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"消费循环未知异常: {e}")
                time.sleep(5)

        logger.info("Kafka消费循环已停止")

    def start(self):
        """启动消费者（后台线程）"""
        if self._running:
            logger.warning("消费者已在运行中")
            return

        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True, name="kafka-consumer")
        self._thread.start()
        logger.info("Kafka消费者已启动（后台线程）")

    def stop(self):
        """停止消费者"""
        self._running = False
        if self._consumer:
            try:
                self._consumer.close()
            except Exception as e:
                logger.error(f"关闭Kafka消费者异常: {e}")
            self._consumer = None
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("Kafka消费者已停止")

    def consume_once(self, timeout_ms: int = 5000) -> Dict[str, list]:
        """
        单次拉取消息（用于测试/手动触发）
        :return: {topic: [message_dict, ...]}
        """
        if self._consumer is None:
            self._consumer = self._create_consumer()

        result = {t: [] for t in self.topics}
        messages = self._consumer.poll(timeout_ms=timeout_ms)

        for topic_partition, records in messages.items():
            for record in records:
                if record.topic in result:
                    result[record.topic].append(record.value)

        return result


# ============================================
# 消息数据格式定义（供参考）
# ============================================

# 气象站数据格式:
# {
#     "station_id": "weather_001",
#     "timestamp": "2026-09-04T10:00:00+08:00",
#     "ghi": 650.5,          # 水平面总辐照度 W/㎡
#     "dhi": 120.3,          # 散射辐照度 W/㎡
#     "temp": 22.5,           # 环境温度 ℃
#     "humidity": 45.2,       # 相对湿度 %
#     "wind_speed": 3.5,      # 风速 m/s
#     "wind_dir": 180,        # 风向 °
#     "pressure": 865.2,      # 气压 hPa
#     "cloud_cover": 20       # 云量 %
# }

# 逆变器数据格式:
# {
#     "inverter_id": "inv_001",
#     "timestamp": "2026-09-04T10:00:00+08:00",
#     "dc_voltage": 620.5,    # 直流电压 V
#     "dc_current": 45.2,     # 直流电流 A
#     "dc_power": 28000.0,    # 直流功率 W
#     "ac_voltage": 380.0,    # 交流电压 V
#     "ac_current": 72.5,     # 交流电流 A
#     "ac_power": 27500.0,    # 交流功率 W
#     "ac_frequency": 50.02,  # 交流频率 Hz
#     "temperature": 45.2,    # 逆变器温度 ℃
#     "status": "running",    # 运行状态
#     "daily_energy": 125.5   # 当日发电量 kWh
# }

# 电表数据格式:
# {
#     "meter_id": "meter_001",
#     "timestamp": "2026-09-04T10:00:00+08:00",
#     "voltage_a": 220.1,     # A相电压 V
#     "voltage_b": 219.8,
#     "voltage_c": 220.3,
#     "current_a": 35.2,      # A相电流 A
#     "current_b": 34.8,
#     "current_c": 35.5,
#     "active_power": 23500.0, # 有功功率 W
#     "reactive_power": 1200.0, # 无功功率 var
#     "power_factor": 0.985,   # 功率因数
#     "frequency": 50.01,      # 频率 Hz
#     "total_energy_import": 15230.5,  # 正向有功总电能 kWh
#     "total_energy_export": 8560.2     # 反向有功总电能 kWh
# }
