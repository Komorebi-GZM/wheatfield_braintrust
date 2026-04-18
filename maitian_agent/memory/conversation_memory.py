"""
对话记忆管理
短期记忆实现
"""
from typing import List, Dict, Any, Optional
from langchain_core.memory import BaseMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory, MongoDBChatMessageHistory
import os
import json
from datetime import datetime


class ConversationMemory:
    """对话记忆管理器

    短期记忆：ConversationBufferWindowMemory，保留最近N轮对话
    """

    def __init__(
        self,
        memory_type: str = "buffer",
        conversation_window: int = 5,
        session_id: str = "default",
        persist_directory: str = "./data/memory"
    ):
        self.memory_type = memory_type
        self.conversation_window = conversation_window
        self.session_id = session_id
        self.persist_directory = persist_directory

        if memory_type == "buffer":
            self.memory = ConversationBufferWindowMemory(
                k=conversation_window,
                chat_memory=self._load_chat_history(),
                return_messages=True
            )
        else:
            self.memory = ConversationBufferWindowMemory(
                k=conversation_window,
                return_messages=True
            )

        os.makedirs(persist_directory, exist_ok=True)

    def _load_chat_history(self):
        """加载聊天历史"""
        history_file = os.path.join(
            self.persist_directory,
            f"chat_history_{self.session_id}.json"
        )

        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    messages = json.load(f)
                    return self._messages_to_objects(messages)
            except Exception:
                pass
        return []

    def _messages_to_objects(self, messages: List[Dict]) -> List[BaseMessage]:
        """将字典转换为消息对象"""
        result = []
        for msg in messages:
            if msg.get("type") == "human":
                result.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("type") == "ai":
                result.append(AIMessage(content=msg.get("content", "")))
        return result

    def _objects_to_dict(self, messages: List[BaseMessage]) -> List[Dict]:
        """将消息对象转换为字典"""
        result = []
        for msg in messages:
            msg_dict = {"type": msg.type, "content": msg.content}
            if hasattr(msg, "additional_kwargs"):
                msg_dict["additional_kwargs"] = msg.additional_kwargs
            result.append(msg_dict)
        return result

    def save_context(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> None:
        """保存对话上下文

        Args:
            input_data: 输入数据
            output_data: 输出数据
        """
        human_message = input_data.get("human_input", "")
        ai_message = output_data.get("response", "")

        if human_message:
            self.memory.chat_memory.add_user_message(human_message)
        if ai_message:
            self.memory.chat_memory.add_ai_message(ai_message)

        self._persist_history()

    def _persist_history(self) -> None:
        """持久化聊天历史"""
        history_file = os.path.join(
            self.persist_directory,
            f"chat_history_{self.session_id}.json"
        )

        messages = self._objects_to_dict(self.memory.chat_memory.messages)
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

    def load_memory_variables(self) -> Dict[str, Any]:
        """加载记忆变量"""
        return self.memory.load_memory_variables({})

    def clear(self) -> None:
        """清除记忆"""
        self.memory.clear()
        self._persist_history()

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史

        Returns:
            对话历史列表
        """
        messages = self.memory.chat_memory.messages
        return [
            {
                "type": msg.type,
                "content": msg.content,
                "timestamp": datetime.now().isoformat()
            }
            for msg in messages
        ]


class RedisMemory(ConversationMemory):
    """Redis记忆（待配置）"""

    def __init__(
        self,
        session_id: str = "default",
        url: str = "redis://localhost:6379/0"
    ):
        # TODO: 实现Redis聊天历史
        raise NotImplementedError("Redis记忆待实现")


class MongoDBMemory(ConversationMemory):
    """MongoDB记忆（待配置）"""

    def __init__(
        self,
        session_id: str = "default",
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "maitian_agent"
    ):
        # TODO: 实现MongoDB聊天历史
        raise NotImplementedError("MongoDB记忆待实现")
