import sys

from magique.ai.constant import DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT
from magique.worker import MagiqueWorker

from ..agent import Agent
from ..team import Team
from ..memory import MemoryManager


class ChatRoom:
    def __init__(
        self,
        agent: Agent | Team,
        memory_manager: MemoryManager,
        name: str = "Pantheon Chat Room",
        description: str = "Chatroom for Pantheon agents",
        worker_params: dict | None = None,
    ):
        self.agent = agent
        self.memory_manager = memory_manager
        self.name = name
        self.description = description
        _worker_params = {
            "service_name": name,
            "server_host": DEFAULT_SERVER_HOST,
            "server_port": DEFAULT_SERVER_PORT,
            "need_auth": False,
        }
        if worker_params is not None:
            _worker_params.update(worker_params)
        self.worker = MagiqueWorker(**_worker_params)
        self.setup_handlers()
        self.chat_memories = {}

    def setup_handlers(self):
        self.worker.register(self.new_chat)
        self.worker.register(self.delete_chat)
        self.worker.register(self.chat)

    async def new_chat(self, name: str | None = None) -> str:
        memory = await self.memory_manager.new_memory(name)
        self.chat_memories[memory.name] = memory
        return {"success": True, "message": "Chat created successfully", "chat_name": memory.name}

    async def delete_chat(self, name: str):
        try:
            await self.memory_manager.delete_memory(name)
            del self.chat_memories[name]
            return {"success": True, "message": "Chat deleted successfully"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    async def chat(
        self,
        chat_name: str,
        message: str,
        process_chunk = None,
        process_step_message = None,
    ):
        memory = self.chat_memories[chat_name]
        resp = await self.agent.run(
            message,
            memory=memory,
            process_chunk=process_chunk,
            process_step_message=process_step_message,
        )
        return {"success": True, "response": resp.content}

    async def run(self, log_level: str = "INFO"):
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level=log_level)
        logger.info(f"Remote Server: {self.worker.server_uri}")
        logger.info(f"Service Name: {self.worker.service_name}")
        logger.info(f"Service ID: {self.worker.service_id}")
        return await self.worker.run()
