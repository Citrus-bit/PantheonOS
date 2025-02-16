import asyncio
from magique.client import connect_to_server, ServiceProxy

from .constant import DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT


async def connect_remote(
        service_name_or_id: str,
        server_host: str = DEFAULT_SERVER_HOST,
        server_port: int = DEFAULT_SERVER_PORT,
        timeout: float = 5.0,
        time_delta: float = 0.5,
        ) -> ServiceProxy:
    server = await connect_to_server(
        server_host,
        server_port,
    )
    service = None

    async def _retry():
        nonlocal service
        while service is None:
            try:
                service = await server.get_service(service_name_or_id)
            except ValueError:
                await asyncio.sleep(time_delta)

    await asyncio.wait_for(_retry(), timeout)

    return service
