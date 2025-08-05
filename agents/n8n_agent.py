from typing import Dict, Any
import aiohttp
import logging

logger = logging.getLogger(__name__)

class N8nAgent:
    """Simple agent to trigger n8n workflows via webhooks."""

    def __init__(self, base_url: str = "http://localhost:5678") -> None:
        self.base_url = base_url.rstrip('/')
        # default workflow mapping
        self.workflow_routes: Dict[str, str] = {
            "create_reminder": f"{self.base_url}/webhook/create-reminder"
        }

    def register_workflow(self, name: str, url: str) -> None:
        self.workflow_routes[name] = url

    async def execute(self, workflow: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow and return JSON response."""
        url = self.workflow_routes.get(workflow)
        if not url:
            raise ValueError(f"Unknown workflow: {workflow}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {"success": True, "data": data}
                    else:
                        text = await resp.text()
                        logger.error("n8n workflow %s failed: %s", workflow, text)
                        return {"success": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            logger.error("n8n request error: %s", e)
            return {"success": False, "error": str(e)}
