from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from openai import AsyncOpenAI

from evolab_local.optovlab.config import AgentProviderConfig


AGENT_INSTRUCTIONS: dict[str, str] = {
    "data_mining": """
You are the OptoVLab Data Mining Agent. Help researchers extract evidence-backed OLED
device and material records from uploaded papers and analyze the resulting device table.
Use deterministic tools for mining status, retrieval, and statistics. Never invent a
reported value or evidence quotation. Distinguish extraction results from interpretation.
When a user asks to start an expensive pipeline, explain the action and rely on the host
application to execute it. Keep responses concise and action oriented.
""",
    "device_modeling": """
You are the OptoVLab Device Modeling Agent. Help researchers inspect datasets, design
graph-neural-network experiments, write reproducible modeling code, and operate the
configured Slurm HPC environment. Treat held-out evaluation, leakage prevention, and
dataset fingerprints as mandatory. Never claim a job was submitted unless the scheduler
tool returned a job ID. Training submission always requires explicit user confirmation.
""",
    "experimental_design": """
You are the OptoVLab Experimental Design Agent. Generate conservative, testable OLED
device recommendations grounded in retrieved database records. Every factual precedent
must cite its DOI and device ID. Separate observations, hypotheses, and proposed
experiments. State uncertainty and applicability limits. Do not invent HOMO, LUMO, PLQY,
mobility, lifetime, or processing values that are absent from the retrieved evidence.
""",
}


class MicrosoftAgentRuntime:
    def __init__(self, config: AgentProviderConfig) -> None:
        self.config = config
        self._client: OpenAIChatCompletionClient | None = None
        self._http_client: httpx.AsyncClient | None = None

    @property
    def available(self) -> bool:
        return bool(self.config.api_key)

    def describe(self) -> dict[str, Any]:
        return {
            "framework": "Microsoft Agent Framework",
            "provider": self.config.provider,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "configured": self.available,
        }

    async def respond(
        self,
        agent_type: str,
        prompt: str,
        *,
        history: Sequence[dict[str, str]] = (),
        context: dict[str, Any] | None = None,
        tools: Sequence[Callable[..., Any]] = (),
    ) -> str:
        if not self.available:
            raise RuntimeError(f"{self.config.api_key_env} is not configured")
        transcript = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in list(history)[-12:]
        )
        context_text = json.dumps(context or {}, ensure_ascii=False, default=str)
        composite_prompt = (
            "Recent conversation:\n"
            f"{transcript or '(new session)'}\n\n"
            "Authoritative application context:\n"
            f"{context_text}\n\n"
            "Current user request:\n"
            f"{prompt}"
        )
        agent = Agent(
            client=self._get_client(),
            name=agent_type.replace("_", "-") + "-agent",
            description=f"OptoVLab {agent_type.replace('_', ' ')} specialist",
            instructions=AGENT_INSTRUCTIONS[agent_type],
            tools=list(tools),
        )
        response = await agent.run(composite_prompt)
        text = response.text.strip()
        if not text:
            raise RuntimeError("Agent returned an empty response")
        return text

    def build_coordinator(self, specialists: dict[str, Agent]) -> Agent:
        tools = [
            agent.as_tool(
                name=f"consult_{name}",
                description=f"Delegate to the OptoVLab {name.replace('_', ' ')} specialist.",
            )
            for name, agent in specialists.items()
        ]
        return Agent(
            client=self._get_client(),
            name="optovlab-coordinator",
            description="Routes cross-domain requests to OptoVLab specialists.",
            instructions=(
                "Route each task to the smallest relevant set of specialists. Preserve tool "
                "evidence, ask for explicit confirmation before model training, and synthesize "
                "a concise final response without fabricating scientific facts."
            ),
            tools=tools,
        )

    async def close(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
            self._client = None

    def _get_client(self) -> OpenAIChatCompletionClient:
        if self._client is None:
            self._http_client = httpx.AsyncClient(
                trust_env=False,
                timeout=httpx.Timeout(self.config.request_timeout_seconds),
            )
            openai_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                http_client=self._http_client,
            )
            self._client = OpenAIChatCompletionClient(
                model=self.config.model,
                async_client=openai_client,
            )
        return self._client
