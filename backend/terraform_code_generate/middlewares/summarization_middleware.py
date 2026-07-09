import logging

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.runtime import Runtime

from backend.terraform_code_generate.agents.agent_state import CodeAgentState

logger = logging.getLogger(__name__)

class ContextSummarizationMiddleware(SummarizationMiddleware):

    def __init__(
            self,
            *args,
            agent_name: str | None = None,
            **kwargs,
    ):
        super().__init__(*args,**kwargs)
        self._agent_name = agent_name

    def before_model(self, state: CodeAgentState, runtime: Runtime) -> dict | None:
        return self._maybe_summarize(state)

    async def abefore_model(self, state: CodeAgentState, runtime: Runtime) -> dict | None:
        return await self._maybe_summarize(state)

    def _maybe_summarize(self, state: CodeAgentState) -> dict | None:
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        logger.info(f" begin to summarization the context message, messages length: {messages.__len__()}")

        new_messages : list[HumanMessage|ToolMessage|AIMessage] = []
        length = messages.__len__()
        if length < 4:
            return None

        for message in messages:
            if isinstance(message, HumanMessage):
                new_messages.append(message)
            elif isinstance(message, ToolMessage) and message.name == "web_search_and_extract":
                if new_messages.__len__() >= 2:
                    new_messages[1]=message
                else:
                    new_messages.append(message)

        new_messages.append(messages[length-2])
        new_messages.append(messages[length-1])

        logger.info(f" end summarization the context message, messages length: {new_messages.__len__()}")

        return {
            "messages": [*new_messages]
        }
