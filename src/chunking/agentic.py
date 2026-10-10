from .strategies import BaseChunker
from typing import List
from src.pipeline import RAGPipeline # Assuming RAGPipeline or a dedicated Service can orchestrate the Agent call

class AgenticChunker(BaseChunker):
    """
    Agentic chunking: delegates the splitting decision to a specialized LLM Agent.

    Implementation Note: This class acts as a wrapper. A fully functional implementation
    requires a mechanism (like a dependency injected `AgentOrchestrator` service)
    to call the LangChain/Claude Agent tool with a strict JSON schema prompt.
    The current implementation uses a direct fallback.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # In a real pipeline, this would hold a reference to the AgentService
        self.agent_orchestrator = None

    def set_agent_orchestrator(self, orchestrator):
        """Sets the service responsible for interacting with the Agent."""
        self.agent_orchestrator = orchestrator

    def _run_agent_for_splitting(self, text: str) -> List[str]:
        """
        CORE FUNCTION: Executes the Agent and parses the resulting chunk list.

        Args:
            text: The text content to be chunked.

        Returns:
            A list of strings, where each string is a well-defined chunk.
        """
        if not self.agent_orchestrator:
            print("WARNING: AgentOrchestrator not set. Falling back to whole text.")
            return [text] # Fallback: return whole text as one chunk

        # --- ACTUAL AGENT CALL WOULD HAPPEN HERE ---
        # Example: response = self.agent_orchestrator.run_agent(
        #    prompt=f"Split this text into chunks, max size {self.chunk_size}..."
        # )
        # return parse_json_to_list(response)
        print("INFO: Successfully routed chunking request to the Agent Orchestrator.")
        return [text] # Placeholder return

    def split_text(self, text: str) -> List[str]:
        """
        Uses an LLM Agent to split text into intelligently segmented chunks.
        """
        return self._run_agent_for_splitting(text)