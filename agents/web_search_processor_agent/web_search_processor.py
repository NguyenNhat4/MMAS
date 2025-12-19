import os
from .web_search_agent import WebSearchAgent
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class WebSearchProcessor:
    """
    Processes web search results and routes them to the appropriate LLM for response generation.
    """
    
    def __init__(self, config):
        self.web_search_agent = WebSearchAgent(config)
        
        # Initialize LLM for processing web search results
        self.llm = config.web_search.llm
    
    def _build_prompt_for_web_search(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Build the prompt for the web search.
        
        Args:
            query: User query
            chat_history: chat history
            
        Returns:
            Complete prompt string
        """
        # Add chat history if provided
        # print("Chat History:", chat_history)
            
        # Build the prompt
        prompt = f"""Đây là một vài tin nhắn cuối cùng từ cuộc trò chuyện của chúng tôi:

        {chat_history}

        Người dùng đã hỏi câu hỏi sau:

        {query}

        Hãy tóm tắt chúng thành một câu hỏi duy nhất, được hình thành tốt chỉ khi cuộc trò chuyện trước đây có liên quan đến truy vấn hiện tại để có thể sử dụng cho tìm kiếm trên web.
        Giữ cho nó ngắn gọn và đảm bảo nắm bắt được ý định chính đằng sau cuộc thảo luận.
        """

        return prompt
    
    def process_web_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Fetches web search results, processes them using LLM, and returns a user-friendly response.
        """
        # print(f"[WebSearchProcessor] Fetching web search results for: {query}")
        web_search_query_prompt = self._build_prompt_for_web_search(query=query, chat_history=chat_history)
        # print("Web Search Query Prompt:", web_search_query_prompt)
        web_search_query = self.llm.invoke(web_search_query_prompt)
        # print("Web Search Query:", web_search_query)
        
        # Retrieve web search results
        web_results = self.web_search_agent.search(web_search_query.content)

        # print(f"[WebSearchProcessor] Fetched results: {web_results}")
        
        # Construct prompt to LLM for processing the results
        llm_prompt = (
            "Bạn là một trợ lý AI chuyên về thông tin y tế. Dưới đây là các kết quả tìm kiếm trên web "
            "được tìm thấy cho truy vấn của người dùng. Hãy tóm tắt và tạo ra một câu trả lời hữu ích, ngắn gọn bằng TIẾNG VIỆT. "
            "Chỉ sử dụng các nguồn đáng tin cậy và đảm bảo tính chính xác về y tế.\n\n"
            f"Câu hỏi: {query}\n\nKết quả tìm kiếm Web:\n{web_results}\n\nCâu trả lời (bằng Tiếng Việt):"
        )
        
        # Invoke the LLM to process the results
        response = self.llm.invoke(llm_prompt)
        
        return response
