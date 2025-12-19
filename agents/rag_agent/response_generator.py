import logging
from typing import List, Dict, Any, Optional, Union

class ResponseGenerator:
    """
    Generates responses based on retrieved context and user query.
    """
    def __init__(self, config):
        """
        Initialize the response generator.
        
        Args:
            config: Configuration object
            llm: Large language model for response generation
        """
        self.logger = logging.getLogger(__name__)
        self.response_generator_model = config.rag.response_generator_model
        self.include_sources = getattr(config.rag, "include_sources", True)

    def _build_prompt(
            self,
            query: str, 
            context: str,
            chat_history: Optional[List[Dict[str, str]]] = None
        ) -> str:
        """
        Build the prompt for the language model.
        
        Args:
            query: User query
            context: Formatted context from retrieved documents
            chat_history: Optional chat history
            
        Returns:
            Complete prompt string
        """

        table_instructions = """
        Một số thông tin được tìm thấy được trình bày dưới dạng bảng. Khi sử dụng thông tin từ bảng:
        1. Trình bày dữ liệu dạng bảng bằng định dạng bảng markdown phù hợp với các tiêu đề, như thế này:
            | Cột 1 | Cột 2 | Cột 3 |
            |-------|-------|-------|
            | Giá trị 1 | Giá trị 2 | Giá trị 3 |
        2. Định dạng lại cấu trúc bảng để dễ đọc và dễ hiểu hơn
        3. Nếu có thành phần mới nào được đưa vào trong quá trình định dạng lại bảng, hãy đề cập rõ ràng
        4. Giải thích rõ ràng dữ liệu dạng bảng trong câu trả lời của bạn
        5. Tham chiếu bảng liên quan khi trình bày các điểm dữ liệu cụ thể
        6. Nếu phù hợp, tóm tắt các xu hướng hoặc mẫu hiển thị trong các bảng
        7. Nếu chỉ có số tham chiếu được đề cập và bạn có thể lấy các giá trị tương ứng như tiêu đề bài báo nghiên cứu hoặc tác giả từ ngữ cảnh, hãy thay thế số tham chiếu bằng các giá trị thực tế
        """

        response_format_instructions = """Hướng dẫn:
        1. Trả lời câu hỏi CHỈ dựa trên thông tin được cung cấp trong ngữ cảnh.
        2. Nếu ngữ cảnh không chứa thông tin liên quan để trả lời câu hỏi, hãy nêu: "Tôi không có đủ thông tin để trả lời câu hỏi này dựa trên ngữ cảnh được cung cấp."
        3. Không sử dụng kiến thức trước đó không có trong ngữ cảnh.
        4. Ngắn gọn và chính xác.
        5. Cung cấp câu trả lời có cấu trúc tốt với tiêu đề, tiêu đề phụ và cấu trúc bảng nếu cần ở định dạng markdown dựa trên kiến thức đã tìm kiếm. Giữ các tiêu đề và tiêu đề phụ ở kích thước nhỏ.
        6. Chỉ cung cấp các phần có ý nghĩa trong câu trả lời của chatbot. Ví dụ: không đề cập rõ ràng đến tài liệu tham khảo.
        7. Nếu liên quan đến các giá trị, hãy đảm bảo phản hồi với các giá trị chính xác có trong ngữ cảnh. Không bịa đặt giá trị.
        8. Không lặp lại câu hỏi trong câu trả lời hoặc phản hồi.
        9. **LUÔN TRẢ LỜI BẰNG TIẾNG VIỆT.**"""
            
        # Build the prompt
        prompt = f"""Bạn là một trợ lý y tế cung cấp thông tin chính xác dựa trên các nguồn y tế đã được xác minh.

        Đây là một vài tin nhắn cuối cùng từ cuộc trò chuyện của chúng tôi:
        
        {chat_history}

        Người dùng đã hỏi câu hỏi sau:
        {query}

        Tôi đã tìm thấy thông tin sau để giúp trả lời câu hỏi này:

        {context}

        {table_instructions}

        {response_format_instructions}

        Dựa trên thông tin được cung cấp, vui lòng trả lời câu hỏi của người dùng một cách kỹ lưỡng nhưng ngắn gọn bằng TIẾNG VIỆT.
        Nếu thông tin không chứa câu trả lời, hãy thừa nhận những hạn chế của thông tin có sẵn.

        Không cung cấp bất kỳ liên kết nguồn nào không có trong ngữ cảnh. Không bịa đặt bất kỳ liên kết nguồn nào.

        Phản hồi của Trợ lý Y tế (bằng Tiếng Việt):"""

        return prompt

    def generate_response(
            self,
            query: str,
            retrieved_docs: List[Dict[str, Any]],
            picture_paths: List[str],
            chat_history: Optional[List[Dict[str, str]]] = None,
        ) -> Dict[str, Any]:
        """
        Generate a response based on retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved document dictionaries
            chat_history: Optional chat history
            
        Returns:
            Dict containing response text and source information
        """
        try:
           
            # Extract content from documents for context
            doc_texts = [doc["content"] for doc in retrieved_docs]
            
            # Combine retrieved documents into a single context
            context = "\n\n===DOCUMENT SECTION===\n\n".join(doc_texts)
            
            # Build the prompt
            prompt = self._build_prompt(query, context, chat_history)
            
            # Generate response
            response = self.response_generator_model.invoke(prompt)
            
            # Extract sources for citation
            sources = self._extract_sources(retrieved_docs) if hasattr(self, 'include_sources') and self.include_sources else []
            
            # Calculate confidence
            confidence = self._calculate_confidence(retrieved_docs)

            # Add sources to response
            if hasattr(self, 'include_sources') and self.include_sources:
                response_with_source = response.content + "\n\n##### Source documents:"
                for current_source in sources:
                    source_path = current_source['path']
                    source_title = current_source['title']
                    response_with_source += f"\n- [{source_title}]({source_path})"
            else:
                response_with_source = response.content
            
            # Add picture paths to response
            response_with_source_and_picture_paths = response_with_source + "\n\n##### Reference images:"
            for picture_path in picture_paths:
                response_with_source_and_picture_paths += f"\n- [{picture_path.split('/')[-1]}]({picture_path})"
            
            # Format final response
            result = {
                "response": response_with_source_and_picture_paths,
                "sources": sources,
                "confidence": confidence
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error while generating a response. Please try rephrasing your question.",
                "sources": [],
                "confidence": 0.0
            }

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract source information from retrieved documents for citation.
        
        Args:
            documents: List of retrieved document dictionaries
            
        Returns:
            List of source information dictionaries
        """
        sources = []
        seen_sources = set()  # Track unique sources to avoid duplicates
        
        for doc in documents:
            # Extract source and source_path
            source = doc.get("source")
            source_path = doc.get("source_path")
            
            # Skip if no source information is available
            if not source:
                continue
                
            # Create a unique identifier for this source
            source_id = f"{source}|{source_path}"
            
            # Skip if we've already included this source
            if source_id in seen_sources:
                continue
                
            # Add to our sources list
            source_info = {
                "title": source,
                "path": source_path,
                "score": doc.get("combined_score", doc.get("rerank_score", doc.get("score", 0.0)))
            }
            
            sources.append(source_info)
            seen_sources.add(source_id)
        
        # Sort sources by score from highest to lowest
        sources.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Format the final sources list, removing the scores which were just used for sorting
        formatted_sources = []
        for source in sources:
            formatted_source = {
                "title": source["title"],
                "path": source["path"]
            }
            formatted_sources.append(formatted_source)
            
        return formatted_sources

    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on retrieved documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Confidence score between 0 and 1
        """
        if not documents:
            return 0.0
            
        # Use combined score (both reranker and cosine similarity) if available, otherwise use original score
        if "combined_score" in documents[0]:
            scores = [doc.get("combined_score", 0) for doc in documents[:3]]
        elif "rerank_score" in documents[0]:
            scores = [doc.get("rerank_score", 0) for doc in documents[:3]]
        else:
            scores = [doc.get("score", 0) for doc in documents[:3]]
            
        # Average of top 3 document scores or fewer if less than 3
        return sum(scores) / len(scores) if scores else 0.0