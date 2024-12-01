# Input: Query
# Ouptut: Answer

# Context: Whole content - divided with titles

from langchain_upstage import UpstageEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class BlockB:
    def __init__(self, api_key, embedded_doc1, doc1, temp=0):
        self.api_key = api_key
        self.embeddings = UpstageEmbeddings(
            api_key = self.api_key,
            model="embedding-query"
        )
        self.embedded_doc1 = embedded_doc1
        self.doc1 = doc1 # doc_1.json
        
        self.llm = ChatUpstage(api_key=api_key,
                               model='solar-1-mini-chat',
                               temperature=temp)


    def search_content_similarity(self, query):
        """
            Get top 3 simliar content.
        """
        query_vec = self.embeddings.embed_query(query)
        
        with open(self.embedded_doc1, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        similarities = []
        
        for idx, entry in data.items():
            content_vec = entry.get("title")
            
            query_vec = np.array(query_vec)
            content_vec = np.array(content_vec[0])
            
            similarity = cosine_similarity([query_vec], [content_vec])[0][0]
            similarities.append((idx, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_3_indices = [sim[0] for sim in similarities[:3]]
        # print("Title: ", top_3_indices[0])
        
        return top_3_indices


    def get_content_text(self, query):
        """
            Get text of top 3 content.
        """
        top_3_indices = self.search_content_similarity(query)
        
        with open(self.doc1, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        top_3_contents = [data[str(idx)].get("content") for idx in top_3_indices]
        
        return top_3_contents
    
    
    def get_response(self, query, top_idx=0):
        """
            Get final response with context.
            - top_idx: Index number of top3
        """
        
        prompt_template = PromptTemplate.from_template(
        """
        Please provide the MOST precise and accurate answer based on the given context.
    
        Response format:
        <Start>
        [Answer]: (A) answer
        [Reason]: Your short reason why.
    
        [Final Answer]: (A) Your final, true answer after reviewing your [Reason] once again.
        <End>
    
        Now, here are the question and context:
        Question: {question}
        Context: {context}
        """)
        
        chain = prompt_template | self.llm
        
        context = self.get_content_text(query)
        response = chain.invoke({"question": query, "context": context[top_idx]})
        
        # print(context[0])

        return response.content
        
        