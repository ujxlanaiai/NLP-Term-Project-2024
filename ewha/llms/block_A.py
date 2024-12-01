# Input: Query
# Ouptut: Answer

# Context: Whole content - divided with naive text length

from langchain_upstage import UpstageEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class BlockA:
    def __init__(self, api_key, embedded_doc0, doc0, temp=0):
        self.api_key = api_key
        self.embeddings = UpstageEmbeddings(
            api_key = self.api_key,
            model="embedding-query"
        )
        self.embedded_doc0 = embedded_doc0
        self.doc0 = doc0 # doc_0.json
        
        self.llm = ChatUpstage(api_key=api_key,
                               model='solar-1-mini-chat',
                               temperature=temp)
        
    
    def search_content_similiarity(self, query):
        """
            Get top 3 similar content.
        """
        query_vec = self.embeddings.embed_query(query)
        
        with open(self.embedded_doc0, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
            
        similarities = []
        
        for idx, entry in data.items():
            
            query_vec = np.array(query_vec)
            content_vec = np.array(entry)
            
            similarity = cosine_similarity([query_vec], content_vec)[0][0]
            similarities.append((idx, similarity))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_3_indices = [sim[0] for sim in similarities[:3]]
        # print("Title: ", top_3_indices[0])
        
        return top_3_indices

    
    def get_content_text(self, query):
        """
            Get text of top 3 content.
        """
        top_3_indices = self.search_content_similiarity(query)
        
        with open(self.doc0, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        top_3_contents = [data[str(idx)] for idx in top_3_indices]
        
        return top_3_contents


    def get_response(self, query, top_idx=0):
        """
            Get final response with context.
            - top_idx: Index number of top3
        """
        
        prompt_template = PromptTemplate.from_template(
        """
        Please provide the MOST precise and accurate answer based on the given context.
        IMPORTANT: If you cannot find the answer from the context, JUST RESPOND 'None'.
    
        Response format:
        <Start>
        [Answer]: (A) answer / (X) If not certain!
        [Reason]: Your short reason why.
    
        [Final Answer]: (A) Your final, true answer after reviewing your [Reason] once again.
        <End>
    
        Now, here are the question and context:
        Question: {question}
        Context: {context}
        """)
        
        chain = prompt_template | self.llm
        context = self.get_content_text(query)
        
        input_context = "".join(c for c in context[:top_idx+1])
        response = chain.invoke({"question": query, "context": input_context})
        
        # print(input_context)
        
        return response.content
        