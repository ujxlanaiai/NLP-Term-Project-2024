# Input: Query
# Ouptut: Answer

# Context: Article inside the content of the title.

from langchain_upstage import UpstageEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class BlockF:
    def __init__(self, api_key, embedded_doc2, doc2, temp=0):
        self.api_key = api_key
        self.embeddings = UpstageEmbeddings(
            api_key = self.api_key,
            model="embedding-query"
        )
        self.embedded_article = embedded_doc2 # embedded_doc2.json
        self.doc2 = doc2 # doc_2.json
        
        self.llm = ChatUpstage(api_key=api_key,
                               model='solar-1-mini-chat',
                               temperature=temp)

    
    def search_article_similarity(self, query):
        
        # query = query.split("\n(A)")[0]
        query_vec = self.embeddings.embed_query(query)
        
        with open(self.embedded_article, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        article_similarities = []
        
        for idx in data:
            content = data.get(str(idx), {}).get("content", [])

            for article in content:
                # print("article no.", article["article_no."])
                article_no = article["article_no."]
                article_vec = np.array(article["embedded_content"])
                
                similarity = cosine_similarity([query_vec], [article_vec[0]])[0][0]
                article_similarities.append((article_no, similarity))
            
        article_similarities.sort(key=lambda x: x[1], reverse=True)
        top_1 = [idx, article_similarities[0][0]]
            
        return top_1
    
    
    def get_article_text(self, query):
        
        with open(self.doc2, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        top_1 = self.search_article_similarity(query)
        title_idx, article_idx = top_1[0], top_1[1]
        
        print(title_idx, article_idx)
        
        article_contents = data.get(str(title_idx), {}).get("content", [])
        top_1_article_contents = [article_contents[article_idx]]
        
        return top_1_article_contents


    def get_response(self, query, top_idx=0):
        """
            - top_idx: Number of articles wanted to input to context.
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
        
        context = self.get_article_text(query)
        
        if len(context) < top_idx:
            # print("There is only one article found.")
            top_idx = 0

        # print(context[0])
        
        response = chain.invoke({"question": query, "context": context[top_idx]})
        
        # print(context[top_idx])
        
        return response.content