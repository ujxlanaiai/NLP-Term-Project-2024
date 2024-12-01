# Input: Query
# Ouptut: Answer

# Context: Article inside the content of the title which is filtered with LLM.

from langchain_upstage import UpstageEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class BlockE:
    def __init__(self, api_key, embedded_title, embedded_doc2, doc2, temp):
        self.api_key = api_key
        self.embeddings = UpstageEmbeddings(
            api_key = self.api_key,
            model="embedding-query"
        )
        self.embedded_title = embedded_title # embedded_title.json
        self.embedded_article = embedded_doc2 # embedded_doc2.json
        self.doc2 = doc2 # doc_1.json
        
        self.llm = ChatUpstage(api_key=api_key,
                               model='solar-1-mini-chat',
                               temperature=temp)
        
    
    def search_title_similarity(self, query):
        """
            Get top 3 similar title with query.
        """
        # query = query.split("\n(A)")[0]
        query_vec = self.embeddings.embed_query(query)
        
        with open(self.embedded_title, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        similarities = []
        
        for idx, entry in data.items():
            title_vec = entry.get("title")
            
            query_vec = np.array(query_vec)
            title_vec = np.array(title_vec[0])
            
            similarity = cosine_similarity([query_vec], [title_vec])[0][0]
            similarities.append((idx, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_3_indices = [sim[0] for sim in similarities[:3]]
        
        return query_vec, top_3_indices
    
    
    def search_article_similarity(self, query):
        
        query_vec, top_3_title_indices = self.search_title_similarity(query)
        top_1_title_idx = top_3_title_indices[0]
        
        with open(self.embedded_article, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        article_similarities = []
        
        for idx in top_3_title_indices:
            content = data.get(str(idx), {}).get("content", [])

            for article in content:
                # print("article no.", article["article_no."])
                article_no = article["article_no."]
                article_vec = np.array(article["embedded_content"])
                
                similarity = cosine_similarity([query_vec], [article_vec[0]])[0][0]
                article_similarities.append((article_no, similarity))
            
            article_similarities.sort(key=lambda x: x[1], reverse=True)
            top_3_article_indices = (
                [sim[0] for sim in article_similarities[:3]]
                if len(article_similarities) >= 3
                else [sim[0] for sim in article_similarities]
                )
        
            break # use only one title
        
        return top_1_title_idx, top_3_article_indices
    
    
    def get_article_text(self, query):
        
        with open(self.doc2, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        title_idx, top_3_article_indices = self.search_article_similarity(query)
        
        article_contents = data.get(str(title_idx), {}).get("content", [])
        
        top_3_article_contents = []
        top_3_article_contents = [article_contents[idx] for idx in top_3_article_indices]
        
        return top_3_article_contents

    
    def filter_article(self, query):
        
        top_3_article_contents = self.get_article_text(query)
        
        prompt_template_parse = PromptTemplate.from_template(
        """
        Identify and extract the WHOLE, SUFFICIENT part from the context that can help to answer the question. 
        Do not modify the excerpt; simply extract it as-is.
        
        Response format:
        <Start>
        [Extracted part] your excerpt
        <End>

        Now, here are the question and context.
        Question: {question}
        Context: {context}
        """
        )
        
        chain = prompt_template_parse | self.llm
        response = chain.invoke({"question": query, "context": top_3_article_contents[0]})

        context_identified = response.content
        # print(context_identified)
        
        return context_identified
        

    def get_response(self, query):

        prompt_template = PromptTemplate.from_template(
        """
        Please provide the MOST exact correct answer based on the given context.
        
        Response format:
        <Start>
        [Answer]: (A) answer / (X) If no certain answer!!
        [Reason]: Your short reason why.
        
        [Final Answer]: (A) Your final, true answer after reviewing your [Reason] once again.
        <End>
        
        Now, here are the question and context:
        Question: {question}
        Context: {context}
        """
        )
        
        chain = prompt_template | self.llm
        context = self.filter_article(query)
        
        # print(context)
        
        response = chain.invoke({"question": query, "context": context})
        
        return response.content