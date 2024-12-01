# main for ewha

from collections import Counter
import re

from ewha.llms.block_A import BlockA
from ewha.llms.block_B import BlockB
from ewha.llms.block_C import BlockC
from ewha.llms.block_D import BlockD
from ewha.llms.block_E import BlockE
from ewha.llms.block_F import BlockF
from ewha.llms.utils import extract_answer


class EwhaQA:
    def __init__(self, api_key, code=0):
        self.api_key = api_key
        
        self.embedded_title = "./ewha/docs/embedded_title.json"
        self.embedded_doc0 = "./ewha/docs/embedded_doc0.json"
        self.embedded_doc1 = "./ewha/docs/embedded_doc1.json"
        self.embedded_doc2 = "./ewha/docs/embedded_doc2.json"
        self.doc0 = "./ewha/docs/doc_0.json"
        self.doc1 = "./ewha/docs/doc_1.json"
        self.doc2 = "./ewha/docs/doc_2.json"
        
        self.llm_block_A = BlockA(self.api_key, self.embedded_doc0, self.doc0, temp=0.1)
        self.llm_block_B = BlockB(self.api_key, self.embedded_doc1, self.doc1, temp=0.1)
        self.llm_block_C = BlockC(self.api_key, self.embedded_title, self.doc1, temp=0.1)
        self.llm_block_D = BlockD(self.api_key, self.embedded_title, self.embedded_doc2, self.doc2, temp=0.1)
        self.llm_block_E = BlockE(self.api_key, self.embedded_title, self.embedded_doc2, self.doc2, temp=0.1)
        self.llm_block_F = BlockF(self.api_key, self.embedded_doc2, self.doc2, temp=0.1)
        
        if code == 0:
            self.llms = [self.llm_block_A,
                     self.llm_block_B,
                     self.llm_block_C,
                     self.llm_block_D,
                     self.llm_block_E]
        elif code == 1:
            self.llms = [self.llm_block_B,
                     self.llm_block_C,
                     self.llm_block_D,
                     self.llm_block_E]
        elif code == 2:
            self.llms = [self.llm_block_B,
                     self.llm_block_D,
                     self.llm_block_E]
        elif code == 3:
            self.llms = [self.llm_block_F,
                     self.llm_block_B,
                     self.llm_block_C,
                     self.llm_block_D,
                     self.llm_block_E]
        elif code == 4:
            self.llms = [self.llm_block_A,
                     self.llm_block_B,
                     self.llm_block_D,
                     self.llm_block_E]
        elif code == 5:
            self.llms = [self.llm_block_A,
                         self.llm_block_E]
        elif code == 6:
            self.llms = [self.llm_block_A]
        elif code == 7:
            self.llms = [self.llm_block_E]
        elif code == 8:
            self.llms = [self.llm_block_A,
                         self.llm_block_B,
                         self.llm_block_E]
        elif code == 9:
            self.llms = [self.llm_block_B]
        elif code == 10:
            self.llms = [self.llm_block_C]
        elif code == 11:
            self.llms = [self.llm_block_D]
        else:
            raise ValueError("Please select code between 0-11.")

            
    
    def change_temperature(self, temp):
        self.llm_block_A = BlockA(self.api_key, self.embedded_doc0, self.doc0, temp=temp)
        self.llm_block_B = BlockB(self.api_key, self.embedded_doc1, self.doc1, temp=temp)
        self.llm_block_C = BlockC(self.api_key, self.embedded_title, self.doc1, temp=temp)
        self.llm_block_D = BlockD(self.api_key, self.embedded_title, self.embedded_doc2, self.doc2, temp=temp)
        self.llm_block_E = BlockE(self.api_key, self.embedded_title, self.embedded_doc2, self.doc2, temp=temp)
        self.llm_block_F = BlockF(self.api_key, self.embedded_doc2, self.doc2, temp=temp)
    
    
    def extract_question(self, text):
        # Use regex to match and remove only the "QUESTION n)" part
        text = re.sub(r"QUESTION\d+\)\s*", "", text)
    
        return text
        
        
    def get_response(self, query, max_iterations=10):
        """
        Ensemble equally for all LLM Blocks.
        Adjusts temperature in case of ties and stops after a maximum number of iterations.
        """
        num = 5  # Number of answers for each LLM block
        answers = []
        temp = 0.1
        
        for iteration in range(max_iterations):
            answers.clear()  # Reset answers for the current iteration

            # Collect responses from all LLM blocks
            for llm in self.llms:
                for _ in range(num):
                    try:
                        # Preprocess question
                        query = self.extract_question(query)
                        response = llm.get_response(query)
                        # print("♥️", response)
                        answer = extract_answer(response)
                        if 'X' not in answer:
                            answers.append(answer)
                        # print(answers)
                    except Exception as e:
                        print(f"Error querying LLM: {e}")
            
            # Count occurrences of answers
            answer_counts = Counter(answers)
            max_count = max(answer_counts.values(), default=0)  # Handle empty answers gracefully
            most_common_answers = [ans for ans, count in answer_counts.items() if count == max_count]

            if len(most_common_answers) == 1:
                self.change_temperature(temp=0.1)
                # print(answers)
                return most_common_answers[0]  # Return the most common answer

            # Adjust temperature to encourage more variability
            for llm in self.llms:
                temp += 0.1
                if temp > 0.9: temp = 0.9
                self.change_temperature(temp=temp)
                
            print(f"Updated temperature to {temp}")
            
        # Return a fallback value if no single answer is found
        print("Exceeded maximum iterations without resolving ties.")
        
        self.change_temperature(temp=0.1)
        return most_common_answers[0]