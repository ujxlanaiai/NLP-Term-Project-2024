import re

def extract_again(response):
    pattern = r"\(([A-JX])\)"  # Extract the last (A)-(J) and (X)
    match = re.search(pattern, response)
    if match:
        return match.group(0)
    else:
        return None

# TODO: modify to extract from final answer
def extract_answer(response):
    """
    Extracts the answer from the response using a regular expression.
    Expected formats: "[Final Answer]: (A)" or "최종답변: (A)"

    If no valid answer is found in the expected format, it falls back to extracting
    the last (A)-(J) or (X).
    """
    # Regular expression to match "[Final Answer]: (A)" or "최종답변: (A)"
    pattern = r"(?:\[Final Answer\]|최종답변):\s*(\((A|B|C|D|E|F|G|H|I|J|X)\))"
    match = re.search(pattern, response)

    if match:
        return match.group(1)  # Return the entire "(A)" part
    else:
        return extract_again(response)


def evaluate_accuracy(answers, responses):
    cnt = 0

    for answer, response in zip(answers, responses):
        print("-"*10)
        generated_answer = extract_answer(response)
        print(response)

        # check
        if generated_answer:
            generated_answer = generated_answer.replace("(", "")
            generated_answer = generated_answer.replace(")", "")
            print(f"generated answer: {generated_answer}, answer: {answer}")
        else:
            print("extraction fail")

        if generated_answer == None:
            continue
        if generated_answer in answer:
            cnt += 1

    print()
    print(f"acc: {(cnt/len(answers))*100}%") # TODO: change here!!!!!!!!!!!!!!!!

    return
