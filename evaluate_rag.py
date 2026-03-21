import requests

test_questions = [
{
"question": "What is Naive Bayes?",
"expected_keywords": ["probability","probabilistic","classifier","classification"]
},

{
"question": "What assumption does Naive Bayes make?",
"expected_keywords": ["independent","independence","features"]
},

{
"question": "What is Logistic Regression used for?",
"expected_keywords": ["classification","predict","probability"]
},

{
"question": "What is a decision boundary in Logistic Regression?",
"expected_keywords": ["boundary","separate","classes"]
},

{
"question": "What is classification in machine learning?",
"expected_keywords": ["categories","classes","groups"]
},

{
"question": "What is binary classification?",
"expected_keywords": ["two","two classes","two categories"]
},

{
"question": "What is model training?",
"expected_keywords": ["learn","learning","data","training"]
}
]

def ask_ai(question):
    url = "http://127.0.0.1:8000/chat"
    response = requests.get(url, params={"prompt": question})
    return response.json()["response"]

def is_correct(answer, keywords):
    answer = answer.lower()

    for keyword in keywords:
        if keyword.lower() in answer:
            return True

    return False

correct = 0
total = len(test_questions)

for i, test in enumerate(test_questions):

    question = test["question"]
    keywords = test["expected_keywords"]

    print(f"\nRunning test {i+1}/{total}")
    print("Question:", question)

    answer = ask_ai(question)

    print("AI Answer:", answer)

    if is_correct(answer, keywords):
        print("Result: CORRECT")
        correct += 1
    else:
        print("Result: WRONG")

print("\n=================")
print("Total Questions:", total)
print("Correct Answers:", correct)
print("Accuracy:", (correct / total) * 100, "%")
