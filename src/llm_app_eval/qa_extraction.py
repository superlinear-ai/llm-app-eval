import instructor
import openai
from pydantic import BaseModel

instructor.patch()
# openai.api_key_path = "../../../openai_key"


class QuestionAnswer(BaseModel):
    question: str
    answer: str


class QuestionAnswerPairs(BaseModel):
    qa_pairs: list[QuestionAnswer]


def extract_question_answer_pairs(chunk: str) -> list[QuestionAnswer]:
    qas: QuestionAnswerPairs = openai.ChatCompletion.create(
        model="gpt-4",
        response_model=QuestionAnswerPairs,
        messages=[
            {
                "role": "system",
                "content": "Turn the provided text into a set of question and answer pairs. Use only the information provided in the text. Make the answers as short as possible.",
            },
            {"role": "user", "content": f"TEXT: {chunk}"},
        ],
    )
    return qas.qa_pairs


def save_qa_pairs(qa_pairs, file_path: str):
    """Store the question and answers into a CSV file."""
    with open(file_path, "w") as f:
        f.write("question,answer\n")
        for qa in qa_pairs:
            # Replace double quotes with single quotes
            answer = qa.answer.replace('"', "'")
            f.write(f'"{qa.question}","{answer}"\n')


def load_qa_pairs(file_path: str):
    """Load the question and answers from a CSV file."""
    qa_pairs = []
    with open(file_path) as f:
        f.readline()  # Skip header
        for line in f.readlines():
            # Split on the comma between the question and answer, indicated by the double quotes
            question, answer = line.split('","')
            # Remove the double quotes from the question and answer
            question = question.replace('"', "")
            answer = answer.replace('"', "")
            qa_pairs.append(QuestionAnswer(question=question, answer=answer))
    return qa_pairs
