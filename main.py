from frontend import create_app
from flask import render_template, request
import os
import csv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader
import pinecone
import openai

os.environ["OPENAI_API_KEY"] = ""  # Include your personal OPEN_API_KEY
openai_api_key = ""  # Include your personal OPEN_API_KEY
PINECONE_API_KEY = "bc217cd8-f93e-4aab-8fe3-f8ba4c145d6d"  # Pinecone API key with "Introduction to Algorithm" vectors
PINECONE_ENV = "us-east4-gcp"  # Personal API Environment with "Introduction to Algorithm" vector

embed_model = "text-embedding-ada-002"  # OpenAI Embedding model

pinecone.init(  # Initializing Pinecone
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # Initializing OpenAI Embeddings model
index = pinecone.Index("comp251-algorithms")  #Index name in Pinecone
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)  # Initializing OpenAI model

app = create_app()


@app.route("/")
def home():
    return render_template("home.html")


@app.route('/process', methods=['POST'])
def process():
    data = request.form.get('data')
    result1 = prompting(data)
    return result1


"""
This method allows you to add pdf files to the vector database. Currently the vector database is filled with all
the documents related to Introduction to Calculus. Please do not add any more files to this vector. This method is shown
just for educational purposes.
"""


def vec_init(pdf):
    loader2 = UnstructuredPDFLoader("" + pdf)
    data = loader2.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    docs = text_splitter.split_documents(data)
    index_name = "comp251-algorithms"
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return docsearch


"""
This method is used to do random testing and checks if the model received the uploaded documents by connecting the 
vector database with a open AI model and just having a question/answer session. Just ask a query and it returns an 
answer.
"""


def chk_init(query):
    index_name = "comp251-algorithms"
    docsearch = Pinecone.from_existing_index(embedding=embeddings, index_name=index_name)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
    answer = qa.run(query)
    return answer


"""
Some PDF's have tables in them as well. Unfortunately, when we upload those tables as a PDF itself, it does not work the
best way possible. Therefore, I created a table parser. I download those tables as CSV files and then parse those tables
accordingly. We can change the parsing technique by making changes to the parsing_tabular_to_text(string) method. Please
do not add any more files to this vector. This method is shown just for educational purposes.
"""


def vec_init_from_csv(csv1):
    index_name = "comp251-algorithms"
    parsing_tabular_to_text(csv1)
    loader = UnstructuredFileLoader("")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
    docs = text_splitter.split_documents(docs)
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return docsearch

'''
You could change the line2 section of this code based on what kind of table information you would be adding to the
datebase. The current format is based of the tabular data I had at the time.
Example-
f' The stats on search algorithms {Binary Search} are - 
{Complexity}: {.....}
{Latency}: {......}
'''
def parsing_tabular_to_text(csv1):
    f_list = list()
    new_line = "\n"

    with open("" + csv1, mode='r') as file:
        csvFile = csv.reader(file)
        list1 = list(csvFile)
        element_length = len(list1[0])
        # print(element_length)
    f = open("", "w")
    for line in list1:
        if list1[1][0] == "1" or list1[0][0] == "1":
            line2 = f'The following stat shows information on {list1[0][1]}- {line[1]} and they are: {new_line}{new_line.join(f"{list1[0][n + 1]}: {line[n + 1]}" for n in range(1, len(list1[0]) - 1))}{new_line}{new_line}{new_line}{new_line}'

            if line2.find(list1[0][2] + ": " + list1[0][2]) == -1:
                f.write(line2)

        else:
            line2 = f'The following table shows information/stats on {list1[0][0]}- {line[0]} and they are: {new_line}{new_line.join(f"{list1[0][n + 1]}: {line[n + 1]}" for n in range(len(list1[0]) - 1))}{new_line}{new_line}{new_line}{new_line}'
            if line2.find(list1[0][1] + ": " + list1[0][1]) == -1:
                f.write(line2)

        # f_list.append(line2)
    list1.remove(list1[0])


"""
This is the main method where the user asks the model its main query and using a stronger model (GPT 3.5), the llm
returns back an answer. To control the model, we can make changes to the 'prompt' section of this function. User can
accordingly do prompt engineering through this specific model. For now, I've kept the prompt as basic as possible.
"""


def prompting(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    xq = res['data'][0]['embedding']
    res = index.query(xq, top_k=10, include_metadata=True)
    context = [item['metadata']['text'] for item in res['matches']]
    augmented_query = "\n\n\n\n---\n\n\n\n".join(context) + "\n\n----\n\n" + query
    prompt = f""" You are Q&A bot. A highly intelligent system that answers user questions based on the information
    provided by the user above for each question. If the information cannot be found in the information provided by
    the user, you truthfully say "I don't know."
    """

    result = openai.ChatCompletion.create(
        model="gpt-35-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    # return result
    print(result['choices'][0]['message']['content'])

    return result['choices'][0]['message']['content']


if __name__ == '__main__':
    app.run(debug=True)
