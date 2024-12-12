# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env/pyvenv.cfg")

import sys

# bring in deps
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer

import nest_asyncio
nest_asyncio.apply()

def summarize(filename, writeFile=True):
    # set up parser
    parser = LlamaParse(
        result_type="text"  # "markdown" and "text" are available
    )

    # use SimpleDirectoryReader to parse our file
    file_extractor = {".pdf": parser}
    try:
        documents = SimpleDirectoryReader(input_files=[f"data/{filename}.pdf"], file_extractor=file_extractor).load_data()
    except ValueError:
        return
    
    content = """"""
    if(writeFile):
        with open(filename+".txt",'w+',encoding="utf-8") as f:
            for i in range(len(documents)):
                f.write(documents[i].text.replace("  ", "")+"\n")
                
        
        with open(filename+".txt", "r", encoding="utf-8") as f:
            content += f.read()
    else:
        for page in documents:
            content += page.text.replace("  ","") + "\n" 
    
    with open("data/question.txt", "r") as f:
        questions = f.read().split("\n")
    model_name = "deepset/roberta-base-squad2"

    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    ans = {}
    for question in questions:
        print(question)
        QA_input = {
            'question': f'{question}: ',
            'context': content
        }
        output = nlp(QA_input)
        if output["score"] > 0.8:
            ans[question] = output.answer
    return ans

def main():
    results = []
    with open(sys.argv[1], "r") as f:
        filename = f.readline().strip()
        while filename:
            result = summarize(filename, writeFile=False)
            results.append((filename, result))
            
            filename = f.readline().strip()
            
    with open(sys.argv[2], "w+") as summaries:
        for name, res in results:
            summaries.write(f"{name}:\n")
            for key,value in res:
                summaries.write(f"\t{key}: value\n")
            summaries.write("\n")
    
if __name__ == "__main__":
    main()