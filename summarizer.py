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
    
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = splitter.split_text(content)
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    res = []
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    for chunk in chunks:
        summary = summarizer(chunk, max_length = len(tokenizer(chunk).input_ids)-1, min_length = len(tokenizer(chunk).input_ids) // 10)
        res.append(summary[0]["summary_text"])
        
    while(len(res) > 1):
        # print("1 iteration")
        result = "\n".join(res)
        temp = []
        temp_chunks = splitter.split_text(result)
        for temp_chunk in temp_chunks:
            summary = summarizer(temp_chunk, max_length = len(tokenizer(temp_chunk).input_ids)-1, min_length = len(tokenizer(temp_chunk).input_ids) // 10)
            temp.append(summary[0]["summary_text"])
        res = temp
    return(res[0])

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
            summaries.write(f"{name}: {res}\n")
    
if __name__ == "__main__":
    main()