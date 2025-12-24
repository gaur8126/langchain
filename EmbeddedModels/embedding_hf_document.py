from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")

document = ["Dehli is the capital of india",
            "Paris is the capital of france",
            "Washington DC is the capital of USA"]

vector = embedding.embed_documents(document)

print(str(vector))