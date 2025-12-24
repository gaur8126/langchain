from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",output_dimensionality=300)

docs = [
    "Virat Kohli is a world-class batter and former all-format captain of India, known for his chasing ability and many international centuries.",
    "Rohit Sharma is a prolific opening batter and current India captain in Tests and ODIs, famous for his double hundreds and success in ICC tournaments.",
    "Jasprit Bumrah is a leading right-arm fast bowler for India, celebrated for his yorkers and achievements like more than 400 international wickets.",
    "Ravindra Jadeja is an all-rounder who contributes with left-arm spin, useful lower-order batting, and exceptional fielding for India.",
    "Rishabh Pant is an aggressive left-hand wicketkeeper-batter who plays impactful innings for India in all formats after making a strong comeback from injury."
]

query = "tell me about Rishabh Pant"
docs_embeddings = embedding.embed_documents(docs)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],docs_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(docs[index])
print("similarity score is: ",score)
