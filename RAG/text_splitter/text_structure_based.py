from langchain.text_splitter import RecursiveCharacterTextSplitter


text = """
Shunga would form the Shunga Empire in the north and north-east of the subcontinent, while the Greco-Bactrian Kingdom would claim the north-west and found the Indo-Greek Kingdom. 
Various parts of India were ruled by numerous dynasties, including the Gupta Empire, in the 4th to 6th centuries CE. 
This period, witnessing a Hindu religious and intellectual resurgence is known as the Classical or Golden Age of India. 
Aspects of Indian civilisation, administration, culture, and religion spread to much of Asia, which led to the establishment of Indianised kingdoms in the region, forming Greater India.[6][5] 
The most significant event between the 7th and 11th centuries was the Tripartite struggle centred on Kannauj. Southern India saw the rise of multiple imperial powers from the middle of the fifth century. The Chola dynasty conquered southern India in the 11th century. 
In the early medieval period, Indian mathematics, including Hindu numerals, influenced the development of mathematics and astronomy in the Arab world, including the creation of the Hindu-Arabic numeral system.[7]
"""


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)