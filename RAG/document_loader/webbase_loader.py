from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()
import os


model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key = os.getenv("GEMINI_API_KEY")
    
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Answer the following question \n {question} from the following text \n {text}.",
        ),
        
        

    ]
)

parser = StrOutputParser()

url = "https://www.flipkart.com/warnercann-hi5-pro-intel-core-i5-12th-gen-8-gb-256-gb-ssd-windows-11-pro-business-laptop/p/itm03fe2b09295a7?pid=COMH6BCGNK3NGTE4&lid=LSTCOMH6BCGNK3NGTE4S7UJUE&marketplace=FLIPKART&fm=neo%2Fmerchandising&iid=M_b7d81e91-863c-4bf6-91b0-6f35cafc7514_2_LCNPC4VE1P08_MC.COMH6BCGNK3NGTE4&ppt=None&ppn=None&ssid=peul83usls0000001749633797442&otracker=clp_pmu_v2_Core%2Bi5%2BLaptops_1_2.productCard.PMU_V2_WarnerCann%2BHi5%2Bpro%2BIntel%2BCore%2Bi5%2B12th%2BGen%2B-%2B%25288%2BGB%252F256%2BGB%2BSSD%252FWindows%2B11%2BPro%2529%2BHi5%2BPro%2BBusiness%2BLaptop%2BBusiness%2BLaptop_laptops-store_COMH6BCGNK3NGTE4_neo%2Fmerchandising_0&otracker1=clp_pmu_v2_PINNED_neo%2Fmerchandising_Core%2Bi5%2BLaptops_LIST_productCard_cc_1_NA_view-all&cid=COMH6BCGNK3NGTE4"
loader = WebBaseLoader(url)


docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question':'available offers','text':docs[0].page_content}))
# print(docs[0].page_content)