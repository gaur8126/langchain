import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid
import re

# Updated LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader # Corrected import path for WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document

from flask import Flask, request, jsonify, render_template_string
import requests
from bs4 import BeautifulSoup


from dotenv import load_dotenv
load_dotenv()

# Configuration
class Config:
    # Corrected: Load API key from environment variable
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    COMPANY_NAME = "Your Company Name"
    WEBSITE_URLS = [
        "https://www.flipkart.com/laptops-store",
        # "https://yourcompany.com/about",
        # "https://yourcompany.com/contact"
    ]
    # For demo, we'll use some sample URLs
    DEMO_URLS = [
        "https://www.flipkart.com/laptops-store",  # Replace with your actual URLs
    ]

@dataclass
class Order:
    order_id: str
    product_name: str
    quantity: int
    customer_email: str
    customer_phone: str
    customer_name: str
    status: str
    timestamp: datetime
    total_price: float
    
    def to_dict(self):
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

class WebsiteDataLoader:
    """Loads and processes website data using WebBaseLoader"""
    
    def __init__(self, urls: List[str]):
        self.urls = urls
        self.documents = []
        self.vectorstore = None
        
    async def load_website_data(self):
        """Load data from website URLs"""
        try:
            # Use WebBaseLoader to load website content
            loader = WebBaseLoader(
                web_paths=self.urls,
                bs_kwargs={
                    "parse_only": BeautifulSoup.SoupStrainer(
                        ["p", "h1", "h2", "h3", "h4", "h5", "h6", "div", "span", "li"]
                    )
                }
            )
            
            documents = await asyncio.to_thread(loader.load) # Use asyncio.to_thread for blocking I/O
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            self.documents = text_splitter.split_documents(documents)
            
            print(f"Loaded {len(self.documents)} document chunks from website")
            return self.documents
            
        except Exception as e:
            print(f"Error loading website data: {e}")
            # Fallback to sample data
            return self.create_sample_documents()
    
    def create_sample_documents(self):
        """Create sample documents when website loading fails"""
        sample_data = [
            {
                "content": """
                Premium Wireless Headphones - $299.99
                
                Our flagship wireless headphones feature advanced noise cancellation technology, 
                premium sound quality, and 30-hour battery life. Perfect for music lovers and 
                professionals who demand the best audio experience.
                
                Key Features:
                - Active Noise Cancellation (ANC)
                - 30-hour battery life with quick charge
                - Bluetooth 5.0 connectivity
                - Premium leather headband
                - Touch controls
                - Compatible with all devices
                
                Available in Black, White, and Rose Gold.
                Stock: 50 units available
                """,
                "metadata": {"source": "products/headphones", "type": "product"}
            },
            {
                "content": """
                Fitness Smartwatch Pro - $199.99
                
                Advanced fitness tracking smartwatch with comprehensive health monitoring, 
                GPS tracking, and waterproof design. Track your workouts, monitor your health, 
                and stay connected throughout the day.
                
                Key Features:
                - Heart rate monitoring
                - GPS tracking for outdoor activities
                - Waterproof up to 50m
                - 7-day battery life
                - Sleep tracking
                - Smartphone notifications
                - 100+ workout modes
                
                Available in Silver, Black, and Blue.
                Stock: 30 units available
                """,
                "metadata": {"source": "products/smartwatch", "type": "product"}
            },
            {
                "content": """
                About Our Company
                
                We are a leading technology company specializing in premium consumer electronics. 
                Founded in 2020, we focus on creating innovative products that enhance your daily life.
                
                Our mission is to provide high-quality, affordable technology products with 
                exceptional customer service. We offer:
                
                - Free shipping on orders over $100
                - 30-day money-back guarantee
                - 2-year warranty on all products
                - 24/7 customer support
                - Easy returns and exchanges
                
                Contact us: support@company.com | 1-800-TECH-HELP
                """,
                "metadata": {"source": "about", "type": "company_info"}
            }
        ]
        
        documents = []
        for data in sample_data:
            doc = Document(
                page_content=data["content"],
                metadata=data["metadata"]
            )
            documents.append(doc)
        
        self.documents = documents
        return documents
    
    async def create_vectorstore(self, embeddings):
        """Create vector store from documents"""
        if not self.documents:
            await self.load_website_data()
        
        self.vectorstore = await asyncio.to_thread(
            FAISS.from_documents,
            documents=self.documents,
            embedding=embeddings
        )
        
        return self.vectorstore

class OrderManager:
    """Enhanced order management with persistence"""
    
    def __init__(self):
        self.orders = {}
        self.order_file = "orders.json"
        self.load_orders()
    
    def load_orders(self):
        """Load orders from file"""
        try:
            with open(self.order_file, 'r') as f:
                data = json.load(f)
                for order_id, order_data in data.items():
                    order_data['timestamp'] = datetime.fromisoformat(order_data['timestamp'])
                    self.orders[order_id] = Order(**order_data)
        except FileNotFoundError:
            self.orders = {}
    
    def save_orders(self):
        """Save orders to file"""
        try:
            data = {order_id: order.to_dict() for order_id, order in self.orders.items()}
            with open(self.order_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving orders: {e}")
    
    def create_order(self, product_name: str, quantity: int, 
                     customer_email: str, customer_phone: str, 
                     customer_name: str, total_price: float) -> str:
        """Create a new order"""
        order_id = f"ORD{str(uuid.uuid4())[:8].upper()}"
        
        order = Order(
            order_id=order_id,
            product_name=product_name,
            quantity=quantity,
            customer_email=customer_email,
            customer_phone=customer_phone,
            customer_name=customer_name,
            status="confirmed",
            timestamp=datetime.now(),
            total_price=total_price
        )
        
        self.orders[order_id] = order
        self.save_orders()
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        if order_id in self.orders:
            self.orders[order_id].status = "cancelled"
            self.save_orders()
            return True
        return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details"""
        return self.orders.get(order_id)
    
    def get_orders_by_email(self, email: str) -> List[Order]:
        """Get all orders for a customer"""
        return [order for order in self.orders.values() if order.customer_email == email]

class ModernSalesChatbot:
    """Modern sales chatbot using updated LangChain patterns"""
    
    def __init__(self, config: Config):
        self.config = config
        self.order_manager = OrderManager()
        
        # Initialize LLM with updated syntax
        # Using a more standard model name. Adjust as needed.
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", # Changed model for better stability
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.1
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=config.GEMINI_API_KEY
        )
        
        # Initialize website data loader
        urls = config.WEBSITE_URLS if hasattr(config, 'WEBSITE_URLS') else config.DEMO_URLS
        self.data_loader = WebsiteDataLoader(urls)
        
        # Chat history storage
        self.store = {}
        
        # Flag to indicate if the system is initialized
        self.initialized = False
        
        # Removed asyncio.create_task here, it will be called in main_chat
        
    async def initialize_system(self):
        """Initialize the RAG system"""
        try:
            # Load website data and create vector store
            await self.data_loader.load_website_data()
            self.vectorstore = await self.data_loader.create_vectorstore(self.embeddings)
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            # Setup prompts and chains
            self.setup_chains()
            self.initialized = True
            print("Sales chatbot initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing chatbot: {e}")
            # Initialize with minimal functionality
            self.setup_fallback_system()
            self.initialized = True # Mark as initialized even if it's fallback
    
    def setup_chains(self):
        """Setup LangChain chains with updated patterns"""
        
        # System prompt for the sales assistant
        system_prompt = f"""
        You are a professional sales assistant for {self.config.COMPANY_NAME}. 
        
        Your responsibilities:
        1. Answer questions about our products using the provided context
        2. Help customers place orders
        3. Assist with order cancellations and tracking
        4. Provide product recommendations
        5. Handle customer inquiries professionally
        
        IMPORTANT GUIDELINES:
        - ONLY discuss products and services from our company
        - Use the provided context to answer product questions accurately
        - For orders, collect: product name, quantity, customer name, email, phone
        - Be helpful, friendly, and professional
        - If asked about unrelated topics, politely redirect to our products
        - Always confirm order details before processing
        
        Context from our website:
        {{context}}
        
        Current conversation:
        {{chat_history}}
        
        Human: {{input}}
        
        Assistant: I'll help you with information about our products and services.
        """
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template(system_prompt)
        
        # Create the question-answer chain
        Youtube_chain = create_stuff_documents_chain(self.llm, self.prompt)
        
        # Create RAG chain
        self.rag_chain = create_retrieval_chain(self.retriever, Youtube_chain)
        
        # Setup conversational RAG with history
        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    
    def setup_fallback_system(self):
        """Setup fallback system when website loading fails"""
        print("Setting up fallback system...")
        
        # Create a simple chain without retrieval
        simple_prompt = ChatPromptTemplate.from_template(
            f"""
            You are a sales assistant for {self.config.COMPANY_NAME}. 
            Help customers with product inquiries and orders.
            
            Available products:
            1. Premium Wireless Headphones - $299.99
            2. Fitness Smartwatch Pro - $199.99
            
            Human: {{input}}
            Assistant:"""
        )
        
        self.simple_chain = simple_prompt | self.llm | StrOutputParser()
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get chat history for a session"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    async def extract_order_info(self, message: str) -> Dict[str, Any]:
        """Extract order information from user message using LLM"""
        extraction_prompt = f"""
        Extract order information from this message: "{message}"
        
        Look for:
        - Product name (e.g., "Premium Wireless Headphones", "Fitness Smartwatch Pro")
        - Quantity (default to 1 if not specified)
        - Customer name
        - Customer email
        - Customer phone
        
        Return JSON format with the extracted information, or indicate what's missing.
        If information is missing, return {{"missing": ["field1", "field2"]}}
        
        Available products:
        - Premium Wireless Headphones ($299.99)
        - Fitness Smartwatch Pro ($199.99)
        """
        
        try:
            # Use await for async LLM invocation
            response_obj = await self.llm.invoke(extraction_prompt)
            # Access the content attribute for the actual response string
            response_str = response_obj.content
            
            # Use regex to find a potential JSON string in the response
            json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if json_match:
                extracted_json_str = json_match.group(0)
                # Attempt to parse the JSON
                extracted_info = json.loads(extracted_json_str)
                # Ensure all expected fields are present or marked as missing
                required_fields = ["product_name", "quantity", "customer_name", "customer_email", "customer_phone"]
                missing_fields = [f for f in required_info.get("missing", []) if extracted_info.get(f) is None] # Better way to check
                
                if missing_fields:
                    return {"needs_more_info": True, "missing": missing_fields}
                else:
                    return extracted_info
            else:
                return {"needs_more_info": True, "missing": ["product_name", "quantity", "customer_name", "customer_email", "customer_phone"]} # Fallback if no JSON found

        except json.JSONDecodeError as e:
            print(f"JSON parsing error in extract_order_info: {e}")
            return {"needs_more_info": True, "missing": ["product_name", "quantity", "customer_name", "customer_email", "customer_phone"]} # Indicate missing info on parsing error
        except Exception as e:
            print(f"Error in extract_order_info: {e}")
            return {"needs_more_info": True, "missing": ["product_name", "quantity", "customer_name", "customer_email", "customer_phone"]} # General error fallback
    
    def process_order(self, order_info: Dict) -> str:
        """Process a complete order"""
        try:
            # Validate product and calculate price
            product_prices = {
                "premium wireless headphones": 299.99,
                "headphones": 299.99,
                "wireless headphones": 299.99,
                "fitness smartwatch pro": 199.99,
                "smartwatch": 199.99,
                "watch": 199.99
            }
            
            product_name = order_info.get("product_name", "").lower()
            quantity = int(order_info.get("quantity", 1))
            
            # Find matching product
            price = None
            actual_product_name = None
            for key, value in product_prices.items():
                if key in product_name or product_name in key:
                    price = value
                    actual_product_name = key.title()
                    break
            
            if not price:
                return "I couldn't find that product. Our available products are Premium Wireless Headphones ($299.99) and Fitness Smartwatch Pro ($199.99)."
            
            total_price = price * quantity
            
            # Create order
            order_id = self.order_manager.create_order(
                product_name=actual_product_name,
                quantity=quantity,
                customer_email=order_info["customer_email"], # Corrected key
                customer_phone=order_info["customer_phone"], # Corrected key
                customer_name=order_info["customer_name"],   # Corrected key
                total_price=total_price
            )
            
            return f"""
🎉 Order Confirmed!

Order ID: {order_id}
Product: {actual_product_name}
Quantity: {quantity}
Total: ${total_price:.2f}

Customer: {order_info['customer_name']}
Email: {order_info['customer_email']}
Phone: {order_info['customer_phone']}

You'll receive a confirmation email shortly. Thank you for your purchase!

Is there anything else I can help you with?
            """.strip()
            
        except Exception as e:
            print(f"Error processing order: {e}") # Added print for debugging
            return f"I encountered an error processing your order: {str(e)}. Please try again or contact our support team."
    
    def handle_order_cancellation(self, message: str) -> str:
        """Handle order cancellation"""
        # Extract order ID
        order_id_pattern = r'ORD[A-Z0-9]{8}'
        match = re.search(order_id_pattern, message.upper())
        
        if not match:
            return "To cancel an order, please provide your order ID (e.g., ORD12345678)."
        
        order_id = match.group()
        
        if self.order_manager.cancel_order(order_id):
            order = self.order_manager.get_order(order_id)
            if order: # Ensure order exists before accessing its attributes
                return f"""
Order {order_id} has been successfully cancelled.

Cancelled Order Details:
- Product: {order.product_name}
- Quantity: {order.quantity}
- Total: ${order.total_price:.2f}

You'll receive a cancellation confirmation email shortly.
                """.strip()
            else:
                return f"Order {order_id} was cancelled, but I couldn't retrieve its details."
        else:
            return f"I couldn't find order {order_id}. Please check your order ID and try again."
    
    async def get_response(self, message: str, session_id: str) -> str:
        """Get chatbot response"""
        # Wait until the system is initialized
        while not self.initialized:
            print("Chatbot is still initializing. Please wait...")
            await asyncio.sleep(0.5) # Reduced sleep time
        
        try:
            # Check for order-related intents
            message_lower = message.lower()
            
            # Handle order cancellation
            if any(word in message_lower for word in ['cancel', 'refund', 'return']) and 'ord' in message_lower:
                return self.handle_order_cancellation(message)
            
            # Handle order placement (simplified detection)
            if any(word in message_lower for word in ['order', 'buy', 'purchase', 'want to get']):
                # Try to extract order info
                order_info = await self.extract_order_info(message) # Await the async method
                if not order_info.get("needs_more_info"):
                    # Check if all required fields were extracted
                    required_fields_extracted = all(order_info.get(field) for field in ["product_name", "quantity", "customer_name", "customer_email", "customer_phone"])
                    if required_fields_extracted:
                        return self.process_order(order_info)
                    else:
                        missing_fields = order_info.get("missing", [])
                        return f"""
To place an order, I still need the following information: {", ".join(missing_fields)}.
Could you please provide them?
For example: "I want to order 1 wireless headphones. My name is John Smith, email john@email.com, phone 123-456-7890"
                        """.strip()

                return """
To place an order, I'll need:
1. Which product you'd like (Wireless Headphones or Smartwatch)
2. Quantity
3. Your full name
4. Email address
5. Phone number

For example: "I want to order 1 wireless headphones. My name is John Smith, email john@email.com, phone 123-456-7890"
                """.strip()
            
            # Use RAG chain for product questions
            if hasattr(self, 'conversational_rag_chain'):
                config = {"configurable": {"session_id": session_id}}
                response = await self.conversational_rag_chain.invoke(
                    {"input": message},
                    config=config
                )
                return response["answer"]
            
            # Fallback to simple chain
            elif hasattr(self, 'simple_chain'):
                response = await self.simple_chain.invoke(
                    {"input": message}
                )
                return response
            
            else:
                return "I'm still initializing. Please try again in a moment."
                
        except Exception as e:
            print(f"Error in get_response: {e}")
            return "I apologize, but I encountered an error. Please try rephrasing your question or contact our support team."

if __name__ == "__main__":
    config = Config()
    # chatbot = ModernSalesChatbot(config) # Removed direct initialization

    # Main loop for terminal interaction
    async def main_chat():
        chatbot = ModernSalesChatbot(config) # Initialize the chatbot inside the async function
        
        # Now, call initialize_system directly after initializing the object
        # It's okay to call it directly as await here because main_chat is already awaited
        await chatbot.initialize_system() 

        # Give the chatbot a moment to ensure setup is complete if not already
        while not chatbot.initialized:
            print("Chatbot is still initializing. Please wait...")
            await asyncio.sleep(0.5) # wait a bit if it's still busy

        print("\n--- Sales Chatbot ---")
        print("Type 'q' to quit.")
        print("---------------------")

        # Using a fixed session_id for this simple terminal demo
        session_id = "terminal_user_session"

        while True:
            user_message = input("You: ")

            if user_message.lower() == "q":
                print("Exiting chat. Goodbye!")
                break

            response = await chatbot.get_response(user_message, session_id)
            print(f"Bot: {response}")

    # Run the async main chat function
    asyncio.run(main_chat())