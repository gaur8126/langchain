# Sales Chatbot with LangChain & Gemini API
# This chatbot handles product queries, orders, and cancellations

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from langchain.llms import GooglePalm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from flask import Flask, request, jsonify, render_template_string
import google.generativeai as genai
from dotenv import load_dotenv
import os 

load_dotenv()

# Configuration
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    COMPANY_NAME = "Ignis"
    PRODUCTS_DATA_PATH = "products.json"

@dataclass
class Order:
    order_id: str
    product_name: str
    quantity: int
    customer_email: str
    customer_phone: str
    status: str
    timestamp: datetime
    total_price: float

class ProductDatabase:
    """Handles product information and inventory"""
    
    def __init__(self, products_file: str):
        self.products_file = products_file
        self.products = self.load_products()
    
    def load_products(self) -> Dict:
        """Load products from JSON file"""
        try:
            with open(self.products_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Sample product data if file doesn't exist
            sample_products = {
                "products": [
                    {
                        "id": "P001",
                        "name": "Premium Wireless Headphones",
                        "price": 299.99,
                        "description": "High-quality wireless headphones with noise cancellation, 30-hour battery life, and premium sound quality.",
                        "features": ["Noise Cancellation", "30-hour battery", "Bluetooth 5.0", "Quick charge"],
                        "category": "Electronics",
                        "stock": 50,
                        "images": ["headphones1.jpg", "headphones2.jpg"]
                    },
                    {
                        "id": "P002",
                        "name": "Fitness Smartwatch",
                        "price": 199.99,
                        "description": "Advanced fitness tracking smartwatch with heart rate monitoring, GPS, and waterproof design.",
                        "features": ["Heart Rate Monitor", "GPS Tracking", "Waterproof", "7-day battery"],
                        "category": "Wearables",
                        "stock": 30,
                        "images": ["watch1.jpg", "watch2.jpg"]
                    }
                ]
            }
            with open(self.products_file, 'w') as f:
                json.dump(sample_products, f, indent=2)
            return sample_products
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        for product in self.products["products"]:
            if product["id"] == product_id:
                return product
        return None
    
    def get_product_by_name(self, product_name: str) -> Optional[Dict]:
        for product in self.products["products"]:
            if product_name.lower() in product["name"].lower():
                return product
        return None
    
    def search_products(self, query: str) -> List[Dict]:
        results = []
        query_lower = query.lower()
        for product in self.products["products"]:
            if (query_lower in product["name"].lower() or 
                query_lower in product["description"].lower() or
                query_lower in product["category"].lower()):
                results.append(product)
        return results

class OrderManager:
    """Handles order creation, tracking, and cancellation"""
    
    def __init__(self):
        self.orders = {}
        self.order_counter = 1000
    
    def create_order(self, product_name: str, quantity: int, 
                    customer_email: str, customer_phone: str, 
                    total_price: float) -> str:
        """Create a new order"""
        order_id = f"ORD{self.order_counter}"
        self.order_counter += 1
        
        order = Order(
            order_id=order_id,
            product_name=product_name,
            quantity=quantity,
            customer_email=customer_email,
            customer_phone=customer_phone,
            status="pending",
            timestamp=datetime.now(),
            total_price=total_price
        )
        
        self.orders[order_id] = order
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        if order_id in self.orders:
            self.orders[order_id].status = "cancelled"
            return True
        return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details"""
        return self.orders.get(order_id)

class SalesChatbot:
    """Main chatbot class that handles all interactions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.product_db = ProductDatabase(config.PRODUCTS_DATA_PATH)
        self.order_manager = OrderManager()
        
        # Configure Gemini API
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Set up the chatbot prompt
        self.setup_chatbot_prompt()
    
    def setup_chatbot_prompt(self):
        """Setup the system prompt for the chatbot"""
        self.system_prompt = f"""
        You are a professional sales assistant for {self.config.COMPANY_NAME}. Your primary role is to:
        
        1. Answer questions about our products only
        2. Help customers place orders
        3. Assist with order cancellations
        4. Provide product recommendations
        5. Handle customer inquiries professionally
        
        IMPORTANT GUIDELINES:
        - Only discuss products from our catalog
        - Don't answer questions unrelated to our products or services
        - Be helpful, friendly, and professional
        - Always confirm order details before processing
        - For order cancellations, verify the order ID
        - If asked about topics outside our products, politely redirect to our products
        
        Available products: {json.dumps(self.product_db.products, indent=2)}
        
        When customers want to place an order, collect:
        - Product name and quantity
        - Customer email
        - Customer phone number
        
        Always respond in a conversational, helpful manner.
        """
    
    def extract_intent(self, user_message: str) -> str:
        """Extract user intent from message"""
        user_message_lower = user_message.lower()
        
        if any(word in user_message_lower for word in ['order', 'buy', 'purchase', 'get']):
            return "place_order"
        elif any(word in user_message_lower for word in ['cancel', 'refund', 'return']):
            return "cancel_order"
        elif any(word in user_message_lower for word in ['track', 'status', 'order id']):
            return "track_order"
        elif any(word in user_message_lower for word in ['product', 'price', 'feature', 'spec']):
            return "product_inquiry"
        else:
            return "general_inquiry"
    
    def handle_product_inquiry(self, user_message: str) -> str:
        """Handle product-related questions"""
        # Search for relevant products
        products = self.product_db.search_products(user_message)
        
        if not products:
            return "I couldn't find any products matching your query. Could you please be more specific about what you're looking for?"
        
        response = "Here are the products I found:\n\n"
        for product in products[:3]:  # Limit to top 3 results
            response += f"**{product['name']}** - ${product['price']}\n"
            response += f"{product['description']}\n"
            response += f"Features: {', '.join(product['features'])}\n"
            response += f"Stock: {product['stock']} available\n\n"
        
        response += "Would you like more details about any of these products or would you like to place an order?"
        return response
    
    def handle_order_placement(self, user_message: str, context: Dict) -> str:
        """Handle order placement process"""
        # Extract order information using Gemini
        prompt = f"""
        Extract order information from this message: "{user_message}"
        
        Look for:
        - Product name
        - Quantity
        - Customer email
        - Customer phone
        
        Return as JSON format or indicate what information is missing.
        Available products: {[p['name'] for p in self.product_db.products['products']]}
        """
        
        try:
            response = self.model.generate_content(prompt)
            
            # For demo purposes, we'll use a simplified approach
            # In production, you'd parse the Gemini response more carefully
            
            return """To place an order, I'll need the following information:
            1. Which product would you like to order?
            2. How many items?
            3. Your email address
            4. Your phone number
            
            Please provide these details and I'll process your order immediately!"""
            
        except Exception as e:
            return "I'd be happy to help you place an order! Please tell me which product you'd like and how many, along with your contact information."
    
    def process_complete_order(self, product_name: str, quantity: int, 
                             email: str, phone: str) -> str:
        """Process a complete order"""
        # Find the product
        product = self.product_db.get_product_by_name(product_name)
        if not product:
            return f"Sorry, I couldn't find a product named '{product_name}'. Please check the product name."
        
        # Check stock
        if product['stock'] < quantity:
            return f"Sorry, we only have {product['stock']} units of {product['name']} in stock."
        
        # Calculate total
        total_price = product['price'] * quantity
        
        # Create order
        order_id = self.order_manager.create_order(
            product_name=product['name'],
            quantity=quantity,
            customer_email=email,
            customer_phone=phone,
            total_price=total_price
        )
        
        return f"""Order confirmed! 🎉
        
        Order ID: {order_id}
        Product: {product['name']}
        Quantity: {quantity}
        Total: ${total_price:.2f}
        
        You'll receive a confirmation email at {email} shortly.
        
        Thank you for your purchase! Is there anything else I can help you with?"""
    
    def handle_order_cancellation(self, user_message: str) -> str:
        """Handle order cancellation"""
        # Extract order ID from message
        words = user_message.split()
        order_id = None
        
        for word in words:
            if word.startswith("ORD"):
                order_id = word
                break
        
        if not order_id:
            return "To cancel an order, please provide your order ID (e.g., ORD1001)."
        
        if self.order_manager.cancel_order(order_id):
            return f"Order {order_id} has been successfully cancelled. You'll receive a confirmation email shortly."
        else:
            return f"I couldn't find order {order_id}. Please check your order ID and try again."
    
    def get_response(self, user_message: str, context: Dict = None) -> str:
        """Get chatbot response based on user message"""
        if context is None:
            context = {}
        
        # Extract intent
        intent = self.extract_intent(user_message)
        
        # Handle different intents
        if intent == "product_inquiry":
            return self.handle_product_inquiry(user_message)
        elif intent == "place_order":
            return self.handle_order_placement(user_message, context)
        elif intent == "cancel_order":
            return self.handle_order_cancellation(user_message)
        elif intent == "track_order":
            return "To track your order, please provide your order ID (e.g., ORD1001)."
        else:
            # Use Gemini to generate response with context
            try:
                full_prompt = f"""
                {self.system_prompt}
                
                User message: {user_message}
                
                Respond as a sales assistant for our company. Keep responses focused on our products and services.
                """
                
                response = self.model.generate_content(full_prompt)
                return response.text
            
            except Exception as e:
                return "I'm here to help you with our products! What would you like to know about our headphones or smartwatch?"

# Flask Web Application
app = Flask(__name__)

# Initialize chatbot
config = Config()
chatbot = SalesChatbot(config)

# Store user sessions
user_sessions = {}

@app.route('/')
def index():
    """Main chatbot page"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{company_name}} - Sales Assistant</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            
            .chat-container {
                width: 90%;
                max-width: 800px;
                height: 80vh;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .chat-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
            }
            
            .chat-header h1 {
                font-size: 1.5rem;
                margin-bottom: 5px;
            }
            
            .chat-header p {
                opacity: 0.9;
                font-size: 0.9rem;
            }
            
            .chat-messages {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                background: #f8f9fa;
            }
            
            .message {
                margin-bottom: 15px;
                display: flex;
                align-items: flex-start;
            }
            
            .message.user {
                justify-content: flex-end;
            }
            
            .message-content {
                max-width: 70%;
                padding: 12px 16px;
                border-radius: 18px;
                position: relative;
            }
            
            .message.bot .message-content {
                background: white;
                border: 1px solid #e0e0e0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }
            
            .message.user .message-content {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .chat-input-container {
                padding: 20px;
                background: white;
                border-top: 1px solid #e0e0e0;
            }
            
            .chat-input-form {
                display: flex;
                gap: 10px;
            }
            
            .chat-input {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e0e0e0;
                border-radius: 25px;
                font-size: 16px;
                outline: none;
                transition: border-color 0.3s;
            }
            
            .chat-input:focus {
                border-color: #667eea;
            }
            
            .send-button {
                padding: 12px 24px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-weight: 600;
                transition: transform 0.2s;
            }
            
            .send-button:hover {
                transform: translateY(-2px);
            }
            
            .send-button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .typing-indicator {
                display: none;
                padding: 12px 16px;
                background: white;
                border-radius: 18px;
                border: 1px solid #e0e0e0;
                max-width: 70px;
            }
            
            .typing-dots {
                display: flex;
                gap: 4px;
            }
            
            .typing-dot {
                width: 8px;
                height: 8px;
                background: #667eea;
                border-radius: 50%;
                animation: typing 1.4s infinite;
            }
            
            .typing-dot:nth-child(2) {
                animation-delay: 0.2s;
            }
            
            .typing-dot:nth-child(3) {
                animation-delay: 0.4s;
            }
            
            @keyframes typing {
                0%, 60%, 100% {
                    transform: translateY(0);
                }
                30% {
                    transform: translateY(-10px);
                }
            }
            
            .welcome-message {
                text-align: center;
                color: #666;
                margin: 20px 0;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>{{company_name}} Sales Assistant</h1>
                <p>Ask me about our products, place orders, or cancel existing orders!</p>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="welcome-message">
                    👋 Welcome! I'm here to help you with our products. What would you like to know?
                </div>
            </div>
            
            <div class="chat-input-container">
                <form class="chat-input-form" id="chatForm">
                    <input type="text" class="chat-input" id="messageInput" 
                           placeholder="Ask about our products, place an order, or get help..." 
                           autocomplete="off" required>
                    <button type="submit" class="send-button" id="sendButton">Send</button>
                </form>
            </div>
        </div>
        
        <script>
            const chatMessages = document.getElementById('chatMessages');
            const chatForm = document.getElementById('chatForm');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            
            chatForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const message = messageInput.value.trim();
                if (!message) return;
                
                // Add user message
                addMessage(message, 'user');
                messageInput.value = '';
                sendButton.disabled = true;
                
                // Show typing indicator
                showTyping();
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ message: message })
                    });
                    
                    const data = await response.json();
                    hideTyping();
                    addMessage(data.response, 'bot');
                } catch (error) {
                    hideTyping();
                    addMessage('Sorry, I encountered an error. Please try again.', 'bot');
                } finally {
                    sendButton.disabled = false;
                    messageInput.focus();
                }
            });
            
            function addMessage(text, sender) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.innerHTML = text.replace(/\n/g, '<br>');
                
                messageDiv.appendChild(contentDiv);
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function showTyping() {
                const typingDiv = document.createElement('div');
                typingDiv.id = 'typingIndicator';
                typingDiv.className = 'message bot';
                typingDiv.innerHTML = `
                    <div class="typing-indicator" style="display: block;">
                        <div class="typing-dots">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                `;
                chatMessages.appendChild(typingDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function hideTyping() {
                const typingIndicator = document.getElementById('typingIndicator');
                if (typingIndicator) {
                    typingIndicator.remove();
                }
            }
            
            // Focus on input when page loads
            messageInput.focus();
        </script>
    </body>
    </html>
    ''', company_name=config.COMPANY_NAME)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        # Get session ID (in production, use proper session management)
        session_id = request.remote_addr
        
        # Get chatbot response
        response = chatbot.get_response(user_message)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'response': 'Sorry, I encountered an error. Please try again.',
            'status': 'error'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print(f"Starting {config.COMPANY_NAME} Sales Chatbot...")
    print("Make sure to set your Gemini API key in the Config class!")
    app.run(debug=True, host='0.0.0.0', port=5000)