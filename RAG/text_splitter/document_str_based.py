from langchain.text_splitter import RecursiveCharacterTextSplitter,Language


text = """
class ProductDatabase:
    "Handles product information and inventory"
    
    def __init__(self, products_file: str):
        self.products_file = products_file
        self.products = self.load_products()
    
    def load_products(self) -> Dict:
        "Load products from JSON file"
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
    
    
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0,
)

result = splitter.split_text(text)

print(len(result))
print(result[0])